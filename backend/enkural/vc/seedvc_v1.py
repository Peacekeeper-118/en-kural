"""Seed-VC v1 engine (zero-shot voice conversion).

Pipeline:  source audio → Whisper-small encoder (multilingual speech content)
           → length regulator → DiT / flow-matching mel generation conditioned on
           the target's CAMPPlus speaker embedding + reference mel prompt
           → BigVGAN vocoder → waveform.

Timing is preserved frame-for-frame (the content sequence keeps the source's
length), so pauses, rhythm and emphasis stay where they were.  Pitch control
uses the F0-conditioned 44.1 kHz model, which explicitly carries the source's
intonation contour shifted to the target's pitch level.

A profile may carry a fine-tuned copy of the DiT + length regulator (see
``enkural.vc.finetune``); its path arrives as ``target.meta["dit_checkpoint"]``
and is swapped into the base model for that conversion only.
"""
from __future__ import annotations

import contextlib
import copy
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import config
from ..audio.io import AudioClip, load_audio, write_wav
from . import fast_ops
from .base import ConversionSettings, ProgressFn, StageTimer, TargetVoice, VoiceConverter
from .hf import HF_CACHE, hf_download, setup_seedvc
from .registry import register

log = logging.getLogger(__name__)

QUALITY_STEPS = {"draft": 10, "standard": 25, "high": 50}
# Flow-matching sampler.  "legacy" reproduces upstream exactly (uniform steps,
# guidance on every step); ENKURAL_SAMPLER=legacy forces it for A/B comparisons.
LEGACY_SAMPLER = {"schedule": "linear", "sway": 0.0, "cfg_from": 0.0, "cfg_until": 1.0, "steps": QUALITY_STEPS}
# Tuned with scripts/benchmark.py against 50 uniform steps from the same noise
# (eval_runs/sampler-*): the sway grid needs 16 steps to match uniform 25, and
# guidance matters in the early (noisy) steps but hardly after t = 0.75.
SAMPLER = {"schedule": "sway", "sway": -1.0, "cfg_from": 0.0, "cfg_until": 0.75,
           "steps": {"draft": 8, "standard": 16, "high": 30}}
REPO = "Plachta/Seed-VC"
BASE_CKPT = ("DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth", "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
F0_CKPT = ("DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth", "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")
BIGVGAN_22K = "nvidia/bigvgan_v2_22khz_80band_256x"
BIGVGAN_44K = "nvidia/bigvgan_v2_44khz_128band_512x"
# the modules fine-tuning updates (Whisper, CAMPPlus and BigVGAN stay frozen)
TRAINABLE_MODULES = ("cfm", "length_regulator")


def load_matching_state(module, state: Dict) -> int:
    """Load the tensors whose names and shapes match; returns how many loaded.
    (Upstream checkpoints carry cache buffers sized for another batch size.)"""
    own = module.state_dict()
    keep = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    module.load_state_dict(keep, strict=False)
    return len(keep)


def time_grid(steps: int, schedule: str = "linear", sway: float = 0.0):
    """ODE time points from 0 (noise) to 1 (mel).  "sway" (F5-TTS) spends more
    of the step budget early, where the flow is least straight."""
    import torch

    t = torch.linspace(0, 1, steps + 1)
    if schedule == "sway" and sway:
        t = t + sway * (torch.cos(torch.pi / 2 * t) - 1 + t)
    return t


def similarity_to_cfg(similarity: float) -> float:
    """0..100 → classifier-free-guidance rate 0.40..0.85 (0.7 ≈ upstream default)."""
    return round(0.40 + 0.45 * float(np.clip(similarity, 0, 100)) / 100.0, 3)


@register
class SeedVCv1(VoiceConverter):
    id = "seedvc_v1"
    name = "Seed-VC v1 · Whisper content + DiT + BigVGAN"
    description = (
        "Zero-shot neural voice conversion. Multilingual Whisper speech features carry "
        "the words and delivery; a CAMPPlus speaker embedding averaged over all your "
        "samples plus a clean reference clip carry your identity. Timing is preserved."
    )
    output_sr = 22050
    capabilities = {"similarity": True, "pitch": True, "speed": True, "multi_reference": True}

    REF_MAX_SECONDS = 12.0   # prompt length; leaves ~18 s of source per DiT window

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._prompt_cache: Dict[Tuple, Dict] = {}
        self._f0_loaded = False
        self._base_state: Optional[Dict] = None   # CPU copy of the base DiT, taken on first swap
        self._active_ckpt: Optional[str] = None   # None = base weights
        legacy = os.environ.get("ENKURAL_SAMPLER", "").lower() == "legacy"
        self.sampler: Dict = copy.deepcopy(LEGACY_SAMPLER if legacy else SAMPLER)
        self.flash_attention = not legacy
        self._flash = None

    @classmethod
    def available(cls) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            import munch  # noqa: F401
            import einops  # noqa: F401
        except Exception:
            return False
        return (config.SEEDVC_DIR / "modules").exists()

    # ------------------------------------------------------------------ loading
    def load(self, device: str) -> None:
        import torch
        import yaml

        setup_seedvc()
        from modules.audio import mel_spectrogram
        from modules.bigvgan import bigvgan
        from modules.campplus.DTDNN import CAMPPlus
        from modules.commons import build_model, load_checkpoint, recursive_munch
        from transformers import AutoFeatureExtractor, WhisperModel

        with self._lock:
            self.device = torch.device(device)
            ckpt, cfg_path = hf_download(REPO, *BASE_CKPT)
            cfg = yaml.safe_load(open(cfg_path))
            params = recursive_munch(cfg["model_params"])
            model = build_model(params, stage="DiT")
            sp = cfg["preprocess_params"]["spect_params"]
            self.sr = int(cfg["preprocess_params"]["sr"])
            self.hop = int(sp["hop_length"])
            model, _, _, _ = load_checkpoint(model, None, ckpt, load_only_params=True, ignore_modules=[], is_distributed=False)
            for k in model:
                model[k].eval()
                model[k].to(self.device)
            model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
            self.model = model
            mel_args = dict(n_fft=sp["n_fft"], win_size=sp["win_length"], hop_size=sp["hop_length"],
                            num_mels=sp["n_mels"], sampling_rate=self.sr, fmin=0, fmax=None, center=False)
            self.to_mel = lambda x: mel_spectrogram(x, **mel_args)

            whisper_name = getattr(params.speech_tokenizer, "whisper_name", "openai/whisper-small")
            wdtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.whisper = WhisperModel.from_pretrained(whisper_name, torch_dtype=wdtype, cache_dir=str(HF_CACHE)).to(self.device)
            del self.whisper.decoder
            self.whisper.eval()
            self.whisper_fe = AutoFeatureExtractor.from_pretrained(whisper_name, cache_dir=str(HF_CACHE))

            camp_path = hf_download("funasr/campplus", "campplus_cn_common.bin")
            self.campplus = CAMPPlus(feat_dim=80, embedding_size=192)
            self.campplus.load_state_dict(torch.load(camp_path, map_location="cpu"))
            self.campplus.eval().to(self.device)

            self.bigvgan = bigvgan.BigVGAN.from_pretrained(BIGVGAN_22K, use_cuda_kernel=False, cache_dir=str(HF_CACHE))
            self.bigvgan.remove_weight_norm()
            self.bigvgan = self.bigvgan.eval().to(self.device)
            self._speed_up_vocoder(self.bigvgan, 80)
            if self.flash_attention:
                self._flash = fast_ops.flash_attention_switch()
            if fast_ops.compile_enabled(self.device) and os.environ.get("ENKURAL_COMPILE_DIT") == "1":
                self._compile_dit(model.cfm.estimator)

            self._prompt_cache.clear()
            self._base_state, self._active_ckpt = None, None
            self._loaded = True
            log.info("Seed-VC v1 loaded on %s (sr=%d, hop=%d)", self.device, self.sr, self.hop)

    def _compile_dit(self, est) -> None:
        """Opt-in (ENKURAL_COMPILE_DIT=1): ~25 % faster DiT steps for ~8 s of
        compiling per server start (~35 s the very first time).  Pays off for
        long recordings; the forward is compiled in place so checkpoint swaps and
        state dicts are unaffected."""
        import torch

        t0 = time.perf_counter()
        est.forward = torch.compile(est.forward, dynamic=True)
        T, C = 64, est.cond_projection.in_features
        flash = self._flash.unpadded() if self._flash is not None else contextlib.nullcontext()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16), flash:
            for b in (2, 1):  # guided and unguided steps
                x = torch.zeros(b, est.in_channels, T, device=self.device)
                est(x, x, torch.LongTensor([T]).to(self.device), torch.zeros(b, device=self.device),
                    torch.zeros(b, 192, device=self.device), torch.zeros(b, T, C, device=self.device))
        log.info("DiT compiled in %.1f s", time.perf_counter() - t0)

    def _speed_up_vocoder(self, vocoder, n_mels: int) -> None:
        """Polyphase anti-aliased activations, compiled with Triton on CUDA
        (exact; ~2× faster vocoder).  The warm-up compiles once per machine
        (~5 s, cached under models/torch) so conversions never pay for it."""
        import torch

        if not fast_ops.compile_enabled(self.device):
            return  # uncompiled, the polyphase form is slower than the original
        n = fast_ops.fuse_bigvgan_activations(vocoder, compile_=True)
        if n:
            t0 = time.perf_counter()
            with torch.inference_mode():
                vocoder(torch.zeros(1, n_mels, 64, device=self.device))
            log.info("BigVGAN activations fused and compiled in %.1f s", time.perf_counter() - t0)

    def _load_f0_models(self) -> None:
        """Lazy: F0-conditioned 44.1 kHz model + BigVGAN 44k + RMVPE pitch extractor."""
        if self._f0_loaded:
            return
        import torch
        import yaml
        from modules.audio import mel_spectrogram
        from modules.bigvgan import bigvgan
        from modules.commons import build_model, load_checkpoint, recursive_munch
        from modules.rmvpe import RMVPE

        with self._lock:
            ckpt, cfg_path = hf_download(REPO, *F0_CKPT)
            cfg = yaml.safe_load(open(cfg_path))
            params = recursive_munch(cfg["model_params"])
            model = build_model(params, stage="DiT")
            sp = cfg["preprocess_params"]["spect_params"]
            self.sr_f0 = int(cfg["preprocess_params"]["sr"])
            self.hop_f0 = int(sp["hop_length"])
            model, _, _, _ = load_checkpoint(model, None, ckpt, load_only_params=True, ignore_modules=[], is_distributed=False)
            for k in model:
                model[k].eval()
                model[k].to(self.device)
            model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
            self.model_f0 = model
            mel_args = dict(n_fft=sp["n_fft"], win_size=sp["win_length"], hop_size=sp["hop_length"],
                            num_mels=sp["n_mels"], sampling_rate=self.sr_f0, fmin=0, fmax=None, center=False)
            self.to_mel_f0 = lambda x: mel_spectrogram(x, **mel_args)
            self.bigvgan44 = bigvgan.BigVGAN.from_pretrained(BIGVGAN_44K, use_cuda_kernel=False, cache_dir=str(HF_CACHE))
            self.bigvgan44.remove_weight_norm()
            self.bigvgan44 = self.bigvgan44.eval().to(self.device)
            self._speed_up_vocoder(self.bigvgan44, 128)
            rmvpe_path = hf_download("lj1995/VoiceConversionWebUI", "rmvpe.pt")
            self.rmvpe = RMVPE(rmvpe_path, is_half=False, device=self.device)
            self._f0_loaded = True
            log.info("Seed-VC v1 F0 model loaded (sr=%d)", self.sr_f0)

    def unload(self) -> None:
        with self._lock:
            for attr in ("model", "model_f0", "whisper", "campplus", "bigvgan", "bigvgan44", "rmvpe"):
                if hasattr(self, attr):
                    delattr(self, attr)
            self._prompt_cache.clear()
            self._base_state, self._active_ckpt = None, None
            self._f0_loaded = False
            self._loaded = False
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # ------------------------------------------------------------------ features
    def _whisper_features(self, audio16k):
        """Whisper encoder states for 16 kHz audio ``(1, n)``; chunked past 30 s."""
        import torch

        def encode(chunk):
            inputs = self.whisper_fe([chunk.squeeze(0).cpu().numpy()], return_tensors="pt",
                                     return_attention_mask=True, sampling_rate=16000)
            feats = self.whisper._mask_input_features(inputs.input_features, attention_mask=inputs.attention_mask).to(self.device)
            out = self.whisper.encoder(feats.to(self.whisper.encoder.dtype), head_mask=None,
                                       output_attentions=False, output_hidden_states=False, return_dict=True)
            return out.last_hidden_state.to(torch.float32)[:, : chunk.size(-1) // 320 + 1]

        if audio16k.size(-1) <= 16000 * 30:
            return encode(audio16k)
        overlap = 5
        parts, buffer, traversed = [], None, 0
        while traversed < audio16k.size(-1):
            if buffer is None:
                chunk = audio16k[:, traversed: traversed + 16000 * 30]
            else:
                chunk = torch.cat([buffer, audio16k[:, traversed: traversed + 16000 * (30 - overlap)]], dim=-1)
            feats = encode(chunk)
            parts.append(feats if traversed == 0 else feats[:, 50 * overlap:])
            buffer = chunk[:, -16000 * overlap:]
            traversed += 30 * 16000 if traversed == 0 else chunk.size(-1) - 16000 * overlap
        return torch.cat(parts, dim=1)

    def _style(self, wave16k):
        import torchaudio

        feat = torchaudio.compliance.kaldi.fbank(wave16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        return self.campplus(feat.unsqueeze(0))  # (1, 192)

    def _embed_clip(self, clip: AudioClip, chunk_s: float = 30.0):
        """Length-weighted mean CAMPPlus embedding over ≤30 s chunks of a clip."""
        import torch

        x = clip.resampled(16000).samples
        n = int(chunk_s * 16000)
        embs, weights = [], []
        for a in range(0, len(x), n):
            seg = x[a: a + n]
            if len(seg) < 16000:  # < 1 s: too short to be meaningful
                continue
            w = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                embs.append(self._style(w)[0].float())
            weights.append(len(seg))
        if not embs:
            w = torch.from_numpy(np.pad(x, (0, max(0, 16000 - len(x)))).astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                return self._style(w)[0].float()
        E = torch.stack(embs)
        wt = torch.tensor(weights, dtype=torch.float32, device=E.device)[:, None]
        return (E * wt).sum(0) / wt.sum()

    def speaker_embedding(self, clip: AudioClip) -> np.ndarray:
        with self._lock:
            return self._embed_clip(clip).cpu().numpy()

    # ------------------------------------------------------------------ target voice
    def build_target(self, reference: AudioClip, samples: List[AudioClip], out_dir: Path,
                     profile_id: str, progress: Optional[ProgressFn] = None) -> TargetVoice:
        import torch

        out_dir.mkdir(parents=True, exist_ok=True)
        ref44 = reference.resampled(44100)
        ref_path = write_wav(out_dir / "reference.wav", ref44, subtype="PCM_24")
        with self._lock, torch.inference_mode():
            sample_embs = []
            for i, s in enumerate(samples):
                sample_embs.append((self._embed_clip(s), s.duration))
                if progress:
                    progress((i + 1) / max(1, len(samples)) * 0.8, f"Embedding sample {i + 1}/{len(samples)}")
            ref_emb = self._embed_clip(reference)
            if sample_embs:
                E = torch.stack([e for e, _ in sample_embs])
                wt = torch.tensor([d for _, d in sample_embs], dtype=torch.float32, device=E.device)[:, None]
                En = E / E.norm(dim=-1, keepdim=True)
                mean_dir = (En * wt).sum(0) / wt.sum()
                style = mean_dir / mean_dir.norm() * E.norm(dim=-1).mean()
                per_sample = [float(torch.nn.functional.cosine_similarity(e, style, dim=0)) for e, _ in sample_embs]
            else:
                style = ref_emb
                per_sample = []
            ref_cos = float(torch.nn.functional.cosine_similarity(ref_emb, style, dim=0))
        np.save(out_dir / "style_v1.npy", style.cpu().numpy())
        np.save(out_dir / "style_ref_v1.npy", ref_emb.cpu().numpy())
        self._prompt_cache = {k: v for k, v in self._prompt_cache.items() if k[0] != profile_id}
        meta = {
            "style_file": "style_v1.npy",
            "reference_seconds": round(ref44.duration, 2),
            "n_samples": len(samples),
            "sample_consistency": [round(c, 3) for c in per_sample],
            "reference_consistency": round(ref_cos, 3),
        }
        if progress:
            progress(1.0, "Voice profile ready")
        return TargetVoice(profile_id=profile_id, engine_id=self.id, dir=out_dir, reference_wav=ref_path, meta=meta)

    # ------------------------------------------------------------------ fine-tuned weights
    def activate_checkpoint(self, ckpt: Optional[Path]) -> None:
        """Swap a fine-tuned DiT + length regulator into the base model
        (``None`` restores the downloaded base weights)."""
        import torch

        key = str(ckpt) if ckpt else None
        if key == self._active_ckpt:
            return
        with self._lock:
            if key is not None and self._base_state is None:
                self._base_state = {m: {n: t.detach().to("cpu", copy=True) for n, t in self.model[m].state_dict().items()}
                                    for m in TRAINABLE_MODULES}
            if key is None:
                state = self._base_state
            else:
                state = torch.load(key, map_location="cpu", weights_only=True)["net"]
            for m in TRAINABLE_MODULES:
                n = load_matching_state(self.model[m], state[m])
                if key is not None and n == 0:
                    raise ValueError(f"Fine-tuned checkpoint {ckpt} does not match the base model.")
            self._active_ckpt = key
            log.info("Seed-VC v1 DiT weights: %s", key or "base")

    # ------------------------------------------------------------------ conversion
    def _prompt(self, target: TargetVoice, use_f0: bool):
        """Reference-derived conditioning, cached per profile (and per fine-tuned
        checkpoint: the length regulator that encodes the prompt is fine-tuned too)."""
        import torch
        import torchaudio

        key = (target.profile_id, use_f0, str(target.reference_wav), None if use_f0 else self._active_ckpt)
        if key in self._prompt_cache:
            return self._prompt_cache[key]
        sr = self.sr_f0 if use_f0 else self.sr
        mel_fn = self.to_mel_f0 if use_f0 else self.to_mel
        module = self.model_f0 if use_f0 else self.model
        ref = load_audio(target.reference_wav, sr=sr)
        ref_np = ref.samples[: int(sr * self.REF_MAX_SECONDS)]
        ref_t = torch.from_numpy(ref_np.astype(np.float32)).unsqueeze(0).to(self.device)
        ref16 = torchaudio.functional.resample(ref_t, sr, 16000)
        with torch.inference_mode():
            S_ori = self._whisper_features(ref16)
            mel2 = mel_fn(ref_t)
            F0_ori = None
            if use_f0:
                F0_ori = torch.from_numpy(np.asarray(self.rmvpe.infer_from_audio(ref16[0], thred=0.03))).to(self.device)[None]
            prompt_cond, *_ = module.length_regulator(S_ori, ylens=torch.LongTensor([mel2.size(2)]).to(self.device),
                                                      n_quantizers=3, f0=F0_ori)
        style_file = target.dir / target.meta.get("style_file", "style_v1.npy")
        if style_file.exists():
            style = torch.from_numpy(np.load(style_file)).float().to(self.device).unsqueeze(0)
        else:
            with torch.inference_mode():
                style = self._style(ref16)
        entry = {"mel2": mel2, "prompt_cond": prompt_cond, "style": style, "F0_ori": F0_ori}
        self._prompt_cache[key] = entry
        return entry

    @staticmethod
    def _crossfade(a: np.ndarray, b: np.ndarray, overlap: int) -> np.ndarray:
        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
        b = b.copy()
        if len(b) < overlap:
            b[:overlap] = b[:overlap] * fade_in[: len(b)] + (a[-overlap:] * fade_out)[: len(b)]
        else:
            b[:overlap] = b[:overlap] * fade_in + a[-overlap:] * fade_out
        return b

    def _sample(self, cfm, mu, prompt, style, f0, steps: int, cfg_rate: float):
        """Euler flow-matching sampler (upstream ``CFM.solve_euler``) with two
        speed knobs: a non-uniform time grid, and classifier-free guidance only
        while ``cfg_from`` <= t < ``cfg_until`` (every guided step costs a second,
        unconditional DiT pass).  With the legacy settings it reproduces upstream exactly,
        including its random draws.

        ``mu`` is (1, T, C) content, ``prompt`` the reference mel (1, 80, P)."""
        import torch

        sp = self.sampler
        est = cfm.estimator
        T = mu.size(1)
        x_lens = torch.LongTensor([T]).to(mu.device)
        x = torch.randn([1, cfm.in_channels, T], device=mu.device)
        t_span = time_grid(steps, sp["schedule"], sp.get("sway", 0.0)).to(mu.device)
        p = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :p] = prompt[..., :p]
        x[..., :p] = 0
        if cfm.zero_prompt_speech_token:
            mu[..., :p] = 0
        # rows are never padded here, so the attention mask is all true and can go
        flash = self._flash.unpadded() if self._flash is not None else contextlib.nullcontext()
        with flash:
            for i in range(steps):
                t, dt = t_span[i], t_span[i + 1] - t_span[i]
                if cfg_rate > 0 and sp.get("cfg_from", 0.0) <= float(t) < sp["cfg_until"]:
                    both = est(torch.cat([x, x]), torch.cat([prompt_x, torch.zeros_like(prompt_x)]), x_lens,
                               torch.stack([t, t]), torch.cat([style, torch.zeros_like(style)]),
                               torch.cat([mu, torch.zeros_like(mu)]))
                    cond, uncond = both.chunk(2, dim=0)
                    v = (1.0 + cfg_rate) * cond - cfg_rate * uncond
                else:
                    v = est(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)
                x = x + dt * v
                x[:, :, :p] = 0
        return x

    def convert(self, source: AudioClip, target: TargetVoice, settings: ConversionSettings,
                progress: Optional[ProgressFn] = None) -> AudioClip:
        import torch
        import torchaudio

        s = settings.clamped()
        steps = int(self.sampler["steps"][s.quality])
        cfg_rate = similarity_to_cfg(s.similarity)
        length_adjust = 1.0 / s.speed
        use_f0 = abs(s.pitch_semitones) >= 0.05
        report = progress or (lambda p, m: None)

        with self._lock:
            timer = StageTimer(self.device)
            if use_f0:
                self._load_f0_models()
            # the fine-tune only covers the base (22 kHz, no-F0) model
            ckpt = target.meta.get("dit_checkpoint") if (s.use_finetuned and not use_f0) else None
            with timer("load_weights"):
                self.activate_checkpoint(Path(ckpt) if ckpt else None)
            module = self.model_f0 if use_f0 else self.model
            mel_fn = self.to_mel_f0 if use_f0 else self.to_mel
            vocoder = self.bigvgan44 if use_f0 else self.bigvgan
            sr = self.sr_f0 if use_f0 else self.sr
            hop = self.hop_f0 if use_f0 else self.hop
            overlap_frames = 16
            overlap_wave = overlap_frames * hop
            max_context = sr // hop * 30

            with timer("prompt"):
                prompt = self._prompt(target, use_f0)
            mel2, prompt_cond, style, F0_ori = prompt["mel2"], prompt["prompt_cond"], prompt["style"], prompt["F0_ori"]

            src = torch.from_numpy(source.resampled(sr).samples.astype(np.float32)).unsqueeze(0).to(self.device)
            src16 = torchaudio.functional.resample(src, sr, 16000)
            report(0.02, "Extracting speech content")
            with torch.inference_mode():
                with timer("content"):
                    S_alt = self._whisper_features(src16)
                    mel = mel_fn(src)
                target_len = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(self.device)

                shifted_f0 = None
                if use_f0:
                    F0_alt = torch.from_numpy(np.asarray(self.rmvpe.infer_from_audio(src16[0], thred=0.03))).to(self.device)[None]
                    voiced_ori = F0_ori[F0_ori > 1]
                    voiced_alt = F0_alt[F0_alt > 1]
                    log_alt = torch.log(F0_alt + 1e-5)
                    shifted = log_alt.clone()
                    if voiced_ori.numel() > 0 and voiced_alt.numel() > 0:
                        # move the source contour to the target's pitch level, keep its shape
                        shifted[F0_alt > 1] = log_alt[F0_alt > 1] - torch.median(torch.log(voiced_alt + 1e-5)) + torch.median(torch.log(voiced_ori + 1e-5))
                    shifted_f0 = torch.exp(shifted)
                    shifted_f0[F0_alt > 1] = shifted_f0[F0_alt > 1] * (2 ** (s.pitch_semitones / 12))

                cond, *_ = module.length_regulator(S_alt, ylens=target_len, n_quantizers=3, f0=shifted_f0)
                total = cond.size(1)
                max_source_window = max(1, max_context - mel2.size(2))
                processed, chunks, prev = 0, [], None
                autocast = (torch.autocast(device_type="cuda", dtype=torch.float16)
                            if self.device.type == "cuda" else contextlib.nullcontext())
                report(0.08, "Converting voice")
                while processed < total:
                    chunk_cond = cond[:, processed: processed + max_source_window]
                    is_last = processed + max_source_window >= total
                    cat = torch.cat([prompt_cond, chunk_cond], dim=1)
                    with timer("dit"), autocast:
                        vc_target = self._sample(module.cfm, cat, mel2, style, None, steps, cfg_rate)
                        vc_target = vc_target[:, :, mel2.size(-1):]
                    with timer("vocoder"):
                        wave = vocoder(vc_target.float())[0]  # (1, n)
                        w = wave[0].float().cpu().numpy()
                    if processed == 0:
                        if is_last:
                            chunks.append(w)
                            break
                        chunks.append(w[:-overlap_wave])
                        prev = w[-overlap_wave:]
                    elif is_last:
                        chunks.append(self._crossfade(prev, w, overlap_wave))
                        break
                    else:
                        chunks.append(self._crossfade(prev, w[:-overlap_wave], overlap_wave))
                        prev = w[-overlap_wave:]
                    processed += vc_target.size(2) - overlap_frames
                    report(0.08 + 0.9 * min(1.0, processed / total), "Converting voice")
            out = np.concatenate(chunks).astype(np.float32)
            expected = int(round(source.duration * length_adjust * sr))
            if expected > 0:
                if len(out) > expected:
                    out = out[:expected]
                elif len(out) < expected:
                    out = np.pad(out, (0, expected - len(out)))
            self.last_timings = timer.rounded()
            self.last_info = {"model": "f0_44k" if use_f0 else "base_22k", "finetuned": bool(ckpt),
                              "diffusion_steps": steps, "cfg_rate": cfg_rate,
                              "schedule": self.sampler["schedule"],
                              "cfg_window": [self.sampler.get("cfg_from", 0.0), self.sampler["cfg_until"]]}
            report(1.0, "Conversion done")
            return AudioClip(np.clip(out, -1.0, 1.0), sr)
