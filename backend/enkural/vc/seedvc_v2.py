"""Seed-VC v2 engine (optional alternative).

Uses the vendored ``VoiceConversionWrapper`` (HuBERT + BSQ discrete content
tokens, CFM decoder, BigVGAN).  Style conversion (the autoregressive branch)
is deliberately *not* used so the source's delivery is kept; only timbre is
replaced.  Heavier download than v1 (~3 GB incl. HuBERT-large).
"""
from __future__ import annotations

import contextlib
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import config
from ..audio.io import AudioClip, load_audio, write_wav
from .base import ConversionSettings, ProgressFn, StageTimer, TargetVoice, VoiceConverter
from .hf import setup_seedvc
from .registry import register

log = logging.getLogger(__name__)

QUALITY_STEPS = {"draft": 10, "standard": 30, "high": 50}


@register
class SeedVCv2(VoiceConverter):
    id = "seedvc_v2"
    name = "Seed-VC v2 · discrete content tokens + CFM"
    description = (
        "Alternative engine with separate intelligibility and similarity guidance. "
        "Style conversion is kept off so the original delivery is preserved. "
        "Larger download (~3 GB)."
    )
    output_sr = 22050
    capabilities = {"similarity": True, "pitch": False, "speed": True, "multi_reference": True}
    REF_MAX_SECONDS = 12.0

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._prompt_cache: Dict[Tuple[str, str], Dict] = {}

    @classmethod
    def available(cls) -> bool:
        try:
            import hydra  # noqa: F401
            import omegaconf  # noqa: F401
            import torch  # noqa: F401
        except Exception:
            return False
        return (config.SEEDVC_DIR / "configs" / "v2" / "vc_wrapper.yaml").exists()

    def load(self, device: str) -> None:
        import torch
        import yaml

        setup_seedvc()
        from hydra.utils import instantiate
        from omegaconf import DictConfig

        with self._lock:
            self.device = torch.device(device)
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            cfg = DictConfig(yaml.safe_load(open(config.SEEDVC_DIR / "configs" / "v2" / "vc_wrapper.yaml")))
            w = instantiate(cfg)
            w.load_checkpoints()
            w.to(self.device)
            w.eval()
            w.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=self.dtype, device=self.device)
            self.w = w
            self._prompt_cache.clear()
            self._loaded = True
            log.info("Seed-VC v2 loaded on %s", self.device)

    def unload(self) -> None:
        with self._lock:
            if hasattr(self, "w"):
                del self.w
            self._prompt_cache.clear()
            self._loaded = False
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _autocast(self):
        import torch

        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return contextlib.nullcontext()

    def _embed_clip(self, clip: AudioClip, chunk_s: float = 30.0):
        import torch

        x = clip.resampled(16000).samples
        n = int(chunk_s * 16000)
        embs, weights = [], []
        for a in range(0, len(x), n):
            seg = x[a: a + n]
            if len(seg) < 16000:
                continue
            t = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                embs.append(self.w.compute_style(t)[0].float())
            weights.append(len(seg))
        if not embs:
            t = torch.from_numpy(np.pad(x, (0, max(0, 16000 - len(x)))).astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                return self.w.compute_style(t)[0].float()
        E = torch.stack(embs)
        wt = torch.tensor(weights, dtype=torch.float32, device=E.device)[:, None]
        return (E * wt).sum(0) / wt.sum()

    def speaker_embedding(self, clip: AudioClip) -> np.ndarray:
        with self._lock:
            return self._embed_clip(clip).cpu().numpy()

    def build_target(self, reference: AudioClip, samples: List[AudioClip], out_dir: Path,
                     profile_id: str, progress: Optional[ProgressFn] = None) -> TargetVoice:
        import torch

        out_dir.mkdir(parents=True, exist_ok=True)
        ref44 = reference.resampled(44100)
        ref_path = write_wav(out_dir / "reference.wav", ref44, subtype="PCM_24")
        with self._lock, torch.inference_mode():
            embs = [(self._embed_clip(s), s.duration) for s in samples]
            ref_emb = self._embed_clip(reference)
            if embs:
                E = torch.stack([e for e, _ in embs])
                wt = torch.tensor([d for _, d in embs], dtype=torch.float32, device=E.device)[:, None]
                En = E / E.norm(dim=-1, keepdim=True)
                mean_dir = (En * wt).sum(0) / wt.sum()
                style = mean_dir / mean_dir.norm() * E.norm(dim=-1).mean()
                per_sample = [float(torch.nn.functional.cosine_similarity(e, style, dim=0)) for e, _ in embs]
            else:
                style, per_sample = ref_emb, []
            ref_cos = float(torch.nn.functional.cosine_similarity(ref_emb, style, dim=0))
        np.save(out_dir / "style_v2.npy", style.cpu().numpy())
        self._prompt_cache = {k: v for k, v in self._prompt_cache.items() if k[0] != profile_id}
        meta = {
            "style_file": "style_v2.npy", "reference_seconds": round(ref44.duration, 2),
            "n_samples": len(samples), "sample_consistency": [round(c, 3) for c in per_sample],
            "reference_consistency": round(ref_cos, 3),
        }
        if progress:
            progress(1.0, "Voice profile ready")
        return TargetVoice(profile_id=profile_id, engine_id=self.id, dir=out_dir, reference_wav=ref_path, meta=meta)

    def _prompt(self, target: TargetVoice) -> Dict:
        import torch
        import torchaudio

        key = (target.profile_id, str(target.reference_wav))
        if key in self._prompt_cache:
            return self._prompt_cache[key]
        w = self.w
        ref = load_audio(target.reference_wav, sr=w.sr)
        ref_np = ref.samples[: int(w.sr * self.REF_MAX_SECONDS)]
        ref_t = torch.from_numpy(ref_np.astype(np.float32)).unsqueeze(0).to(self.device)
        ref16 = torchaudio.functional.resample(ref_t, w.sr, 16000)
        with torch.inference_mode():
            target_mel = w.mel_fn(ref_t)
            with self._autocast():
                tgt_idx = w._process_content_features(ref16, is_narrow=False)
                prompt_condition, _ = w.cfm_length_regulator(tgt_idx, ylens=torch.LongTensor([target_mel.size(2)]).to(self.device))
        style_file = target.dir / target.meta.get("style_file", "style_v2.npy")
        if style_file.exists():
            style = torch.from_numpy(np.load(style_file)).float().to(self.device).unsqueeze(0)
        else:
            with torch.inference_mode():
                style = w.compute_style(ref16)
        entry = {"target_mel": target_mel, "prompt_condition": prompt_condition.float(), "style": style}
        self._prompt_cache[key] = entry
        return entry

    def convert(self, source: AudioClip, target: TargetVoice, settings: ConversionSettings,
                progress: Optional[ProgressFn] = None) -> AudioClip:
        import torch
        import torchaudio

        s = settings.clamped()
        steps = QUALITY_STEPS[s.quality]
        sim_cfg = round(0.40 + 0.50 * s.similarity / 100.0, 3)
        intel_cfg = 0.7
        length_adjust = 1.0 / s.speed
        report = progress or (lambda p, m: None)
        w = self.w
        sr, hop = w.sr, w.hop_size

        with self._lock:
            timer = StageTimer(self.device)
            with timer("prompt"):
                prompt = self._prompt(target)
            target_mel, prompt_condition, style = prompt["target_mel"], prompt["prompt_condition"], prompt["style"]
            target_mel_len = target_mel.size(2)
            src_t = torch.from_numpy(source.resampled(sr).samples.astype(np.float32)).unsqueeze(0).to(self.device)
            src16 = torchaudio.functional.resample(src_t, sr, 16000)
            max_context = sr // hop * w.dit_max_context_len
            overlap_frames = w.overlap_frame_len
            overlap_wave = overlap_frames * hop
            report(0.02, "Extracting speech content")
            with torch.inference_mode():
                with timer("content"):
                    source_mel_len = w.mel_fn(src_t).size(2)
                    with self._autocast():
                        src_idx = w._process_content_features(src16, is_narrow=False)
                cond, _ = w.cfm_length_regulator(src_idx, ylens=torch.LongTensor([int(source_mel_len * length_adjust)]).to(self.device))
                cond = cond.float()
                total = cond.size(1)
                max_source_window = max(1, max_context - target_mel_len)
                processed, chunks, prev = 0, [], None
                report(0.08, "Converting voice")
                while processed < total:
                    chunk_cond = cond[:, processed: processed + max_source_window]
                    is_last = processed + max_source_window >= total
                    cat = torch.cat([prompt_condition, chunk_cond], dim=1)
                    orig_len = cat.size(1)
                    with timer("dit"):
                        vc_mel = w.cfm.inference(cat, torch.LongTensor([orig_len]).to(self.device), target_mel, style, steps,
                                                 inference_cfg_rate=[intel_cfg, sim_cfg], random_voice=False)
                        vc_mel = vc_mel[:, :, target_mel_len:orig_len]
                    with timer("vocoder"):
                        wave = w.vocoder(vc_mel.float()).squeeze()[None]
                        x = wave[0].float().cpu().numpy()
                    if processed == 0:
                        if is_last:
                            chunks.append(x)
                            break
                        chunks.append(x[:-overlap_wave])
                        prev = x[-overlap_wave:]
                    elif is_last:
                        chunks.append(_crossfade(prev, x, overlap_wave))
                        break
                    else:
                        chunks.append(_crossfade(prev, x[:-overlap_wave], overlap_wave))
                        prev = x[-overlap_wave:]
                    processed += vc_mel.size(2) - overlap_frames
                    report(0.08 + 0.9 * min(1.0, processed / total), "Converting voice")
            out = np.concatenate(chunks).astype(np.float32)
            expected = int(round(source.duration * length_adjust * sr))
            if expected > 0:
                out = out[:expected] if len(out) > expected else np.pad(out, (0, expected - len(out)))
            self.last_timings = timer.rounded()
            self.last_info = {"model": "v2", "finetuned": False, "diffusion_steps": steps, "cfg_rate": sim_cfg}
            report(1.0, "Conversion done")
            return AudioClip(np.clip(out, -1.0, 1.0), sr)


def _crossfade(a: np.ndarray, b: np.ndarray, overlap: int) -> np.ndarray:
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    b = b.copy()
    if len(b) < overlap:
        b[:overlap] = b[:overlap] * fade_in[: len(b)] + (a[-overlap:] * fade_out)[: len(b)]
    else:
        b[:overlap] = b[:overlap] * fade_in + a[-overlap:] * fade_out
    return b
