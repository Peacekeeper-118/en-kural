"""Objective quality metrics for voice conversion, independent of the engine.

The in-app "voice match" uses CAMPPlus, the very embedding the engine is
conditioned on, so it mostly tells you the engine hit its own target.  The
metrics here use models the engine never sees:

  * speaker similarity – WavLM-base-plus-SV x-vectors, cosine (the model card's
                         same-speaker threshold is ~0.86)
  * intelligibility    – Whisper ASR character error rate (Tamil script)
  * naturalness        – UTMOS22 predicted MOS (1–5)
  * prosody            – log-F0 contour correlation with the source
  * timing             – output duration / expected duration

Models load lazily and can be released with ``release()`` so they do not
compete with the conversion engine for GPU memory.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from .audio import vad
from .audio.io import AudioClip

log = logging.getLogger(__name__)

SV_MODEL = "microsoft/wavlm-base-plus-sv"
SV_SAME_SPEAKER_THRESHOLD = 0.86
MOS_REPO = "tarepan/SpeechMOS:v1.2.0"
MOS_MODEL = "utmos22_strong"
WHISPER_BASE_FOR_GENERATION = "openai/whisper-small"


def _device(pref: Optional[str] = None) -> str:
    if pref:
        return pref
    from .vc.registry import resolve_device

    return resolve_device()


def speech_only(clip: AudioClip, sr: int = 16000) -> AudioClip:
    """The clip's speech regions concatenated (falls back to the whole clip)."""
    c = clip.resampled(sr)
    segs = vad.speech_segments(c)
    if not segs:
        return c
    s = vad.extract_speech(c, segs)
    return s if s.duration >= 0.5 else c


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _chunks(x: np.ndarray, sr: int, chunk_s: float, min_s: float = 1.0) -> List[np.ndarray]:
    n = int(chunk_s * sr)
    parts = [x[a: a + n] for a in range(0, len(x), n)]
    keep = [p for p in parts if len(p) >= int(min_s * sr)]
    return keep or [np.pad(x, (0, max(0, int(min_s * sr) - len(x))))]


# --------------------------------------------------------------------------- speaker similarity
class SpeakerVerifier:
    """WavLM-base-plus-SV speaker embeddings (independent of the engine)."""

    def __init__(self, device: Optional[str] = None):
        self.device = _device(device)
        self.model = None

    def _load(self):
        if self.model is None:
            from transformers import AutoFeatureExtractor, WavLMForXVector

            cache = str(config.MODELS_DIR / "hf")
            self.fe = AutoFeatureExtractor.from_pretrained(SV_MODEL, cache_dir=cache)
            self.model = WavLMForXVector.from_pretrained(SV_MODEL, cache_dir=cache).to(self.device).eval()
        return self.model

    def embed(self, clip: AudioClip, chunk_s: float = 10.0) -> np.ndarray:
        import torch

        model = self._load()
        x = speech_only(clip).samples
        embs, weights = [], []
        for seg in _chunks(x, 16000, chunk_s):
            inputs = self.fe(seg, sampling_rate=16000, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                e = model(**inputs).embeddings
            embs.append(torch.nn.functional.normalize(e, dim=-1)[0].float().cpu().numpy())
            weights.append(len(seg))
        v = np.average(np.stack(embs), axis=0, weights=np.asarray(weights, dtype=np.float64))
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)

    def centroid(self, clips: Sequence[AudioClip]) -> np.ndarray:
        E = np.stack([self.embed(c) for c in clips])
        v = E.mean(axis=0)
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)

    def release(self) -> None:
        self.model = None
        _empty_cache()


# --------------------------------------------------------------------------- intelligibility
_PUNCT = re.compile(r"[\s]+")


def normalize_text(text: str) -> str:
    """Lower-case, drop punctuation / symbols, collapse whitespace (script-agnostic)."""
    out = []
    for ch in unicodedata.normalize("NFC", text or "").lower():
        cat = unicodedata.category(ch)
        out.append(" " if cat[0] in ("P", "S") else ch)
    return _PUNCT.sub(" ", "".join(out)).strip()


def asr_windows(clip: AudioClip, max_s: float = 12.0) -> List[Tuple[float, float]]:
    """Speech spans of at most ``max_s`` seconds, split at pauses."""
    segs = vad.speech_segments(clip.resampled(16000))
    out: List[Tuple[float, float]] = []
    for s, e in segs:
        if out and e - out[-1][0] <= max_s:
            out[-1] = (out[-1][0], e)
        else:
            while e - s > max_s:  # one very long phrase: hard split
                out.append((s, s + max_s))
                s += max_s
            out.append((s, e))
    pad = 0.15
    return [(max(0.0, s - pad), e + pad) for s, e in out] or [(0.0, clip.duration)]


def cer(reference: str, hypothesis: str) -> Optional[float]:
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        return None
    import jiwer

    return float(jiwer.cer(ref, hyp))


class Transcriber:
    """Whisper ASR over the whole clip (chunked), forced to one language."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None,
                 language: Optional[str] = None):
        self.model_name = model_name or config.EVAL_ASR_MODEL
        self.language = language or config.EVAL_LANGUAGE
        self.device = _device(device)
        self.pipe = None

    def _load(self):
        if self.pipe is None:
            import torch
            from transformers import pipeline

            from transformers import GenerationConfig

            cache = str(config.MODELS_DIR / "hf")
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.pipe = pipeline("automatic-speech-recognition", model=self.model_name, torch_dtype=dtype,
                                 device=self.device, chunk_length_s=30, model_kwargs={"cache_dir": cache})
            if not getattr(self.pipe.model.generation_config, "lang_to_id", None):
                # Fine-tunes made with older transformers ship a generation config without the
                # language table (and with empty suppress_tokens); borrow the base model's.
                gc = GenerationConfig.from_pretrained(WHISPER_BASE_FOR_GENERATION, cache_dir=cache)
                self.pipe.model.generation_config = gc
                if hasattr(self.pipe, "generation_config"):  # pipelines keep their own copy
                    self.pipe.generation_config = gc
        return self.pipe

    def transcribe(self, clip: AudioClip, windows: Optional[List[Tuple[float, float]]] = None) -> str:
        """Transcript of the clip.  With ``windows`` each (start, end) span is
        transcribed on its own: Whisper often drops whole sentences when given
        a long clip in one pass, which would be scored as garbled speech."""
        pipe = self._load()
        x = clip.resampled(16000).samples.astype(np.float32)
        spans = windows or [(0.0, len(x) / 16000)]
        inputs = [{"raw": x[int(a * 16000): max(int(a * 16000) + 1600, int(b * 16000))], "sampling_rate": 16000}
                  for a, b in spans]
        outs = pipe(inputs, batch_size=4, generate_kwargs={"language": self.language, "task": "transcribe"})
        return " ".join(o["text"].strip() for o in outs).strip()

    def release(self) -> None:
        self.pipe = None
        _empty_cache()


# --------------------------------------------------------------------------- naturalness
class NaturalnessPredictor:
    """UTMOS22 (strong learner) predicted MOS, averaged over ≤10 s speech chunks."""

    def __init__(self, device: Optional[str] = None):
        self.device = _device(device)
        self.model = None

    def _load(self):
        if self.model is None:
            import torch

            self.model = torch.hub.load(MOS_REPO, MOS_MODEL, trust_repo=True).to(self.device).eval()
        return self.model

    def score(self, clip: AudioClip, chunk_s: float = 10.0) -> float:
        import torch

        model = self._load()
        x = speech_only(clip).samples
        scores, weights = [], []
        for seg in _chunks(x, 16000, chunk_s):
            w = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                scores.append(float(model(w, 16000).item()))
            weights.append(len(seg))
        return float(np.average(scores, weights=weights))

    def release(self) -> None:
        self.model = None
        _empty_cache()


# --------------------------------------------------------------------------- prosody / timing
def f0_contour(clip: AudioClip, frame_period_ms: float = 10.0) -> np.ndarray:
    """F0 in Hz per 10 ms frame (0 = unvoiced), WORLD DIO + StoneMask."""
    import pyworld

    c = clip.resampled(16000)
    x = c.samples.astype(np.float64)
    f0, t = pyworld.dio(x, c.sr, f0_floor=60.0, f0_ceil=600.0, frame_period=frame_period_ms)
    return pyworld.stonemask(x, f0, t, c.sr)


def pitch_correlation(source: AudioClip, converted: AudioClip, speed: float = 1.0) -> Dict[str, Optional[float]]:
    """Pearson correlation of log-F0 over frames voiced in both clips.

    High values mean the converted speech keeps the source's intonation shape
    (it is expected to move to *your* pitch level, so the absolute offset is
    reported separately and is not an error).
    """
    fs, fo = f0_contour(source), f0_contour(converted)
    idx = np.round(np.arange(len(fo)) * speed).astype(int)
    ok = idx < len(fs)
    fo, fs_al = fo[ok], fs[idx[ok]]
    both = (fo > 0) & (fs_al > 0)
    if both.sum() < 20:
        return {"f0_corr": None, "f0_offset_st": None, "voiced_overlap": float(both.mean()) if len(both) else 0.0}
    ls, lo = np.log(fs_al[both]), np.log(fo[both])
    r = float(np.corrcoef(ls, lo)[0, 1])
    offset = float(12 * (np.median(lo) - np.median(ls)) / np.log(2))
    voiced_src = max(1, int((fs_al > 0).sum()))
    return {"f0_corr": r, "f0_offset_st": offset, "voiced_overlap": float(both.sum() / voiced_src)}


def mel_distance(a: AudioClip, b: AudioClip, floor_db: float = 60.0) -> float:
    """Mean absolute log-mel difference (dB) between two time-aligned renditions
    of the same speech, over frames within ``floor_db`` of the loudest one.

    With a fixed seed every sampler setting starts from the same noise, so the
    distance to a many-step reference measures how far a faster setting drifts."""
    import torch
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=1024, hop_length=256, n_mels=80)
    xa = torch.from_numpy(a.resampled(16000).samples.astype(np.float32))
    xb = torch.from_numpy(b.resampled(16000).samples.astype(np.float32))
    n = min(len(xa), len(xb))
    A = 10 * torch.log10(mel(xa[:n]) + 1e-10)
    B = 10 * torch.log10(mel(xb[:n]) + 1e-10)
    frame_db = torch.maximum(A.max(0).values, B.max(0).values)
    active = frame_db > frame_db.max() - floor_db
    return float((A - B).abs()[:, active].mean())


# --------------------------------------------------------------------------- one-shot helper
class Evaluator:
    """Bundles the judges; call ``release()`` before handing the GPU back."""

    def __init__(self, device: Optional[str] = None, asr_model: Optional[str] = None,
                 use_asr: bool = True, use_mos: bool = True):
        self.sv = SpeakerVerifier(device)
        self.asr = Transcriber(asr_model, device) if use_asr else None
        self.mos = NaturalnessPredictor(device) if use_mos else None
        self._transcripts: Dict[int, str] = {}

    def transcript(self, clip: AudioClip, windows=None, key: Optional[int] = None) -> Optional[str]:
        if self.asr is None:
            return None
        if key is not None and key in self._transcripts:
            return self._transcripts[key]
        t = self.asr.transcribe(clip, windows)
        if key is not None:
            self._transcripts[key] = t
        return t

    def score(self, source: AudioClip, converted: AudioClip, target_centroid: np.ndarray,
              speed: float = 1.0, source_text: Optional[str] = None,
              source_key: Optional[int] = None) -> Dict:
        """Metrics for one conversion.  ``source_text`` is a known transcript;
        without it the ASR transcript of the source is the reference."""
        out: Dict = {}
        conv_emb = self.sv.embed(converted)
        out["speaker_sim"] = cosine(conv_emb, target_centroid)
        out["speaker_sim_to_source"] = cosine(conv_emb, self.sv.embed(source))
        expected = source.duration / max(speed, 1e-6)
        out["duration_ratio"] = converted.duration / expected if expected > 0 else None
        out.update(pitch_correlation(source, converted, speed))
        if self.asr is not None:
            # the converter keeps timing, so the source's phrase windows (scaled by
            # the speed change) cut the converted audio at the same words
            win = asr_windows(source)
            ref = source_text if source_text else self.transcript(source, win, source_key)
            hyp = self.asr.transcribe(converted, [(a / speed, b / speed) for a, b in win])
            out["cer"] = cer(ref, hyp)
            out["asr_reference"] = "known text" if source_text else "ASR of source"
            out["transcript"] = hyp
        if self.mos is not None:
            out["utmos"] = self.mos.score(converted)
        return out

    def release(self) -> None:
        self.sv.release()
        if self.asr:
            self.asr.release()
        if self.mos:
            self.mos.release()


def _empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
