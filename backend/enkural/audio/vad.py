"""Voice activity detection (Silero VAD) helpers.

Silero VAD is language-agnostic, so it works for Tamil / Tanglish speech
without any tuning.  It is tiny and runs on the CPU.
"""
from __future__ import annotations

import threading
from typing import List, Tuple

import numpy as np

from ..config import ANALYSIS_SR
from .io import AudioClip

_lock = threading.Lock()
_model = None


def _get_model():
    global _model
    if _model is None:
        import torch
        from silero_vad import load_silero_vad

        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        _model = load_silero_vad(onnx=False)
    return _model


def speech_segments(
    clip: AudioClip,
    threshold: float = 0.5,
    min_speech_ms: int = 120,
    min_silence_ms: int = 150,
    pad_ms: int = 40,
) -> List[Tuple[float, float]]:
    """Return ``[(start_s, end_s), ...]`` speech regions of the clip."""
    import torch
    from silero_vad import get_speech_timestamps

    c = clip.resampled(ANALYSIS_SR)
    if len(c.samples) < ANALYSIS_SR // 10:
        return []
    audio = torch.from_numpy(np.ascontiguousarray(c.samples, dtype=np.float32))
    with _lock:
        model = _get_model()
        model.reset_states()
        stamps = get_speech_timestamps(
            audio,
            model,
            sampling_rate=ANALYSIS_SR,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=pad_ms,
            return_seconds=False,
        )
    return [(s["start"] / ANALYSIS_SR, s["end"] / ANALYSIS_SR) for s in stamps]


def segments_to_mask(segments: List[Tuple[float, float]], n_samples: int, sr: int) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)
    for s, e in segments:
        a, b = int(max(0, s * sr)), int(min(n_samples, e * sr))
        if b > a:
            mask[a:b] = True
    return mask


def speech_duration(segments: List[Tuple[float, float]]) -> float:
    return float(sum(max(0.0, e - s) for s, e in segments))


def pauses(segments: List[Tuple[float, float]], min_pause: float = 0.15) -> List[float]:
    """Durations of gaps *between* speech segments (leading/trailing excluded)."""
    gaps = []
    for (s0, e0), (s1, e1) in zip(segments[:-1], segments[1:]):
        g = s1 - e0
        if g >= min_pause:
            gaps.append(float(g))
    return gaps


def extract_speech(clip: AudioClip, segments: List[Tuple[float, float]], pad_s: float = 0.08,
                   fade_s: float = 0.01) -> AudioClip:
    """Concatenate speech regions (with small pads and fades) into one clip."""
    sr = clip.sr
    n = len(clip.samples)
    parts = []
    fade = int(fade_s * sr)
    for s, e in segments:
        a = int(max(0, (s - pad_s) * sr))
        b = int(min(n, (e + pad_s) * sr))
        if b - a < fade * 2 + 1:
            continue
        seg = clip.samples[a:b].copy()
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        seg[:fade] *= ramp
        seg[-fade:] *= ramp[::-1]
        parts.append(seg)
    if not parts:
        return AudioClip(np.zeros(0, dtype=np.float32), sr)
    return AudioClip(np.concatenate(parts).astype(np.float32), sr)
