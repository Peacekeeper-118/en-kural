"""Voice / background separation with Hybrid Demucs (bundled with torchaudio).

Reels and videos usually carry a music bed under the speech.  Spectral-gating
denoisers cannot remove music, and the voice converter handles it badly in
both directions: Seed-VC copies the reference clip's acoustic character (music
included) into every output, and it turns music in the source into voice-like
artefacts.  Hybrid Demucs splits a mix into drums / bass / other / vocals; the
vocals stem is the voice and the sum of the rest is the background (which also
absorbs most steady noise).

The model (~320 MB) downloads once into ``models/torch`` and is moved to the
GPU only while it runs, so it does not compete with the conversion engine.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .io import AudioClip

log = logging.getLogger(__name__)

SEP_SR = 44100
SEGMENT_S = 10.0
OVERLAP_S = 1.0
# Background energy relative to the voice above which separation is applied in
# "auto" mode.  Clean speech measures ≈ -30…-40 dB (Demucs leaks a little);
# speech over a music bed is typically -15…0 dB.
BACKGROUND_THRESHOLD_DB = -22.0

_lock = threading.Lock()
_model = None


@dataclass
class Separation:
    vocals: AudioClip          # at the input clip's sample rate
    background: AudioClip      # at SEP_SR (kept full-band for remixing)
    background_db: float       # background energy relative to the vocals, dB

    @property
    def significant(self) -> bool:
        return bool(self.background_db > BACKGROUND_THRESHOLD_DB)

    def summary(self, applied: bool) -> dict:
        return {"applied": bool(applied), "background_db": round(float(self.background_db), 1),
                "threshold_db": BACKGROUND_THRESHOLD_DB}


def available() -> bool:
    from .. import config

    if config.SEPARATION == "off":
        return False
    try:
        from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS  # noqa: F401
    except Exception:
        return False
    return True


def _get_model():
    global _model
    if _model is None:
        from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

        _model = HDEMUCS_HIGH_MUSDB_PLUS.get_model().eval()
    return _model


def _window(n: int, ramp: int):
    import torch

    w = torch.ones(n)
    if ramp > 0 and n > 2 * ramp:
        r = torch.linspace(1.0 / ramp, 1.0, ramp)
        w[:ramp], w[-ramp:] = r, r.flip(0)
    return w


def separate(clip: AudioClip, device: Optional[str] = None,
             progress: Optional[Callable[[float], None]] = None) -> Separation:
    """Split ``clip`` into voice and background (overlap-add over 10 s segments)."""
    import torch

    from ..vc.registry import resolve_device

    dev = torch.device(device or resolve_device())
    x = clip.resampled(SEP_SR).samples.astype(np.float32)
    mix = torch.from_numpy(x)[None, None, :].repeat(1, 2, 1)  # model expects stereo
    mean, std = mix.mean(), mix.std() + 1e-8
    mix = (mix - mean) / std

    n = mix.shape[-1]
    seg, ov = int(SEGMENT_S * SEP_SR), int(OVERLAP_S * SEP_SR)
    hop = seg - ov
    out = torch.zeros(4, n)          # sources × samples (mono)
    weight = torch.zeros(n)
    with _lock:
        model = _get_model().to(dev)
        try:
            with torch.inference_mode():
                start = 0
                while start < n:
                    chunk = mix[..., start: start + seg]
                    length = chunk.shape[-1]
                    if length < SEP_SR:  # very short tail: pad so the model has context
                        chunk = torch.nn.functional.pad(chunk, (0, SEP_SR - length))
                    y = model(chunk.to(dev))[0, :, :, :length].mean(1).float().cpu()  # (4, L)
                    w = _window(length, ov)
                    out[:, start: start + length] += y * w
                    weight[start: start + length] += w
                    if progress:
                        progress(min(1.0, (start + length) / n))
                    if start + seg >= n:
                        break
                    start += hop
        finally:
            model.to("cpu")
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    out = out / weight.clamp_min(1e-6) * std + mean
    names = list(getattr(model, "sources", ["drums", "bass", "other", "vocals"]))
    vi = names.index("vocals")
    vocals = out[vi].numpy()
    background = (out.sum(0) - out[vi]).numpy()
    e_v = float(np.mean(vocals.astype(np.float64) ** 2)) + 1e-12
    e_b = float(np.mean(background.astype(np.float64) ** 2)) + 1e-12
    voc = AudioClip(np.clip(vocals, -1, 1).astype(np.float32), SEP_SR)
    return Separation(
        vocals=voc.resampled(clip.sr) if clip.sr != SEP_SR else voc,
        background=AudioClip(np.clip(background, -1, 1).astype(np.float32), SEP_SR),
        background_db=float(10.0 * np.log10(e_b / e_v)),
    )
