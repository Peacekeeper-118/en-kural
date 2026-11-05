"""Audio clean-up and post-processing.

* ``denoise``            – DeepFilterNet (if installed) or spectral gating
* ``normalize_loudness`` – EBU R128 loudness + true-peak limiter
* ``match_envelope``     – transfer the source's loudness contour onto the
                           converted audio (preserves emphasis / dynamics and
                           implements the "energy" control)
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d

from .io import AudioClip, resample
from .quality import EPS, frame_rms

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- filters
def highpass(x: np.ndarray, sr: int, cutoff: float = 60.0, order: int = 2) -> np.ndarray:
    sos = signal.butter(order, cutoff, btype="highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)


def fade_edges(x: np.ndarray, sr: int, ms: float = 8.0) -> np.ndarray:
    n = min(len(x) // 2, int(sr * ms / 1000))
    if n <= 0:
        return x
    y = x.copy()
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


# --------------------------------------------------------------------------- denoise
_df_state = None


def _deepfilter(x: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """DeepFilterNet3 (48 kHz) if the optional ``deepfilternet`` package is present."""
    global _df_state
    try:
        import torch
        from df.enhance import enhance, init_df
    except Exception:
        return None
    try:
        if _df_state is None:
            model, df_state, _ = init_df(log_level="ERROR")
            _df_state = (model, df_state)
        model, df_state = _df_state
        x48 = resample(x, sr, 48000)
        with torch.no_grad():
            y = enhance(model, df_state, torch.from_numpy(x48[None, :].astype(np.float32)))
        y = y.squeeze(0).cpu().numpy()
        return resample(y, 48000, sr)[: len(x)]
    except Exception as e:  # pragma: no cover - optional path
        log.warning("DeepFilterNet failed, falling back to spectral gating: %s", e)
        return None


def denoise(x: np.ndarray, sr: int, strength: float = 0.75, method: str = "auto") -> np.ndarray:
    """Reduce steady background noise.  ``strength`` in [0, 1]."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return x
    if method in ("auto", "deepfilter"):
        y = _deepfilter(x, sr)
        if y is not None:
            # blend to keep the result natural when strength < 1
            return (strength * y + (1 - strength) * x[: len(y)]).astype(np.float32)
        if method == "deepfilter":
            raise RuntimeError("deepfilternet is not installed")
    import noisereduce as nr

    y = nr.reduce_noise(
        y=x.astype(np.float32), sr=sr, stationary=False, prop_decrease=strength,
        time_constant_s=1.5, freq_mask_smooth_hz=250, time_mask_smooth_ms=40,
        n_fft=1024, use_torch=False,
    )
    return y.astype(np.float32)


# --------------------------------------------------------------------------- loudness
def measure_lufs(x: np.ndarray, sr: int) -> float:
    import pyloudnorm as pyln

    if len(x) < int(0.4 * sr):
        return float("-inf")
    meter = pyln.Meter(sr)
    try:
        return float(meter.integrated_loudness(x.astype(np.float64)))
    except Exception:
        return float("-inf")


def true_peak_db(x: np.ndarray) -> float:
    if len(x) == 0:
        return -120.0
    up = signal.resample_poly(x.astype(np.float64), 4, 1)
    return float(20 * np.log10(np.max(np.abs(up)) + EPS))


def limiter(x: np.ndarray, sr: int, ceiling_db: float = -1.5, lookahead_ms: float = 2.0,
            release_ms: float = 80.0) -> np.ndarray:
    """Transparent look-ahead peak limiter working on 1 ms blocks (true-peak aware)."""
    ceiling = 10 ** (ceiling_db / 20)
    block = max(1, int(sr * 0.001))
    n_blocks = int(np.ceil(len(x) / block))
    if n_blocks == 0:
        return x
    xp = np.pad(x.astype(np.float64), (0, n_blocks * block - len(x)))
    up = signal.resample_poly(xp, 4, 1)  # 4x oversampled for true-peak estimate
    peaks = np.abs(up).reshape(n_blocks, block * 4).max(axis=1)
    g_req = np.minimum(1.0, ceiling / (peaks + EPS))
    if np.all(g_req >= 0.999):
        return x
    la = max(1, int(round(lookahead_ms)))
    g_att = minimum_filter1d(g_req, size=2 * la + 1, mode="nearest")
    alpha = float(np.exp(-1.0 / max(release_ms, 1.0)))
    g = np.empty_like(g_att)
    prev = 1.0
    for i in range(n_blocks):
        target = g_att[i]
        prev = target if target < prev else target + (prev - target) * alpha
        g[i] = prev
    centres = (np.arange(n_blocks) + 0.5) * block
    gain = np.interp(np.arange(len(x)), centres, g)
    y = x * gain
    return np.clip(y, -ceiling, ceiling).astype(np.float32)


def normalize_loudness(x: np.ndarray, sr: int, target_lufs: float = -14.0, true_peak: float = -1.5,
                       max_gain_db: float = 30.0) -> np.ndarray:
    lufs = measure_lufs(x, sr)
    if not np.isfinite(lufs):
        peak = np.max(np.abs(x)) + EPS
        return (x * min(10 ** (true_peak / 20) / peak, 10 ** (max_gain_db / 20))).astype(np.float32)
    gain_db = float(np.clip(target_lufs - lufs, -max_gain_db, max_gain_db))
    y = (x * 10 ** (gain_db / 20)).astype(np.float32)
    return limiter(y, sr, ceiling_db=true_peak)


# --------------------------------------------------------------------------- dynamics transfer
def match_envelope(out: AudioClip, src: AudioClip, energy: float = 1.0, strength: float = 1.0,
                   max_db: float = 8.0, smooth_frames: int = 7) -> AudioClip:
    """Shape the converted audio's loudness contour after the source's.

    Both clips are assumed time-aligned (the converter preserves timing; a
    speed change scales the whole axis).  ``energy`` > 1 exaggerates the
    source's dynamics, < 1 flattens them; 1.0 reproduces them exactly.
    """
    if strength <= 0 or out.duration < 0.2 or src.duration < 0.2:
        return out
    ar = 16000
    o = out.resampled(ar).samples
    s = src.resampled(ar).samples
    o_rms, o_t = frame_rms(o, ar, win_ms=30, hop_ms=10)
    s_rms, s_t = frame_rms(s, ar, win_ms=30, hop_ms=10)
    o_db = 20 * np.log10(o_rms + EPS)
    s_db = 20 * np.log10(s_rms + EPS)
    scale = out.duration / src.duration
    s_on_o = np.interp(o_t, s_t * scale, s_db)
    floor_o = np.percentile(o_db, 10)
    floor_s = np.percentile(s_db, 10)
    active = (o_db > floor_o + 10) & (s_on_o > floor_s + 6)
    if active.sum() < 10:
        return out
    o_mean = float(np.mean(o_db[active]))
    s_mean = float(np.mean(s_on_o[active]))
    desired = o_mean + energy * (s_on_o - s_mean)
    gain_db = np.zeros_like(o_db)
    gain_db[active] = strength * (desired[active] - o_db[active])
    # pauses in the source: gently pull down whatever the converter produced there
    quiet_src = (s_on_o <= floor_s + 6) & (o_db > floor_o + 10)
    gain_db[quiet_src] = np.minimum(0.0, strength * (floor_o + 6 - o_db[quiet_src]))
    gain_db = np.clip(gain_db, -max_db, max_db)
    k = max(1, smooth_frames)
    gain_db = np.convolve(gain_db, np.ones(k) / k, mode="same")
    times = np.arange(len(out.samples)) / out.sr
    gain = 10 ** (np.interp(times, o_t, gain_db) / 20)
    return AudioClip((out.samples * gain).astype(np.float32), out.sr)


# --------------------------------------------------------------------------- misc
def crossfade_concat(chunks: List[np.ndarray], sr: int, fade_ms: float = 20.0) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    n = int(sr * fade_ms / 1000)
    out = chunks[0].astype(np.float32)
    for c in chunks[1:]:
        c = c.astype(np.float32)
        m = min(n, len(out), len(c))
        if m > 0:
            ramp = np.linspace(0, 1, m, dtype=np.float32)
            out[-m:] = out[-m:] * (1 - ramp) + c[:m] * ramp
            out = np.concatenate([out, c[m:]])
        else:
            out = np.concatenate([out, c])
    return out


def trim_to_segments(clip: AudioClip, segments: List[Tuple[float, float]], pad_s: float = 0.25) -> Tuple[AudioClip, float]:
    """Trim leading/trailing silence; returns the clip and the trimmed-off start offset."""
    if not segments:
        return clip, 0.0
    start = max(0.0, segments[0][0] - pad_s)
    end = min(clip.duration, segments[-1][1] + pad_s)
    a, b = int(start * clip.sr), int(end * clip.sr)
    return AudioClip(clip.samples[a:b].copy(), clip.sr), start
