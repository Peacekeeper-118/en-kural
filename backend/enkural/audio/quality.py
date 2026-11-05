"""Recording-quality assessment for voice samples and source recordings.

Detects: background noise (low SNR), very low volume, clipping, too little
speech, and excessive background sound between phrases.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import ANALYSIS_SR
from . import vad
from .io import AudioClip

EPS = 1e-9


def _db(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), EPS)))


def frame_rms(x: np.ndarray, sr: int, win_ms: float = 30.0, hop_ms: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """Short-time RMS (linear) and frame centre times in seconds."""
    win = max(1, int(sr * win_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + EPS)
    times = (np.arange(n) * hop + win / 2) / sr
    return rms.astype(np.float32), times.astype(np.float32)


def clipping_stats(x: np.ndarray, thresh: float = 0.985) -> Tuple[float, int]:
    """Fraction of clipped samples and number of clipped runs (>=3 samples)."""
    clipped = np.abs(x) >= thresh
    frac = float(clipped.mean()) if len(x) else 0.0
    if not clipped.any():
        return frac, 0
    d = np.diff(np.concatenate(([0], clipped.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    runs = int(np.sum((ends - starts) >= 3))
    return frac, runs


def noise_and_speech_levels(x: np.ndarray, sr: int, segments: List[Tuple[float, float]]) -> Dict[str, float]:
    rms, times = frame_rms(x, sr)
    rms_db = 20 * np.log10(rms + EPS)
    mask = np.zeros(len(times), dtype=bool)
    for s, e in segments:
        mask |= (times >= s) & (times <= e)
    speech_db = rms_db[mask]
    nonspeech_db = rms_db[~mask]

    if speech_db.size:
        # energy-weighted level of speech frames, ignoring the quietest 20 %
        # (pauses inside VAD segments)
        lo = np.percentile(speech_db, 20)
        core = speech_db[speech_db >= lo]
        speech_level = float(10 * np.log10(np.mean(10 ** (core / 10)) + EPS))
    else:
        speech_level = -100.0

    if nonspeech_db.size >= 30:  # >= 0.3 s of non-speech frames
        noise_floor = float(np.percentile(nonspeech_db, 50))
        noise_p10 = float(np.percentile(nonspeech_db, 10))
        # frames between phrases that are much louder than the floor: bursts of
        # background sound (music, other people, traffic, keyboard, ...)
        bg_activity = float(np.mean(nonspeech_db > noise_p10 + 12.0))
    else:
        noise_floor = float(np.percentile(rms_db, 5))
        noise_p10 = noise_floor
        bg_activity = 0.0

    return {
        "speech_level_db": speech_level,
        "noise_floor_db": noise_floor,
        "noise_min_db": noise_p10,
        "snr_db": float(speech_level - noise_floor),
        "background_activity": bg_activity,
        "peak_db": _db(np.max(np.abs(x)) if len(x) else 0.0),
    }


def assess(clip: AudioClip, purpose: str = "source", segments: Optional[List[Tuple[float, float]]] = None) -> Dict:
    """Assess a recording.  ``purpose`` is ``"profile"`` (voice sample) or ``"source"``."""
    c = clip.resampled(ANALYSIS_SR)
    x = c.samples.astype(np.float32)
    duration = c.duration
    if segments is None:
        segments = vad.speech_segments(c)
    speech_dur = vad.speech_duration(segments)
    speech_ratio = speech_dur / duration if duration > 0 else 0.0
    levels = noise_and_speech_levels(x, ANALYSIS_SR, segments)
    clip_frac, clip_runs = clipping_stats(x)
    dc = float(np.mean(x)) if len(x) else 0.0

    issues: List[Dict] = []
    score = 100.0

    def issue(code: str, severity: str, title: str, detail: str, penalty: float):
        nonlocal score
        issues.append({"code": code, "severity": severity, "title": title, "detail": detail})
        score -= penalty

    # --- amount of speech --------------------------------------------------
    if purpose == "profile":
        if speech_dur < 15:
            issue("too_little_speech", "poor", "Too little speech",
                  f"Only {speech_dur:.0f} s of speech detected. Record at least 1 minute (1–5 minutes is ideal).", 40)
        elif speech_dur < 60:
            issue("too_little_speech", "warn", "Short recording",
                  f"{speech_dur:.0f} s of speech detected. 1–5 minutes of speech gives a more accurate voice profile.", 15)
    else:
        if speech_dur < 0.4:
            issue("too_little_speech", "poor", "No speech detected",
                  "The recording does not seem to contain speech.", 60)

    # --- volume ------------------------------------------------------------
    lvl = levels["speech_level_db"]
    if speech_dur > 0.3:
        if lvl < -38:
            issue("low_volume", "poor", "Very low volume",
                  f"Speech level is {lvl:.0f} dBFS. Move closer to the microphone or raise the input gain.", 25)
        elif lvl < -30:
            issue("low_volume", "warn", "Low volume",
                  f"Speech level is {lvl:.0f} dBFS. A level around -20 dBFS is ideal.", 10)

    # --- clipping ----------------------------------------------------------
    if clip_runs >= 20 or clip_frac > 0.002:
        issue("clipping", "poor", "Clipping / distortion",
              f"{clip_runs} clipped bursts detected (peak {levels['peak_db']:.1f} dBFS). Lower the input gain or speak further from the mic.", 25)
    elif clip_runs > 0 or levels["peak_db"] > -0.3:
        issue("clipping", "warn", "Occasional clipping",
              f"{clip_runs} short clipped bursts. Leave more headroom (peaks below -3 dBFS).", 8)

    # --- background noise (SNR) -------------------------------------------
    snr = levels["snr_db"]
    if speech_dur > 0.5:
        if snr < 12:
            issue("background_noise", "poor", "Loud background noise",
                  f"Signal-to-noise ratio is only {snr:.0f} dB. Record in a quieter room, away from fans and traffic.", 30)
        elif snr < 20:
            issue("background_noise", "warn", "Background noise",
                  f"Signal-to-noise ratio is {snr:.0f} dB (aim for 25 dB or more). Noise reduction will be applied.", 12)

    # --- excessive background sound between phrases -----------------------
    if levels["background_activity"] > 0.35 and speech_dur > 2:
        issue("excessive_background", "warn", "Excessive background sound",
              "Music, other voices or traffic are audible between phrases. Use a quieter space.", 12)
    elif levels["noise_floor_db"] > -40 and speech_dur > 2:
        issue("excessive_background", "warn", "Constant background sound",
              f"Noise floor is high ({levels['noise_floor_db']:.0f} dBFS): a fan, AC or hum is probably present.", 8)

    if abs(dc) > 0.02:
        issue("dc_offset", "warn", "DC offset", "The recording has a DC offset; it will be filtered out.", 3)

    score = float(max(0.0, min(100.0, score)))
    has_poor = any(i["severity"] == "poor" for i in issues)
    return {
        "score": round(score),
        "ok": not has_poor,
        "duration": round(duration, 2),
        "speech_duration": round(speech_dur, 2),
        "speech_ratio": round(speech_ratio, 3),
        "speech_level_db": round(lvl, 1),
        "noise_floor_db": round(levels["noise_floor_db"], 1),
        "snr_db": round(snr, 1),
        "peak_db": round(levels["peak_db"], 1),
        "clipping_ratio": round(clip_frac, 5),
        "clipped_events": clip_runs,
        "background_activity": round(levels["background_activity"], 3),
        "issues": issues,
        "segments": [[round(s, 3), round(e, 3)] for s, e in segments],
    }
