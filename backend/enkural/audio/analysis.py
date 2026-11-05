"""Voice-characteristic analysis used to describe a voice profile.

The numbers here are *descriptive* (shown on the profile card and used for
sanity checks such as automatic pitch alignment).  The identity used for
conversion itself comes from the neural speaker embedding + reference clip
built by the conversion engine (see enkural.vc).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

from ..config import ANALYSIS_SR
from . import vad
from .io import AudioClip
from .quality import EPS, frame_rms

Segments = List[Tuple[float, float]]


# --------------------------------------------------------------------------- pitch
def pitch_track(x: np.ndarray, sr: int, frame_ms: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """F0 contour (Hz, 0 = unvoiced) and frame times using WORLD (DIO + StoneMask)."""
    import pyworld as pw

    xd = np.ascontiguousarray(x, dtype=np.float64)
    f0, t = pw.dio(xd, sr, f0_floor=55.0, f0_ceil=600.0, frame_period=frame_ms)
    f0 = pw.stonemask(xd, f0, t, sr)
    return f0.astype(np.float32), t.astype(np.float32)


def _frame_mask(times: np.ndarray, segments: Segments) -> np.ndarray:
    m = np.zeros(len(times), dtype=bool)
    for s, e in segments:
        m |= (times >= s) & (times <= e)
    return m


def pitch_summary(f0: np.ndarray, t: np.ndarray, segments: Segments) -> Dict:
    voiced = f0[(f0 > 0) & _frame_mask(t, segments)]
    if voiced.size < 20:
        return {"available": False}
    median = float(np.median(voiced))
    p5, p95 = float(np.percentile(voiced, 5)), float(np.percentile(voiced, 95))
    st = 12 * np.log2(voiced / median)
    std_st = float(np.std(st))
    range_st = float(12 * np.log2(max(p95, 1) / max(p5, 1)))
    if median < 110:
        level = "low"
    elif median < 150:
        level = "medium-low"
    elif median < 200:
        level = "medium"
    elif median < 260:
        level = "medium-high"
    else:
        level = "high"
    if std_st < 1.6:
        variability = "flat / monotone"
    elif std_st < 3.0:
        variability = "moderately varied"
    else:
        variability = "expressive"
    return {
        "available": True,
        "median_hz": round(median, 1),
        "mean_hz": round(float(np.mean(voiced)), 1),
        "p5_hz": round(p5, 1),
        "p95_hz": round(p95, 1),
        "range_semitones": round(range_st, 1),
        "std_semitones": round(std_st, 2),
        "level": level,
        "variability": variability,
        "voiced_ratio": round(float(np.mean(f0[_frame_mask(t, segments)] > 0)), 3),
    }


# --------------------------------------------------------------------------- texture
def _hnr_acf(x: np.ndarray, sr: int, f0: np.ndarray, t: np.ndarray, mask: np.ndarray,
             max_frames: int = 400) -> float:
    """Harmonics-to-noise ratio (dB) from the normalised autocorrelation peak at
    the pitch period of voiced frames (Praat-style)."""
    win = int(0.04 * sr)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return 0.0
    if idx.size > max_frames:
        idx = idx[np.linspace(0, idx.size - 1, max_frames).astype(int)]
    vals = []
    window = np.hanning(win)
    for i in idx:
        c = int(t[i] * sr)
        a, b = c - win // 2, c - win // 2 + win
        if a < 0 or b > len(x) or f0[i] <= 0:
            continue
        frame = x[a:b].astype(np.float64)
        frame = (frame - frame.mean()) * window
        e = np.dot(frame, frame)
        if e < 1e-8:
            continue
        lag0 = int(round(sr / f0[i]))
        lo, hi = max(2, int(lag0 * 0.8)), min(win - 1, int(lag0 * 1.2))
        if hi <= lo:
            continue
        ac = np.correlate(frame, frame, mode="full")[win - 1:]
        wac = np.correlate(window, window, mode="full")[win - 1:]  # window autocorrelation
        r = (ac[lo:hi] / (wac[lo:hi] + 1e-12)) / (ac[0] / wac[0])
        peak = float(np.clip(np.max(r), 1e-4, 1 - 1e-4))
        vals.append(10 * np.log10(peak / (1 - peak)))
    return float(np.median(vals)) if vals else 0.0


def texture_summary(x: np.ndarray, sr: int, f0: np.ndarray, t: np.ndarray, segments: Segments,
                    max_seconds: float = 90.0) -> Dict:
    """Breathiness / harmonicity / stability estimates."""
    import pyworld as pw

    n = min(len(x), int(max_seconds * sr))
    xd = np.ascontiguousarray(x[:n], dtype=np.float64)
    nf = min(len(f0), int(n / sr / (t[1] - t[0])) if len(t) > 1 else len(f0))
    f0d, td = f0[:nf].astype(np.float64), t[:nf].astype(np.float64)
    mask = (f0d > 0) & _frame_mask(td, segments)
    if mask.sum() < 20:
        return {"available": False}
    hnr_db = _hnr_acf(x[:n], sr, f0d, td, mask)
    ap = pw.d4c(xd, f0d, td, sr)  # (frames, bins) in [0, 1]
    bins = ap.shape[1]
    hi = max(2, int(bins * (2000.0 / (sr / 2))))
    ap_mean = float(np.clip(np.mean(ap[mask, :hi]), 1e-3, 1 - 1e-3))

    # jitter: cycle-to-cycle F0 irregularity (approximated from 10 ms frames)
    idx = np.flatnonzero(mask)
    consecutive = idx[1:][np.diff(idx) == 1]
    if consecutive.size > 10:
        d = np.abs(f0d[consecutive] - f0d[consecutive - 1])
        jitter = float(np.mean(d) / np.mean(f0d[consecutive]) * 100)
    else:
        jitter = 0.0
    rms, rt = frame_rms(x[:n], sr)
    rmask = np.interp(rt, td, mask.astype(np.float32)) > 0.5
    ridx = np.flatnonzero(rmask)
    rcons = ridx[1:][np.diff(ridx) == 1]
    if rcons.size > 10:
        shimmer = float(np.mean(np.abs(rms[rcons] - rms[rcons - 1])) / (np.mean(rms[rcons]) + EPS) * 100)
    else:
        shimmer = 0.0

    if hnr_db > 15:
        quality = "clear, clean tone"
    elif hnr_db > 10:
        quality = "slightly breathy tone"
    else:
        quality = "breathy, husky tone"
    stability = "steady" if jitter < 1.5 else ("lightly textured" if jitter < 3 else "rough / textured")
    return {
        "available": True,
        "hnr_db": round(hnr_db, 1),
        "aperiodicity": round(ap_mean, 3),
        "jitter_pct": round(jitter, 2),
        "shimmer_pct": round(shimmer, 2),
        "quality": quality,
        "stability": stability,
    }


# --------------------------------------------------------------------------- timbre
def _formants(x: np.ndarray, sr: int, f0: np.ndarray, t: np.ndarray, segments: Segments,
              max_frames: int = 300) -> Dict:
    import librosa

    order = 2 + sr // 1000
    win = int(0.025 * sr)
    mask = (f0 > 0) & _frame_mask(t, segments)
    idx = np.flatnonzero(mask)
    if idx.size < 10:
        return {}
    if idx.size > max_frames:
        idx = idx[np.linspace(0, idx.size - 1, max_frames).astype(int)]
    pre = np.append(x[0], x[1:] - 0.97 * x[:-1]).astype(np.float64)
    window = np.hamming(win)
    found: List[List[float]] = []
    for i in idx:
        c = int(t[i] * sr)
        a, b = c - win // 2, c - win // 2 + win
        if a < 0 or b > len(pre):
            continue
        frame = pre[a:b] * window
        if np.max(np.abs(frame)) < 1e-4:
            continue
        try:
            A = librosa.lpc(frame, order=order)
        except Exception:
            continue
        roots = np.roots(A)
        roots = roots[np.imag(roots) >= 0.01]
        freqs = np.angle(roots) * sr / (2 * np.pi)
        bws = -0.5 * (sr / (2 * np.pi)) * np.log(np.abs(roots) + EPS)
        cand = sorted(f for f, bw in zip(freqs, bws) if 90 < f < 4800 and bw < 450)
        if len(cand) >= 3:
            found.append(cand[:3])
    if len(found) < 5:
        return {}
    arr = np.array(found)
    return {
        "f1_hz": round(float(np.median(arr[:, 0]))),
        "f2_hz": round(float(np.median(arr[:, 1]))),
        "f3_hz": round(float(np.median(arr[:, 2]))),
    }


def timbre_summary(speech: np.ndarray, sr: int, x: np.ndarray, f0: np.ndarray, t: np.ndarray,
                   segments: Segments, max_seconds: float = 120.0) -> Dict:
    import librosa

    y = speech[: int(max_seconds * sr)].astype(np.float32)
    if len(y) < sr // 2:
        return {"available": False}
    n_fft, hop = 1024, 256
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    energy = S.sum(axis=0)
    keep = energy > np.percentile(energy, 30)  # drop the quietest frames
    S = S[:, keep] if keep.sum() > 10 else S
    centroid = float(librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr).mean())
    bandwidth = float(librosa.feature.spectral_bandwidth(S=np.sqrt(S), sr=sr).mean())
    rolloff = float(librosa.feature.spectral_rolloff(S=np.sqrt(S), sr=sr, roll_percent=0.85).mean())
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    ltas = 10 * np.log10(S.mean(axis=1) + EPS)
    band = (freqs >= 100) & (freqs <= 4000)
    slope = float(np.polyfit(np.log2(freqs[band]), ltas[band], 1)[0])  # dB / octave
    lo = S[(freqs >= 0) & (freqs < 2000)].sum()
    hi = S[(freqs >= 2000) & (freqs < 8000)].sum()
    hf_ratio_db = float(10 * np.log10((hi + EPS) / (lo + EPS)))

    if centroid < 1400:
        brightness = "warm / dark"
    elif centroid < 2300:
        brightness = "balanced"
    else:
        brightness = "bright"
    if hf_ratio_db > -10:
        articulation = "crisp consonants"
    elif hf_ratio_db > -18:
        articulation = "clear articulation"
    else:
        articulation = "soft articulation"
    out = {
        "available": True,
        "spectral_centroid_hz": round(centroid),
        "spectral_bandwidth_hz": round(bandwidth),
        "rolloff85_hz": round(rolloff),
        "spectral_tilt_db_per_octave": round(slope, 2),
        "hf_energy_ratio_db": round(hf_ratio_db, 1),
        "brightness": brightness,
        "articulation": articulation,
        "mfcc_mean": [round(float(v), 2) for v in mfcc.mean(axis=1)],
        "mfcc_std": [round(float(v), 2) for v in mfcc.std(axis=1)],
    }
    out.update(_formants(x, sr, f0, t, segments))
    return out


# --------------------------------------------------------------------------- rhythm
def rhythm_summary(x: np.ndarray, sr: int, segments: Segments) -> Dict:
    """Syllable-nucleus based speaking rate and rhythm regularity (language-agnostic)."""
    speech_dur = vad.speech_duration(segments)
    if speech_dur < 1.0:
        return {"available": False}
    sos = signal.butter(4, [300, 2500], btype="bandpass", fs=sr, output="sos")
    bp = signal.sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)
    env, times = frame_rms(bp, sr, win_ms=40, hop_ms=10)
    env = np.convolve(env, np.ones(5) / 5, mode="same")
    mask = _frame_mask(times, segments)
    if mask.sum() < 20:
        return {"available": False}
    ref = np.percentile(env[mask], 95) + EPS
    envm = np.where(mask, env, 0.0)
    peaks, _ = signal.find_peaks(envm, distance=8, height=0.12 * ref, prominence=0.06 * ref)
    n_syl = int(len(peaks))
    if n_syl < 4:
        return {"available": False}
    first, last = segments[0][0], segments[-1][1]
    active = max(last - first, 1e-3)
    articulation_rate = n_syl / speech_dur
    speaking_rate = n_syl / active

    # normalised pairwise variability index of inter-nucleus intervals
    pt = times[peaks]
    seg_id = np.zeros(len(pt), dtype=int)
    for k, (s, e) in enumerate(segments):
        seg_id[(pt >= s) & (pt <= e)] = k
    intervals = []
    for k in np.unique(seg_id):
        p = pt[seg_id == k]
        if len(p) >= 3:
            intervals.append(np.diff(p))
    if intervals:
        d = np.concatenate(intervals)
        pv = np.abs(np.diff(d)) / ((d[1:] + d[:-1]) / 2 + EPS)
        npvi = float(100 * np.mean(pv)) if pv.size else 0.0
    else:
        npvi = 0.0

    if articulation_rate < 4.0:
        pace = "slow, deliberate"
    elif articulation_rate < 5.5:
        pace = "measured"
    elif articulation_rate < 7.0:
        pace = "brisk"
    else:
        pace = "fast"
    regularity = "even" if npvi < 40 else ("natural" if npvi < 62 else "varied / dramatic")
    return {
        "available": True,
        "syllables_estimated": n_syl,
        "articulation_rate_sps": round(articulation_rate, 2),
        "speaking_rate_sps": round(speaking_rate, 2),
        "npvi": round(npvi, 1),
        "pace": pace,
        "regularity": regularity,
    }


def pause_summary(segments: Segments, duration: float) -> Dict:
    gaps = vad.pauses(segments, 0.15)
    if not segments:
        return {"available": False}
    first, last = segments[0][0], segments[-1][1]
    active = max(last - first, 1e-3)
    n = len(gaps)
    per_min = n / (active / 60.0)
    mean = float(np.mean(gaps)) if gaps else 0.0
    median = float(np.median(gaps)) if gaps else 0.0
    longest = float(np.max(gaps)) if gaps else 0.0
    if not gaps:
        style = "continuous (almost no pauses)"
    elif mean < 0.3:
        style = "short, quick pauses"
    elif mean < 0.6:
        style = "natural pauses"
    else:
        style = "long, deliberate pauses"
    return {
        "available": True,
        "count": n,
        "per_minute": round(per_min, 1),
        "mean_s": round(mean, 2),
        "median_s": round(median, 2),
        "longest_s": round(longest, 2),
        "speech_ratio": round(vad.speech_duration(segments) / max(duration, 1e-3), 3),
        "style": style,
    }


def loudness_summary(speech: np.ndarray, sr: int) -> Dict:
    import pyloudnorm as pyln

    y = speech.astype(np.float64)
    if len(y) < int(0.5 * sr):
        return {"available": False}
    meter = pyln.Meter(sr)
    try:
        lufs = float(meter.integrated_loudness(y))
    except Exception:
        lufs = float("-inf")
    win, hop = int(3 * sr), int(1 * sr)
    st = []
    for a in range(0, max(1, len(y) - win + 1), hop):
        seg = y[a : a + win]
        if len(seg) < int(0.4 * sr):
            break
        try:
            v = meter.integrated_loudness(seg)
        except Exception:
            continue
        if np.isfinite(v):
            st.append(v)
    lra = float(np.percentile(st, 95) - np.percentile(st, 10)) if len(st) >= 3 else 0.0
    dyn = "steady" if lra < 4 else ("moderately dynamic" if lra < 8 else "very dynamic")
    return {
        "available": True,
        "integrated_lufs": round(lufs, 1) if np.isfinite(lufs) else None,
        "loudness_range_lu": round(lra, 1),
        "dynamics": dyn,
    }


# --------------------------------------------------------------------------- top level
def analyze(clip: AudioClip, segments: Optional[Segments] = None, max_seconds: float = 300.0) -> Dict:
    c = clip.resampled(ANALYSIS_SR)
    sr = ANALYSIS_SR
    x = c.samples.astype(np.float32)
    if segments is None:
        segments = vad.speech_segments(c)
    # cap the analysed region for speed (keeps whole segments)
    kept: Segments = []
    total = 0.0
    for s, e in segments:
        if total >= max_seconds:
            break
        kept.append((s, e))
        total += e - s
    segments = kept
    if segments:
        x = x[: int(min(len(x), (segments[-1][1] + 0.5) * sr))]
    speech = vad.extract_speech(AudioClip(x, sr), segments).samples

    f0, t = pitch_track(x, sr)
    result = {
        "speech_seconds_analyzed": round(vad.speech_duration(segments), 1),
        "pitch": pitch_summary(f0, t, segments),
        "texture": texture_summary(x, sr, f0, t, segments),
        "timbre": timbre_summary(speech, sr, x, f0, t, segments),
        "rhythm": rhythm_summary(x, sr, segments),
        "pauses": pause_summary(segments, c.duration),
        "loudness": loudness_summary(speech, sr),
    }
    result["summary"] = summarize(result)
    return result


def summarize(a: Dict) -> str:
    parts = []
    p = a.get("pitch", {})
    if p.get("available"):
        parts.append(f"{p['level']} pitch (median {p['median_hz']:.0f} Hz), {p['variability']} intonation")
    tm = a.get("timbre", {})
    tx = a.get("texture", {})
    if tm.get("available") and tx.get("available"):
        parts.append(f"{tm['brightness']} timbre with a {tx['quality']}")
    elif tm.get("available"):
        parts.append(f"{tm['brightness']} timbre")
    r = a.get("rhythm", {})
    if r.get("available"):
        parts.append(f"{r['pace']} pace ({r['articulation_rate_sps']:.1f} syllables/s), {r['regularity']} rhythm")
    ps = a.get("pauses", {})
    if ps.get("available"):
        parts.append(ps["style"])
    if tm.get("available"):
        parts.append(tm["articulation"])
    if not parts:
        return "Not enough speech to analyse."
    text = "; ".join(parts) + "."
    return text[0].upper() + text[1:]
