"""Voice-profile building.

    samples (uploads / recordings)
      → voice / background separation where music or loud background is found
      → quality re-assessment + clean-up (high-pass, optional denoise)
      → usable-sample selection (poor recordings left out when enough clean speech exists)
      → reference selection (cleanest, speech-dense phrases, ~11 s)
      → voice-characteristics analysis (descriptive)
      → engine target (speaker embedding averaged over the usable samples + reference)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config, store
from .audio import analysis, enhance, quality, separate, vad
from .audio.io import AudioClip, load_audio, probe, write_wav
from .jobs import JobContext
from .vc.base import TargetVoice
from .vc.registry import get_engine

log = logging.getLogger(__name__)

WORK_SR = 44100
REFERENCE_TARGET_S = 10.0
REFERENCE_MIN_S = 4.0
REFERENCE_MAX_S = 11.0
# clean speech needed before poor recordings are dropped from the identity embedding
MIN_IDENTITY_SPEECH_S = 20.0
# recording-condition problems that make a sample "poor" (too little speech is
# not one of them: several short clean samples are fine together)
CONDITION_ISSUES = {"background_noise", "clipping", "low_volume"}

CONSENT_STATEMENT = (
    "I confirm that this is my own voice, or that the person whose voice this is has given me "
    "explicit permission to create a voice profile from these recordings and to use it for "
    "voice conversion. I will not use it to impersonate anyone or to deceive."
)


# ------------------------------------------------------------------ samples
def add_sample(profile: Dict, original_name: str, data: bytes, kind: str = "upload") -> Dict:
    ext = Path(original_name).suffix.lower() or ".bin"
    if ext not in config.ALLOWED_AUDIO_EXT and ext != ".bin":
        raise ValueError(f"Unsupported audio format '{ext}'.")
    sid = store.new_id()
    dest = store.profile_dir(profile["id"]) / "samples" / f"{sid}{ext}"
    store.save_upload(dest, data)
    try:
        info = probe(dest)
        if info["duration"] > config.MAX_SAMPLE_SECONDS:
            raise ValueError(f"Sample is longer than {config.MAX_SAMPLE_SECONDS // 60} minutes.")
        clip = load_audio(dest, sr=config.ANALYSIS_SR)
        q = quality.assess(clip, purpose="profile")
    except Exception:
        try:
            os.remove(dest)
        except OSError:
            pass
        raise
    sample = {
        "id": sid,
        "file": dest.name,
        "original_name": original_name[:120],
        "kind": kind,
        "duration": round(info["duration"] or clip.duration, 2),
        "sample_rate": info.get("sample_rate"),
        "codec": info.get("codec"),
        "created_at": store.now_iso(),
        "quality": {k: v for k, v in q.items() if k != "segments"},
    }
    profile["samples"].append(sample)
    profile["status"] = "needs_build"
    store.save_profile(profile)
    return sample


# ------------------------------------------------------------------ clean-up
def _clean(clip: AudioClip, q: Dict) -> AudioClip:
    x = enhance.highpass(clip.samples, clip.sr, 60.0)
    snr = q.get("snr_db", 99)
    if snr < 25:
        strength = 0.75 if snr < 15 else 0.55
        x = enhance.denoise(x, clip.sr, strength=strength)
    return AudioClip(x.astype(np.float32), clip.sr)


def _phrases(segments: List[Tuple[float, float]], max_gap: float = 0.35, max_len: float = 8.0) -> List[Tuple[float, float]]:
    """Merge VAD segments separated by short gaps into phrases."""
    out: List[Tuple[float, float]] = []
    for s, e in segments:
        if out and s - out[-1][1] <= max_gap and (e - out[-1][0]) <= max_len:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def select_reference(items: List[Dict]) -> Tuple[AudioClip, List[Dict]]:
    """Pick the cleanest, most speech-dense phrases across all samples.

    ``items``: [{"sample": sample_meta, "clip": cleaned AudioClip @WORK_SR,
                 "segments": [...], "quality": {...}}]
    Returns the concatenated reference clip and the list of chosen phrases.
    """
    cands = []
    for idx, it in enumerate(items):
        clip, q = it["clip"], it["quality"]
        sr = clip.sr
        noise_floor = q.get("noise_floor_db", -60.0)
        for (s, e) in _phrases(it["segments"]):
            dur = e - s
            if dur < 0.8:
                continue
            seg = clip.samples[int(s * sr): int(e * sr)]
            if len(seg) < sr // 4:
                continue
            rms_db = 20 * np.log10(np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-9)
            local_snr = rms_db - noise_floor
            clip_frac, clip_runs = quality.clipping_stats(seg)
            score = (
                q.get("score", 50) * 0.5
                + float(np.clip(local_snr, 0, 40)) * 1.5
                - clip_runs * 5.0
                - (25.0 if rms_db < -35 else 0.0)
                + (6.0 if 2.0 <= dur <= 6.0 else 0.0)
            )
            cands.append({"item": idx, "start": s, "end": e, "dur": dur, "score": score,
                          "snr": local_snr, "sample_id": it["sample"]["id"]})
    chosen: List[Dict] = []
    total = 0.0
    for c in sorted(cands, key=lambda c: c["score"], reverse=True):
        if total >= REFERENCE_TARGET_S:
            break
        if total + c["dur"] > REFERENCE_MAX_S:
            continue
        chosen.append(c)
        total += c["dur"]
    if total < REFERENCE_MIN_S and cands:
        # fall back: longest contiguous stretch of the best-scoring sample
        best_item = max(range(len(items)), key=lambda i: items[i]["quality"].get("score", 0))
        it = items[best_item]
        segs = it["segments"]
        if segs:
            start = segs[0][0]
            end = min(segs[-1][1], start + REFERENCE_MAX_S)
        else:
            start, end = 0.0, min(it["clip"].duration, REFERENCE_MAX_S)
        chosen = [{"item": best_item, "start": start, "end": end, "dur": end - start, "score": 0.0,
                   "snr": it["quality"].get("snr_db", 0.0), "sample_id": it["sample"]["id"]}]
    chosen.sort(key=lambda c: (c["item"], c["start"]))
    parts = []
    sr = WORK_SR
    gap = np.zeros(int(0.22 * sr), dtype=np.float32)
    for c in chosen:
        clip = items[c["item"]]["clip"]
        a = int(max(0.0, c["start"] - 0.10) * sr)
        b = int(min(clip.duration, c["end"] + 0.12) * sr)
        seg = enhance.fade_edges(clip.samples[a:b].copy(), sr, 12.0)
        if parts:
            parts.append(gap)
        parts.append(seg)
    ref = np.concatenate(parts) if parts else np.zeros(sr, dtype=np.float32)
    # consistent level for the prompt (-20 LUFS, well below clipping)
    ref = enhance.normalize_loudness(ref.astype(np.float32), sr, target_lufs=-20.0, true_peak=-3.0)
    return AudioClip(ref, sr), [{"sample_id": c["sample_id"], "start": round(c["start"], 2),
                                 "end": round(c["end"], 2), "snr_db": round(float(c["snr"]), 1)} for c in chosen]


# ------------------------------------------------------------------ build
def load_sample_clip(profile: Dict, sample: Dict) -> AudioClip:
    return load_audio(store.profile_dir(profile["id"]) / "samples" / sample["file"], sr=WORK_SR)


def _separated(profile: Dict, sample: Dict, clip: AudioClip) -> Tuple[AudioClip, Dict]:
    """The sample's voice stem when it has music / loud background, else the
    clip itself.  The decision and the stem are cached in the profile folder."""
    cache = store.secure_dir(store.sample_cache_dir(profile["id"]))
    stem = cache / f"{sample['id']}.vocals.wav"
    meta = cache / f"{sample['id']}.separation.json"
    if meta.exists():
        info = json.loads(meta.read_text())
        if not info.get("applied"):
            return clip, info
        if stem.exists():
            return load_audio(stem, sr=WORK_SR), info
    sep = separate.separate(clip)
    info = sep.summary(sep.significant)
    if info["applied"]:
        clip = sep.vocals.resampled(WORK_SR)
        write_wav(stem, clip, subtype="PCM_24")
    meta.write_text(json.dumps(info))
    os.chmod(meta, 0o600)
    return clip, info


def _prepare_items(profile: Dict, ctx: Optional[JobContext] = None) -> List[Dict]:
    items = []
    n = len(profile["samples"])
    for i, s in enumerate(profile["samples"]):
        clip = load_sample_clip(profile, s)
        sep_info = None
        if separate.available():
            if ctx:
                ctx.progress(i / max(1, n), f"Separating voice from background {i + 1}/{n}")
            try:
                clip, sep_info = _separated(profile, s, clip)
            except Exception as e:  # an enhancement: never fail the build over it
                log.warning("Voice separation unavailable for sample %s: %s", s["id"], e)
        segments = vad.speech_segments(clip)
        if sep_info and sep_info["applied"]:  # judge the voice we will actually use
            q = quality.assess(clip, purpose="profile", segments=segments)
        else:
            q = s.get("quality") or quality.assess(clip, purpose="profile", segments=segments)
        clean = _clean(clip, q)
        items.append({"sample": s, "clip": clean, "segments": segments, "quality": q, "separation": sep_info})
        if ctx:
            ctx.progress((i + 1) / max(1, n), f"Cleaned sample {i + 1}/{n}")
    return items


def is_poor(q: Dict) -> bool:
    return any(i.get("severity") == "poor" and i.get("code") in CONDITION_ISSUES for i in q.get("issues", []))


def select_usable(items: List[Dict]) -> Dict:
    """Which samples feed the reference clip and the identity embedding.

    The engine copies the reference clip's acoustics into every output and the
    identity is an average, so one noisy or clipped recording drags the whole
    profile down.  Poor recordings are left out whenever the good ones carry
    enough speech on their own; otherwise everything is used (and flagged).
    """
    good = [it for it in items if not is_poor(it["quality"])]
    good_speech = sum(vad.speech_duration(it["segments"]) for it in good)
    return {
        "reference": good if good and good_speech >= REFERENCE_MIN_S else items,
        "identity": good if good and good_speech >= MIN_IDENTITY_SPEECH_S else items,
        "good_speech": good_speech,
        "all_poor": not good,
    }


def _record_preparation(items: List[Dict], usable: Dict) -> List[str]:
    """Store per-sample preparation facts on the samples; return build notes."""
    ref_ids = {it["sample"]["id"] for it in usable["reference"]}
    id_ids = {it["sample"]["id"] for it in usable["identity"]}
    for it in items:
        sep = it.get("separation") or {}
        it["sample"]["prepared"] = {
            "separated": bool(sep.get("applied")),
            "background_db": sep.get("background_db"),
            "quality_score": it["quality"].get("score"),
            "poor": is_poor(it["quality"]),
            "used_for_reference": it["sample"]["id"] in ref_ids,
            "used_for_identity": it["sample"]["id"] in id_ids,
        }
    notes = []
    n_sep = sum(1 for it in items if (it.get("separation") or {}).get("applied"))
    if n_sep:
        notes.append(f"Background music / noise was removed from {n_sep} sample(s) before use.")
    left_out = [it["sample"]["original_name"] for it in items if it["sample"]["id"] not in (ref_ids | id_ids)]
    if left_out:
        notes.append(f"Left out because of poor recording quality: {', '.join(left_out)}.")
    if usable["all_poor"]:
        notes.append("Every sample has poor recording quality (noise, clipping or low volume). The converted "
                     "voice will inherit those problems — re-record in a quiet room for better results.")
    return notes


def build(profile_id: str, engine_id: Optional[str], ctx: JobContext) -> Dict:
    profile = store.get_profile(profile_id)
    if not profile:
        raise KeyError("Profile not found")
    if not profile.get("consent", {}).get("confirmed"):
        raise PermissionError("Consent is required before a voice profile can be built.")
    if not profile["samples"]:
        raise ValueError("Add at least one voice sample first.")
    engine_id = engine_id or config.DEFAULT_ENGINE
    profile["status"] = "building"
    profile["error"] = None
    store.save_profile(profile)
    try:
        ctx.stage("analyze", "Loading and cleaning samples", label="Analyzing voice...")
        items = _prepare_items(profile, ctx)
        usable = select_usable(items)
        profile["build_notes"] = _record_preparation(items, usable)

        ctx.stage("extract", "Selecting reference speech", label="Extracting speech...")
        reference, chosen = select_reference(usable["reference"])
        pdir = store.profile_dir(profile_id)
        write_wav(pdir / "reference.wav", reference, subtype="PCM_24")

        # descriptive analysis on the cleaned samples (with their natural pauses)
        ctx.progress(0.3, "Analyzing pitch, timbre and rhythm")
        analysis_clip = _concat_for_analysis(usable["identity"], max_seconds=300.0)
        profile["analysis"] = analysis.analyze(analysis_clip)
        ctx.progress(0.9, "Analysis complete")

        ctx.stage("convert", "Building the voice identity", label="Building voice profile...")
        target = _build_engine_target(profile, engine_id, usable["identity"], reference, ctx)

        ctx.stage("finalize", "Saving profile", label="Finalizing...")
        profile["reference"] = {"seconds": round(reference.duration, 2), "phrases": chosen}
        profile["engines"][engine_id] = target.to_dict()
        profile["status"] = "ready"
        profile["built_at"] = store.now_iso()
        store.save_profile(profile)
        return {"profile_id": profile_id, "engine": engine_id, "done_label": "Voice profile ready."}
    except Exception as e:
        profile = store.get_profile(profile_id) or profile
        profile["status"] = "error"
        profile["error"] = f"{type(e).__name__}: {e}"[:300]
        store.save_profile(profile)
        raise


def _concat_for_analysis(items: List[Dict], max_seconds: float) -> AudioClip:
    parts, total = [], 0.0
    for it in sorted(items, key=lambda i: i["quality"].get("score", 0), reverse=True):
        clip = it["clip"]
        take = min(clip.duration, max_seconds - total)
        if take <= 0.5:
            break
        parts.append(clip.samples[: int(take * clip.sr)])
        total += take
    return AudioClip(np.concatenate(parts).astype(np.float32), WORK_SR)


def speech_clips(items: List[Dict], min_seconds: float = 1.0) -> List[AudioClip]:
    """The cleaned samples with pauses removed."""
    clips = [vad.extract_speech(it["clip"], it["segments"]) for it in items]
    return [c for c in clips if c.duration >= min_seconds]


def _build_engine_target(profile: Dict, engine_id: str, items: List[Dict], reference: AudioClip,
                         ctx: Optional[JobContext]) -> TargetVoice:
    engine = get_engine(engine_id)
    clips = speech_clips(items)
    out_dir = store.profile_dir(profile["id"]) / "engines" / engine_id
    store.secure_dir(out_dir)

    def progress(f: float, msg: str) -> None:
        if ctx:
            ctx.progress(f, msg)

    return engine.build_target(reference, clips, out_dir, profile["id"], progress)


def ensure_engine_target(profile: Dict, engine_id: str, ctx: Optional[JobContext] = None) -> TargetVoice:
    """Return the engine's TargetVoice for the profile, building it on demand
    (e.g. after switching to a newly added engine)."""
    entry = profile.get("engines", {}).get(engine_id)
    if entry:
        tv = TargetVoice.from_dict(entry)
        if tv.reference_wav.exists():
            return tv
    if not profile.get("consent", {}).get("confirmed"):
        raise PermissionError("This voice profile has no recorded consent.")
    ref_path = store.profile_dir(profile["id"]) / "reference.wav"
    items = _prepare_items(profile, ctx)
    usable = select_usable(items)
    if ref_path.exists():
        reference = load_audio(ref_path, sr=WORK_SR)
    else:
        reference, _ = select_reference(usable["reference"])
        write_wav(ref_path, reference, subtype="PCM_24")
    tv = _build_engine_target(profile, engine_id, usable["identity"], reference, ctx)
    profile["engines"][engine_id] = tv.to_dict()
    store.save_profile(profile)
    return tv
