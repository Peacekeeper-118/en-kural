"""The conversion pipeline (one job = one source recording → target voice).

    source → [voice / background separation] → high-pass → [denoise]
           → engine.convert (with the profile's fine-tuned weights, if any)
           → loudness-contour restore → [background remix] → loudness normalise

``convert_clip`` does the work and is shared by the job runner, the benchmark
script and fine-tune checkpoint selection; ``run_conversion`` adds the job
bookkeeping (consent check, files on disk, voice-match score).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import config, profiles, store
from .audio import enhance, quality, separate
from .audio.io import AudioClip, encode_mp3, load_audio, time_stretch, write_wav
from .jobs import JobContext
from .vc.base import ConversionSettings, TargetVoice
from .vc.registry import get_engine

log = logging.getLogger(__name__)


class QuietContext:
    """Stand-in for JobContext outside the job queue (benchmarks, evaluation)."""

    def stage(self, key: str, detail: str = "", label: Optional[str] = None) -> None:
        pass

    def progress(self, fraction: float, detail: Optional[str] = None) -> None:
        pass

    def check(self) -> None:
        pass


def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ------------------------------------------------------------------ target voice
def finetuned_checkpoint(profile: Dict, engine_id: str) -> Optional[Path]:
    """The profile's fine-tuned checkpoint for ``engine_id``, if one is ready."""
    ft = profile.get("finetune") or {}
    if ft.get("status") != "ready" or ft.get("engine") != engine_id or not ft.get("checkpoint"):
        return None
    path = store.profile_dir(profile["id"]) / ft["checkpoint"]
    return path if path.exists() else None


def target_for(profile: Dict, engine_id: str, settings: ConversionSettings, ctx=None) -> TargetVoice:
    target = profiles.ensure_engine_target(profile, engine_id, ctx)
    ckpt = finetuned_checkpoint(profile, engine_id) if settings.use_finetuned else None
    if ckpt:
        target.meta = {**target.meta, "dit_checkpoint": str(ckpt)}
    return target


# ------------------------------------------------------------------ source preparation
def prepare_source(clip: AudioClip, settings: ConversionSettings, source_quality: Optional[Dict] = None,
                   ctx=None) -> Dict:
    """Isolate and clean the voice to convert.

    Returns ``{"voice", "background" (44.1 kHz or None), "separation", "denoised", "snr_db"}``.
    """
    ctx = ctx or QuietContext()
    if clip.sr < 16000:
        clip = clip.resampled(16000)
    voice, background, sep_info = clip, None, None
    if settings.separate != "off" and separate.available():
        ctx.progress(0.1, "Separating the voice from background sound")
        try:
            sep = separate.separate(clip)
        except Exception as e:  # an enhancement: never fail the conversion over it
            log.warning("Voice separation unavailable, converting the recording as is: %s", e)
            sep = None
        if sep is not None:
            applied = settings.separate == "on" or sep.significant
            sep_info = sep.summary(applied)
            if applied:
                voice, background = sep.vocals, sep.background
    if sep_info and sep_info["applied"]:
        q = quality.assess(voice, purpose="source")
    else:
        q = source_quality or quality.assess(clip, purpose="source")
    x = enhance.highpass(voice.samples, voice.sr, 60.0)
    snr = float(q.get("snr_db", 99.0))
    denoised = settings.denoise == "on" or (settings.denoise == "auto" and snr < 24.0)
    if denoised:
        ctx.progress(0.6, "Reducing background noise")
        x = enhance.denoise(x, voice.sr, strength=0.75 if snr < 15 else 0.55)
    ctx.progress(1.0, "Source ready")
    return {"voice": AudioClip(x.astype(np.float32), voice.sr), "background": background,
            "separation": sep_info, "denoised": denoised, "snr_db": snr}


def remix_background(voice_out: AudioClip, voice_src: AudioClip, background: AudioClip, speed: float) -> AudioClip:
    """Put the separated background back under the converted voice (at 44.1 kHz).

    The converter outputs its own level, so the voice is first brought back to
    the source voice's loudness: the voice / music balance stays as it was.
    """
    sr = background.sr
    v = voice_out.resampled(sr)
    lo = enhance.measure_lufs(v.samples, sr)
    ls = enhance.measure_lufs(voice_src.samples, voice_src.sr)
    gain_db = float(np.clip(ls - lo, -12.0, 12.0)) if np.isfinite(lo) and np.isfinite(ls) else 0.0
    bg = time_stretch(background, speed) if abs(speed - 1.0) > 1e-3 else background
    b = bg.samples[: len(v.samples)]
    if len(b) < len(v.samples):
        b = np.pad(b, (0, len(v.samples) - len(b)))
    return AudioClip((v.samples * 10 ** (gain_db / 20) + b).astype(np.float32), sr)


# ------------------------------------------------------------------ conversion
def convert_clip(profile: Dict, clip: AudioClip, settings: ConversionSettings, engine_id: Optional[str] = None,
                 ctx=None, source_quality: Optional[Dict] = None) -> Dict:
    """Convert one recording to the profile's voice.

    Returns ``{"final": mastered output, "voice": converted voice only (before
    remix / normalisation, for scoring), "source_voice": the cleaned source
    voice, "target": TargetVoice, "engine": engine, "info": facts + timings}``.
    """
    ctx = ctx or QuietContext()
    engine_id = engine_id or config.DEFAULT_ENGINE
    t0 = time.perf_counter()

    # 1. target voice ------------------------------------------------------
    ctx.stage("analyze", "Loading your voice profile")
    engine = get_engine(engine_id)
    target = target_for(profile, engine_id, settings, ctx)
    ctx.progress(1.0, "Voice profile loaded")
    t_profile = time.perf_counter() - t0

    # 2. source clean-up ---------------------------------------------------
    ctx.stage("extract", "Preparing the source recording")
    t1 = time.perf_counter()
    src = prepare_source(clip, settings, source_quality, ctx)
    src_voice = src["voice"]
    t_source = time.perf_counter() - t1

    # 3. conversion --------------------------------------------------------
    ctx.stage("convert", "Converting voice")
    t2 = time.perf_counter()
    out = engine.convert(src_voice, target, settings, progress=lambda f, m: ctx.progress(f, m))
    t_convert = time.perf_counter() - t2

    # 4. enhancement -------------------------------------------------------
    ctx.stage("enhance", "Restoring loudness contour")
    t3 = time.perf_counter()
    if settings.preserve_dynamics:
        out = enhance.match_envelope(out, src_voice, energy=settings.energy, strength=1.0)
    voice = AudioClip(enhance.highpass(out.samples, out.sr, 50.0), out.sr)
    mix, remixed = voice, False
    if src["background"] is not None and settings.keep_background:
        ctx.progress(0.4, "Mixing the background back in")
        mix, remixed = remix_background(voice, src_voice, src["background"], settings.speed), True
    y = enhance.fade_edges(mix.samples, mix.sr, 6.0)
    ctx.progress(0.7, "Normalising loudness")
    y = enhance.normalize_loudness(y, mix.sr, target_lufs=config.TARGET_LUFS, true_peak=config.TRUE_PEAK_DB)
    final = AudioClip(y.astype(np.float32), mix.sr)
    t_enhance = time.perf_counter() - t3

    total = time.perf_counter() - t0
    sep = dict(src["separation"] or {"applied": False})
    sep["background_remixed"] = remixed
    info = {
        "engine": engine_id,
        "engine_info": dict(engine.last_info),
        "finetuned": bool(engine.last_info.get("finetuned")),
        "denoised": src["denoised"],
        "source_snr_db": round(src["snr_db"], 1),
        "separation": sep,
        "timings": {
            "total_s": round(total, 3),
            "realtime_factor": round(total / clip.duration, 3) if clip.duration else None,
            "profile_s": round(t_profile, 3),
            "source_prep_s": round(t_source, 3),
            "convert_s": round(t_convert, 3),
            "enhance_s": round(t_enhance, 3),
            "engine_stages_s": dict(engine.last_timings),
        },
    }
    return {"final": final, "voice": voice, "source_voice": src_voice, "target": target,
            "engine": engine, "info": info}


def run_conversion(ctx: JobContext) -> Dict:
    job = ctx.job
    settings = ConversionSettings.from_dict(job.get("settings"))
    job["settings"] = settings.to_dict()
    profile = store.get_profile(job["profile_id"])
    if not profile:
        raise KeyError("Voice profile not found.")
    if not profile.get("consent", {}).get("confirmed"):
        raise PermissionError("Conversion refused: the target voice profile has no recorded consent.")
    source = store.get_source(job["source_id"])
    if not source:
        raise KeyError("Source recording not found.")
    engine_id = job.get("engine") or config.DEFAULT_ENGINE

    clip = load_audio(store.source_dir(source["id"]) / source["file"], sr=None)
    res = convert_clip(profile, clip, settings, engine_id, ctx, source.get("quality"))
    final, info, target, engine = res["final"], res["info"], res["target"], res["engine"]

    ctx.stage("finalize", "Writing audio")
    jdir = store.secure_dir(store.job_dir(job["id"]))
    out_path = write_wav(jdir / "output.wav", final, subtype="PCM_24")
    match = None
    try:
        style_file = target.dir / target.meta.get("style_file", "")
        style = np.load(style_file) if style_file.exists() else None
        # score the voice alone: a remixed music bed would skew the embedding
        match = {
            "converted_vs_target": _cosine(engine.speaker_embedding(res["voice"]), style),
            "source_vs_target": _cosine(engine.speaker_embedding(res["source_voice"]), style),
        }
    except Exception as e:  # evaluation is optional
        log.debug("speaker match not computed: %s", e)
    ctx.progress(1.0, "Done")
    return {
        "output": {
            "file": out_path.name,
            "duration": round(final.duration, 2),
            "sample_rate": final.sr,
            "lufs": round(enhance.measure_lufs(final.samples, final.sr), 1),
            "engine": engine_id,
            "denoised": info["denoised"],
            "source_snr_db": info["source_snr_db"],
            "voice_match": match,
            "separation": info["separation"],
            "finetuned": info["finetuned"],
            "timings": info["timings"],
        },
        "done_label": "Voice conversion complete.",
    }


def export_path(job: Dict, fmt: str) -> Path:
    """Return (creating on demand) the export file for a finished job."""
    jdir = store.job_dir(job["id"])
    src = jdir / job["output"]["file"]
    if fmt == "wav":
        dest = jdir / "export.wav"
        if not dest.exists():
            clip = load_audio(src, sr=config.EXPORT_SR)
            write_wav(dest, clip, subtype="PCM_16")
        return dest
    if fmt == "mp3":
        dest = jdir / "export.mp3"
        if not dest.exists():
            encode_mp3(src, dest, bitrate="320k", sr=config.EXPORT_SR)
        return dest
    raise ValueError("format must be wav or mp3")
