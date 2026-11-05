"""Opt-in speaker fine-tuning for Seed-VC v1.

Zero-shot conversion only *imitates* the reference clip.  Fine-tuning adapts
the DiT + length regulator to one speaker from a few minutes of their clean
speech, which usually raises similarity at some risk to clarity.  So every
run is checked: candidate checkpoints are compared with the base model on
recordings of another speaker, scored against *held-out* clips of the target
voice by engine-independent judges, and a checkpoint is kept only if it
sounds more like the target without losing intelligibility or naturalness.

Privacy: training needs a separate, explicit, revocable opt-in per profile
(``profile["training_consent"]``).  Training data is cut into the profile's
own folder and deleted after the run; the kept model lives in
``<profile>/finetune/model.pth`` and is deleted with the profile, when the
consent is withdrawn, or on request.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .. import config, store
from ..audio import vad
from ..audio.io import AudioClip, write_wav

log = logging.getLogger(__name__)

ENGINE_ID = "seedvc_v1"
TRAIN_SR = 22050
STEP_PRESETS = {"quick": 300, "standard": 600, "thorough": 1000}
BATCH_SIZE = 2
WINDOW_MIN_S, WINDOW_MAX_S = 2.0, 12.0
HELDOUT_FRACTION, HELDOUT_MIN_S = 0.15, 6.0
PROBE_SECONDS = 20.0
# checkpoint acceptance (vs the base model, on the probe recordings)
MIN_SIM_GAIN = 0.005
MAX_CER_INCREASE = 0.05
MAX_UTMOS_DROP = 0.15
MAX_F0_CORR_DROP = 0.08   # the source's intonation must survive

TRAINING_CONSENT_STATEMENT = (
    "I allow En Kural to train a voice model on this voice's recordings, on this computer only. "
    "The voice is mine, or its owner clearly agreed to it being used for training. "
    "The trained model stays with this voice and is deleted with it, or when I stop allowing training."
)


def finetune_dir(pid: str) -> Path:
    return store.profile_dir(pid) / "finetune"


# ------------------------------------------------------------------ consent
def set_training_consent(profile: Dict, allow: bool) -> Dict:
    if allow:
        profile["allow_training"] = True
        profile["training_consent"] = {"confirmed": True, "statement": TRAINING_CONSENT_STATEMENT,
                                       "timestamp": store.now_iso()}
    else:  # withdrawing consent removes everything training produced
        profile["allow_training"] = False
        profile["training_consent"] = {"confirmed": False, "withdrawn_at": store.now_iso()}
        delete_finetune(profile, save=False)
    return store.save_profile(profile)


def delete_finetune(profile: Dict, save: bool = True) -> None:
    d = finetune_dir(profile["id"])
    if d.exists():
        shutil.rmtree(d)
    profile["finetune"] = None
    if save:
        store.save_profile(profile)


def training_allowed(profile: Dict) -> bool:
    return bool(profile.get("allow_training") and (profile.get("training_consent") or {}).get("confirmed")
                and profile.get("consent", {}).get("confirmed"))


# ------------------------------------------------------------------ data
def _windows(segments, max_gap: float = 0.6):
    """Merge speech segments into WINDOW_MIN_S..WINDOW_MAX_S training windows."""
    out = []
    for s, e in segments:
        if out and s - out[-1][1] <= max_gap and e - out[-1][0] <= WINDOW_MAX_S:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    res = []
    for s, e in out:
        while e - s > WINDOW_MAX_S:
            res.append((s, s + WINDOW_MAX_S))
            s += WINDOW_MAX_S
        res.append((s, e))
    return [(s, e) for s, e in res if e - s >= WINDOW_MIN_S]


def prepare_dataset(profile: Dict, work: Path, ctx=None) -> Dict:
    """Cut the profile's usable (clean, separated) speech into training clips and
    a held-out set.  Poor recordings are excluded exactly as for the profile."""
    from .. import profiles

    items = profiles._prepare_items(profile, ctx)
    usable = profiles.select_usable(items)
    if usable["all_poor"]:
        raise ValueError("Every sample has poor recording quality; fine-tuning on it would teach the model "
                         "the noise. Add clean recordings first.")
    clips: List[AudioClip] = []
    for it in usable["identity"]:
        c = it["clip"]
        for s, e in _windows(it["segments"]):
            a, b = int(max(0.0, s - 0.1) * c.sr), int(min(c.duration, e + 0.1) * c.sr)
            clips.append(AudioClip(c.samples[a:b], c.sr).resampled(TRAIN_SR))
    total = sum(c.duration for c in clips)
    if total < config.MIN_TRAINING_SPEECH_SECONDS:
        raise ValueError(f"Only {total:.0f} s of clean speech found; fine-tuning needs at least "
                         f"{config.MIN_TRAINING_SPEECH_SECONDS:.0f} s (1–5 minutes is better).")
    # hold out every k-th clip (spread across recordings) for checkpoint selection
    want = max(HELDOUT_MIN_S, HELDOUT_FRACTION * total)
    k = max(2, int(round(total / max(want, 1e-6))))
    spread = list(range(k // 2, len(clips), k))
    order = spread + [i for i in range(len(clips)) if i not in spread]
    held_idx, held_s = [], 0.0
    for i in order:
        if held_s >= want or len(clips) - len(held_idx) <= 2:
            break
        held_idx.append(i)
        held_s += clips[i].duration
    train_dir, held_dir = store.secure_dir(work / "train"), store.secure_dir(work / "heldout")
    train_s = 0.0
    for i, c in enumerate(clips):
        peak = float(np.max(np.abs(c.samples))) if len(c.samples) else 0.0
        if peak > 0.99:
            c = AudioClip(c.samples * (0.95 / peak), c.sr)
        if i in held_idx:
            write_wav(held_dir / f"{i:04d}.wav", c)
        else:
            write_wav(train_dir / f"{i:04d}.wav", c)
            train_s += c.duration
    return {"train_dir": str(train_dir), "heldout_dir": str(held_dir),
            "heldout_files": sorted(str(p) for p in held_dir.glob("*.wav")),
            "n_train": len(clips) - len(held_idx), "n_heldout": len(held_idx),
            "train_seconds": round(train_s, 1), "heldout_seconds": round(held_s, 1)}


def probe_files(limit: int = 2) -> List[str]:
    """Recordings of *other* speakers to test candidates on: the user's own
    source uploads first, then the bundled test clip."""
    files = []
    for s in store.list_sources()[:limit]:
        p = store.source_dir(s["id"]) / s["file"]
        if p.exists():
            files.append(str(p))
    bundled = config.ROOT / "tests" / "data" / "friend.wav"
    if len(files) < limit and bundled.exists():
        files.append(str(bundled))
    return files


# ------------------------------------------------------------------ job
def run_finetune(ctx, profile_id: str, steps: int) -> Dict:
    from .registry import resolve_device, unload_all

    profile = store.get_profile(profile_id)
    if not profile:
        raise KeyError("Profile not found")
    if not training_allowed(profile):
        raise PermissionError("Fine-tuning needs the separate training consent for this profile.")
    if profile.get("status") != "ready":
        raise ValueError("Build the voice profile first.")
    device = resolve_device()
    if device != "cuda" and os.environ.get("ENKURAL_ALLOW_CPU_TRAINING") != "1":
        raise RuntimeError("Fine-tuning needs an NVIDIA GPU (set ENKURAL_ALLOW_CPU_TRAINING=1 to try the CPU; "
                           "expect many hours).")

    ctx.stage("analyze", "Preparing training data", label="Preparing training data...")
    root = store.secure_dir(finetune_dir(profile_id))
    work = root / "run"
    if work.exists():
        shutil.rmtree(work)
    store.secure_dir(work)
    try:
        ds = prepare_dataset(profile, work, ctx)
        ctx.progress(1.0, f"{ds['n_train']} training clips ({ds['train_seconds']:.0f} s), "
                          f"{ds['n_heldout']} held out ({ds['heldout_seconds']:.0f} s)")
        save_at = sorted({max(1, steps // 2), max(1, (3 * steps) // 4), steps})
        spec = {
            "profile_id": profile_id, "work_dir": str(work), "train_dir": ds["train_dir"],
            "heldout_files": ds["heldout_files"], "probe_files": probe_files(), "probe_seconds": PROBE_SECONDS,
            "steps": steps, "save_at": save_at, "batch_size": BATCH_SIZE, "device": "cuda:0" if device == "cuda" else "cpu",
            "seed": 0, "min_sim_gain": MIN_SIM_GAIN, "max_cer_increase": MAX_CER_INCREASE,
            "max_utmos_drop": MAX_UTMOS_DROP, "max_f0_corr_drop": MAX_F0_CORR_DROP,
        }
        (work / "spec.json").write_text(json.dumps(spec, indent=2))
        unload_all()  # hand the whole GPU to the worker
        stats = _run_worker(ctx, work, spec)
        result = json.loads((work / "result.json").read_text())
        return _record(profile_id, steps, ds, result, stats, root)
    finally:
        if work.exists():  # training clips and spare checkpoints never outlive the run
            shutil.rmtree(work, ignore_errors=True)


def _run_worker(ctx, work: Path, spec: Dict) -> Dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(config.ROOT / "backend") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["ENKURAL_DATA_DIR"] = str(config.DATA_DIR)
    env["ENKURAL_MODELS_DIR"] = str(config.MODELS_DIR)
    log_path = work / "worker.log"
    stats: Dict = {}
    with open(log_path, "w") as logf:
        proc = subprocess.Popen([sys.executable, "-m", "enkural.vc.finetune_worker", "--spec", str(work / "spec.json")],
                                cwd=str(work), env=env, stdout=subprocess.PIPE, stderr=logf, text=True, bufsize=1)
        ctx.stage("convert", "Starting the trainer (loading models)", label="Training...")
        try:
            for line in proc.stdout:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("event") == "train":
                    ctx.progress(ev["step"] / ev["steps"], f"Step {ev['step']}/{ev['steps']} · loss {ev['loss']:.3f}")
                elif ev.get("event") == "trained":
                    stats = {"training_time_s": ev.get("seconds"), "peak_gpu_gb": ev.get("peak_gpu_gb")}
                    ctx.stage("enhance", "Comparing checkpoints with the base model",
                              label="Checking the result...")
                elif ev.get("event") == "eval":
                    what = "Converting test recordings" if ev.get("phase") == "convert" else "Scoring"
                    frac = ev["done"] / max(1, ev["total"])
                    ctx.progress(0.5 * frac if ev.get("phase") == "convert" else 0.5 + 0.5 * frac,
                                 f"{what} {ev['done']}/{ev['total']}")
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        code = proc.wait()
    if code != 0:
        tail = deque(open(log_path, errors="ignore"), maxlen=12)
        raise RuntimeError(f"Fine-tuning worker failed (exit {code}): {''.join(tail)[-800:]}")
    return stats


def _record(profile_id: str, steps: int, ds: Dict, result: Dict, stats: Dict, root: Path) -> Dict:
    chosen = result.get("chosen")
    profile = store.get_profile(profile_id)
    prev = profile.get("finetune") if (profile.get("finetune") or {}).get("status") == "ready" else None
    info = {
        "engine": ENGINE_ID, "trained_at": store.now_iso(), "steps": steps,
        "train_clips": ds["n_train"], "train_seconds": ds["train_seconds"],
        "heldout_clips": ds["n_heldout"], "heldout_seconds": ds["heldout_seconds"],
        "evaluated": result.get("evaluated", False), "probes": result.get("probes", []),
        "candidates": [{k: v for k, v in c.items() if k != "path"} for c in result.get("candidates", [])],
        **stats,
    }
    if chosen and chosen.get("path"):
        dest = root / "model.pth"
        shutil.move(chosen["path"], dest)
        os.chmod(dest, 0o600)
        info.update({"status": "ready", "checkpoint": str(dest.relative_to(store.profile_dir(profile_id))),
                     "chosen_step": chosen["step"]})
        msg = ("Fine-tuned model ready." if info["evaluated"] else
               "Fine-tuned model ready (not verified: no test recordings were available).")
    else:
        info.update({"status": "rejected", "checkpoint": None, "chosen_step": None})
        if prev and (root / "model.pth").exists():  # an earlier verified model stays in use
            info = {**prev, "last_run": info}
            msg = "This run did not beat the base model; your earlier fine-tuned model stays in use."
        else:
            msg = "Fine-tuning did not beat the base model on held-out audio, so the base model stays in use."
    profile["finetune"] = info
    store.save_profile(profile)
    return {"profile_id": profile_id, "finetune": info, "done_label": msg}
