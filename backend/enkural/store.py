"""On-disk storage of profiles, sources and jobs.

Layout (all under ``DATA_DIR``, owner-only permissions):

    profiles/<id>/profile.json
    profiles/<id>/samples/<sample_id>.<ext>     original uploaded/recorded audio
    profiles/<id>/engines/<engine_id>/...       engine artefacts (reference, embeddings)
    profiles/<id>/cache/<sample_id>.*           separated voice stems of samples (derived)
    profiles/<id>/finetune/model.pth            opt-in fine-tuned model (see vc/finetune.py)
    sources/<id>/source.json + original.<ext>
    jobs/<id>/job.json + output.wav (+ export.mp3 / export.wav caches)

Nothing here is ever served by a static file handler; every read goes through
an authenticated API route.  Deleting an item removes its whole directory.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from . import config

_lock = threading.RLock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex


def _safe_id(value: str) -> str:
    v = str(value)
    if not v or not all(c in "0123456789abcdef" for c in v) or len(v) != 32:
        raise KeyError("invalid id")
    return v


def _write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def secure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def save_upload(dest: Path, data: bytes) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    try:
        os.chmod(dest, 0o600)
    except OSError:
        pass
    return dest


# ------------------------------------------------------------------ profiles
def profile_dir(pid: str) -> Path:
    return config.DATA_DIR / "profiles" / _safe_id(pid)


def sample_cache_dir(pid: str) -> Path:
    """Derived per-sample artefacts (separated voice stems); safe to delete."""
    return profile_dir(pid) / "cache"


def list_profiles() -> List[Dict]:
    out = []
    root = config.DATA_DIR / "profiles"
    if root.exists():
        for d in sorted(root.iterdir()):
            p = _read_json(d / "profile.json")
            if p:
                out.append(p)
    out.sort(key=lambda p: p.get("created_at", ""), reverse=True)
    return out


def get_profile(pid: str) -> Optional[Dict]:
    return _read_json(profile_dir(pid) / "profile.json")


def save_profile(profile: Dict) -> Dict:
    with _lock:
        profile["updated_at"] = now_iso()
        _write_json(profile_dir(profile["id"]) / "profile.json", profile)
    return profile


def create_profile(name: str, consent: Dict) -> Dict:
    pid = new_id()
    secure_dir(profile_dir(pid))
    secure_dir(profile_dir(pid) / "samples")
    profile = {
        "id": pid,
        "name": name.strip()[:80] or "My Voice",
        "created_at": now_iso(),
        "consent": consent,
        "allow_training": False,   # training only after the separate, revocable opt-in (vc/finetune.py)
        "training_consent": None,
        "finetune": None,
        "samples": [],
        "analysis": None,
        "engines": {},             # engine_id -> TargetVoice dict
        "status": "empty",         # empty | needs_build | building | ready | error
        "error": None,
    }
    return save_profile(profile)


def delete_profile(pid: str) -> bool:
    d = profile_dir(pid)
    if d.exists():
        shutil.rmtree(d)
        return True
    return False


def delete_sample(pid: str, sample_id: str) -> bool:
    with _lock:
        p = get_profile(pid)
        if not p:
            return False
        keep, removed = [], False
        for s in p["samples"]:
            if s["id"] == sample_id:
                try:
                    os.remove(profile_dir(pid) / "samples" / s["file"])
                except OSError:
                    pass
                cache = sample_cache_dir(pid)
                for f in (cache.glob(f"{_safe_id(sample_id)}.*") if cache.exists() else []):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
                removed = True
            else:
                keep.append(s)
        p["samples"] = keep
        if removed:
            p["status"] = "needs_build" if keep else "empty"
            save_profile(p)
        return removed


# ------------------------------------------------------------------ sources
def source_dir(sid: str) -> Path:
    return config.DATA_DIR / "sources" / _safe_id(sid)


def list_sources() -> List[Dict]:
    out = []
    root = config.DATA_DIR / "sources"
    if root.exists():
        for d in sorted(root.iterdir()):
            s = _read_json(d / "source.json")
            if s:
                out.append(s)
    out.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return out


def get_source(sid: str) -> Optional[Dict]:
    return _read_json(source_dir(sid) / "source.json")


def save_source(src: Dict) -> Dict:
    with _lock:
        _write_json(source_dir(src["id"]) / "source.json", src)
    return src


def delete_source(sid: str) -> bool:
    d = source_dir(sid)
    if d.exists():
        shutil.rmtree(d)
        return True
    return False


# ------------------------------------------------------------------ jobs
def job_dir(jid: str) -> Path:
    return config.DATA_DIR / "jobs" / _safe_id(jid)


def list_jobs() -> List[Dict]:
    out = []
    root = config.DATA_DIR / "jobs"
    if root.exists():
        for d in sorted(root.iterdir()):
            j = _read_json(d / "job.json")
            if j:
                out.append(j)
    out.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return out


def get_job(jid: str) -> Optional[Dict]:
    return _read_json(job_dir(jid) / "job.json")


def save_job(job: Dict) -> Dict:
    with _lock:
        job["updated_at"] = now_iso()
        _write_json(job_dir(job["id"]) / "job.json", job)
    return job


def delete_job(jid: str) -> bool:
    d = job_dir(jid)
    if d.exists():
        shutil.rmtree(d)
        return True
    return False


def delete_everything() -> Dict[str, int]:
    counts = {"profiles": 0, "sources": 0, "jobs": 0}
    for kind in counts:
        root = config.DATA_DIR / kind
        if root.exists():
            for d in list(root.iterdir()):
                if d.is_dir():
                    shutil.rmtree(d)
                    counts[kind] += 1
    tmp = config.DATA_DIR / "tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    return counts


def storage_summary() -> Dict:
    def size(p: Path) -> int:
        total = 0
        if p.exists():
            for f in p.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        return total

    return {
        "data_dir": str(config.DATA_DIR),
        "profiles": len(list_profiles()),
        "sources": len(list_sources()),
        "jobs": len(list_jobs()),
        "bytes": size(config.DATA_DIR),
        "models_bytes": size(config.MODELS_DIR),
    }
