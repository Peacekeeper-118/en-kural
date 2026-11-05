from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .. import __version__, config, pipeline, profiles, store
from ..audio import quality, separate
from ..audio.io import load_audio, probe
from ..jobs import manager
from ..vc import finetune
from ..vc.base import ConversionSettings
from ..vc.registry import list_engines, resolve_device

log = logging.getLogger(__name__)
router = APIRouter()

MEDIA_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".mp4": "audio/mp4",
               ".aac": "audio/aac", ".ogg": "audio/ogg", ".oga": "audio/ogg", ".opus": "audio/ogg",
               ".webm": "audio/webm", ".flac": "audio/flac"}


def _media_type(path: Path) -> str:
    return MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def _no_store(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


async def _read_upload(f: UploadFile) -> bytes:
    data = await f.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    if not data:
        raise HTTPException(400, "Empty file.")
    return data


# ------------------------------------------------------------------ system
@router.get("/system")
def system_info():
    return {
        "version": __version__,
        "device": resolve_device(),
        "engines": list_engines(),
        "default_engine": config.DEFAULT_ENGINE,
        "storage": store.storage_summary(),
        "consent_statement": profiles.CONSENT_STATEMENT,
        "limits": {"max_upload_mb": config.MAX_UPLOAD_MB, "max_source_seconds": config.MAX_SOURCE_SECONDS,
                   "max_sample_seconds": config.MAX_SAMPLE_SECONDS},
        "defaults": ConversionSettings().to_dict(),
        "export": {"sample_rate": config.EXPORT_SR, "target_lufs": config.TARGET_LUFS},
        "separation_available": separate.available(),
        "finetune": {
            "engine": finetune.ENGINE_ID,
            "presets": finetune.STEP_PRESETS,
            "consent_statement": finetune.TRAINING_CONSENT_STATEMENT,
            "min_speech_seconds": config.MIN_TRAINING_SPEECH_SECONDS,
            "gpu": resolve_device() == "cuda",
        },
    }


# ------------------------------------------------------------------ profiles
class ProfileCreate(BaseModel):
    name: str = Field(default="My Voice", max_length=80)
    consent: bool = False


@router.get("/profiles")
def get_profiles():
    return store.list_profiles()


@router.post("/profiles")
def create_profile(body: ProfileCreate):
    if not body.consent:
        raise HTTPException(400, "You must confirm that you are authorised to use this voice.")
    consent = {"confirmed": True, "statement": profiles.CONSENT_STATEMENT, "timestamp": store.now_iso()}
    return store.create_profile(body.name, consent)


@router.get("/profiles/{pid}")
def get_profile(pid: str):
    p = store.get_profile(pid)
    if not p:
        raise HTTPException(404, "Profile not found")
    return p


@router.delete("/profiles/{pid}")
def delete_profile(pid: str):
    if not store.delete_profile(pid):
        raise HTTPException(404, "Profile not found")
    return {"deleted": pid}


@router.post("/profiles/{pid}/samples")
async def add_samples(pid: str, files: List[UploadFile] = File(...), kind: str = Form("upload")):
    p = store.get_profile(pid)
    if not p:
        raise HTTPException(404, "Profile not found")
    added, errors = [], []
    for f in files:
        try:
            data = await _read_upload(f)
            name = f.filename or ("recording.webm" if kind == "recording" else "sample")
            added.append(profiles.add_sample(p, name, data, kind=kind if kind in ("upload", "recording") else "upload"))
        except HTTPException as e:
            errors.append({"file": f.filename, "error": e.detail})
        except Exception as e:  # noqa: BLE001
            errors.append({"file": f.filename, "error": str(e)[:300]})
    return {"added": added, "errors": errors, "profile": store.get_profile(pid)}


@router.delete("/profiles/{pid}/samples/{sid}")
def delete_sample(pid: str, sid: str):
    if not store.delete_sample(pid, sid):
        raise HTTPException(404, "Sample not found")
    return {"deleted": sid, "profile": store.get_profile(pid)}


@router.get("/profiles/{pid}/samples/{sid}/audio")
def sample_audio(pid: str, sid: str):
    p = store.get_profile(pid)
    if not p:
        raise HTTPException(404, "Profile not found")
    for s in p["samples"]:
        if s["id"] == sid:
            path = store.profile_dir(pid) / "samples" / s["file"]
            if path.exists():
                return _no_store(FileResponse(path, media_type=_media_type(path)))
    raise HTTPException(404, "Sample not found")


@router.get("/profiles/{pid}/reference")
def reference_audio(pid: str):
    path = store.profile_dir(pid) / "reference.wav"
    if not path.exists():
        raise HTTPException(404, "Reference not built yet")
    return _no_store(FileResponse(path, media_type="audio/wav"))


class BuildRequest(BaseModel):
    engine: Optional[str] = None


@router.post("/profiles/{pid}/build")
def build_profile(pid: str, body: BuildRequest = BuildRequest()):
    p = store.get_profile(pid)
    if not p:
        raise HTTPException(404, "Profile not found")
    if not p["samples"]:
        raise HTTPException(400, "Add at least one voice sample first.")
    engine = body.engine or config.DEFAULT_ENGINE
    if engine not in [e["id"] for e in list_engines()]:
        raise HTTPException(400, "Unknown engine")
    job = {"id": store.new_id(), "kind": "profile_build", "profile_id": pid, "engine": engine}
    manager.submit(job, lambda ctx: profiles.build(pid, engine, ctx))
    return job


# ------------------------------------------------------------------ fine-tuning (opt-in)
class TrainingConsent(BaseModel):
    allow: bool = False


@router.post("/profiles/{pid}/training-consent")
def training_consent(pid: str, body: TrainingConsent):
    p = store.get_profile(pid)
    if not p:
        raise HTTPException(404, "Profile not found")
    if body.allow and not p.get("consent", {}).get("confirmed"):
        raise HTTPException(403, "This voice profile has no recorded consent.")
    if not body.allow and _finetune_running(pid):
        raise HTTPException(409, "Cancel the running fine-tuning job first.")
    return finetune.set_training_consent(p, body.allow)


class FinetuneRequest(BaseModel):
    preset: str = "standard"


def _finetune_running(pid: str) -> bool:
    return any(j.get("kind") == "finetune" and j.get("profile_id") == pid and j.get("status") in ("queued", "running")
               for j in store.list_jobs())


@router.post("/profiles/{pid}/finetune")
def start_finetune(pid: str, body: FinetuneRequest = FinetuneRequest()):
    p = store.get_profile(pid)
    if not p:
        raise HTTPException(404, "Profile not found")
    if not finetune.training_allowed(p):
        raise HTTPException(403, "Allow fine-tuning for this profile first (separate training consent).")
    if p.get("status") != "ready":
        raise HTTPException(400, "Build the voice profile first.")
    if body.preset not in finetune.STEP_PRESETS:
        raise HTTPException(400, f"preset must be one of {', '.join(finetune.STEP_PRESETS)}")
    if _finetune_running(pid):
        raise HTTPException(409, "Fine-tuning is already running for this profile.")
    steps = finetune.STEP_PRESETS[body.preset]
    job = {"id": store.new_id(), "kind": "finetune", "profile_id": pid, "engine": finetune.ENGINE_ID,
           "preset": body.preset, "steps": steps}
    manager.submit(job, lambda ctx: finetune.run_finetune(ctx, pid, steps))
    return job


@router.delete("/profiles/{pid}/finetune")
def delete_finetune(pid: str):
    p = store.get_profile(pid)
    if not p:
        raise HTTPException(404, "Profile not found")
    if _finetune_running(pid):
        raise HTTPException(409, "Cancel the running fine-tuning job first.")
    finetune.delete_finetune(p)
    return store.get_profile(pid)


# ------------------------------------------------------------------ sources
@router.get("/sources")
def get_sources():
    return store.list_sources()


@router.post("/sources")
async def add_source(file: UploadFile = File(...)):
    data = await _read_upload(file)
    name = file.filename or "source"
    ext = Path(name).suffix.lower() or ".bin"
    if ext not in config.ALLOWED_AUDIO_EXT and ext != ".bin":
        raise HTTPException(400, f"Unsupported audio format '{ext}'.")
    sid = store.new_id()
    d = store.secure_dir(store.source_dir(sid))
    path = store.save_upload(d / f"original{ext}", data)
    try:
        info = probe(path)
        if info["duration"] > config.MAX_SOURCE_SECONDS:
            raise ValueError(f"Recording is longer than {config.MAX_SOURCE_SECONDS // 60} minutes.")
        clip = load_audio(path, sr=config.ANALYSIS_SR)
        q = quality.assess(clip, purpose="source")
    except Exception as e:
        store.delete_source(sid)
        raise HTTPException(400, str(e)[:300])
    src = {
        "id": sid, "file": path.name, "original_name": name[:120], "created_at": store.now_iso(),
        "duration": round(info["duration"] or clip.duration, 2), "sample_rate": info.get("sample_rate"),
        "codec": info.get("codec"), "quality": {k: v for k, v in q.items() if k != "segments"},
    }
    return store.save_source(src)


@router.get("/sources/{sid}/audio")
def source_audio(sid: str):
    s = store.get_source(sid)
    if not s:
        raise HTTPException(404, "Source not found")
    path = store.source_dir(sid) / s["file"]
    return _no_store(FileResponse(path, media_type=_media_type(path)))


@router.delete("/sources/{sid}")
def delete_source(sid: str):
    if not store.delete_source(sid):
        raise HTTPException(404, "Source not found")
    return {"deleted": sid}


# ------------------------------------------------------------------ conversion jobs
class ConvertRequest(BaseModel):
    profile_id: str
    source_id: str
    engine: Optional[str] = None
    settings: dict = Field(default_factory=dict)


@router.post("/convert")
def convert(body: ConvertRequest):
    p = store.get_profile(body.profile_id)
    if not p:
        raise HTTPException(404, "Profile not found")
    if not p.get("consent", {}).get("confirmed"):
        raise HTTPException(403, "This voice profile has no recorded consent.")
    if p.get("status") not in ("ready",):
        raise HTTPException(400, "Build the voice profile first.")
    if not store.get_source(body.source_id):
        raise HTTPException(404, "Source not found")
    engine = body.engine or config.DEFAULT_ENGINE
    if engine not in [e["id"] for e in list_engines()]:
        raise HTTPException(400, "Unknown engine")
    settings = ConversionSettings.from_dict(body.settings).to_dict()
    job = {"id": store.new_id(), "kind": "convert", "profile_id": body.profile_id,
           "source_id": body.source_id, "engine": engine, "settings": settings}
    manager.submit(job, pipeline.run_conversion)
    return job


@router.get("/jobs")
def get_jobs():
    return store.list_jobs()


@router.get("/jobs/{jid}")
def get_job(jid: str):
    j = store.get_job(jid)
    if not j:
        raise HTTPException(404, "Job not found")
    return j


@router.post("/jobs/{jid}/cancel")
def cancel_job(jid: str):
    return {"cancelled": manager.cancel(jid)}


@router.delete("/jobs/{jid}")
def delete_job(jid: str):
    manager.cancel(jid)
    if not store.delete_job(jid):
        raise HTTPException(404, "Job not found")
    return {"deleted": jid}


@router.get("/jobs/{jid}/audio")
def job_audio(jid: str):
    j = store.get_job(jid)
    if not j or j.get("status") != "done" or not j.get("output"):
        raise HTTPException(404, "No output yet")
    path = store.job_dir(jid) / j["output"]["file"]
    return _no_store(FileResponse(path, media_type="audio/wav"))


@router.get("/jobs/{jid}/export")
def job_export(jid: str, format: str = "wav", name: Optional[str] = None):
    j = store.get_job(jid)
    if not j or j.get("status") != "done" or not j.get("output"):
        raise HTTPException(404, "No output yet")
    fmt = format.lower()
    if fmt not in ("wav", "mp3"):
        raise HTTPException(400, "format must be wav or mp3")
    try:
        path = pipeline.export_path(j, fmt)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Export failed: {e}")
    base = "".join(c for c in (name or "my-voice") if c.isalnum() or c in "-_ ").strip() or "my-voice"
    return _no_store(FileResponse(path, media_type="audio/wav" if fmt == "wav" else "audio/mpeg",
                                  filename=f"{base}.{fmt}"))


# ------------------------------------------------------------------ privacy / data
@router.delete("/data")
def delete_all_data():
    counts = store.delete_everything()
    return {"deleted": counts}
