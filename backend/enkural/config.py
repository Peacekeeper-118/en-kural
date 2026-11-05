"""Central configuration: paths, network binding and processing limits.

Everything is overridable through ``ENKURAL_*`` environment variables so the
app can be relocated (or containerised) without code changes.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(os.environ.get("ENKURAL_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get("ENKURAL_DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.environ.get("ENKURAL_MODELS_DIR", ROOT / "models"))
VENDOR_DIR = ROOT / "vendor"
SEEDVC_DIR = Path(os.environ.get("ENKURAL_SEEDVC_DIR", VENDOR_DIR / "seed-vc"))
FRONTEND_DIR = ROOT / "frontend"
BIN_DIR = ROOT / "bin"

HOST = os.environ.get("ENKURAL_HOST", "127.0.0.1")
PORT = int(os.environ.get("ENKURAL_PORT", "7860"))

# Which conversion engine is used by default; see enkural.vc.registry.
DEFAULT_ENGINE = os.environ.get("ENKURAL_ENGINE", "seedvc_v1")
# auto | cuda | cpu
DEVICE = os.environ.get("ENKURAL_DEVICE", "auto")

# Sample rate used for all analysis (VAD, pitch, quality checks).
ANALYSIS_SR = 16000
# Export sample rate for WAV/MP3 deliverables.
EXPORT_SR = 44100

MAX_UPLOAD_MB = int(os.environ.get("ENKURAL_MAX_UPLOAD_MB", "300"))
MAX_SOURCE_SECONDS = int(os.environ.get("ENKURAL_MAX_SOURCE_SECONDS", "900"))
MAX_SAMPLE_SECONDS = int(os.environ.get("ENKURAL_MAX_SAMPLE_SECONDS", "900"))

# Loudness target for exported audio (LUFS integrated / dBTP ceiling).
# -14 LUFS / -1.5 dBTP matches YouTube / Instagram normalisation well.
TARGET_LUFS = float(os.environ.get("ENKURAL_TARGET_LUFS", "-14"))
TRUE_PEAK_DB = float(os.environ.get("ENKURAL_TRUE_PEAK_DB", "-1.5"))

# Voice / background separation (Hybrid Demucs): "auto" isolates the voice wherever
# music or loud background sound is detected; "off" disables it everywhere.
SEPARATION = os.environ.get("ENKURAL_SEPARATION", "auto").lower()

MIN_TRAINING_SPEECH_SECONDS = float(os.environ.get("ENKURAL_MIN_TRAINING_SPEECH_SECONDS", "20"))

# Objective-evaluation models (scripts/benchmark.py and fine-tune checkpoint selection).
# The ASR judge is a Tamil-specialised Whisper; openai/whisper-small also works but
# transcribes Tamil poorly, which makes character error rates noisy.
EVAL_ASR_MODEL = os.environ.get("ENKURAL_EVAL_ASR_MODEL", "vasista22/whisper-tamil-small")
EVAL_LANGUAGE = os.environ.get("ENKURAL_EVAL_LANGUAGE", "ta")

ALLOWED_AUDIO_EXT = {
    ".wav", ".mp3", ".m4a", ".aac", ".mp4", ".ogg", ".oga", ".opus", ".flac",
    ".webm", ".wma", ".aiff", ".aif", ".caf", ".amr", ".3gp", ".mov", ".mkv",
}


def _find_binary(name: str) -> str:
    env = os.environ.get(f"ENKURAL_{name.upper()}")
    if env and Path(env).exists():
        return env
    local = BIN_DIR / name
    if local.exists():
        return str(local)
    found = shutil.which(name)
    if found:
        return found
    try:  # last resort: the imageio-ffmpeg wheel bundles a static ffmpeg
        if name == "ffmpeg":
            import imageio_ffmpeg  # type: ignore

            return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise RuntimeError(
        f"{name} not found. Put a static build in {BIN_DIR} (see scripts/setup.sh) "
        f"or set ENKURAL_{name.upper()}=/path/to/{name}."
    )


def ffmpeg_path() -> str:
    return _find_binary("ffmpeg")


def ffprobe_path() -> str:
    return _find_binary("ffprobe")


def ensure_dirs() -> None:
    """Create the private data directories (owner-only permissions)."""
    for d in (DATA_DIR, DATA_DIR / "profiles", DATA_DIR / "sources", DATA_DIR / "jobs", MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(d, 0o700)
        except OSError:
            pass
    # Make the vendored models cache the default Hugging Face / torch.hub cache too.
    os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
    os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
    # compiled GPU kernels survive reboots (the default lives in /tmp)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(MODELS_DIR / "torch" / "inductor"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
