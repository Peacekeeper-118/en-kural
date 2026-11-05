"""Model download helpers + vendored Seed-VC import setup."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from .. import config

log = logging.getLogger(__name__)

HF_CACHE = config.MODELS_DIR / "hf"


def hf_download(repo_id: str, filename: str | None = None, config_filename: str | None = None,
                model_filename: str | None = None):
    """Download (once) a file from the Hugging Face Hub into the models dir.

    Signature-compatible with Seed-VC's ``load_custom_model_from_hf`` (which
    some callers invoke with ``model_filename=``).
    """
    from huggingface_hub import hf_hub_download

    filename = filename or model_filename or "pytorch_model.bin"
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(HF_CACHE))
    if config_filename is None:
        return path
    cfg = hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir=str(HF_CACHE))
    return path, cfg


_patched = False


def setup_seedvc() -> Path:
    """Put the vendored Seed-VC repo on ``sys.path`` and redirect its checkpoint
    downloads into our models directory.  Safe to call repeatedly."""
    global _patched
    d = config.SEEDVC_DIR
    if not (d / "modules").exists():
        raise RuntimeError(
            f"Seed-VC not found at {d}. Run scripts/setup.sh (it clones "
            "https://github.com/Plachtaa/seed-vc into vendor/)."
        )
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    if not _patched:
        import hf_utils  # from the vendored repo

        hf_utils.load_custom_model_from_hf = hf_download  # type: ignore[attr-defined]
        _patched = True
    return d
