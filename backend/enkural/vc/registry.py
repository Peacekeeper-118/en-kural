"""Engine registry: maps engine ids to implementations and manages the
singleton instance that holds models in memory."""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Type

from .. import config
from .base import VoiceConverter

log = logging.getLogger(__name__)

_ENGINES: Dict[str, Type[VoiceConverter]] = {}
_instances: Dict[str, VoiceConverter] = {}
_lock = threading.RLock()


def register(cls: Type[VoiceConverter]) -> Type[VoiceConverter]:
    _ENGINES[cls.id] = cls
    return cls


def _ensure_builtin_registered() -> None:
    if _ENGINES:
        return
    # Import side-effects register the built-in engines.
    from . import seedvc_v1  # noqa: F401
    from . import seedvc_v2  # noqa: F401


def engine_ids() -> List[str]:
    _ensure_builtin_registered()
    return list(_ENGINES)


def resolve_device(pref: Optional[str] = None) -> str:
    pref = (pref or config.DEVICE or "auto").lower()
    if pref in ("cuda", "cpu"):
        return pref
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def list_engines() -> List[Dict]:
    _ensure_builtin_registered()
    out = []
    for eid, cls in _ENGINES.items():
        inst = _instances.get(eid)
        out.append({
            "id": eid, "name": cls.name, "description": cls.description,
            "available": cls.available(), "loaded": bool(inst and inst.loaded),
            "capabilities": cls.capabilities, "output_sr": cls.output_sr,
            "default": eid == config.DEFAULT_ENGINE,
        })
    return out


def unload_all() -> None:
    """Free every engine's GPU memory (e.g. before evaluation or training)."""
    with _lock:
        for inst in _instances.values():
            if inst.loaded:
                inst.unload()


def get_engine(engine_id: Optional[str] = None, load: bool = True) -> VoiceConverter:
    """Return the (lazily loaded) engine instance.  Only one engine is kept on
    the GPU at a time; switching unloads the previous one."""
    _ensure_builtin_registered()
    eid = engine_id or config.DEFAULT_ENGINE
    if eid not in _ENGINES:
        raise KeyError(f"Unknown engine '{eid}'. Available: {', '.join(_ENGINES)}")
    with _lock:
        inst = _instances.get(eid)
        if inst is None:
            inst = _ENGINES[eid]()
            _instances[eid] = inst
        if load and not inst.loaded:
            for other_id, other in _instances.items():
                if other_id != eid and other.loaded:
                    log.info("Unloading engine %s to free memory", other_id)
                    other.unload()
            device = resolve_device()
            log.info("Loading engine %s on %s", eid, device)
            inst.load(device)
        return inst
