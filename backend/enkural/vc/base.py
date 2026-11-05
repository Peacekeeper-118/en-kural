from __future__ import annotations

import contextlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..audio.io import AudioClip

ProgressFn = Callable[[float, str], None]

QUALITY_LEVELS = ("draft", "standard", "high")
SEPARATION_MODES = ("auto", "on", "off")


class StageTimer:
    """Accumulates wall time per named stage; synchronises CUDA so GPU work is
    attributed to the stage that queued it."""

    def __init__(self, device=None) -> None:
        self.totals: Dict[str, float] = {}
        self._cuda = getattr(device, "type", str(device)) == "cuda"

    def _sync(self) -> None:
        if self._cuda:
            import torch

            torch.cuda.synchronize()

    @contextlib.contextmanager
    def __call__(self, name: str):
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.totals[name] = self.totals.get(name, 0.0) + time.perf_counter() - t0

    def rounded(self) -> Dict[str, float]:
        return {k: round(v, 3) for k, v in self.totals.items()}


@dataclass
class ConversionSettings:
    """User-facing controls.  All values are clamped to *natural* ranges so the
    result can never become robotic through extreme settings."""

    similarity: float = 65.0        # 0..100  how strongly to pull towards the target voice
    pitch_semitones: float = 0.0    # -4..+4  fine pitch adjustment (0 = let the voice decide)
    speed: float = 1.0              # 0.85..1.15 playback-speed multiplier (timing preserved otherwise)
    energy: float = 1.0             # 0.7..1.3  dynamics / emphasis (1 = same as source)
    quality: str = "standard"       # draft | standard | high
    denoise: str = "auto"           # auto | on | off  (source clean-up before conversion)
    preserve_dynamics: bool = True  # transfer the source loudness contour onto the output
    separate: str = "auto"          # auto | on | off  isolate the voice from music / background first
    keep_background: bool = True    # mix the separated background back under the converted voice
    use_finetuned: bool = True      # use the profile's fine-tuned model when one exists

    def clamped(self) -> "ConversionSettings":
        def c(v, lo, hi):
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = lo if lo <= 0 <= hi else (lo + hi) / 2
            return float(min(hi, max(lo, v)))

        return ConversionSettings(
            similarity=c(self.similarity, 0, 100),
            pitch_semitones=c(self.pitch_semitones, -4, 4),
            speed=c(self.speed, 0.85, 1.15),
            energy=c(self.energy, 0.7, 1.3),
            quality=self.quality if self.quality in QUALITY_LEVELS else "standard",
            denoise=self.denoise if self.denoise in ("auto", "on", "off") else "auto",
            preserve_dynamics=bool(self.preserve_dynamics),
            separate=self.separate if self.separate in SEPARATION_MODES else "auto",
            keep_background=bool(self.keep_background),
            use_finetuned=bool(self.use_finetuned),
        )

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "ConversionSettings":
        d = d or {}
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known).clamped()


@dataclass
class TargetVoice:
    """Engine-specific artefacts describing one authorised target voice."""

    profile_id: str
    engine_id: str
    dir: Path
    reference_wav: Path          # clean reference speech (the "prompt"), any sample rate
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "profile_id": self.profile_id,
            "engine_id": self.engine_id,
            "dir": str(self.dir),
            "reference_wav": str(self.reference_wav),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TargetVoice":
        return cls(
            profile_id=d["profile_id"], engine_id=d["engine_id"], dir=Path(d["dir"]),
            reference_wav=Path(d["reference_wav"]), meta=d.get("meta", {}),
        )


class VoiceConverter(ABC):
    """Interface every conversion engine implements.

    Contract:
      * ``convert`` must preserve the *content* and *timing* of the source: the
        output has the same duration as the input (times ``1/speed``) so that
        pauses, rhythm and emphasis line up.
      * The speaker identity must come only from the ``TargetVoice`` built
        from the user's authorised samples.
    """

    id: str = "abstract"
    name: str = "Abstract engine"
    description: str = ""
    output_sr: int = 22050
    capabilities: Dict[str, bool] = {
        "similarity": False, "pitch": False, "speed": False, "multi_reference": False,
    }

    def __init__(self) -> None:
        self.device = None
        self._loaded = False
        self.last_timings: Dict[str, float] = {}   # per-stage seconds of the last convert()
        self.last_info: Dict = {}                  # engine facts about the last convert()

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def available(cls) -> bool:
        """Whether the engine's dependencies are importable."""
        return True

    @abstractmethod
    def load(self, device: str) -> None: ...

    def unload(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------ work
    @abstractmethod
    def build_target(self, reference: AudioClip, samples: List[AudioClip], out_dir: Path,
                     profile_id: str, progress: Optional[ProgressFn] = None) -> TargetVoice: ...

    @abstractmethod
    def convert(self, source: AudioClip, target: TargetVoice, settings: ConversionSettings,
                progress: Optional[ProgressFn] = None) -> AudioClip: ...

    def speaker_embedding(self, clip: AudioClip):
        """Optional: a speaker embedding (numpy vector) used for evaluation."""
        raise NotImplementedError

    def info(self) -> Dict:
        return {
            "id": self.id, "name": self.name, "description": self.description,
            "output_sr": self.output_sr, "capabilities": self.capabilities,
            "loaded": self.loaded, "device": str(self.device) if self.device else None,
        }
