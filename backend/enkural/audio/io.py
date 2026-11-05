"""Audio decoding / encoding via a static ffmpeg build.

Any format ffmpeg understands (wav, mp3, m4a/aac, ogg/opus, flac, webm from the
browser recorder, video containers, ...) is decoded straight to float32 mono.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from ..config import ffmpeg_path, ffprobe_path


@dataclass
class AudioClip:
    """A mono float32 waveform in [-1, 1] plus its sample rate."""

    samples: np.ndarray
    sr: int

    @property
    def duration(self) -> float:
        return float(len(self.samples)) / self.sr if self.sr else 0.0

    def resampled(self, sr: int) -> "AudioClip":
        if sr == self.sr:
            return self
        return AudioClip(resample(self.samples, self.sr, sr), sr)

    def copy(self) -> "AudioClip":
        return AudioClip(self.samples.copy(), self.sr)


def resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return x
    try:
        import soxr

        return soxr.resample(x.astype(np.float32), orig_sr, target_sr, quality="VHQ").astype(np.float32)
    except Exception:
        import librosa

        return librosa.resample(x.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_vhq")


def probe(path: str | Path) -> dict:
    """Return ffprobe metadata (duration, codec, sample rate, channels)."""
    cmd = [
        ffprobe_path(), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise ValueError(f"Could not read audio file: {out.stderr.strip()[:300]}")
    info = json.loads(out.stdout or "{}")
    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise ValueError("The file does not contain an audio stream.")
    s = streams[0]
    fmt = info.get("format", {})
    duration = float(s.get("duration") or fmt.get("duration") or 0.0)
    return {
        "duration": duration,
        "codec": s.get("codec_name"),
        "sample_rate": int(s.get("sample_rate") or 0),
        "channels": int(s.get("channels") or 0),
        "format": fmt.get("format_name"),
        "bit_rate": int(fmt.get("bit_rate") or 0),
    }


def load_audio(path: str | Path, sr: Optional[int] = None, mono: bool = True) -> AudioClip:
    """Decode any audio/video file to float32 (mono) at ``sr`` (native if None)."""
    path = str(path)
    if sr is None:
        try:
            sr = probe(path)["sample_rate"] or 44100
        except Exception:
            sr = 44100
    cmd = [
        ffmpeg_path(), "-v", "error", "-nostdin", "-i", path,
        "-vn", "-sn", "-dn",
        "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(sr),
    ]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-"]
    out = subprocess.run(cmd, capture_output=True, timeout=600)
    if out.returncode != 0:
        raise ValueError(f"Could not decode audio: {out.stderr.decode(errors='ignore').strip()[:300]}")
    data = np.frombuffer(out.stdout, dtype=np.float32)
    if data.size == 0:
        raise ValueError("The file decoded to zero audio samples.")
    data = np.clip(data, -1.0, 1.0).astype(np.float32)
    return AudioClip(data, int(sr))


def load_bytes(data: bytes, suffix: str, sr: Optional[int] = None) -> AudioClip:
    with tempfile.NamedTemporaryFile(suffix=suffix or ".bin", delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        return load_audio(tmp, sr=sr)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def write_wav(path: str | Path, clip: AudioClip, subtype: str = "PCM_16") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.clip(clip.samples, -1.0, 1.0).astype(np.float32)
    sf.write(str(path), samples, clip.sr, subtype=subtype)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def time_stretch(clip: AudioClip, rate: float) -> AudioClip:
    """Play ``rate``× faster without changing pitch (ffmpeg ``atempo``)."""
    if abs(rate - 1.0) < 1e-3:
        return clip
    with tempfile.TemporaryDirectory() as d:
        src = write_wav(Path(d) / "in.wav", clip, subtype="FLOAT")
        cmd = [ffmpeg_path(), "-v", "error", "-nostdin", "-i", str(src), "-filter:a", f"atempo={rate:.6f}",
               "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(clip.sr), "-ac", "1", "-"]
        out = subprocess.run(cmd, capture_output=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(f"Time-stretch failed: {out.stderr.decode(errors='ignore')[:300]}")
    return AudioClip(np.frombuffer(out.stdout, dtype=np.float32).copy(), clip.sr)


def encode_mp3(wav_path: str | Path, mp3_path: str | Path, bitrate: str = "320k", sr: int = 44100) -> Path:
    mp3_path = Path(mp3_path)
    cmd = [
        ffmpeg_path(), "-v", "error", "-nostdin", "-y", "-i", str(wav_path),
        "-vn", "-ar", str(sr), "-ac", "1", "-codec:a", "libmp3lame", "-b:a", bitrate,
        "-id3v2_version", "3", str(mp3_path),
    ]
    out = subprocess.run(cmd, capture_output=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(f"MP3 encoding failed: {out.stderr.decode(errors='ignore')[:300]}")
    try:
        os.chmod(mp3_path, 0o600)
    except OSError:
        pass
    return mp3_path
