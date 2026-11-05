"""Unit tests for the audio analysis / enhancement layer (no models needed).

They use the Tamil test recordings in tests/data (generate them with
scripts/make_test_audio.py); tests are skipped when the files are missing.
"""
from pathlib import Path

import numpy as np
import pytest

from enkural.audio import enhance, quality
from enkural.audio.io import AudioClip, load_audio

DATA = Path(__file__).resolve().parent / "data"
SR = 16000


def speech_clip(name="my_voice_1.wav", sr=SR) -> AudioClip:
    p = DATA / name
    if not p.exists():
        pytest.skip(f"missing test audio {p}; run scripts/make_test_audio.py")
    return load_audio(p, sr=sr)


def all_my_voice(sr=SR) -> AudioClip:
    files = sorted(DATA.glob("my_voice_*.wav"))
    if not files:
        pytest.skip("missing test audio; run scripts/make_test_audio.py")
    return AudioClip(np.concatenate([load_audio(f, sr=sr).samples for f in files]), sr)


def with_noise(clip: AudioClip, noise_db: float, seed=0) -> AudioClip:
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(len(clip.samples)).astype(np.float32) * 10 ** (noise_db / 20)
    return AudioClip(np.clip(clip.samples + n, -1, 1), clip.sr)


def scaled(clip: AudioClip, gain_db: float) -> AudioClip:
    return AudioClip(np.clip(clip.samples * 10 ** (gain_db / 20), -1, 1).astype(np.float32), clip.sr)


@pytest.fixture(scope="module")
def clean():
    c = speech_clip()
    return AudioClip(enhance.normalize_loudness(c.samples, c.sr, target_lufs=-20, true_peak=-3), c.sr)


def test_quality_clean_recording_is_ok(clean):
    q = quality.assess(clean, purpose="source")
    assert q["ok"], q["issues"]
    assert q["speech_duration"] > 5
    assert q["snr_db"] > 25
    assert not any(i["code"] in ("clipping", "low_volume", "background_noise") for i in q["issues"])


def test_quality_detects_low_volume(clean):
    q = quality.assess(scaled(clean, -22), purpose="source")
    assert any(i["code"] == "low_volume" for i in q["issues"])


def test_quality_detects_noise(clean):
    q = quality.assess(with_noise(clean, -32), purpose="source")
    assert any(i["code"] == "background_noise" for i in q["issues"])
    assert q["snr_db"] < 20


def test_quality_detects_clipping(clean):
    q = quality.assess(scaled(clean, +24), purpose="source")
    assert any(i["code"] == "clipping" for i in q["issues"])


def test_quality_profile_needs_enough_speech(clean):
    short = AudioClip(clean.samples[: 4 * SR], SR)
    q = quality.assess(short, purpose="profile")
    assert any(i["code"] == "too_little_speech" and i["severity"] == "poor" for i in q["issues"])
    assert not q["ok"]


def test_normalize_loudness_hits_target(clean):
    y = enhance.normalize_loudness(clean.samples, SR, target_lufs=-14.0, true_peak=-1.5)
    lufs = enhance.measure_lufs(y, SR)
    assert abs(lufs - (-14.0)) < 1.5
    assert enhance.true_peak_db(y) <= -1.3


def test_limiter_respects_ceiling():
    x = np.sin(2 * np.pi * 440 * np.arange(SR) / SR).astype(np.float32) * 1.5
    y = enhance.limiter(x, SR, ceiling_db=-1.0)
    assert np.max(np.abs(y)) <= 10 ** (-1.0 / 20) + 1e-3


def test_match_envelope_transfers_dynamics(clean):
    src = clean
    # a "converted" version with flattened dynamics: compress the envelope hard
    r, t = quality.frame_rms(src.samples, SR)
    env = np.interp(np.arange(len(src.samples)) / SR, t, r)
    flat = src.samples / np.maximum(env, 1e-3) ** 0.8 * 0.02
    out = AudioClip(flat.astype(np.float32), SR)
    matched = enhance.match_envelope(out, src, energy=1.0)
    assert matched.sr == out.sr and len(matched.samples) == len(out.samples)
    r_src, _ = quality.frame_rms(src.samples, SR)
    r_out, _ = quality.frame_rms(out.samples, SR)
    r_m, _ = quality.frame_rms(matched.samples, SR)
    ds, do, dm = (20 * np.log10(r + 1e-9) for r in (r_src, r_out, r_m))
    active = ds > np.percentile(ds, 10) + 10
    corr_before = np.corrcoef(ds[active], do[active])[0, 1]
    corr_after = np.corrcoef(ds[active], dm[active])[0, 1]
    assert corr_after > corr_before
    assert corr_after > 0.8


def test_denoise_reduces_noise_floor(clean):
    noisy = with_noise(clean, -38)
    y = enhance.denoise(noisy.samples, SR, strength=0.8)
    assert len(y) == len(noisy.samples)
    q_before = quality.assess(noisy, purpose="source")
    q_after = quality.assess(AudioClip(y, SR), purpose="source")
    assert q_after["snr_db"] > q_before["snr_db"] + 3


def test_settings_are_clamped():
    from enkural.vc.base import ConversionSettings

    s = ConversionSettings(similarity=500, pitch_semitones=-30, speed=3, energy=0, quality="ultra").clamped()
    assert s.similarity == 100 and s.pitch_semitones == -4 and s.speed == 1.15 and s.energy == 0.7
    assert s.quality == "standard"


def test_load_m4a_and_mp3_formats():
    for name in ("friend.m4a", "friend.wav"):
        c = speech_clip(name, sr=22050)
        assert c.sr == 22050 and c.duration > 5
