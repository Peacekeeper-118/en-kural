"""Tests for the measurement tools, sample selection, separation and the
fine-tuning helpers.  Model-dependent tests skip when the weights are not
downloaded (run the app once, or scripts/benchmark.py, to fetch them)."""
from pathlib import Path

import numpy as np
import pytest

from enkural import config, evaluation, pipeline, profiles
from enkural.audio import separate
from enkural.audio.io import AudioClip, time_stretch
from enkural.vc.base import ConversionSettings
from enkural.vc.finetune import _windows
from enkural.vc.finetune_worker import select
from tests.test_audio import speech_clip


def _music(n: int, sr: int, seed: int = 0) -> np.ndarray:
    """A crude music bed: sustained chords plus a kick drum every half second."""
    t = np.arange(n) / sr
    chord = sum(np.sin(2 * np.pi * f * t) for f in (110.0, 138.6, 164.8, 220.0, 329.6)) / 5
    kick = np.zeros(n)
    for start in range(0, n, sr // 2):
        m = min(n - start, sr // 10)
        kick[start:start + m] = np.sin(2 * np.pi * 60 * t[:m]) * np.exp(-t[:m] * 30)
    rng = np.random.default_rng(seed)
    return (0.6 * chord + 0.8 * kick + 0.02 * rng.standard_normal(n)).astype(np.float32)


# ------------------------------------------------------------------ settings
def test_new_settings_are_clamped():
    s = ConversionSettings.from_dict({"separate": "weird", "keep_background": 0, "use_finetuned": 1})
    assert s.separate == "auto" and s.keep_background is False and s.use_finetuned is True
    assert ConversionSettings.from_dict({"separate": "off"}).separate == "off"


# ------------------------------------------------------------------ evaluation helpers
def test_text_normalisation_and_cer():
    assert evaluation.normalize_text("வணக்கம்,  Hello!!") == "வணக்கம் hello"
    assert evaluation.cer("அ ஆ இ", "அ ஆ இ") == 0.0
    assert evaluation.cer("", "anything") is None
    assert 0 < evaluation.cer("abcd", "abxd") < 0.5


def test_pitch_correlation_identity_and_speed():
    c = speech_clip("friend.wav")
    same = evaluation.pitch_correlation(c, c)
    assert same["f0_corr"] > 0.99 and abs(same["f0_offset_st"]) < 0.01
    faster = time_stretch(c, 1.1)  # same contour, 10 % faster
    r = evaluation.pitch_correlation(c, faster, speed=1.1)
    assert r["f0_corr"] > 0.8


def test_asr_windows_cover_speech_in_short_spans():
    c = speech_clip("friend.wav")
    wins = evaluation.asr_windows(c, max_s=8.0)
    assert len(wins) >= 2
    assert all(e - s <= 8.0 + 0.31 for s, e in wins)
    assert wins[0][0] < 1.0 and wins[-1][1] > c.duration - 2.0


def test_time_stretch_changes_duration_only():
    c = speech_clip("friend.wav")
    y = time_stretch(c, 1.15)
    assert abs(y.duration - c.duration / 1.15) < 0.1 and y.sr == c.sr


# ------------------------------------------------------------------ sample selection
def _item(sid, speech_s, poor=False):
    issues = [{"code": "background_noise", "severity": "poor"}] if poor else []
    return {"sample": {"id": sid, "original_name": f"{sid}.wav"}, "segments": [(0.0, speech_s)],
            "quality": {"issues": issues, "score": 20 if poor else 90}, "separation": None}


def test_poor_samples_left_out_when_enough_clean_speech():
    items = [_item("a", 30), _item("b", 40, poor=True)]
    u = profiles.select_usable(items)
    assert [it["sample"]["id"] for it in u["identity"]] == ["a"]
    assert [it["sample"]["id"] for it in u["reference"]] == ["a"]
    notes = profiles._record_preparation(items, u)
    assert items[1]["sample"]["prepared"]["used_for_identity"] is False
    assert any("b.wav" in n for n in notes)


def test_poor_samples_kept_when_clean_speech_is_scarce():
    items = [_item("a", 10), _item("b", 40, poor=True)]
    u = profiles.select_usable(items)
    assert len(u["identity"]) == 2          # 10 s clean is too little for the identity alone…
    assert [it["sample"]["id"] for it in u["reference"]] == ["a"]  # …but enough for the prompt
    only_poor = profiles.select_usable([_item("b", 40, poor=True)])
    assert only_poor["all_poor"] and len(only_poor["identity"]) == 1


def test_too_little_speech_alone_is_not_poor():
    q = {"issues": [{"code": "too_little_speech", "severity": "poor"}]}
    assert not profiles.is_poor(q)


# ------------------------------------------------------------------ fine-tuning helpers
def test_training_windows_merge_and_split():
    w = _windows([(0.0, 1.0), (1.2, 3.0), (10.0, 10.5), (20.0, 45.0)])
    assert w[0] == (0.0, 3.0)                 # merged across a short pause
    assert all(e - s <= 12.0 for s, e in w)   # long phrase split
    assert all(e - s >= 2.0 for s, e in w)    # fragments dropped


def test_checkpoint_selection_guards():
    base = {"step": 0, "speaker_sim": 0.80, "cer": 0.10, "utmos": 3.0, "f0_corr": 0.8}
    better = {"step": 300, "speaker_sim": 0.85, "cer": 0.11, "utmos": 3.0, "f0_corr": 0.78}
    garbled = {"step": 600, "speaker_sim": 0.90, "cer": 0.30, "utmos": 3.0, "f0_corr": 0.8}
    flat = {"step": 900, "speaker_sim": 0.91, "cer": 0.10, "utmos": 3.0, "f0_corr": 0.5}
    pick = select([base, better, garbled, flat], max_cer_increase=0.05, max_utmos_drop=0.15,
                  min_sim_gain=0.005, max_f0_corr_drop=0.08)
    assert pick["step"] == 300
    assert select([base, dict(better, speaker_sim=0.802)], 0.05, 0.15, 0.005) is None  # not a real gain


def test_finetuned_checkpoint_only_when_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    pid = "a" * 32
    ckpt = tmp_path / "profiles" / pid / "finetune" / "model.pth"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")
    prof = {"id": pid, "finetune": {"status": "ready", "engine": "seedvc_v1", "checkpoint": "finetune/model.pth"}}
    assert pipeline.finetuned_checkpoint(prof, "seedvc_v1") == ckpt
    assert pipeline.finetuned_checkpoint(prof, "seedvc_v2") is None
    assert pipeline.finetuned_checkpoint({**prof, "finetune": {**prof["finetune"], "status": "rejected"}},
                                         "seedvc_v1") is None


# ------------------------------------------------------------------ separation / remix
def _hdemucs_cached() -> bool:
    return (config.MODELS_DIR / "torch" / "hub" / "torchaudio" / "models" / "hdemucs_high_trained.pt").exists()


needs_hdemucs = pytest.mark.skipif(not _hdemucs_cached(), reason="Hybrid Demucs weights not downloaded")


@pytest.fixture
def separation_on(monkeypatch):
    monkeypatch.setattr(config, "SEPARATION", "auto")
    config.ensure_dirs()


@needs_hdemucs
def test_separation_leaves_clean_speech_alone(separation_on):
    c = speech_clip("friend.wav", sr=44100)
    sep = separate.separate(c)
    assert not sep.significant and sep.background_db < separate.BACKGROUND_THRESHOLD_DB
    assert len(sep.vocals.samples) == len(c.samples)


@needs_hdemucs
def test_separation_detects_and_removes_music(separation_on):
    c = speech_clip("friend.wav", sr=44100)
    m = _music(len(c.samples), 44100)
    m *= np.sqrt(np.mean(c.samples ** 2) / np.mean(m ** 2)) * 10 ** (-3 / 20)  # music 3 dB under the voice
    mix = AudioClip((c.samples + m).astype(np.float32), 44100)
    sep = separate.separate(mix)
    assert sep.significant
    err_before = np.mean((mix.samples - c.samples) ** 2)
    err_after = np.mean((sep.vocals.samples - c.samples) ** 2)
    assert err_after < 0.3 * err_before  # most of the music is gone from the voice stem


def test_remix_keeps_length_and_voice_level():
    sr = 44100
    voice_src = speech_clip("friend.wav", sr=22050)
    voice_out = AudioClip(voice_src.samples * 0.25, 22050)  # converter came out 12 dB quieter
    bg = AudioClip(np.zeros(int(voice_src.duration * sr) + 500, dtype=np.float32), sr)
    mix = pipeline.remix_background(voice_out, voice_src, bg, speed=1.0)
    assert mix.sr == sr and abs(mix.duration - voice_src.duration) < 0.01
    from enkural.audio import enhance

    assert abs(enhance.measure_lufs(mix.samples, sr) - enhance.measure_lufs(voice_src.samples, 22050)) < 1.0
