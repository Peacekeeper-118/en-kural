"""API-level tests with a stub engine (no GPU / models required)."""
import io
import os
import time
import wave
from pathlib import Path

import numpy as np
import pytest

os.environ["ENKURAL_NO_AUTH"] = "1"
os.environ["ENKURAL_ENGINE"] = "stub"

from enkural import config  # noqa: E402
from enkural.vc import registry  # noqa: E402

config.SEPARATION = "off"  # no Demucs download / GPU in API tests (covered in test_quality_tools)
from enkural.vc.base import ConversionSettings, TargetVoice, VoiceConverter  # noqa: E402
from tests.test_audio import all_my_voice, speech_clip  # noqa: E402


@registry.register
class StubEngine(VoiceConverter):
    id = "stub"
    name = "Stub"
    description = "test engine"
    output_sr = 16000
    capabilities = {"similarity": True, "pitch": False, "speed": True, "multi_reference": True}

    def load(self, device):
        self.device = device
        self._loaded = True

    def build_target(self, reference, samples, out_dir, profile_id, progress=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        from enkural.audio.io import write_wav

        p = write_wav(out_dir / "reference.wav", reference)
        np.save(out_dir / "style.npy", np.ones(4, dtype=np.float32))
        return TargetVoice(profile_id, self.id, out_dir, p, {"style_file": "style.npy", "n_samples": len(samples)})

    def convert(self, source, target, settings, progress=None):
        c = source.resampled(self.output_sr)
        n = int(round(len(c.samples) / settings.clamped().speed))
        return type(c)(np.resize(c.samples, n) * 0.5, self.output_sr)

    def speaker_embedding(self, clip):
        return np.ones(4, dtype=np.float32)


def wav_bytes(clip):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(clip.sr)
        w.writeframes((np.clip(clip.samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from enkural.main import app

    with TestClient(app) as c:
        yield c


def wait_job(client, jid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.2)
    raise TimeoutError


def test_full_flow(client):
    r = client.get("/api/system")
    assert r.status_code == 200 and any(e["id"] == "stub" for e in r.json()["engines"])

    # consent is mandatory
    assert client.post("/api/profiles", json={"name": "X", "consent": False}).status_code == 400
    p = client.post("/api/profiles", json={"name": "Me", "consent": True}).json()
    assert p["consent"]["confirmed"] and p["allow_training"] is False and p["finetune"] is None

    sample = wav_bytes(all_my_voice())
    r = client.post(f"/api/profiles/{p['id']}/samples", files=[("files", ("me.wav", sample, "audio/wav"))], data={"kind": "upload"})
    assert r.status_code == 200
    added = r.json()["added"]
    assert len(added) == 1 and added[0]["quality"]["speech_duration"] > 30

    # conversion refused before build
    src_clip = speech_clip("friend.wav")
    src = wav_bytes(src_clip)
    s = client.post("/api/sources", files={"file": ("friend.wav", src, "audio/wav")}).json()
    assert s["quality"]["ok"]
    assert client.post("/api/convert", json={"profile_id": p["id"], "source_id": s["id"]}).status_code == 400

    job = client.post(f"/api/profiles/{p['id']}/build", json={"engine": "stub"}).json()
    j = wait_job(client, job["id"])
    assert j["status"] == "done", j.get("error")
    prof = client.get(f"/api/profiles/{p['id']}").json()
    assert prof["status"] == "ready" and prof["analysis"]["pitch"]["available"]
    assert "stub" in prof["engines"]

    job = client.post("/api/convert", json={"profile_id": p["id"], "source_id": s["id"], "engine": "stub",
                                            "settings": {"similarity": 80, "speed": 1.1, "quality": "draft"}}).json()
    assert job["settings"]["speed"] == 1.1
    j = wait_job(client, job["id"])
    assert j["status"] == "done", j.get("error")
    assert abs(j["output"]["duration"] - src_clip.duration / 1.1) < 0.2
    assert abs(j["output"]["lufs"] - config.TARGET_LUFS) < 2

    out = j["output"]
    assert out["separation"]["applied"] is False and out["finetuned"] is False
    assert out["timings"]["total_s"] > 0 and "convert_s" in out["timings"]

    audio = client.get(f"/api/jobs/{job['id']}/audio")
    assert audio.status_code == 200 and audio.headers["content-type"].startswith("audio/wav")
    mp3 = client.get(f"/api/jobs/{job['id']}/export", params={"format": "mp3", "name": "test"})
    assert mp3.status_code == 200 and mp3.content[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3")
    wavx = client.get(f"/api/jobs/{job['id']}/export", params={"format": "wav"})
    assert wavx.content[:4] == b"RIFF"

    # deletion really removes files
    jdir = Path(config.DATA_DIR) / "jobs" / job["id"]
    assert jdir.exists()
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 200
    assert not jdir.exists()
    assert client.delete(f"/api/profiles/{p['id']}").status_code == 200
    assert not (Path(config.DATA_DIR) / "profiles" / p["id"]).exists()
    assert client.delete("/api/data").status_code == 200


def test_finetune_requires_separate_consent(client):
    from enkural import store

    p = client.post("/api/profiles", json={"name": "Me", "consent": True}).json()
    # no training without the separate opt-in, whatever the profile state
    r = client.post(f"/api/profiles/{p['id']}/finetune", json={"preset": "quick"})
    assert r.status_code == 403
    r = client.post(f"/api/profiles/{p['id']}/training-consent", json={"allow": True})
    assert r.status_code == 200 and r.json()["allow_training"] is True
    assert r.json()["training_consent"]["confirmed"] and r.json()["training_consent"]["timestamp"]
    # consented but not built yet
    assert client.post(f"/api/profiles/{p['id']}/finetune", json={"preset": "quick"}).status_code == 400
    prof = store.get_profile(p["id"])
    prof["status"] = "ready"
    store.save_profile(prof)
    assert client.post(f"/api/profiles/{p['id']}/finetune", json={"preset": "huge"}).status_code == 400

    # withdrawing consent deletes the fine-tuned model
    ftdir = store.profile_dir(p["id"]) / "finetune"
    ftdir.mkdir(parents=True)
    (ftdir / "model.pth").write_bytes(b"x")
    prof = store.get_profile(p["id"])
    prof["finetune"] = {"status": "ready", "engine": "seedvc_v1", "checkpoint": "finetune/model.pth"}
    store.save_profile(prof)
    r = client.post(f"/api/profiles/{p['id']}/training-consent", json={"allow": False})
    assert r.status_code == 200 and r.json()["allow_training"] is False and r.json()["finetune"] is None
    assert not ftdir.exists()
    assert client.post(f"/api/profiles/{p['id']}/finetune", json={"preset": "quick"}).status_code == 403

    # explicit delete also works (and is idempotent)
    assert client.delete(f"/api/profiles/{p['id']}/finetune").status_code == 200
    assert client.delete(f"/api/profiles/{p['id']}").status_code == 200


def test_system_reports_new_capabilities(client):
    s = client.get("/api/system").json()
    assert s["separation_available"] is False
    assert set(s["finetune"]["presets"]) == {"quick", "standard", "thorough"}
    assert "train" in s["finetune"]["consent_statement"]
    assert {"separate", "keep_background", "use_finetuned"} <= set(s["defaults"])
