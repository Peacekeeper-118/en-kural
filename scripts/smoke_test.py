"""End-to-end smoke test with the real engine (no server needed).

  1. creates a profile from tests/data/my_voice_*.wav (consent recorded)
  2. builds it (reference selection, analysis, speaker embedding)
  3. converts tests/data/friend.wav and exports WAV + MP3
  4. prints timings and the voice-match metrics

Usage: PYTHONPATH=backend .venv/bin/python scripts/smoke_test.py [--engine seedvc_v1] [--quality standard]
       [--source tests/data/friend.wav] [--samples tests/data/my_voice_1.wav ...] [--keep]
"""
from __future__ import annotations

import functools
print = functools.partial(print, flush=True)  # noqa: A001 - progress must show up in logs immediately

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import os  # noqa: E402

os.environ.setdefault("ENKURAL_DATA_DIR", str(ROOT / "data_smoke"))

from enkural import config  # noqa: E402

config.ensure_dirs()
from enkural import pipeline, profiles, store  # noqa: E402
from enkural.jobs import JobContext  # noqa: E402


class Ctx(JobContext):
    """JobContext that prints progress instead of persisting it."""

    def __init__(self, job):
        super().__init__(job)
        self.t0 = time.time()
        self._last = ""

    def stage(self, key, detail="", label=None):
        self.job["stage"] = key
        print(f"[{time.time() - self.t0:6.1f}s] {label or key:<28} {detail}")

    def progress(self, fraction, detail=None):
        msg = f"{detail or ''} {fraction * 100:.0f}%"
        if msg != self._last:
            self._last = msg
            print(f"[{time.time() - self.t0:6.1f}s]    {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=config.DEFAULT_ENGINE)
    ap.add_argument("--quality", default="standard")
    ap.add_argument("--similarity", type=float, default=65)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--energy", type=float, default=1.0)
    ap.add_argument("--source", default=str(ROOT / "tests/data/friend.wav"))
    ap.add_argument("--samples", nargs="*", default=sorted(str(p) for p in (ROOT / "tests/data").glob("my_voice_*.wav")))
    ap.add_argument("--out", default=str(ROOT / "tests/out"))
    ap.add_argument("--keep", action="store_true", help="keep the temporary data dir")
    args = ap.parse_args()

    t0 = time.time()
    profile = store.create_profile("Smoke test voice", {"confirmed": True, "statement": profiles.CONSENT_STATEMENT, "timestamp": store.now_iso()})
    for s in args.samples:
        smp = profiles.add_sample(profile, Path(s).name, Path(s).read_bytes())
        q = smp["quality"]
        print(f"sample {Path(s).name}: {smp['duration']}s, speech {q['speech_duration']}s, score {q['score']}, issues {[i['code'] for i in q['issues']]}")
    profile = store.get_profile(profile["id"])

    print("\n== build profile ==")
    ctx = Ctx({"id": store.new_id(), "kind": "profile_build"})
    profiles.build(profile["id"], args.engine, ctx)
    profile = store.get_profile(profile["id"])
    print("analysis:", profile["analysis"]["summary"])
    print("reference:", profile["reference"])
    print("engine meta:", profile["engines"][args.engine]["meta"])

    src_bytes = Path(args.source).read_bytes()
    sid = store.new_id()
    d = store.secure_dir(store.source_dir(sid))
    path = store.save_upload(d / ("original" + Path(args.source).suffix), src_bytes)
    from enkural.audio import quality
    from enkural.audio.io import load_audio, probe

    clip = load_audio(path, sr=config.ANALYSIS_SR)
    q = quality.assess(clip, purpose="source")
    source = store.save_source({"id": sid, "file": path.name, "original_name": Path(args.source).name, "created_at": store.now_iso(),
                                "duration": clip.duration, "quality": {k: v for k, v in q.items() if k != "segments"}})
    print(f"\nsource: {clip.duration:.1f}s, snr {q['snr_db']} dB, issues {[i['code'] for i in q['issues']]}")

    print("\n== convert ==")
    job = {"id": store.new_id(), "kind": "convert", "profile_id": profile["id"], "source_id": sid, "engine": args.engine,
           "settings": {"similarity": args.similarity, "pitch_semitones": args.pitch, "speed": args.speed,
                        "energy": args.energy, "quality": args.quality}}
    store.save_job(job)
    ctx = Ctx(job)
    t1 = time.time()
    result = pipeline.run_conversion(ctx)
    t_conv = time.time() - t1
    job.update(result)
    out = job["output"]
    print(f"\nconverted in {t_conv:.1f}s ({t_conv / clip.duration:.2f}x realtime)  -> {out}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.engine}_{args.quality}_sim{int(args.similarity)}_p{args.pitch:+.1f}_s{args.speed:.2f}_e{args.energy:.2f}"
    for fmt in ("wav", "mp3"):
        p = pipeline.export_path(job, fmt)
        shutil.copy(p, outdir / f"converted_{tag}.{fmt}")
    shutil.copy(store.profile_dir(profile["id"]) / "reference.wav", outdir / "reference.wav")
    shutil.copy(store.job_dir(job["id"]) / "output.wav", outdir / f"output_{tag}.wav")
    print(f"exports in {outdir}/converted_{tag}.wav|.mp3   total {time.time() - t0:.1f}s")
    if not args.keep:
        store.delete_everything()


if __name__ == "__main__":
    main()
