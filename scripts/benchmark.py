"""Benchmark conversions with objective metrics the engine never sees.

  * builds a throw-away profile from --samples (deleted afterwards), or uses an
    existing app profile with --profile-id
  * converts every --sources recording under every combination of --grid values
  * scores each result: WavLM speaker similarity to *held-out* recordings of the
    target voice, Tamil character error rate (Whisper), UTMOS naturalness,
    F0-contour correlation with the source, duration, speed (+ per-stage timing)
  * writes report.md, results.csv and results.json plus the audio under --out;
    --listening adds a blind A/B folder (shuffled letters + key.json)

Held-out audio matters: similarity to the very clips the profile was built
from is inflated.  Pass --heldout recordings of the target voice that are not
in --samples (e.g. a different day or topic).

Examples:
  # quality levels on the bundled test clips
  PYTHONPATH=backend .venv/bin/python scripts/benchmark.py --consent \\
      --samples tests/data/my_voice_1.wav tests/data/my_voice_2.wav tests/data/my_voice_3.wav \\
      --heldout tests/data/my_voice_4.wav --sources tests/data/friend.wav \\
      --grid quality=draft,standard,high

  # sampler tuning: steps × guidance window, distance to the 50-step upstream result
  PYTHONPATH=backend .venv/bin/python scripts/benchmark.py --consent --samples ... --sources ... \\
      --grid sampler.steps=8,12,16 --grid sampler.cfg_until=0.5,1.0 --reference "sampler.steps=50"

  # an app profile with and without its fine-tuned model
  PYTHONPATH=backend .venv/bin/python scripts/benchmark.py --profile-id <id> \\
      --heldout me_on_another_day.wav --sources clip1.wav clip2.wav --grid use_finetuned=false,true
"""
from __future__ import annotations

import functools
print = functools.partial(print, flush=True)  # noqa: A001 - progress must show up in logs immediately

import argparse
import csv
import itertools
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--samples", nargs="+", help="recordings of the target voice to build a throw-away profile from")
    who.add_argument("--profile-id", help="use an existing app profile instead")
    ap.add_argument("--consent", action="store_true",
                    help="confirm you are authorised to use the --samples voice (required with --samples)")
    ap.add_argument("--heldout", nargs="*", default=[], help="target-voice recordings NOT used to build the profile")
    ap.add_argument("--sources", nargs="+", required=True, help="recordings to convert (another speaker)")
    ap.add_argument("--grid", action="append", default=[], metavar="KEY=V1,V2",
                    help="setting to sweep; repeatable. Keys: engine, any ConversionSettings field, or "
                         "sampler.<steps|schedule|sway|cfg_until> (engine sampler; steps applies to every quality)")
    ap.add_argument("--also", action="append", default=[], metavar="KEY=V[,KEY=V]",
                    help="one extra setting added to the grid (e.g. a reference); repeatable")
    ap.add_argument("--reference", default=None, metavar="SETTING",
                    help="setting label (as printed in the report) to measure every output's mel distance against")
    ap.add_argument("--repeats", type=int, default=1, help="conversions per setting (timing is averaged)")
    ap.add_argument("--seed", type=int, default=0,
                    help="diffusion noise seed, reset before every conversion so settings are compared "
                         "on the same noise (the engine is otherwise random)")
    ap.add_argument("--use-transcripts", action="store_true",
                    help="score CER against <source>.txt when present (default: ASR of the source)")
    ap.add_argument("--no-asr", action="store_true", help="skip the character-error-rate metric")
    ap.add_argument("--no-mos", action="store_true", help="skip the UTMOS naturalness metric")
    ap.add_argument("--asr-model", default=None, help="ASR judge (default: ENKURAL_EVAL_ASR_MODEL)")
    ap.add_argument("--listening", action="store_true", help="also write a blind A/B listening folder")
    ap.add_argument("--out", default=None, help="output folder (default: eval_runs/<timestamp>)")
    ap.add_argument("--keep-profile", action="store_true", help="keep the throw-away profile data")
    args = ap.parse_args()
    if args.samples and not args.consent:
        ap.error("--samples requires --consent: confirm this is your voice or you have the speaker's permission")
    return args


def parse_grid(items):
    grid = {}
    for item in items:
        key, _, vals = item.partition("=")
        if not vals:
            raise SystemExit(f"--grid expects KEY=V1,V2 (got {item!r})")
        grid[key.strip()] = [v.strip() for v in vals.split(",") if v.strip()]
    return grid


def coerce(key, value, defaults):
    if key == "engine":
        return value
    if key.startswith("sampler."):
        try:
            return int(value) if key == "sampler.steps" else float(value)
        except ValueError:
            return value
    if key not in defaults:
        raise SystemExit(f"unknown setting {key!r}; choose from engine, {', '.join(defaults)}")
    d = defaults[key]
    if isinstance(d, bool):
        return value.lower() in ("1", "true", "yes", "on")
    return type(d)(value)


def label(cfg):
    return " · ".join(f"{k}={v}" for k, v in cfg.items()) or "default"


def slug(text):
    """File-name-safe label; long labels are cut and made unique with a hash."""
    safe = "".join(c if c.isalnum() or c in "-_=." else "_" for c in text)
    if len(safe) <= 80:
        return safe
    import hashlib

    return f"{safe[:70]}-{hashlib.sha1(text.encode()).hexdigest()[:8]}"


def fmt(v, nd=3):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def main():
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out or ROOT / "eval_runs" / stamp).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    if args.samples:  # isolated store for the throw-away profile
        os.environ["ENKURAL_DATA_DIR"] = str(out / "_data")
    sys.path.insert(0, str(ROOT / "backend"))

    from enkural import config

    config.ensure_dirs()
    import numpy as np
    import torch

    from enkural import evaluation, pipeline, profiles, store
    from enkural.audio.io import load_audio, write_wav
    from enkural.vc.base import ConversionSettings
    from enkural.vc.registry import get_engine, resolve_device, unload_all

    defaults = ConversionSettings().to_dict()
    grid = parse_grid(args.grid)
    keys = list(grid)
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*grid.values())] if keys else [{}]
    for extra in args.also:
        configs.append(dict(kv.strip().split("=", 1) for kv in extra.split(",") if "=" in kv))
    engines = sorted({c.get("engine", config.DEFAULT_ENGINE) for c in configs})
    print(f"{len(configs)} setting(s) × {len(args.sources)} source(s) on {resolve_device()}  →  {out}")

    # ---------------------------------------------------------------- profile
    if args.samples:
        consent = {"confirmed": True, "statement": profiles.CONSENT_STATEMENT, "timestamp": store.now_iso(),
                   "via": "benchmark --consent"}
        profile = store.create_profile("benchmark", consent)
        for f in args.samples:
            profiles.add_sample(profile, Path(f).name, Path(f).read_bytes())
        t = time.time()
        profiles.build(profile["id"], engines[0], pipeline.QuietContext())
        print(f"profile built in {time.time() - t:.1f}s")
    else:
        profile = store.get_profile(args.profile_id)
        if not profile:
            raise SystemExit(f"profile {args.profile_id} not found in {config.DATA_DIR}")
        if not profile.get("consent", {}).get("confirmed") or profile.get("status") != "ready":
            raise SystemExit("that profile has no consent or is not built")
    profile = store.get_profile(profile["id"])
    for note in profile.get("build_notes") or []:
        print("  note:", note)

    heldout_paths = args.heldout
    if not heldout_paths:
        heldout_paths = args.samples or [str(store.profile_dir(profile["id"]) / "samples" / s["file"])
                                         for s in profile["samples"]]
        print("  WARNING: no --heldout audio; similarity is measured against the profile's own samples (inflated)")

    # ---------------------------------------------------------------- conversions
    audio_dir = out / "audio"
    audio_dir.mkdir(exist_ok=True)
    rows = []
    sources = [(Path(p), load_audio(p, sr=None)) for p in args.sources]
    warm = sources[0][1]
    for eng in engines:  # first call per engine pays for model loading; keep it out of the timing
        print(f"warming up {eng}…")
        pipeline.convert_clip(profile, type(warm)(warm.samples[: warm.sr * 5], warm.sr),
                              ConversionSettings.from_dict({"quality": "draft"}), eng)
    for (spath, clip), cfg in itertools.product(sources, configs):
        settings = ConversionSettings.from_dict({k: coerce(k, v, defaults) for k, v in cfg.items()
                                                 if k != "engine" and not k.startswith("sampler.")})
        eng = cfg.get("engine", config.DEFAULT_ENGINE)
        name = f"{slug(spath.stem)}__{slug(label(cfg))}"
        engine_obj = get_engine(eng)
        saved_sampler = json.loads(json.dumps(getattr(engine_obj, "sampler", {})))
        for k, v in cfg.items():
            if k.startswith("sampler."):
                key = k.split(".", 1)[1]
                if not hasattr(engine_obj, "sampler"):
                    raise SystemExit(f"engine {eng} has no tunable sampler")
                if key == "steps":
                    engine_obj.sampler["steps"] = {q: coerce(k, v, defaults) for q in engine_obj.sampler["steps"]}
                else:
                    engine_obj.sampler[key] = coerce(k, v, defaults)
        timings = []
        try:
            for r in range(max(1, args.repeats)):
                torch.manual_seed(args.seed)
                res = pipeline.convert_clip(profile, clip, settings, eng)
                timings.append(res["info"]["timings"])
        finally:
            if saved_sampler:
                engine_obj.sampler = saved_sampler
        info = res["info"]
        write_wav(audio_dir / f"{name}.wav", res["final"], subtype="PCM_24")
        write_wav(audio_dir / f"{name}.voice.wav", res["voice"], subtype="PCM_24")
        write_wav(audio_dir / f"{name}.source_voice.wav", res["source_voice"], subtype="PCM_24")
        engine = res["engine"]
        campplus = None
        try:
            style = np.load(res["target"].dir / res["target"].meta["style_file"])
            campplus = evaluation.cosine(engine.speaker_embedding(res["voice"]), style)
        except Exception:
            pass
        stages = {}
        for t in timings:
            for k, v in t["engine_stages_s"].items():
                stages[k] = stages.get(k, 0.0) + v / len(timings)
        rows.append({
            "source": spath.name, "setting": label(cfg), "file": f"audio/{name}.wav", "_name": name,
            "speed": settings.speed, "duration_s": round(clip.duration, 2),
            "total_s": mean([t["total_s"] for t in timings]),
            "rtf": mean([t["realtime_factor"] for t in timings]),
            "source_prep_s": mean([t["source_prep_s"] for t in timings]),
            **{f"{k}_s": round(v, 3) for k, v in stages.items()},
            "separated": info["separation"].get("applied"), "finetuned": info["finetuned"],
            "model": info["engine_info"].get("model"), "campplus_sim": campplus,
        })
        print(f"  {spath.name:24s} {label(cfg):40s} {rows[-1]['total_s']:.2f}s  rtf {rows[-1]['rtf']:.3f}")
    unload_all()

    # ---------------------------------------------------------------- scoring
    print("scoring…")
    ev = evaluation.Evaluator(asr_model=args.asr_model, use_asr=not args.no_asr, use_mos=not args.no_mos)
    heldout = [load_audio(p, sr=16000) for p in heldout_paths]
    centroid = ev.sv.centroid(heldout)
    levels = {}
    if len(heldout) >= 2:  # genuine-vs-genuine: each held-out clip against the others
        loo = []
        for i, h in enumerate(heldout):
            others = [x for j, x in enumerate(heldout) if j != i]
            loo.append(evaluation.cosine(ev.sv.embed(h), ev.sv.centroid(others)))
        levels["genuine_target_sim"] = mean(loo)
    else:
        ref = store.profile_dir(profile["id"]) / "reference.wav"
        if ref.exists():
            levels["genuine_target_sim"] = evaluation.cosine(ev.sv.embed(load_audio(ref, sr=16000)), centroid)
    for spath, clip in sources:
        levels[f"source_sim[{spath.name}]"] = evaluation.cosine(ev.sv.embed(clip), centroid)
        if not args.no_mos:
            levels[f"source_utmos[{spath.name}]"] = ev.mos.score(clip)
    for row in rows:
        name = row["_name"]
        conv = load_audio(audio_dir / f"{name}.voice.wav", sr=16000)
        src_voice = load_audio(audio_dir / f"{name}.source_voice.wav", sr=16000)
        text = None
        if args.use_transcripts:
            tp = Path(next(p for p, _ in sources if p.name == row["source"])).with_suffix(".txt")
            text = tp.read_text(encoding="utf-8").strip() if tp.exists() else None
        m = ev.score(src_voice, conv, centroid, speed=row["speed"], source_text=text,
                     source_key=hash((src_voice.samples.tobytes(), row["speed"])))
        row.update({k: v for k, v in m.items()})
        row["sim_gain_vs_source"] = m["speaker_sim"] - levels[f"source_sim[{row['source']}]"]
    ev.release()
    if args.reference:
        refs = {r["source"]: r for r in rows if r["setting"] == args.reference}
        if not refs:
            print(f"  WARNING: --reference {args.reference!r} matches no setting")
        for r in rows:
            ref = refs.get(r["source"])
            if ref:
                r["mel_dist_to_ref"] = evaluation.mel_distance(
                    load_audio(out / r["file"].replace(".wav", ".voice.wav"), sr=16000),
                    load_audio(out / ref["file"].replace(".wav", ".voice.wav"), sr=16000))

    # ---------------------------------------------------------------- report
    for r in rows:
        r.pop("_name", None)
    (out / "results.json").write_text(json.dumps({"levels": levels, "rows": rows, "args": vars(args)},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
    cols = sorted({k for r in rows for k in r if k != "transcript"}, key=lambda k: (k not in ("source", "setting"), k))
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    by_setting = {}
    for r in rows:
        by_setting.setdefault(r["setting"], []).append(r)
    stage_cols = [c for c in ("content_s", "dit_s", "vocoder_s") if any(c in r for r in rows)]
    lines = [f"# Benchmark {stamp}", "",
             f"- device: {resolve_device()} · judges: WavLM-SV (similarity), "
             f"{'none' if args.no_asr else (args.asr_model or config.EVAL_ASR_MODEL)} (CER), "
             f"{'none' if args.no_mos else 'UTMOS22'} (naturalness)",
             f"- target: {'samples ' + ', '.join(Path(s).name for s in args.samples) if args.samples else 'profile ' + profile['id']}",
             f"- held-out target audio: {', '.join(Path(p).name for p in heldout_paths)}",
             f"- sources: {', '.join(p.name for p, _ in sources)} · seed {args.seed}", ""]
    lines += [f"- note: {n}" for n in profile.get("build_notes") or []]
    lines += ["", "## Reference levels", "", "| measure | value |", "|---|---|"]
    lines += [f"| {k} | {fmt(v)} |" for k, v in levels.items()]
    lines += ["", "## Averages per setting", "",
              "| setting | RTF ↓ | total s | " + " | ".join(stage_cols) + " | speaker sim ↑ | gain vs source ↑ | CER ↓ | UTMOS ↑ | F0 corr ↑ | duration ratio | CAMPPlus (engine's own) | mel dist to ref ↓ |",
              "|" + "---|" * (11 + len(stage_cols))]
    for s, rs in by_setting.items():
        cells = [s, fmt(mean([r["rtf"] for r in rs])), fmt(mean([r["total_s"] for r in rs]), 2)]
        cells += [fmt(mean([r.get(c) for r in rs]), 2) for c in stage_cols]
        cells += [fmt(mean([r.get(k) for r in rs])) for k in
                  ("speaker_sim", "sim_gain_vs_source", "cer", "utmos", "f0_corr", "duration_ratio", "campplus_sim",
                   "mel_dist_to_ref")]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Per recording", "", "| source | setting | speaker sim | CER | UTMOS | F0 corr | RTF | file |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['source']} | {r['setting']} | {fmt(r.get('speaker_sim'))} | {fmt(r.get('cer'))} | "
                     f"{fmt(r.get('utmos'), 2)} | {fmt(r.get('f0_corr'))} | {fmt(r['rtf'])} | {r['file']} |")
    lines += ["", "## Reading the numbers", "",
              f"- **speaker sim**: WavLM-SV cosine to the held-out target audio. Same-speaker pairs usually score "
              f"above {evaluation.SV_SAME_SPEAKER_THRESHOLD}; compare with *genuine_target_sim* (the ceiling) and "
              "*source_sim* (the starting point).",
              "- **CER**: character error rate of the converted speech's transcript against the source's "
              "(or the known text with --use-transcripts). Lower is better; it rises when words get garbled.",
              "- **UTMOS**: predicted mean opinion score, 1–5. Trained on English speech, so compare settings "
              "with each other rather than reading the absolute value.",
              "- **F0 corr**: how closely the intonation contour follows the source (the pitch *level* moving "
              "to your voice is expected).",
              "- **mel dist to ref**: mean log-mel difference (dB) to the --reference setting's output from the same "
              "noise seed; how far a faster setting drifts from it.",
              "- **CAMPPlus**: the app's in-UI voice match. The engine is conditioned on this embedding, so it is "
              "optimistic by construction — shown only for comparison.",
              "- Objective scores guide the search; a blind listen (--listening) makes the final call."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.listening:
        lst = out / "listening"
        key = {}
        for spath, _ in sources:
            d = lst / slug(spath.stem)
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(spath, d / f"source{spath.suffix}")
            rs = [r for r in rows if r["source"] == spath.name]
            random.shuffle(rs)
            for letter, r in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", rs):
                shutil.copy(out / r["file"], d / f"{letter}.wav")
                key.setdefault(spath.name, {})[letter] = r["setting"]
        (lst / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"blind listening set: {lst} (open key.json only after choosing)")

    if args.samples and not args.keep_profile:
        shutil.rmtree(out / "_data", ignore_errors=True)
    print((out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
