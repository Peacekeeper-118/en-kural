"""Fine-tuning worker (runs as a subprocess of the server; see ``finetune.py``).

    python -m enkural.vc.finetune_worker --spec <work_dir>/spec.json

1. trains the Seed-VC v1 DiT + length regulator on the profile's clean speech
   with upstream's ``Trainer.train_one_step`` (timbre-perturbed content, random
   prompt split), saving candidate checkpoints along the way
2. converts probe recordings of *another* speaker with the base model and each
   candidate, then scores them with the engine-independent judges against the
   held-out clips of the target voice
3. writes ``result.json``; progress goes to stdout as one JSON object per line

Running out of process frees every byte of GPU memory when training ends and
keeps a crash from taking the server down.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


def emit(**event) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


# --------------------------------------------------------------------------- training
def train(spec: Dict) -> List[Dict]:
    import torch
    import yaml

    from .. import config
    from .hf import HF_CACHE, hf_download, setup_seedvc
    from .seedvc_v1 import BASE_CKPT, REPO, TRAINABLE_MODULES

    setup_seedvc()
    import huggingface_hub  # noqa: F401  (import before train.py, which rewrites HF_HUB_CACHE)

    cwd = os.getcwd()
    os.chdir(config.SEEDVC_DIR)  # upstream resolves a few relative paths (configs/...)
    try:
        import train as upstream  # vendor/seed-vc/train.py
    finally:
        os.chdir(cwd)
    os.environ.pop("HF_HUB_CACHE", None)

    work = Path(spec["work_dir"])
    device = spec["device"]
    ckpt, cfg_path = hf_download(REPO, *BASE_CKPT)
    cfg = yaml.safe_load(open(cfg_path))
    cfg["log_dir"] = str(work / "logs")
    run_cfg = work / "train_config.yml"
    run_cfg.write_text(yaml.safe_dump(cfg))

    class Trainer(upstream.Trainer):
        """Upstream trainer with models loaded from our cache and no unused vocoder."""

        def build_semantic_fn(self, device, config_):
            from transformers import AutoFeatureExtractor, WhisperModel

            name = config_["model_params"]["speech_tokenizer"]["name"]
            dtype = torch.float16 if "cuda" in str(device) else torch.float32
            self.whisper_model = WhisperModel.from_pretrained(name, torch_dtype=dtype, cache_dir=str(HF_CACHE)).to(device)
            self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(name, cache_dir=str(HF_CACHE))
            del self.whisper_model.decoder
            self.whisper_model.eval()

            def semantic_fn(waves_16k):
                inputs = self.whisper_feature_extractor([w.cpu().numpy() for w in waves_16k], return_tensors="pt",
                                                        return_attention_mask=True, sampling_rate=16000)
                feats = self.whisper_model._mask_input_features(inputs.input_features,
                                                                attention_mask=inputs.attention_mask).to(device)
                with torch.no_grad():
                    out = self.whisper_model.encoder(feats.to(self.whisper_model.encoder.dtype), head_mask=None,
                                                     output_attentions=False, output_hidden_states=False,
                                                     return_dict=True)
                return out.last_hidden_state.to(torch.float32)[:, : waves_16k.size(-1) // 320 + 1]

            self.semantic_fn = semantic_fn

        def build_vocoder(self, device, config_):
            self.vocoder_fn = None  # not used by train_one_step

    torch.manual_seed(spec.get("seed", 0))
    steps, save_at = int(spec["steps"]), sorted(set(int(s) for s in spec["save_at"]))
    t = Trainer(config_path=str(run_cfg), pretrained_ckpt_path=ckpt, data_dir=spec["train_dir"], run_name="run",
                batch_size=int(spec["batch_size"]), num_workers=0, steps=steps, save_interval=10 ** 9,
                max_epochs=10 ** 6, device=device)
    for m in t.model.values():
        m.train()

    def save(step: int) -> Path:
        path = work / f"ckpt_{step:05d}.pth"
        net = {}
        for m in TRAINABLE_MODULES:
            params = dict(t.model[m].named_parameters())
            # parameters only: the remaining state (rotary tables, masks, KV caches) is not learned
            net[m] = {k: v.detach().cpu().clone() for k, v in t.model[m].state_dict().items() if k in params}
        torch.save({"net": net, "step": step}, path)
        return path

    saved, step, t0, ema = [], 0, time.time(), None
    while step < steps:
        for batch in t.train_dataloader:
            batch = [b.to(device) for b in batch]
            loss = t.train_one_step(batch)
            ema = loss if ema is None else 0.95 * ema + 0.05 * loss
            step += 1
            if step % 5 == 0 or step == steps:
                emit(event="train", step=step, steps=steps, loss=round(ema, 4),
                     elapsed=round(time.time() - t0, 1))
            if step in save_at:
                saved.append({"step": step, "path": str(save(step))})
            if step >= steps:
                break
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
    emit(event="trained", seconds=round(time.time() - t0, 1), peak_gpu_gb=peak and round(peak, 2))
    del t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return saved


# --------------------------------------------------------------------------- evaluation
def evaluate(spec: Dict, candidates: List[Dict]) -> Dict:
    import torch

    from .. import evaluation, pipeline, store
    from ..audio.io import AudioClip, load_audio
    from .base import ConversionSettings
    from .registry import unload_all

    profile = store.get_profile(spec["profile_id"])
    pdir = store.profile_dir(profile["id"])
    settings = ConversionSettings.from_dict({"quality": "standard"})
    probes = []
    for p in spec["probe_files"]:
        c = load_audio(p, sr=None)
        probes.append(AudioClip(c.samples[: int(c.sr * spec.get("probe_seconds", 20))], c.sr))
    runs = [{"step": 0, "label": "base", "path": None}] + [{"label": f"step {c['step']}", **c} for c in candidates]

    outputs = []  # (run index, probe index, source voice, converted voice)
    total = len(runs) * len(probes)
    for ri, run in enumerate(runs):
        prof = copy.deepcopy(profile)
        prof["finetune"] = ({"status": "ready", "engine": "seedvc_v1",
                             "checkpoint": str(Path(run["path"]).relative_to(pdir))} if run["path"] else None)
        for pi, probe in enumerate(probes):
            torch.manual_seed(spec.get("seed", 0))
            res = pipeline.convert_clip(prof, probe, settings, "seedvc_v1")
            outputs.append((ri, pi, res["source_voice"], res["voice"]))
            emit(event="eval", done=ri * len(probes) + pi + 1, total=total, phase="convert")
    unload_all()

    ev = evaluation.Evaluator(use_asr=spec.get("use_asr", True), use_mos=spec.get("use_mos", True))
    heldout = [load_audio(p, sr=16000) for p in spec["heldout_files"]]
    centroid = ev.sv.centroid(heldout)
    scores: Dict[int, List[Dict]] = {}
    for k, (ri, pi, src, conv) in enumerate(outputs):
        m = ev.score(src, conv, centroid, source_key=pi)
        m.pop("transcript", None)
        scores.setdefault(ri, []).append(m)
        emit(event="eval", done=k + 1, total=len(outputs), phase="score")
    ev.release()

    def avg(ms: List[Dict], key: str) -> Optional[float]:
        vals = [m[key] for m in ms if m.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    table = []
    for ri, run in enumerate(runs):
        ms = scores.get(ri, [])
        table.append({"label": run["label"], "step": run["step"], "path": run["path"],
                      **{k: avg(ms, k) for k in ("speaker_sim", "cer", "utmos", "f0_corr")}})
    return {"candidates": table, "probes": [Path(p).name for p in spec["probe_files"]]}


def select(table: List[Dict], max_cer_increase: float, max_utmos_drop: float, min_sim_gain: float,
           max_f0_corr_drop: float = 1.0) -> Optional[Dict]:
    """Best fine-tuned candidate that beats the base on speaker similarity
    without costing intelligibility, naturalness or intonation; ``None`` keeps the base."""
    base = table[0]
    best = None
    for c in table[1:]:
        if c["speaker_sim"] is None or base["speaker_sim"] is None:
            continue
        if c["speaker_sim"] < base["speaker_sim"] + min_sim_gain:
            continue
        if c.get("cer") is not None and base.get("cer") is not None and c["cer"] > base["cer"] + max_cer_increase:
            continue
        if c.get("utmos") is not None and base.get("utmos") is not None and c["utmos"] < base["utmos"] - max_utmos_drop:
            continue
        if (c.get("f0_corr") is not None and base.get("f0_corr") is not None
                and c["f0_corr"] < base["f0_corr"] - max_f0_corr_drop):
            continue
        if best is None or c["speaker_sim"] > best["speaker_sim"]:
            best = c
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    spec = json.loads(Path(ap.parse_args().spec).read_text())
    from .. import config

    config.ensure_dirs()
    candidates = train(spec)
    result: Dict = {"trained_steps": spec["steps"], "candidates": [{"step": c["step"]} for c in candidates]}
    if spec.get("probe_files") and spec.get("heldout_files"):
        result.update(evaluate(spec, candidates))
        chosen = select(result["candidates"], spec["max_cer_increase"], spec["max_utmos_drop"], spec["min_sim_gain"],
                        spec.get("max_f0_corr_drop", 1.0))
        result["chosen"] = chosen
        result["evaluated"] = True
    else:  # nothing to compare against: keep the last checkpoint, flagged as unverified
        result["chosen"] = {"step": candidates[-1]["step"], "path": candidates[-1]["path"], "label": "final"}
        result["evaluated"] = False
    Path(spec["work_dir"], "result.json").write_text(json.dumps(result, indent=2))
    emit(event="done")


if __name__ == "__main__":
    main()
