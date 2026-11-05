"""Objective checks for a conversion result (use when swapping engines).

  * speaker similarity: cosine similarity of CAMPPlus embeddings
      converted ↔ target profile   (higher = sounds more like you)
      converted ↔ source           (lower  = less of the friend's identity)
  * content preservation: Whisper ASR transcripts of source and converted
      audio compared with character error rate (Tamil script), optional.
  * timing: duration match between source and converted audio.

Usage:
  PYTHONPATH=backend .venv/bin/python scripts/evaluate.py --source S.wav --converted C.wav --reference R.wav [--asr]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from enkural import config  # noqa: E402

config.ensure_dirs()
from enkural.audio.io import load_audio  # noqa: E402
from enkural.vc.registry import get_engine  # noqa: E402


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def transcribe(path: str, name: str = "openai/whisper-small"):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained(name, cache_dir=str(config.MODELS_DIR / "hf"))
    mdl = WhisperForConditionalGeneration.from_pretrained(name, cache_dir=str(config.MODELS_DIR / "hf"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()
    clip = load_audio(path, sr=16000)
    x = clip.samples[: 16000 * 30]
    feats = proc(x, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
    with torch.inference_mode():
        kw = {"language": "ta", "task": "transcribe"} if "openai/" in name else {}
        ids = mdl.generate(feats, max_new_tokens=220, **kw)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--converted", required=True)
    ap.add_argument("--reference", required=True, help="clean target-voice audio (e.g. the profile reference)")
    ap.add_argument("--engine", default=config.DEFAULT_ENGINE)
    ap.add_argument("--asr", action="store_true", help="also compare Whisper transcripts (Tamil)")
    ap.add_argument("--asr-model", default="openai/whisper-small",
                    help="ASR judge, e.g. vasista22/whisper-tamil-small for a Tamil-specialised model")
    ap.add_argument("--truth-text", default=None, help="known transcript of the source (optional)")
    args = ap.parse_args()

    eng = get_engine(args.engine)
    src, conv, ref = (load_audio(p, sr=16000) for p in (args.source, args.converted, args.reference))
    e_src, e_conv, e_ref = (eng.speaker_embedding(c) for c in (src, conv, ref))
    print(f"duration          source {src.duration:.2f}s  converted {conv.duration:.2f}s")
    print(f"speaker sim       converted↔target {cosine(e_conv, e_ref):.3f}   converted↔source {cosine(e_conv, e_src):.3f}   source↔target {cosine(e_src, e_ref):.3f}")
    if args.asr:
        import jiwer

        t_src, t_conv = transcribe(args.source, args.asr_model), transcribe(args.converted, args.asr_model)
        print("ASR source     :", t_src)
        print("ASR converted  :", t_conv)
        print(f"CER converted vs source transcript: {jiwer.cer(t_src, t_conv):.3f}   (0 = identical)")
        if args.truth_text:
            norm = lambda x: " ".join(x.replace(".", " ").replace(",", " ").split())  # noqa: E731
            print(f"CER vs known text:  source {jiwer.cer(norm(args.truth_text), norm(t_src)):.3f}   "
                  f"converted {jiwer.cer(norm(args.truth_text), norm(t_conv)):.3f}   "
                  "(similar values = content preserved as far as the ASR can tell)")


if __name__ == "__main__":
    main()
