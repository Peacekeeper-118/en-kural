"""Pre-download every checkpoint the default engine needs (safe to re-run)."""
from __future__ import annotations

import functools
print = functools.partial(print, flush=True)  # noqa: A001 - progress must show up in logs immediately

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from enkural import config  # noqa: E402

config.ensure_dirs()
from enkural.vc.hf import HF_CACHE, hf_download  # noqa: E402

V1 = [
    ("Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"),
    ("Plachta/Seed-VC", "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"),
    ("funasr/campplus", "campplus_cn_common.bin"),
    ("nvidia/bigvgan_v2_22khz_80band_256x", "bigvgan_generator.pt"),
    ("nvidia/bigvgan_v2_22khz_80band_256x", "config.json"),
]
V1_F0 = [
    ("Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth"),
    ("Plachta/Seed-VC", "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"),
    ("nvidia/bigvgan_v2_44khz_128band_512x", "bigvgan_generator.pt"),
    ("nvidia/bigvgan_v2_44khz_128band_512x", "config.json"),
    ("lj1995/VoiceConversionWebUI", "rmvpe.pt"),
]
TRAINING = [  # opt-in fine-tuning (timbre perturbation used by upstream's trainer)
    ("myshell-ai/OpenVoiceV2", "converter/checkpoint.pth"),
    ("myshell-ai/OpenVoiceV2", "converter/config.json"),
    ("Plachta/Seed-VC", "se_db.pt"),
]
V2 = [
    ("Plachta/Seed-VC", "v2/cfm_small.pth"),
    ("Plachta/Seed-VC", "v2/ar_base.pth"),
    ("Plachta/ASTRAL-quantization", "bsq32/bsq32_light.pth"),
    ("Plachta/ASTRAL-quantization", "bsq2048/bsq2048_light.pth"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also download the pitch-control and v2 models")
    ap.add_argument("--quality-tools", action="store_true",
                    help="also download voice separation, fine-tuning and evaluation models (~2 GB)")
    args = ap.parse_args()
    files = list(V1)
    if args.all:
        files += V1_F0 + V2
    if args.quality_tools:
        files += TRAINING
    for repo, name in files:
        print(f"  {repo}/{name}")
        hf_download(repo, name)
    from transformers import AutoFeatureExtractor, WhisperModel

    print("  openai/whisper-small")
    WhisperModel.from_pretrained("openai/whisper-small", cache_dir=str(HF_CACHE))
    AutoFeatureExtractor.from_pretrained("openai/whisper-small", cache_dir=str(HF_CACHE))
    if args.all:
        from transformers import HubertModel

        print("  facebook/hubert-large-ll60k")
        HubertModel.from_pretrained("facebook/hubert-large-ll60k", cache_dir=str(HF_CACHE))
    if args.quality_tools:
        from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

        print("  Hybrid Demucs (voice / background separation)")
        HDEMUCS_HIGH_MUSDB_PLUS.get_model()
        from enkural import evaluation

        print("  evaluation judges: WavLM-SV, UTMOS22, " + config.EVAL_ASR_MODEL)
        evaluation.SpeakerVerifier("cpu")._load()
        evaluation.NaturalnessPredictor("cpu")._load()
        evaluation.Transcriber(device="cpu")._load()
    print(f"Models are in {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
