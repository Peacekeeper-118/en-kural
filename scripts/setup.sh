#!/usr/bin/env bash
# One-time setup: Python 3.11 venv (via uv), dependencies, static ffmpeg, Seed-VC checkout, model download.
# No sudo required.  Re-running is safe.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "==> uv"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "==> Python 3.11 virtualenv"
uv python install 3.11 >/dev/null
[ -d .venv ] || uv venv --python 3.11 .venv

echo "==> Python dependencies (PyTorch CUDA 12.8 build; falls back to CPU automatically)"
UV_HTTP_TIMEOUT=900 uv pip install --python .venv/bin/python -r backend/requirements.txt

echo "==> ffmpeg (static build, no sudo needed)"
mkdir -p bin
if [ ! -x bin/ffmpeg ]; then
  TMP="$(mktemp -d)"
  curl -L -o "$TMP/ffmpeg.tar.xz" https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-linux64-gpl.tar.xz
  tar -xJf "$TMP/ffmpeg.tar.xz" -C "$TMP"
  cp "$TMP"/ffmpeg-*/bin/ffmpeg "$TMP"/ffmpeg-*/bin/ffprobe bin/
  rm -rf "$TMP"
fi

echo "==> Seed-VC (vendored)"
mkdir -p vendor
if [ ! -d vendor/seed-vc/modules ]; then
  git clone --depth 1 https://github.com/Plachtaa/seed-vc.git vendor/seed-vc
fi

echo "==> Model checkpoints (downloaded once into ./models)"
PYTHONPATH=backend .venv/bin/python scripts/download_models.py --quality-tools

echo
echo "Setup complete.  Start the app with:  ./run.sh"
