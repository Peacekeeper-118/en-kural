#!/usr/bin/env bash
# Start the En Kural server (local only by default).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Virtualenv missing. Run ./scripts/setup.sh first." >&2
  exit 1
fi
export PYTHONPATH="backend${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -m enkural.main "$@"
