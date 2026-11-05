#!/usr/bin/env bash
# Start En Kural and open it in your web browser.
#
#   ./start.sh                     start the app (or reuse a running one) and open the browser
#   ./start.sh --no-open           start without opening the browser
#   ./start.sh --port 7870         use another port (default 7860, or $ENKURAL_PORT)
#   ./start.sh --install-shortcut  add "En Kural" to your applications menu, then exit
#
# Close the app with Ctrl+C in this window (or close the window).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
ROOT="$(pwd)"

PORT="${ENKURAL_PORT:-7860}"
HOST=127.0.0.1
OPEN=1

usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="${2:?--port needs a number}"; shift 2 ;;
    --no-open) OPEN=0; shift ;;
    --install-shortcut)
      dir="$HOME/.local/share/applications"
      mkdir -p "$dir"
      cat > "$dir/en-kural.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=En Kural
Comment=Turn a recording into your own voice
Exec="$ROOT/start.sh"
Path=$ROOT
Terminal=true
Categories=AudioVideo;Audio;
EOF
      echo "Added \"En Kural\" to your applications menu ($dir/en-kural.desktop)."
      exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "En Kural is not installed yet. Run this once (it downloads about 6 GB):"
  echo "    ./scripts/setup.sh"
  exit 1
fi

BASE="http://$HOST:$PORT"
DATA_DIR="${ENKURAL_DATA_DIR:-$ROOT/data}"

# answers 0 when En Kural itself responds on the port
is_enkural() {
  "$PY" - "$BASE/healthz" <<'EOF' >/dev/null 2>&1
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=2) as r:
    sys.exit(0 if json.load(r).get("ok") else 1)
EOF
}
port_busy() {
  "$PY" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('$HOST', $PORT)) == 0 else 1)"
}
app_url() {
  local token=""
  [ -f "$DATA_DIR/.token" ] && token="$(tr -d '[:space:]' < "$DATA_DIR/.token")"
  if [ -n "$token" ] && [ "${ENKURAL_NO_AUTH:-0}" != "1" ]; then echo "$BASE/?token=$token"; else echo "$BASE/"; fi
}
open_browser() {
  [ "$OPEN" = 1 ] || return 0
  for opener in xdg-open gio sensible-browser; do
    if command -v "$opener" >/dev/null 2>&1; then
      if [ "$opener" = gio ]; then gio open "$1" >/dev/null 2>&1 & else "$opener" "$1" >/dev/null 2>&1 & fi
      return 0
    fi
  done
  echo "Could not open a browser automatically. Open the link above yourself."
}

if is_enkural; then
  URL="$(app_url)"
  echo "En Kural is already running. Opening it:"
  echo "    $URL"
  open_browser "$URL"
  exit 0
fi
if port_busy; then
  echo "Port $PORT is used by another program. Start En Kural on a different port, for example:" >&2
  echo "    ./start.sh --port $((PORT + 1))" >&2
  exit 1
fi

echo "Starting En Kural…"
export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}" ENKURAL_LAUNCHER=1
"$PY" -m enkural.main --host "$HOST" --port "$PORT" --preload &
SERVER=$!
stop() { kill "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; }
trap 'echo; echo "Stopping En Kural…"; stop; exit 0' INT TERM HUP
trap stop EXIT

for _ in $(seq 1 120); do            # up to 60 s for the web server to answer
  if ! kill -0 "$SERVER" 2>/dev/null; then
    echo "En Kural stopped while starting. See the messages above." >&2
    exit 1
  fi
  if is_enkural; then break; fi
  sleep 0.5
done
if ! is_enkural; then
  echo "En Kural did not start within a minute. See the messages above." >&2
  exit 1
fi

URL="$(app_url)"
echo
echo "  En Kural is open in your browser. If it did not open, use this link:"
echo "    $URL"
echo "  The voice model loads in the background (about 20 seconds)."
echo "  Keep this window open while you use the app. Press Ctrl+C to stop it."
echo
open_browser "$URL"
wait "$SERVER"
