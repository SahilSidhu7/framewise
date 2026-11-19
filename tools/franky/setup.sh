#!/usr/bin/env bash
# Set up a Framewise clip-making project on a Linux server (tested target: the "franky" home server).
#
#   ./setup.sh            # install or update; safe to run again
#
# Layout (under $CLIPS, default ~/framewise-clips):
#   app/                Framewise engine source (copied from this repo by deploy.ps1)
#   .venv/              Python environment with the engine installed
#   inbox/              drop recordings here -> clips appear in inbox/output/<file name>/
#   inbox/framewise-watch.yaml   what the watch folder makes (recipe, style, platform, count)
#   home/               engine data: jobs, uploads, API keys
#   n8n/                optional n8n (Docker) that talks to the engine on 127.0.0.1:8765
set -euo pipefail

CLIPS="${CLIPS:-$HOME/framewise-clips}"
HERE="$(cd "$(dirname "$0")" && pwd)"

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1 ($2)" >&2; exit 1; }; }
need python3 "sudo apt install python3 python3-venv"
python3 - <<'PY' || { echo "Python 3.12+ is needed" >&2; exit 1; }
import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)
PY

mkdir -p "$CLIPS"/{inbox,home,n8n}
[ -d "$CLIPS/app/src/aieditor_engine" ] || { echo "engine source missing in $CLIPS/app (run deploy.ps1 from Windows)" >&2; exit 1; }

# Python environment
[ -d "$CLIPS/.venv" ] || python3 -m venv "$CLIPS/.venv"
"$CLIPS/.venv/bin/pip" install --quiet --upgrade pip
"$CLIPS/.venv/bin/pip" install --quiet -e "$CLIPS/app"
FW="$CLIPS/.venv/bin/framewise"

# Watch-folder settings (kept if you already edited them)
[ -f "$CLIPS/inbox/framewise-watch.yaml" ] || cp "$HERE/framewise-watch.yaml" "$CLIPS/inbox/"
cp "$HERE/docker-compose.yml" "$CLIPS/n8n/"

# Pro is needed for serve + watch folders: keep an existing licence, else start the 14-day trial
if ! "$FW" licence status | grep -q '"pro": true'; then
  "$FW" licence trial >/dev/null || true
fi
"$FW" licence status

# Engine as a user service (starts at boot once lingering is on)
mkdir -p "$HOME/.config/systemd/user"
sed "s#@CLIPS@#$CLIPS#g" "$HERE/framewise-clips.service" > "$HOME/.config/systemd/user/framewise-clips.service"
systemctl --user daemon-reload
systemctl --user enable --now framewise-clips.service
loginctl enable-linger "$USER" 2>/dev/null || echo "note: run 'sudo loginctl enable-linger $USER' so the service starts at boot"

echo
echo "Framewise clips project ready in $CLIPS"
echo "  drop a video into  $CLIPS/inbox/"
echo "  clips appear in    $CLIPS/inbox/output/<video name>/"
echo "  API (this server only): http://127.0.0.1:8765/docs"
echo "  first API key: journalctl --user -u framewise-clips | grep -A1 'first API key'"
echo "  optional n8n: cd $CLIPS/n8n && docker compose up -d   (then http://127.0.0.1:5678 over an SSH tunnel)"
