#!/usr/bin/env bash
# Serve the static website locally: http://localhost:8000
#
# The site needs an exported brain in web/model/ (brain.json + brain.flyb[.gz]):
#     fly export-web --run <run-name>
# Without it the page loads but shows "No fly brain found".
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8000}"
if [[ ! -f "$ROOT/web/model/brain.json" ]]; then
  echo "note: web/model/brain.json not found — run 'fly export-web --run <name>' first (the page will show an error until then)." >&2
fi
echo "serving $ROOT/web at http://localhost:$PORT  (Ctrl-C to stop)"
cd "$ROOT/web"
exec python3 -m http.server "$PORT" --bind 127.0.0.1
