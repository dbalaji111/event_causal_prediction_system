#!/usr/bin/env bash
# Starts the gold Red Flag API and opens redflag_dashboard_gold.html (with the Scenario tests button).
#
#   ./start_gold_dashboard.sh                      # uses GOLD_DATA_ROOT, or demo/data_root_gold_real
#   ./start_gold_dashboard.sh "/path/to/gold/data"  # the folder with memory_train/, memory_test/, data/, kde_output/
#
# First run: creates a Python environment in .venv and installs what the API needs (a few minutes).
# Then: checks the data, builds missing KDE results, starts the API on port 8010 and opens the dashboard.
# Press Ctrl+C in this window to stop.
set -euo pipefail

cd "$(dirname "$0")"
KIT="$(pwd)"
PORT=8010
URL="http://localhost:${PORT}/redflag_dashboard_gold.html"
MARKET="markets/gold.json"

say()  { printf '\033[1;33m▸\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ── 1. data folder ──────────────────────────────────────────────────────────────
ROOT="${1:-${GOLD_DATA_ROOT:-$KIT/demo/data_root_gold_real}}"
[ -d "$ROOT" ] || fail "Data folder not found: $ROOT
   Pass it as the first argument, e.g.  ./start_gold_dashboard.sh \"/path/to/gold/data\""
ROOT="$(cd "$ROOT" && pwd)"
[ -e "$ROOT/memory_train/price_event_memory.json" ] || fail "$ROOT has no memory_train/price_event_memory.json: is this the gold data folder?"
[ -e "$ROOT/data/gold_prices.csv" ] || fail "$ROOT has no data/gold_prices.csv"
say "Data folder: $ROOT"

# ── 2. python ───────────────────────────────────────────────────────────────────
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"                                   # use an existing interpreter, skip the .venv
else
  command -v python3 >/dev/null || fail "python3 is not installed (macOS: install it from python.org)"
  if [ ! -x .venv/bin/python ]; then
    say "Creating .venv (first run only)…"
    python3 -m venv .venv
  fi
  PY="$KIT/.venv/bin/python"
  if [ ! -f .venv/.deps_ok ]; then
    say "Installing packages (first run only)…"
    "$PY" -m pip install --quiet --upgrade pip
    "$PY" -m pip install --quiet numpy pandas scipy fastapi "uvicorn[standard]" openpyxl
    touch .venv/.deps_ok
  fi
fi
"$PY" -c "import fastapi, uvicorn, pandas, scipy" 2>/dev/null || fail "Python packages missing. Delete .venv and run again."

CLI=("$PY" -m redflag_kit.cli --market "$MARKET" --root "$ROOT")

# ── 3. KDE results (the backtest and the scenario tests read them) ─────────────
missing=()
for h in 7 15 30; do [ -f "$ROOT/kde_output/kde_v3_results_h${h}.csv" ] || missing+=("$h"); done
if [ ${#missing[@]} -gt 0 ]; then
  if [ -z "$(ls -A "$ROOT/retrieval_output/per_event" 2>/dev/null)" ]; then
    say "Retrieving analogues for the test events (first run only)…"
    "${CLI[@]}" retrieve
  fi
  hz="$(IFS=,; echo "${missing[*]}")"
  say "Building KDE results for horizons $hz…"
  "${CLI[@]}" kde --horizons "$hz"
fi

# ── 4. start the API (or reuse one already running) ─────────────────────────────
open_browser() {
  if command -v open >/dev/null; then open "$URL"
  elif command -v xdg-open >/dev/null; then xdg-open "$URL" >/dev/null 2>&1 &
  else say "Open $URL in your browser"; fi
}

if curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/scenarios/catalog" 2>/dev/null | grep -q 200; then
  say "The API is already running on port $PORT."
  open_browser
  exit 0
fi
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  fail "Port $PORT is used by another program (not this API). Close it, or stop it with:  lsof -ti tcp:$PORT | xargs kill"
fi

say "Starting the API on port $PORT…"
"${CLI[@]}" serve --port "$PORT" &
API_PID=$!
trap 'echo; say "Stopping the API…"; kill $API_PID 2>/dev/null; wait $API_PID 2>/dev/null; exit 0' INT TERM

for _ in $(seq 1 60); do
  if curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/scenarios/catalog" 2>/dev/null | grep -q 200; then
    say "Ready: $URL"
    say "Use the Validation panel's  Scenario tests ↗  button. Press Ctrl+C here to stop."
    open_browser
    wait $API_PID
    exit $?
  fi
  kill -0 $API_PID 2>/dev/null || fail "The API stopped while starting: see the messages above."
  sleep 1
done
kill $API_PID 2>/dev/null
fail "The API did not answer within 60 seconds."
