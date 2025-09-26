#!/bin/bash
# Simple KG status tailer (log + progress.json)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR_PRIMARY="$REPO_ROOT/logs/kg"
LOG_DIR_FALLBACK="$REPO_ROOT/logs/kg"
LOG_DIR="$LOG_DIR_PRIMARY"
[ -d "$LOG_DIR" ] || LOG_DIR="$LOG_DIR_FALLBACK"
PROGRESS="$REPO_ROOT/data/kg_model/progress.json"

latest_log() {
  ls -t "$LOG_DIR"/train_*.log 2>/dev/null | head -n 1
}

LOG_FILE="$(latest_log || true)"
if [[ -z "${LOG_FILE:-}" ]]; then
  echo "(no KG log files found in $LOG_DIR)"
else
  echo "--- Tailing: $LOG_FILE (Ctrl-C to stop) ---"
fi

echo "--- Progress: $PROGRESS ---"
if [[ -f "$PROGRESS" ]]; then
  cat "$PROGRESS" || true
else
  echo "(no progress.json)"
fi

echo
if [[ -n "${LOG_FILE:-}" ]]; then
  tail -f "$LOG_FILE"
else
  echo "No log to tail. Exiting."
fi

