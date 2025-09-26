#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python3"
if [ ! -x "$PY" ]; then
  PY="python3"
fi
cd "$REPO_ROOT"
mkdir -p logs/kg
TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="logs/kg/build_${TS}.log"
echo "[KG] Starting knowledge graph build..." | tee "$LOGFILE"
"$PY" scripts/kg/build_kg.py --snapshot "$@" 2>&1 | tee -a "$LOGFILE"
echo "[KG] Done. See $LOGFILE"

