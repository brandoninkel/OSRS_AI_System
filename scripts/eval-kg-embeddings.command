#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY_DEFAULT="$REPO_ROOT/.venv/bin/python3"
PY_KG="$REPO_ROOT/.kg-venv/bin/python3"
if [ ! -x "$PY_DEFAULT" ]; then PY_DEFAULT="python3"; fi
if [ ! -x "$PY_KG" ]; then PY_KG="$PY_DEFAULT"; fi

USE_PY="$PY_KG"

cd "$REPO_ROOT"
mkdir -p logs/kg
TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="logs/kg/eval_${TS}.log"
echo "[KG] Evaluating KG embeddings (py=$USE_PY)..." | tee "$LOGFILE"
"$USE_PY" scripts/kg/eval_kg_embeddings.py "$@" 2>&1 | tee -a "$LOGFILE"
echo "[KG] Eval done. See $LOGFILE"

