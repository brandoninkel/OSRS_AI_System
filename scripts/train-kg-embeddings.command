#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY_DEFAULT="$REPO_ROOT/.venv/bin/python3"
PY_KG="$REPO_ROOT/.kg-venv/bin/python3"
if [ ! -x "$PY_DEFAULT" ]; then PY_DEFAULT="python3"; fi
if [ ! -x "$PY_KG" ]; then PY_KG="$PY_DEFAULT"; fi

# Choose interpreter: use KG venv when explicitly requesting pykeen backend
USE_PY="$PY_DEFAULT"
case " $* " in
  *"--backend pykeen"*)
    USE_PY="$PY_KG";;
  *"--backend auto"*)
    # Prefer KG venv for auto so PyKEEN is available
    USE_PY="$PY_KG";;
  *) ;;
esac

cd "$REPO_ROOT"
mkdir -p logs/kg
TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="logs/kg/train_${TS}.log"
echo "[KG] Training KG embeddings (py=$USE_PY)..." | tee "$LOGFILE"
"$USE_PY" scripts/kg/train_kg_embeddings.py --resume "$@" 2>&1 | tee -a "$LOGFILE"
echo "[KG] Done. See $LOGFILE"

