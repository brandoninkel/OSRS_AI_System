#!/bin/bash
# macOS launcher for OSRS AI: starts Ollama (if needed), RAG API, and Admin GUI
# Usage:
#   Double-click in Finder, or run from terminal:
#     ./start-gui.command [--with-embedder] [--with-watchdog] [--with-kg] [--basic-only]
#
# Options:
#   --with-embedder   Also start the incremental embedding watcher (follow mode)
#                     so new/changed wiki pages are embedded automatically.
#   --with-watchdog   Also start the streamlined wiki watchdog (auto-enables embedder and KG)
#   --with-kg         Also start the KG auto-updater service
#   --basic-only      Start only core services (API, RAG) without watchdog/embedder/KG

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
API_DIR="${REPO_ROOT}/api"
GUI_DIR="${REPO_ROOT}/gui"
FRONTEND_DIR="${REPO_ROOT}/frontend"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
DATA_DIR="${REPO_ROOT}/data"
LOG_DIR="${REPO_ROOT}/logs/osrs_ai"
mkdir -p "${LOG_DIR}" "${DATA_DIR}"

# --- Config ---
# Default to reranker OFF to match validated modality tests; enable by setting OSRS_USE_RERANKER=1
export OSRS_USE_RERANKER="0"
export OSRS_RERANKER_MODEL="BAAI/bge-reranker-large"
export OSRS_RERANK_CANDIDATES="30"
export OSRS_RERANK_EXPAND_TOP="8"
export OSRS_RERANK_EXPAND_NEIGHBORS="4"
export OSRS_EXCERPTS_PER_DOC="2"
# Increase Python stdout flushing for real-time logs
export PYTHONUNBUFFERED="1"

# --- Helpers (define functions first) ---
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

WITH_EMBEDDER=0
WITH_WATCHDOG=0
WITH_KG=0
BASIC_ONLY=0
# Parse flags (support multiple)
for arg in "$@"; do
  case "$arg" in
    --with-embedder)
      WITH_EMBEDDER=1
      ;;
    --with-watchdog)
      WITH_WATCHDOG=1
      ;;
    --with-kg)
      WITH_KG=1
      ;;
    --basic-only)
      BASIC_ONLY=1
      ;;
  esac
done

# If basic-only mode, disable all extras
if [[ ${BASIC_ONLY} -eq 1 ]]; then
  log "🔧 Basic-only mode: Starting core services only (API, RAG)"
  WITH_EMBEDDER=0
  WITH_WATCHDOG=0
  WITH_KG=0
fi

check_cmd() {
  command -v "$1" >/dev/null 2>&1
}

ensure_ollama_running() {
  if check_cmd curl; then
    if curl -sSf http://localhost:11434/api/tags >/dev/null 2>&1; then
      log "Ollama already running"
      return 0
    fi
  fi
  if check_cmd ollama; then
    log "Starting Ollama daemon..."
    nohup ollama serve >"${LOG_DIR}/ollama.out" 2>&1 &
    sleep 2
    if curl -sSf http://localhost:11434/api/tags >/dev/null 2>&1; then
      log "Ollama is up"
    else
      log "Warning: Ollama did not respond on :11434 yet (continuing)"
    fi
  else
    log "Warning: 'ollama' not found in PATH. Ensure Ollama is installed and running."
  fi
}

# Choose a Python interpreter (prefer pyenv if available)
PY_BIN="${HOME}/.pyenv/shims/python3"
if [[ ! -x "${PY_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then PY_BIN="$(command -v python3)"; else PY_BIN="/usr/bin/python3"; fi
fi
log "Using Python: ${PY_BIN} ($(${PY_BIN} -V 2>&1))"

# Preflight: check required Python deps (Flask & flask_cors)
if ! ${PY_BIN} - <<'PY' >/dev/null 2>&1
import flask, flask_cors
PY
then
  log "Error: Missing Python deps in ${PY_BIN}. Install with:"
  log "  ${PY_BIN} -m pip install flask flask-cors"
  log "Aborting startup."
  exit 1
fi

start_api() {
  log "Starting RAG API server (Flask) ..."
  cd "${API_DIR}"
  nohup "${PY_BIN}" osrs_api_server.py >"${LOG_DIR}/api.out" 2>&1 &
  echo $! >"${LOG_DIR}/api.pid"
  log "RAG API PID $(cat "${LOG_DIR}/api.pid") | Logs: ${LOG_DIR}/api.out"
}

start_gui() {
  log "Starting Modern Admin GUI (DearPyGui) ..."
  cd "${REPO_ROOT}/admin"
  nohup "${PY_BIN}" modern_admin_gui.py >"${LOG_DIR}/admin_gui.out" 2>&1 &
  echo $! >"${LOG_DIR}/admin_gui.pid"
  log "Modern Admin GUI PID $(cat "${LOG_DIR}/admin_gui.pid") | Logs: ${LOG_DIR}/admin_gui.out"
}

start_embedder_if_requested() {
  if [[ ${WITH_EMBEDDER} -eq 1 ]]; then
    if [[ -f "${SCRIPTS_DIR}/create_osrs_embeddings.py" ]]; then
      log "Starting incremental embedding watcher ..."
      cd "${SCRIPTS_DIR}"
      # Adjust concurrency if desired
      nohup /usr/bin/env python3 create_osrs_embeddings.py --incremental --async --max-concurrency 12 --follow \
        >"${LOG_DIR}/embedder.out" 2>&1 &
      echo $! >"${LOG_DIR}/embedder.pid"
      log "Embedder PID $(cat "${LOG_DIR}/embedder.pid") | Logs: ${LOG_DIR}/embedder.out"
    else
      log "Embedder script not found at ${SCRIPTS_DIR}/create_osrs_embeddings.py (skipping)"
    fi
  fi
}

start_kg_if_requested() {
  if [[ ${WITH_KG} -eq 1 ]]; then
    # Initialize KG if not already built
    if [[ ! -f "${REPO_ROOT}/data/osrs_kg_triples.csv" ]]; then
      log "Building initial KG triples ..."
      cd "${SCRIPTS_DIR}"
      if [[ -f "knowledge-graph.command" ]]; then
        bash knowledge-graph.command --workers 4 >"${LOG_DIR}/kg_build.out" 2>&1
        log "KG triples built | Logs: ${LOG_DIR}/kg_build.out"
      fi
    fi

    # Start KG auto-update service
    if [[ -f "${SCRIPTS_DIR}/kg_auto_updater.py" ]]; then
      log "Starting KG auto-update service ..."
      cd "${SCRIPTS_DIR}"
      nohup /usr/bin/env python3 kg_auto_updater.py --follow \
        >"${LOG_DIR}/kg_updater.out" 2>&1 &
      echo $! >"${LOG_DIR}/kg_updater.pid"
      log "KG Updater PID $(cat "${LOG_DIR}/kg_updater.pid") | Logs: ${LOG_DIR}/kg_updater.out"
    else
      log "KG auto-updater not found at ${SCRIPTS_DIR}/kg_auto_updater.py (skipping)"
    fi
  fi
}

start_watchdog_if_requested() {
  if [[ ${WITH_WATCHDOG} -eq 1 ]]; then
    if [[ -f "${SCRIPTS_DIR}/streamlined-watchdog.js" ]]; then
      log "Starting streamlined wiki watchdog ..."
      cd "${SCRIPTS_DIR}"
      nohup /usr/bin/env node streamlined-watchdog.js >"${LOG_DIR}/watchdog.out" 2>&1 &
      echo $! >"${LOG_DIR}/watchdog.pid"
      log "Watchdog PID $(cat "${LOG_DIR}/watchdog.pid") | Logs: ${LOG_DIR}/watchdog.out"
      # Ensure embedder follows when watchdog is enabled
      if [[ ${WITH_EMBEDDER} -eq 0 ]]; then
        log "Enabling embedder because --with-watchdog was requested"
        WITH_EMBEDDER=1
      fi
      # Ensure KG follows when watchdog is enabled
      if [[ ${WITH_KG} -eq 0 ]]; then
        log "Enabling KG auto-updates because --with-watchdog was requested"
        WITH_KG=1
      fi
    else
      log "Watchdog script not found at ${SCRIPTS_DIR}/streamlined-watchdog.js (skipping)"
    fi
  fi
}

print_endpoints() {
  log "Ready:"
  log "  Admin GUI: Desktop application (DearPyGui)"
  log "  API:       http://localhost:5002 (health: /health, stats: /stats)"
  log ""
  log "The Admin GUI should open automatically in a desktop window."
  log "If it doesn't appear, check the logs: ${LOG_DIR}/admin_gui.out"
}

main() {
  log "Repo root: ${REPO_ROOT}"
  ensure_ollama_running
  start_api
  # Give API a brief moment to bind before GUI proxies/requests
  sleep 1
  start_gui
  start_watchdog_if_requested
  start_embedder_if_requested
  start_kg_if_requested
  print_endpoints
}

main "$@"

