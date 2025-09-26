#!/bin/bash
# macOS launcher for data pipeline: starts Watchdog + Embedder (+ API optional)
# Usage:
#   ./start-data.command [--no-api]
# Notes:
#   - Writes PID files to logs/osrs_ai
#   - Logs tail: logs/osrs_ai/{watchdog.out, embedder.out, api.out}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
API_DIR="${REPO_ROOT}/api"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
DATA_DIR="${REPO_ROOT}/data"
LOG_DIR="${REPO_ROOT}/logs/osrs_ai"
mkdir -p "${LOG_DIR}" "${DATA_DIR}"

# --- Config (keep consistent with start-gui) ---
export OSRS_USE_RERANKER="1"
export OSRS_RERANKER_MODEL="BAAI/bge-reranker-large"
export OSRS_RERANK_CANDIDATES="20"
export OSRS_RERANK_EXPAND_TOP="5"
export OSRS_RERANK_EXPAND_NEIGHBORS="3"
export OSRS_EXCERPTS_PER_DOC="1"
export PYTHONUNBUFFERED="1"

NO_API=0
if [[ "${1:-}" == "--no-api" ]]; then NO_API=1; fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
check_cmd() { command -v "$1" >/dev/null 2>&1; }

ensure_ollama_running() {
  if check_cmd curl; then
    if curl -sSf http://localhost:11434/api/tags >/dev/null 2>&1; then
      log "Ollama already running"; return 0; fi
  fi
  if check_cmd ollama; then
    log "Starting Ollama daemon..."
    nohup ollama serve >"${LOG_DIR}/ollama.out" 2>&1 &
    sleep 2
  else
    log "Warning: 'ollama' not found. Ensure it is installed and running."
  fi
}

# Choose Python interpreter
PY_BIN="${HOME}/.pyenv/shims/python3"
if [[ ! -x "${PY_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then PY_BIN="$(command -v python3)"; else PY_BIN="/usr/bin/python3"; fi
fi
log "Using Python: ${PY_BIN} ($(${PY_BIN} -V 2>&1))"

start_api() {
  log "Starting RAG API server ..."
  cd "${API_DIR}"
  nohup "${PY_BIN}" osrs_api_server.py --host localhost --port 5001 >"${LOG_DIR}/api.out" 2>&1 &
  echo $! >"${LOG_DIR}/api.pid"
  log "RAG API PID $(cat "${LOG_DIR}/api.pid") | Logs: ${LOG_DIR}/api.out"
}

start_watchdog() {
  if [[ -f "${SCRIPTS_DIR}/streamlined-watchdog.js" ]]; then
    log "Starting streamlined wiki watchdog ..."
    cd "${SCRIPTS_DIR}"
    nohup /usr/bin/env node streamlined-watchdog.js >"${LOG_DIR}/watchdog.out" 2>&1 &
    echo $! >"${LOG_DIR}/watchdog.pid"
    log "Watchdog PID $(cat "${LOG_DIR}/watchdog.pid") | Logs: ${LOG_DIR}/watchdog.out"
  else
    log "Watchdog script not found at ${SCRIPTS_DIR}/streamlined-watchdog.js"
  fi
}

start_embedder() {
  if [[ -f "${SCRIPTS_DIR}/create_osrs_embeddings.py" ]]; then
    log "Starting incremental embedding watcher ..."
    cd "${SCRIPTS_DIR}"
    nohup /usr/bin/env python3 create_osrs_embeddings.py --incremental --async --max-concurrency 12 --follow \
      >"${LOG_DIR}/embedder.out" 2>&1 &
    echo $! >"${LOG_DIR}/embedder.pid"
    log "Embedder PID $(cat "${LOG_DIR}/embedder.pid") | Logs: ${LOG_DIR}/embedder.out"
  else
    log "Embedder script not found at ${SCRIPTS_DIR}/create_osrs_embeddings.py"
  fi
}

print_endpoints() {
  log "Ready:"
  if [[ ${NO_API} -eq 0 ]]; then
    log "  API:   http://localhost:5001 (health: /health, stats: /stats)"
  fi
  log "  Logs:  tail -f ${LOG_DIR}/{watchdog.out,embedder.out,api.out}"
}

main() {
  ensure_ollama_running
  if [[ ${NO_API} -eq 0 ]]; then start_api; fi
  start_watchdog
  start_embedder
  print_endpoints
}

main "$@"

