#!/bin/bash
# Start only the core OSRS AI services (no GUI)
# Used by PyQt6 GUI to start backend services

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
API_DIR="${REPO_ROOT}/api"
LOG_DIR="${REPO_ROOT}/logs/osrs_ai"
mkdir -p "${LOG_DIR}"

# --- Helpers ---
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

check_cmd() {
  command -v "$1" >/dev/null 2>&1
}

ensure_ollama_running() {
  if check_cmd curl; then
    if curl -sSf http://localhost:11434/api/tags >/dev/null 2>&1; then
      log "✅ Ollama already running"
      return 0
    fi
  fi
  if check_cmd ollama; then
    log "🚀 Starting Ollama daemon..."
    nohup ollama serve >"${LOG_DIR}/ollama.out" 2>&1 &
    sleep 2
    if curl -sSf http://localhost:11434/api/tags >/dev/null 2>&1; then
      log "✅ Ollama is up"
    else
      log "⚠️ Warning: Ollama did not respond on :11434 yet (continuing)"
    fi
  else
    log "⚠️ Warning: 'ollama' not found in PATH. Ensure Ollama is installed and running."
  fi
}

# Choose a Python interpreter (prefer system python for consistency)
if command -v python3 >/dev/null 2>&1; then 
  PY_BIN="$(command -v python3)"
elif [[ -x "${HOME}/.pyenv/shims/python3" ]]; then
  PY_BIN="${HOME}/.pyenv/shims/python3"
else 
  PY_BIN="/usr/bin/python3"
fi

log "Using Python: ${PY_BIN} ($(${PY_BIN} -V 2>&1))"

start_api() {
  log "🚀 Starting RAG API server (Flask)..."
  cd "${API_DIR}"
  nohup "${PY_BIN}" osrs_api_server.py >"${LOG_DIR}/api.out" 2>&1 &
  echo $! >"${LOG_DIR}/api.pid"
  log "✅ RAG API PID $(cat "${LOG_DIR}/api.pid") | Logs: ${LOG_DIR}/api.out"
}

main() {
  log "🔧 Starting core OSRS AI services (API only, no GUI)"
  log "📁 Repo root: ${REPO_ROOT}"
  
  ensure_ollama_running
  start_api
  
  # Give API a moment to start
  sleep 2
  
  log "✅ Core services started successfully"
  log "🌐 API: http://localhost:5002 (health: /health, stats: /stats)"
}

main "$@"
