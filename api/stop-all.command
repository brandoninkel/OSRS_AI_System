#!/bin/bash
# macOS stopper for OSRS AI: stops GUI, RAG API, and optional embedder
# Usage:
#   ./stop-all.command [--and-ollama]
#
# Stops processes using PID files in logs/osrs_ai and falls back to ports if needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/osrs_ai"
API_PID_FILE="${LOG_DIR}/api.pid"
GUI_PID_FILE="${LOG_DIR}/gui.pid"
FRONTEND_PID_FILE="${LOG_DIR}/frontend.pid"
EMB_PID_FILE="${LOG_DIR}/embedder.pid"
WATCHDOG_PID_FILE="${LOG_DIR}/watchdog.pid"

AND_OLLAMA=0
if [[ "${1:-}" == "--and-ollama" ]]; then
  AND_OLLAMA=1
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

t_kill() {
  local pid="$1" name="$2"
  if kill -0 "$pid" >/dev/null 2>&1; then
    log "Stopping ${name} (PID ${pid}) ..."
    kill -TERM "$pid" || true
    for _ in {1..20}; do
      if kill -0 "$pid" >/dev/null 2>&1; then sleep 0.2; else break; fi
    done
    if kill -0 "$pid" >/dev/null 2>&1; then
      log "${name} not exiting, forcing..."
      kill -KILL "$pid" || true
    fi
  else
    log "${name} not running (PID ${pid} invalid)"
  fi
}

stop_by_pidfile() {
  local file="$1" name="$2"
  if [[ -f "$file" ]]; then
    local pid
    pid=$(cat "$file" || true)
    if [[ -n "${pid}" ]]; then
      t_kill "$pid" "$name"
    fi
    rm -f "$file"
  else
    log "${name} pid file not found (${file})"
  fi
}

stop_by_port() {
  local port="$1" name="$2"
  # macOS: lsof -iTCP -sTCP:LISTEN -n -P | grep :PORT
  local pids
  pids=$(lsof -t -iTCP:"${port}" -sTCP:LISTEN || true)
  if [[ -n "${pids}" ]]; then
    for p in ${pids}; do t_kill "$p" "$name (port ${port})"; done
  else
    log "No ${name} listening on port ${port}"
  fi
}

main() {
  mkdir -p "${LOG_DIR}"
  stop_by_pidfile "${API_PID_FILE}" "RAG API"
  stop_by_pidfile "${GUI_PID_FILE}" "GUI"
  stop_by_pidfile "${FRONTEND_PID_FILE}" "Frontend (Vite)"
  stop_by_pidfile "${EMB_PID_FILE}" "Embedder"
  stop_by_pidfile "${WATCHDOG_PID_FILE}" "Watchdog"

  # Fallback by ports
  stop_by_port 5001 "RAG API"
  stop_by_port 3002 "GUI"
  stop_by_port 3003 "Frontend (Vite)"

  if [[ ${AND_OLLAMA} -eq 1 ]]; then
    stop_by_port 11434 "Ollama"
  fi

  log "All requested processes have been stopped."
}

main "$@"

