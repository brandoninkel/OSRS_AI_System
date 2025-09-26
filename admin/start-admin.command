#!/bin/bash
# Launch OSRS Admin Console (DearPyGui)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Prefer project-local venv if present, then pyenv, then system python
if [[ -x "${REPO_ROOT}/.venv/bin/python3" ]]; then
  PY_BIN="${REPO_ROOT}/.venv/bin/python3"
elif [[ -x "${HOME}/.pyenv/shims/python3" ]]; then
  PY_BIN="${HOME}/.pyenv/shims/python3"
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="$(command -v python3)"
else
  PY_BIN="/usr/bin/python3"
fi
echo "Using Python: ${PY_BIN}"
cd "${REPO_ROOT}"

# Note: launching under nohup/background may crash on macOS; prefer launching from Terminal or Finder.
if [[ ! -t 1 ]]; then
  echo "(Notice) Starting GUI without a TTY. If this was started via nohup/background and crashes, launch it from Terminal or Finder instead."
fi

# Preflight DearPyGui import check with helpful guidance
if ! "${PY_BIN}" -c "import dearpygui.dearpygui as dpg; print('DearPyGui OK')" >/dev/null 2>&1; then
  echo "DearPyGui not installed for ${PY_BIN}."
  echo "Install with: \"${PY_BIN}\" -m pip install dearpygui requests psutil"
  exit 1
fi
exec "${PY_BIN}" "${REPO_ROOT}/admin/admin_gui.py"

