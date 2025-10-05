#!/bin/bash
# Start the Professional OSRS AI Admin GUI (PyQt6 Version)
# Modern, responsive interface with proper process management

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Find the correct Python with PyQt6 installed
# Try different Python installations in order of preference
PYTHON_CANDIDATES=(
    "/usr/bin/python3"                    # System Python (most likely to have PyQt6)
    "${REPO_ROOT}/.venv/bin/python3"      # Project virtual environment
    "${HOME}/.pyenv/shims/python3"       # pyenv Python
    "$(which python3 2>/dev/null || echo '')"  # Whatever's in PATH
)

PY_BIN=""
for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if [[ -x "$candidate" ]] && "$candidate" -c "from PyQt6.QtWidgets import QApplication" 2>/dev/null; then
        PY_BIN="$candidate"
        break
    fi
done

# Fallback if none found with PyQt6
if [[ -z "$PY_BIN" ]]; then
    PY_BIN="/usr/bin/python3"
fi

echo "Using Python: ${PY_BIN}"

echo "🚀 Starting Professional OSRS AI Admin GUI (PyQt6)"
echo "=" * 60
echo "✨ Features:"
echo "  • Modern dark theme with professional styling"
echo "  • Real-time system monitoring (CPU, Memory, Disk)"
echo "  • Intelligent process lifecycle management"
echo "  • Responsive layout with resizable sections"
echo "  • Comprehensive logging with tabbed interface"
echo "  • Proper cleanup on exit (no orphaned processes)"
echo ""
echo "📁 Repository: ${REPO_ROOT}"
echo "🖥️  GUI File: pyqt6_admin_gui.py"
echo ""

# Final verification
echo "🔍 Using Python: ${PY_BIN}"
if ! "${PY_BIN}" -c "from PyQt6.QtWidgets import QApplication" 2>/dev/null; then
    echo "❌ PyQt6 not available with any Python installation"
    echo ""
    echo "📦 Install PyQt6 with system Python:"
    echo "   /usr/bin/python3 -m pip install PyQt6 psutil"
    echo ""
    echo "🔧 Or install with the Python we found:"
    echo "   ${PY_BIN} -m pip install PyQt6 psutil"
    echo ""
    exit 1
fi

echo "✅ PyQt6 verified"
echo ""

# Change to admin directory and start GUI
cd "${SCRIPT_DIR}"

echo "🎮 Launching OSRS AI Control Center..."
echo "💡 Tips:"
echo "  • Use the splitter bars to resize sections"
echo "  • Check different log tabs for detailed information"
echo "  • All processes will be cleaned up when you close the GUI"
echo "  • System status updates every 2 seconds"
echo ""

# Start the PyQt6 GUI with the correct Python
"${PY_BIN}" pyqt6_admin_gui.py

echo ""
echo "👋 OSRS AI Control Center closed"
echo "✅ All processes cleaned up"
