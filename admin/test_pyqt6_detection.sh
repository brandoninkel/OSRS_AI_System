#!/bin/bash
###############################################################################
# Test PyQt6 Detection for OSRS AI Admin App
# Simulates what the app bundle launch script does
###############################################################################

echo "🔍 Testing PyQt6 Detection"
echo "=" | tr '=' '='
echo ""

# Set up PATH like the app does
export PATH="/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin:$HOME/.pyenv/shims:$PATH"

# Function to test if Python has PyQt6
test_python_pyqt6() {
    local python_cmd="$1"
    echo -n "  Testing $python_cmd... "
    
    if ! command -v "$python_cmd" &> /dev/null; then
        echo "❌ Not found"
        return 1
    fi
    
    local python_path=$(which "$python_cmd")
    local python_version=$("$python_cmd" --version 2>&1)
    
    if "$python_cmd" -c "import PyQt6" 2>/dev/null; then
        local pyqt6_path=$("$python_cmd" -c "import PyQt6; print(PyQt6.__file__)" 2>/dev/null)
        echo "✅ Found"
        echo "     Path: $python_path"
        echo "     Version: $python_version"
        echo "     PyQt6: $pyqt6_path"
        return 0
    else
        echo "❌ No PyQt6"
        echo "     Path: $python_path"
        echo "     Version: $python_version"
        return 1
    fi
}

echo "Testing Python commands:"
echo ""

# Try common Python commands
FOUND_PYTHON=""
for cmd in python3 /usr/bin/python3 /usr/local/bin/python3 /opt/homebrew/bin/python3 python; do
    if test_python_pyqt6 "$cmd"; then
        FOUND_PYTHON="$cmd"
        echo ""
        echo "✅ SUCCESS: Will use $FOUND_PYTHON"
        break
    fi
    echo ""
done

if [ -z "$FOUND_PYTHON" ]; then
    echo "❌ FAILED: No Python with PyQt6 found"
    echo ""
    echo "To fix, install PyQt6:"
    echo "  python3 -m pip install --user PyQt6 psutil"
    exit 1
fi

echo ""
echo "🎉 PyQt6 detection working correctly!"
echo "   The app should launch successfully."

