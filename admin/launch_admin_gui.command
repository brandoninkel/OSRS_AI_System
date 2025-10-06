#!/bin/bash
# macOS .command file to launch OSRS AI Admin GUI
# Double-click this file in Finder to launch the GUI

# Get the directory where this script is located
cd "$(dirname "$0")"

# Launch the GUI
echo "🚀 Launching OSRS AI Admin GUI..."
python3 pyqt6_admin_gui.py

# Keep terminal open if there was an error
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Error launching GUI. Press any key to close..."
    read -n 1
fi

