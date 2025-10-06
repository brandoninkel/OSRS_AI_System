#!/bin/bash
# Simple wrapper to launch the GUI
# This avoids macOS security restrictions on app bundles

cd "$(dirname "$0")"
exec python3 pyqt6_admin_gui.py

