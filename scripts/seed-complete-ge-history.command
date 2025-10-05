#!/bin/bash
# macOS launcher for Complete GE Historical Data Seed

cd "$(dirname "$0")/.."
python3 scripts/seed_complete_ge_history.py

echo ""
echo "Press any key to close..."
read -n 1

