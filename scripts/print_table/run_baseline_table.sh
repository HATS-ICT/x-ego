#!/bin/bash
# Run the baseline results table generator
# Usage: ./run_baseline_table.sh [output_dir]

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Running baseline results table generator..."
echo "Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

# Run the Python script
if [ -n "$1" ]; then
    python scripts/print_table/baseline.py "$1"
else
    python scripts/print_table/baseline.py
fi

echo ""
echo "Done!"

