#!/bin/bash
# Script: evaluate_all_siglip.sh
# Purpose: Evaluate all siglip experiments that have not yet been tested.

MAIN_PY="/home1/yunzhewa/projects/x-ego/main.py"
OUTPUT_DIR="/project2/ustun_1726/x-ego/output"

cd "$OUTPUT_DIR" || exit 1

# Loop over all siglip experiments
for exp in siglip-*; do
    # Skip non-directories
    [[ -d "$exp" ]] || continue

    # Determine task type from experiment name
    if [[ "$exp" == *"en-nowcast"* ]]; then
        TASK="enemy_location_nowcast"
        TASK_SHORT="enemy_location"
    elif [[ "$exp" == *"tm-nowcast"* ]]; then
        TASK="teammate_location_nowcast"
        TASK_SHORT="enemy_location"
    else
        echo "⚠️ Could not determine task for $exp, skipping..."
        continue
    fi

    # Paths to expected result files
    BEST_JSON="$OUTPUT_DIR/$exp/test_analysis/${TASK_SHORT}_best/test_results.json"
    LAST_JSON="$OUTPUT_DIR/$exp/test_analysis/${TASK_SHORT}_last/test_results.json"

    # Check if both results already exist
    if [[ -f "$BEST_JSON" && -f "$LAST_JSON" ]]; then
        echo "✅ $exp already evaluated (both result files found). Skipping..."
        continue
    fi

    echo "=========================================="
    echo "Evaluating: $exp"
    echo "Task: $TASK"
    echo "=========================================="

    # Run evaluation
    python "$MAIN_PY" --mode test --task "$TASK" meta.resume_exp="$exp"
done
