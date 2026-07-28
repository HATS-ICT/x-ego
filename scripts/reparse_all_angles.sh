#!/usr/bin/env bash
# Re-parse every demo on every map with view angles and engine spotted state.
#
# CPU only, no GPU. Writes to {map}/trajectory_angles/ and touches nothing that
# the existing pipeline reads, so it is safe to run while other jobs are queued.
# Expect roughly 10 to 30 seconds per demo.
#
# Usage:
#   scripts/reparse_all_angles.sh --data-dir /project2/ustun_1726/x-ego/data
#   scripts/reparse_all_angles.sh --data-dir $DATA --limit 1    # smoke test first

set -uo pipefail

DATA_DIR="data"
LIMIT=""
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --limit)    LIMIT="$2";    shift 2 ;;
    --force)    FORCE=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

MAPS=(inferno dust2 mirage)

# Confirm the props exist before committing to a long run. A parser build without
# approximate_spotted_by silently degrades the relay conditions to geometry.
echo "=== prop availability check on the first ${MAPS[0]} demo"
python -m src.scripts.data_processing.parse_traj_with_angles   --map "${MAPS[0]}" --data-dir "$DATA_DIR" --list-props
echo
echo "If pitch or yaw were REJECTED above, stop here and fix --props."
echo "Continuing in 5 seconds."
sleep 5

ok=0; failed=0
for map in "${MAPS[@]}"; do
  echo
  echo "=== $map"
  ARGS=(--map "$map" --data-dir "$DATA_DIR")
  [[ -n "$LIMIT" ]] && ARGS+=(--limit "$LIMIT")
  [[ $FORCE -eq 0 ]] && ARGS+=(--skip-existing)
  if python -m src.scripts.data_processing.parse_traj_with_angles "${ARGS[@]}"; then
    ok=$((ok + 1))
  else
    echo "  FAILED for $map" >&2
    failed=$((failed + 1))
  fi
done

echo
echo "maps parsed: $ok, failed: $failed"
echo
echo "Next, per map, validate before building any conditions:"
for map in "${MAPS[@]}"; do
  echo "  python -m src.scripts.eval_subsets.validate_visibility --map $map --data-dir $DATA_DIR"
done
