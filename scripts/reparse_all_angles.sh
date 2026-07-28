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

# .env holds DATA_BASE_PATH, but only the Python side loads it via dotenv, so a
# plain shell run would not see it.
if [[ -z "${DATA_BASE_PATH:-}" && -f .env ]]; then
  set -a; . ./.env; set +a
fi

DATA_DIR="${DATA_BASE_PATH:-/project2/ustun_1726/x-ego/data}"
LIMIT=""
FORCE=0

# An unset $DATA would otherwise make "--data-dir $DATA --limit 1" collapse into
# "--data-dir --limit", silently taking a flag as the path.
need_value() {
  if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
    echo "$1 needs a value (got '${2:-}'). An unset shell variable is the usual cause." >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) need_value "$1" "${2:-}"; DATA_DIR="$2"; shift 2 ;;
    --limit)    need_value "$1" "${2:-}"; LIMIT="$2";    shift 2 ;;
    --force)    FORCE=1; shift ;;
    -h|--help)  sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -d "$DATA_DIR" ]]; then
  echo "data dir not found: $DATA_DIR" >&2
  echo "pass --data-dir, or set DATA_BASE_PATH" >&2
  exit 1
fi

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
