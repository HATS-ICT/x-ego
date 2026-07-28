#!/usr/bin/env bash
# Build relay conditions C1-C4 for every map and enemy-location task.
#
# Prerequisites, in order. None involve a GPU.
#   1. Angle-augmented parse:  scripts/reparse_all_angles.sh
#   2. Collision mesh per map: python -m src.scripts.data_processing.build_map_tri --map <map>
#      (needs a .vphys extracted once with Source 2 Viewer, see that script)
#   3. Validate the signal:    python -m src.scripts.eval_subsets.validate_visibility --map <map>
#      Take the reported bit offset and pass it here via MASK_OFFSET.
#
# Usage:
#   scripts/build_all_relay.sh --data-dir /project2/ustun_1726/x-ego/data
#   MASK_OFFSET=1 scripts/build_all_relay.sh --data-dir $DATA
#   ENEMY_BACKEND=los scripts/build_all_relay.sh --data-dir $DATA   # if the mask failed validation

set -uo pipefail

# .env holds DATA_BASE_PATH, but only the Python side loads it via dotenv, so a
# plain shell run would not see it.
if [[ -z "${DATA_BASE_PATH:-}" && -f .env ]]; then
  set -a; . ./.env; set +a
fi

DATA_DIR="${DATA_BASE_PATH:-/project2/ustun_1726/x-ego/data}"
OUT_DIR="output/eval_subsets"
TRAJ_FOLDER="${TRAJ_FOLDER:-trajectory_angles}"
MASK_OFFSET="${MASK_OFFSET:-0}"
ENEMY_BACKEND="${ENEMY_BACKEND:-mask}"
TEAMMATE_BACKEND="${TEAMMATE_BACKEND:-los}"
FORCE=0

# An unset $DATA would otherwise make "--data-dir $DATA --out-dir x" collapse into
# "--data-dir --out-dir", silently taking a flag as the path.
need_value() {
  if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
    echo "$1 needs a value (got '${2:-}'). An unset shell variable is the usual cause." >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) need_value "$1" "${2:-}"; DATA_DIR="$2"; shift 2 ;;
    --out-dir)  need_value "$1" "${2:-}"; OUT_DIR="$2";  shift 2 ;;
    --force)    FORCE=1; shift ;;
    -h|--help)  sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -d "$DATA_DIR" ]]; then
  echo "data dir not found: $DATA_DIR" >&2
  echo "pass --data-dir, or set DATA_BASE_PATH" >&2
  exit 1
fi

MAPS=(inferno dust2 mirage)
# Only the enemy tasks: the relay question is about an opponent seen by a
# teammate. Teammate-location has no analogous condition.
TASKS=(enemy_location_0s enemy_location_5s enemy_location_10s)

echo "data dir         : $DATA_DIR"
echo "trajectory folder: $TRAJ_FOLDER"
echo "enemy backend    : $ENEMY_BACKEND (mask offset $MASK_OFFSET)"
echo "teammate backend : $TEAMMATE_BACKEND"
echo

built=0; skipped=0; failed=0
declare -a FAILED=()

for map in "${MAPS[@]}"; do
  if [[ ! -d "$DATA_DIR/$map/$TRAJ_FOLDER" ]]; then
    echo "SKIP $map: no $TRAJ_FOLDER. Run scripts/reparse_all_angles.sh first." >&2
    continue
  fi
  TRI="$DATA_DIR/$map/mesh/de_${map}.tri"
  TRI_ARG=()
  if [[ -f "$TRI" ]]; then
    TRI_ARG=(--tri-path "$TRI")
  elif [[ "$ENEMY_BACKEND" == "los" || "$TEAMMATE_BACKEND" == "los" ]]; then
    echo "SKIP $map: no mesh at $TRI and a los backend was requested." >&2
    echo "      Build it, or rerun with TEAMMATE_BACKEND=fov to accept over-counting." >&2
    continue
  fi

  for task in "${TASKS[@]}"; do
    out="$OUT_DIR/${map}-${task}-relay-${ENEMY_BACKEND}-${TEAMMATE_BACKEND}.csv"
    if [[ -f "$out" && $FORCE -eq 0 ]]; then
      echo "skip  $out (exists; --force to rebuild)"
      skipped=$((skipped + 1))
      continue
    fi
    echo "=== $map / $task"
    if python -m src.scripts.eval_subsets.build_relay_conditions         --map "$map" --task "$task" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"         --trajectory-folder "$TRAJ_FOLDER"         --enemy-backend "$ENEMY_BACKEND" --teammate-backend "$TEAMMATE_BACKEND"         --mask-bit-offset "$MASK_OFFSET" "${TRI_ARG[@]}"; then
      built=$((built + 1))
    else
      echo "  FAILED" >&2
      FAILED+=("$map/$task")
      failed=$((failed + 1))
    fi
  done
done

echo
echo "built $built, skipped $skipped, failed $failed"
if ((${#FAILED[@]})); then
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo
echo "Next: python -m src.scripts.eval_subsets.analyze_subsets --out reviews/subset-results.md"
echo "The report gains a 'relay conditions' section per task automatically."
