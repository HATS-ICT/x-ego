#!/usr/bin/env bash
# Build every evaluation-subset flag file: 3 maps x 6 tasks = 18 CSVs.
#
# CPU only, no GPU and no model involved. Reads label CSVs, trajectories, and
# round metadata, and writes per-sample flags keyed by (partition, idx) that
# analyze_subsets.py later joins to the dumped predictions.
#
# The `moved` subsets load trajectories for every labelled row, so they take a
# couple of minutes each. Existing outputs are skipped unless --force is given.
#
# Usage:
#   scripts/build_all_eval_subsets.sh
#   scripts/build_all_eval_subsets.sh --data-dir /project2/ustun_1726/x-ego/data
#   scripts/build_all_eval_subsets.sh --force
#   scripts/build_all_eval_subsets.sh --maps inferno --partition all

set -uo pipefail

DATA_DIR="${DATA_BASE_PATH:-/project2/ustun_1726/x-ego/data}"
OUT_DIR="output/eval_subsets"
MAPS="inferno dust2 mirage"
PARTITION="test"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)  DATA_DIR="$2"; shift 2 ;;
    --out-dir)   OUT_DIR="$2";  shift 2 ;;
    --maps)      MAPS="$2";     shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --force)     FORCE=1; shift ;;
    -h|--help)   sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -d "$DATA_DIR" ]]; then
  echo "data dir not found: $DATA_DIR" >&2
  echo "pass --data-dir, or set DATA_BASE_PATH" >&2
  exit 1
fi

RUNNER=(python)
command -v uv >/dev/null 2>&1 && RUNNER=(uv run python)

# subset kind -> tasks it applies to
MOVED_TASKS="enemy_location_10s enemy_location_5s teammate_location_10s"
BOMB_TASKS="global_bombPlanted"
ALIVE_TASKS="enemy_aliveCount teammate_aliveCount"

echo "data dir : $DATA_DIR"
echo "out dir  : $OUT_DIR"
echo "maps     : $MAPS"
echo "partition: $PARTITION"
echo

built=0; skipped=0; failed=0
declare -a FAILURES=()

build_one() {
  local kind="$1" map="$2" task="$3"
  local target="${OUT_DIR}/${map}-${task}-${kind}.csv"

  if [[ $FORCE -eq 0 && -f "$target" ]]; then
    echo "skip  $target (exists; --force to rebuild)"
    skipped=$((skipped + 1))
    return
  fi

  echo "build $kind  $map  $task"
  if "${RUNNER[@]}" -m src.scripts.eval_subsets.build_eval_subsets "$kind" \
      --map "$map" --task "$task" \
      --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --partition "$PARTITION"; then
    built=$((built + 1))
  else
    echo "  FAILED: $kind $map $task" >&2
    FAILURES+=("$kind $map $task")
    failed=$((failed + 1))
  fi
}

for map in $MAPS; do
  for task in $MOVED_TASKS; do build_one moved "$map" "$task"; done
  for task in $BOMB_TASKS;  do build_one bomb  "$map" "$task"; done
  for task in $ALIVE_TASKS; do build_one alive "$map" "$task"; done
done

echo
echo "built $built, skipped $skipped, failed $failed"
if ((${#FAILURES[@]})); then
  echo "failures:"
  printf '  %s\n' "${FAILURES[@]}"
  exit 1
fi

echo
echo "flag files in $OUT_DIR:"
ls -1 "$OUT_DIR"/*.csv 2>/dev/null | sed 's#.*/#  #'
