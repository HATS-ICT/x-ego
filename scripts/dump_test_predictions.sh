#!/usr/bin/env bash
# Re-run test-only evaluation for every run in the manifest, dumping per-sample
# logits. Nothing is retrained: each probe checkpoint already holds the frozen
# encoder plus its trained head, so this is one forward pass over the test split.
#
# Idempotent. A run whose test_predictions_best.parquet already exists is skipped,
# so if the job hits walltime you can just resubmit.
#
# Usage:
#   scripts/dump_test_predictions.sh src/scripts/eval_subsets/dump_manifest.tsv
#   scripts/dump_test_predictions.sh <manifest> <task-filter-regex>
#
# Example, headline task only:
#   scripts/dump_test_predictions.sh src/scripts/eval_subsets/dump_manifest.tsv enemy_location_10s

set -uo pipefail

MANIFEST="${1:?usage: $0 <manifest.tsv> [task-regex]}"
TASK_FILTER="${2:-.}"

OUTPUT_BASE="${OUTPUT_BASE_PATH:-/project2/ustun_1726/x-ego/output}"
export XEGO_DUMP_TEST_PREDICTIONS=1

# Fewer DataLoader workers. The default of 8 exhausts /dev/shm on some nodes,
# surfacing as "unable to allocate shared memory(shm) for file ...". Test is a
# single pass, so throughput barely matters here.
NUM_WORKERS="${XEGO_TEST_NUM_WORKERS:-2}"

# One wandb run per checkpoint would create 120 junk runs and slow every step.
export WANDB_MODE="${WANDB_MODE:-disabled}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "manifest not found: $MANIFEST" >&2
  exit 1
fi

RUNNER=(python)
command -v uv >/dev/null 2>&1 && RUNNER=(uv run python)

total=0; done_already=0; ok=0; failed=0
declare -a FAILED_RUNS=()

while IFS=$'\t' read -r arm map enc task seed stage1 run_dir; do
  [[ "$arm" == "arm" ]] && continue          # header
  [[ -z "${run_dir:-}" ]] && continue
  [[ "$task" =~ $TASK_FILTER ]] || continue
  total=$((total + 1))

  # --mode test evaluates the "last" checkpoint, so that is the file produced.
  target="${OUTPUT_BASE}/${run_dir}/test_predictions_last.parquet"
  alt="${OUTPUT_BASE}/${run_dir}/test_predictions_best.parquet"
  if [[ -f "$target" || -f "$alt" ]]; then
    done_already=$((done_already + 1))
    continue
  fi

  if [[ ! -d "${OUTPUT_BASE}/${run_dir}/checkpoint" ]]; then
    echo "SKIP (no checkpoint dir): $run_dir" >&2
    FAILED_RUNS+=("$run_dir (missing checkpoint dir)")
    failed=$((failed + 1))
    continue
  fi

  echo "=== [$total] $arm $map/$enc $task seed$seed"
  echo "    $run_dir"
  if "${RUNNER[@]}" main.py --mode test --task downstream        "meta.resume_exp=${run_dir}" "data.num_workers=${NUM_WORKERS}"; then
    if [[ -f "$target" || -f "$alt" ]]; then
      ok=$((ok + 1))
    else
      echo "    WARNING: exited 0 but no parquet written" >&2
      FAILED_RUNS+=("$run_dir (no parquet)")
      failed=$((failed + 1))
    fi
  else
    echo "    FAILED" >&2
    FAILED_RUNS+=("$run_dir (nonzero exit)")
    failed=$((failed + 1))
  fi
done < "$MANIFEST"

echo
echo "matched $total runs: $ok dumped, $done_already already present, $failed failed"
if ((${#FAILED_RUNS[@]})); then
  echo "failures:"
  printf '  %s\n' "${FAILED_RUNS[@]}"
  exit 1
fi
