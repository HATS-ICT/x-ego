# Evaluation subsets

Slices existing probe results into controlled conditions, to test whether CECL's
gains survive when the trivial explanation is removed. **Nothing is retrained.**
Each probe checkpoint already holds the frozen encoder plus its trained head, so
every step here is either a forward pass or offline arithmetic.

## What the subsets are

| Subset | Condition | Answers |
|---|---|---|
| `moved` | target's map region at the prediction tick differs from its region at the end of the observed window | the forecast gain is not carried forward from the present |
| `bomb` | the plant occurred before the observation window opened | the target state's defining event is outside every teammate's input, and the HUD is masked |
| `alive` | no death inside the window, but some occurred earlier | the count reflects events nobody could see in the input |

`moved` also emits a continuous `jaccard` column (overlap between the present and
future region sets). Binning on it uses the full test split and has far more power
than the disjoint subset alone, so prefer it as the primary analysis.

## Server run, start to finish

```bash
cd /home1/yunzhewa/projects/x-ego
git pull

DATA=/project2/ustun_1726/x-ego/data
export OUTPUT_BASE_PATH=/project2/ustun_1726/x-ego/output

# 1. Build the subset flags: 3 maps x 6 tasks = 18 CSVs. CPU only, no GPU.
#    Skips anything already built; pass --force to rebuild.
bash scripts/build_all_eval_subsets.sh --data-dir $DATA

# 2. Dump per-sample predictions for all 120 runs in the manifest.
#    Interactive (grab a GPU first, then run the loop directly):
#      salloc --account=ustun_1726 --partition=gpu --gres=gpu:1 \
#             --cpus-per-task=26 --mem=80G --constraint=a40\|a100 --time=8:00:00
bash scripts/dump_test_predictions.sh src/scripts/eval_subsets/dump_manifest.tsv
#    ...or one task at a time, which is the safer way interactively:
# bash scripts/dump_test_predictions.sh src/scripts/eval_subsets/dump_manifest.tsv enemy_location_10s
#
#    Batch alternative if you prefer to queue it:
# sbatch jobs/dump_test_predictions.job [task-regex]

# 3. Once it finishes, check the dumps reproduce the published metrics
python -m src.scripts.eval_subsets.analyze_subsets --verify

# 4. Produce the report
python -m src.scripts.eval_subsets.analyze_subsets --out reviews/subset-results.md
```

Then copy back only `reviews/subset-results.md` and `output/eval_subsets/*.csv`.
Do not copy the checkpoints; they are ~500 MB each.

## Read step 3 before trusting step 4

`--verify` recomputes each run's full-split metric from the dumped logits and
compares it against that run's own `test_results_best.json`. Median absolute
difference should be under 1e-3. If it is not, the dumped predictions are not the
ones behind the published numbers and every subset slice is meaningless. The
script prints a warning but does not stop you, so check the output.

## Notes

- The job is idempotent: runs with an existing `test_predictions_best.parquet` are
  skipped, so resubmit freely if it hits walltime.
- `dump_manifest.tsv` lists all 120 runs with arm, map, encoder, task, seed, and
  the Stage 1 checkpoint each CECL run used. It is needed because **baseline and
  CECL run directories are indistinguishable by name** — the only reliable marker
  is whether `model.stage1_checkpoint` is set in the run's `hparam.yaml`.
- Prediction dumping is gated on `data.dump_test_predictions` or the
  `XEGO_DUMP_TEST_PREDICTIONS` env var. The env var exists because `--mode test`
  reloads a saved experiment config, and configs written before that key existed
  cannot accept it as an override.
- `mirage/clip` is excluded: it has baseline runs but no CECL counterpart.
- `build_relay_conditions.py` (the C1–C4 relay ladder) is **not** part of this
  flow. It needs view angles, which the current trajectory parse does not extract.
  Run `src/scripts/data_processing/parse_traj_with_angles.py` first.
