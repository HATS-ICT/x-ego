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
| `bomb`, complement and balanced variants | see `CONDITIONS` | `plant_before_window` selects **only label==1 rows**, so its accuracy is positive-class recall. `plant not before window` gives the complement the full-split average is taken over, and `plant before window or never planted` restores both classes while still excluding every row containing the plant event |
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

# 4b. Per-cell breakdown, needed whenever a row is not unanimous or is thinly
#     sampled. See REBUTTAL_Q1.md for which rebuttal sentence each column answers.
#     sampled. Every condition row also carries a min-to-max spread across cells.
python -m src.scripts.eval_subsets.analyze_subsets --per-cell \
    --out reviews/subset-results-percell.md
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
- **The `alive` conditions delete a class.** Requiring an earlier death removes the
  "nobody has died yet" class entirely. The training metric is macro recall and
  scores an empty class as 0, so the restricted delta is divided by the full class
  count rather than the surviving one, which deflates it by 1/6 on
  `enemy_aliveCount` and 1/5 on `teammate_aliveCount`. The report now prints a
  `classes present only` row alongside each, and that row is the fair comparison.
- The C1–C4 relay ladder is a separate flow with its own prerequisites. See below.

## The relay ladder (C1–C4)

Tests whether the POV agent's representation encodes an enemy it cannot see but a
teammate can. Reuses the same 120 dumps, so **no retraining and no new forward
passes** — only new per-sample flags and a re-slice.

| Condition | Meaning |
|---|---|
| C1 | B sees an enemy directly |
| C2 | B sees no enemy, a teammate A does, and B can see A |
| C3 | same, but B cannot see any informed teammate |
| C4 | nobody on B's team sees any enemy |

C2 versus C3 is the discriminator. C4 is the Area Chair's condition.

### Two visibility backends, deliberately

The engine tracks spotted state for **enemies only**. `m_bSpottedByMask` (exposed
by demoparser2 as `approximate_spotted_by`) names exactly which players have
spotted a given enemy, but teammates are on the radar regardless of line of sight
and never appear in it. So:

- **enemy leg** (`does teammate A see enemy E`) uses the mask, which is exact and
  accounts for occlusion, smoke, and the engine's own rules
- **teammate leg** (`does B see teammate A`) uses awpy line-of-sight against the
  map collision mesh, intersected with a field-of-view cone

### Steps, all CPU-only

```bash
# 1. Re-parse with pitch/yaw/spotted mask. Smoke-test one demo first.
bash scripts/reparse_all_angles.sh --data-dir $DATA --limit 1
bash scripts/reparse_all_angles.sh --data-dir $DATA

# 2. Collision mesh, once per map. Needs a .vphys extracted with Source 2 Viewer;
#    build_map_tri.py prints the exact export steps.
python -m src.scripts.data_processing.build_map_tri --map inferno --data-dir $DATA

# 3. VALIDATE. Do not skip. Prints the mask bit offset to use in step 4.
python -m src.scripts.eval_subsets.validate_visibility --map inferno \
    --data-dir $DATA --tri-path $DATA/inferno/mesh/de_inferno.tri

# 4. Build the conditions with the offset from step 3.
MASK_OFFSET=0 bash scripts/build_all_relay.sh --data-dir $DATA

# 5. The report picks up a relay section per task automatically.
python -m src.scripts.eval_subsets.analyze_subsets --out reviews/subset-results.md
```

### Why step 3 is mandatory

`m_bSpottedByMask` is indexed by entity slot, and the offset between `entity_id`
and the bit index has changed across parser versions. A wrong offset yields
plausible C2/C3 counts that mean nothing. `validate_visibility.py` resolves the
offset by agreement with geometry and checks three things that must hold
regardless of the data: no player is spotted by themselves, set bits name
opponents rather than teammates, and the mask implies the team-level `spotted`
flag. If any of those warn, use `ENEMY_BACKEND=los` instead and accept that
geometry over-counts because it does not model smoke or flashes.

### Known limits

- The labels are a **team-level multi-hot** over regions for all enemies jointly,
  so a condition attaches to a row, not to one enemy. Regions with exactly one
  occupant are attributable, and `n_relay_regions` / `relay_regions` record them
  for a finer label-level test. Do not report per-enemy conditions from the
  aggregate columns.
- C2 is the rarest cell by construction, being a four-way conjunction. The builder
  warns below n=100, at which point the C2 versus C3 comparison has no power and
  maps must be pooled.
- `EYE_HEIGHT` is the standing offset. Stance is not recorded, so crouched players
  are modelled slightly too tall.
