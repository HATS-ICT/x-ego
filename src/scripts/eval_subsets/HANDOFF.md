# Handoff: the relay experiment (C1-C4)

Written for an agent taking this over on the cluster. Read all of section 1 before
running anything; the point of the experiment determines which shortcuts are and
are not acceptable.

## 1. What we are trying to achieve

### Context

NeurIPS 2026 submission 20576, "The Allocentric Shift: Visual Cross-Ego Alignment
Exhibits Emergent Shared Mental Models". Scores 5 / 3 / 2. We are in the rebuttal
period.

The method, CECL, is two-stage. Stage 1 trains an encoder with a contrastive
objective that aligns simultaneous views of the same moment from different
teammates' first-person cameras. Stage 2 freezes that encoder and trains a linear
probe on downstream state-prediction tasks. The paper's finding is that alignment
improves prediction of *allocentric* state (where teammates and opponents are, how
many are alive, team-level events) while leaving *egocentric* state flat or worse.

### The Area Chair's question, which this experiment answers

> Are there global or team tasks where mere commonality-encoding should not help,
> for example predicting events invisible to all teammates, yet CECL improves
> performance?

The concern is deflationary. If cross-ego alignment merely teaches the encoder
"which features are shared across viewpoints", then gains should appear only where
the answer is visible to somebody. If gains persist where the answer is visible to
nobody, or reachable only by inference from teammate behaviour, then something
stronger than commonality-encoding is happening.

### The specific experiment

Inference is single-agent. At test time a POV agent B is given only B's own video
clip. Nothing is transmitted between agents. So if B's representation encodes an
enemy that B cannot see but a teammate A can see, the only route is that alignment
taught the encoder to read teammate behaviour, meaning stance, aim direction, peek
posture, as evidence about the unseen part of the world.

Four conditions, evaluated at the label's prediction tick for POV agent B:

| Condition | Definition |
|---|---|
| C1 | B sees at least one enemy directly |
| C2 | B sees no enemy, some teammate A sees an enemy B does not, and B can see A |
| C3 | same as C2 but B cannot see any such informed teammate |
| C4 | nobody on B's team sees any enemy |

**C2 versus C3 is the discriminator.** If the CECL-minus-baseline delta is much
larger in C2 than C3, the transfer depends on visual contact with the informed
teammate. If C2 and C3 are equal, the representation is team-aware regardless of
contact, which is a weaker relay story but still supports shared latent state. C4
isolates prior-based inference and is the closest thing to the AC's literal
"invisible to all teammates" case.

### Critical: this needs no retraining

Every probe checkpoint already contains the frozen encoder plus its trained head.
The per-sample predictions for all 120 relevant runs are **already dumped** to
`test_predictions_last.parquet`. This experiment only adds a per-sample condition
label and re-slices existing predictions offline. It is CPU-only arithmetic.

**Do not retrain anything. Do not modify or move any checkpoint.**

### Priority

This is upside, not critical path. The rebuttal already has a working answer to the
AC's question from a different analysis: on `global_bombPlanted`, restricted to
samples where the plant occurred before the observation window opened, the delta is
+0.0911 against +0.0559 on the full split. The relay ladder would sharpen the
argument considerably but the response stands without it. So prefer reporting an
honest negative or an underpowered result over forcing a number.

## 2. Current state

Done and verified:

- 120 per-sample prediction dumps exist under `$OUTPUT_BASE_PATH`, one per run,
  covering 6 tasks x 5 (map, encoder) cells x 2 arms x 2 seeds.
- Those dumps reproduce the published metrics. `analyze_subsets.py --verify` passes
  check A at median 0.0002.
- `reviews/subset-results.md` holds the two conditions that did not need a
  re-parse: forecast-target-moved, and event-precedes-window.

Not done, and the reason:

- The relay conditions need to know where each player is looking and who has
  spotted whom. The original trajectory parse requested `player_props=[]` and kept
  only tick, steamid, side, X, Y, Z, place, health. Everything in section 3 exists
  to fix that.

## 3. Your task, in order

All steps are CPU-only. Do not request a GPU.

### Step 0, confirm the starting point

```bash
cd ~/projects/x-ego && git pull
find /project2/ustun_1726/x-ego/output -name 'test_predictions_last.parquet' | wc -l
```

Expect 120. If not, stop and report; the rest is pointless without the dumps.

### Step 1, smoke-test the re-parse on one demo

```bash
bash scripts/reparse_all_angles.sh --limit 1
```

`.env` supplies `DATA_BASE_PATH`, so pass no paths. This first prints a prop probe.
Two lines decide whether the experiment is possible at all:

- `approximate_spotted_by` must read `OK`. This is `m_bSpottedByMask`, the engine's
  own record of which players have spotted a given player. Expect `LIST-VALUED`,
  with samples like `[8, 0]`, being two 32-bit words. If the samples look like
  steamids rather than small bitmasks, say so and stop.
- `entity_id` must read `OK`. Without it the mask bits cannot be tied to a player.

`pitch` and `yaw` are already confirmed present.

Then verify a CSV actually landed and the mask survived the write:

```bash
head -2 $(find /project2/ustun_1726/x-ego/data/inferno/trajectory_angles -name 'round_*.csv' | head -1)
```

The mask column should look like `8;0`. Nested lists are joined on write because
CSV has no nested type and, more importantly, because `polars.to_pandas()`
segfaults on them (see section 5).

### Step 2, full re-parse

```bash
bash scripts/reparse_all_angles.sh
```

51 demos across inferno, dust2, mirage. Roughly 10 to 30 seconds each, so under an
hour. Writes to `{map}/trajectory_angles/`, touching nothing the existing pipeline
reads. Safe to run while other jobs are queued.

### Step 3, validate the visibility signal. Do not skip this.

```bash
python -m src.scripts.eval_subsets.validate_visibility --map inferno
```

This is a gate, not a formality. It prints four checks and, most importantly, the
**mask bit offset** you must pass in step 5. `m_bSpottedByMask` is indexed by
entity slot, and the offset between `entity_id` and the bit index has changed
across parser versions. A wrong offset produces perfectly plausible C2 and C3
counts that mean nothing.

The four checks and how to read them:

1. **Bit offset.** Reports agreement with geometry per candidate offset. Wrong
   offsets are exposed by self-spotting, which is impossible. Needs a margin of at
   least 0.05 over the runner-up.
2. **Mask implies the team flag.** Whenever any bit is set, `spotted` must be true.
   Above 5% violations means the fields are not what we assume.
3. **Directionality.** Set bits must name opponents. Above 10% teammate bits means
   the field cannot support the enemy leg.
4. **Base rates.** If nearly every pair or nearly no pair is visible, the
   conditions will not partition the data usefully.

**Decision gate.** If checks 1 to 3 pass, use `ENEMY_BACKEND=mask` with the
reported offset. If any warns, use `ENEMY_BACKEND=los` instead and say so in your
report, because line-of-sight geometry does not model smoke or flashbangs and
therefore over-counts visibility.

### Step 4, the collision mesh, only if you need line of sight

The teammate leg of the condition ("does B see teammate A") cannot come from the
mask, because the engine maintains spotted state for enemies only; teammates appear
on the radar regardless of line of sight and never appear in the mask. So the
teammate leg needs geometry, and good geometry needs the map collision mesh.

```bash
python -m src.scripts.data_processing.build_map_tri --map inferno
```

This requires a `.vphys` extracted once per map with Source 2 Viewer, a GUI tool.
The script prints the exact export steps. **This is the one step that may be
impossible headless.** If the `.vphys` files are absent and cannot be obtained,
fall back to:

```bash
TEAMMATE_BACKEND=fov bash scripts/build_all_relay.sh
```

and state plainly in your report that the teammate leg is a field-of-view cone
without occlusion, so C2 is over-counted relative to C3 and the comparison is
biased toward finding no difference.

### Step 5, build the conditions

```bash
MASK_OFFSET=<from step 3> bash scripts/build_all_relay.sh
```

Watch the printed condition counts. The script warns when C2 or C3 falls below 100
samples. **C2 is a four-way conjunction and is rare by construction.** If C2 comes
in under 100 per cell, the C2-versus-C3 comparison has no power; report the counts
and stop rather than reporting a delta.

### Step 6, the report

```bash
python -m src.scripts.eval_subsets.analyze_subsets --out reviews/subset-results.md
```

A "relay conditions" section appears per task automatically. Run `--verify` first
and confirm check A still passes.

## 4. What to report back

1. The prop probe output from step 1, verbatim.
2. The full `validate_visibility` output for each map, verbatim, and which backend
   you consequently used.
3. The C1 to C4 counts per map, and whether C2 cleared 100.
4. The relay section of the report.
5. Anything you changed, and why.

State explicitly which of these hold: mask usable or not, mesh available or not,
C2 adequately powered or not. Those three facts determine how the result can be
described in the rebuttal, and describing it wrongly is worse than not having it.

## 5. Traps already hit, so you do not repeat them

- **`to_pandas()` segfaults on nested columns.** `pyarrow.pandas_compat.
  table_to_dataframe` releases the GIL and kills the interpreter with
  "PyThreadState_Get: the function must be called with the GIL held". The writer
  now uses `polars.write_csv` after joining lists to `"w0;w1"`. Do not reintroduce
  a pandas round trip in the parse path.
- **Do not filter empty fields when splitting the mask string.** Position carries
  meaning: element i covers bits [32i, 32i+32). Dropping a null first word shifts
  every later bit down by 32, silently.
- **`--data-dir $DATA` with `$DATA` unset** silently consumed the next flag as a
  path. The scripts now source `.env` and reject flag-shaped values. Pass no paths.
- **Default DataLoader workers exhaust `/dev/shm`** on some nodes. Only relevant if
  you re-dump predictions, which you should not need to.
- **The published `acc` for multi-class tasks is macro-averaged recall,** not plain
  accuracy, because `MulticlassAccuracy` is built with no `average` argument and
  torchmetrics defaults to macro. Do not "fix" that to micro; it changes
  `enemy_aliveCount` by 0.09.
- **`reference_metrics.tsv` holds best-checkpoint values for every run,** and its
  `which` column is meaningless since all 1133 rows are duplicated across best and
  last. The dumps are the last checkpoint. A small best-versus-last gap in check B
  is expected and is not a failure.

## 6. Known limits to state honestly, not to fix

- Labels are a **team-level multi-hot over map regions for all enemies jointly**, so
  a condition attaches to a row, not to one enemy. Regions with exactly one occupant
  are attributable and recorded in `n_relay_regions` / `relay_regions`, which
  supports a finer label-level test. Do not report per-enemy conditions from the
  aggregate columns.
- `EYE_HEIGHT` is the standing offset. Stance is not recorded, so crouched players
  are modelled slightly too tall.
- Geometry knows about walls but not smoke, flashes, or the engine's spotting
  rules. The mask knows all of them, which is why it is preferred for the enemy leg.

## 7. File map

| File | Role |
|---|---|
| `src/scripts/data_processing/parse_traj_with_angles.py` | re-parse with angles and spotted mask |
| `src/scripts/data_processing/build_map_tri.py` | `.vphys` to `.tri` collision mesh |
| `src/scripts/eval_subsets/visibility.py` | mask / line-of-sight / FOV / team-flag backends |
| `src/scripts/eval_subsets/validate_visibility.py` | the step 3 gate |
| `src/scripts/eval_subsets/build_relay_conditions.py` | assigns C1 to C4 |
| `src/scripts/eval_subsets/analyze_subsets.py` | verification and the report |
| `src/scripts/eval_subsets/relay_io.py` | folder-parameterised trajectory loader |
| `scripts/reparse_all_angles.sh`, `scripts/build_all_relay.sh` | drivers |
| `src/scripts/eval_subsets/README.md` | the other two conditions, already complete |

Unit tests for mask decoding, FOV geometry, offset recovery, and C1 to C4
assignment were run on a Windows machine without polars or awpy, so the polars and
awpy call sites are **unverified against real versions**. Expect to fix an API
detail there and prefer fixing it over working around it.
