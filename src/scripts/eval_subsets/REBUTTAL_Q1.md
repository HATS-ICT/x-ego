# Handout: four numbers the AC Q1 paragraph is waiting on

For whoever runs this on the cluster. Everything here is CPU-only offline
arithmetic over prediction dumps that already exist. **Do not retrain. Do not
re-dump. Do not touch a checkpoint.**

Read section 3 before pasting anything back. Two of the six existing conditions
mean something narrower than the rebuttal draft currently claims, and saying it
wrongly is worse than leaving the placeholder in.

## 1. What is being filled in

The rebuttal answers the Area Chair's Q1 (is the allocentric shift more than
expected commonality encoding?) with a restricted evaluation: rescore existing
predictions on subsets where co-visible evidence of the target is absent. The
aggregate table is already written from `reviews/subset-results.md`:

| Cond. | Task | Full split | Restricted | Cells | Median n |
| --- | --- | --- | --- | --- | --- |
| 1 | `global_bombPlanted` | +0.056 | +0.091 | 4/5 | 348 |
| 1 | `enemy_aliveCount` | +0.029 | +0.026 | 5/5 | 393 |
| 1 | `teammate_aliveCount` | +0.026 | +0.014 | 4/5 | 430 |
| 2 | `enemy_location_10s` | +0.039 | +0.043 | 4/5 | 117 |
| 2 | `enemy_location_5s` | +0.064 | +0.049 | 5/5 | 45 |
| 2 | `teammate_location_10s` | +0.033 | +0.023 | 4/5 | 146 |

Four sentences in the prose still carry placeholders, because
`analyze_subsets.py` only ever printed a mean, a k/5 count, and a median n. It
never emitted per-cell deltas and never scored the bomb complement, so these
numbers do not exist anywhere yet.

| # | Placeholder in the draft | What answers it |
|---|---|---|
| A | "In the four rows with a single dissenting cell, that cell's magnitude is under **[X]**." | the one negative per-cell delta in each of the four 4/5 rows: `global_bombPlanted`, `teammate_aliveCount`, `enemy_location_10s`, `teammate_location_10s` |
| B | "the restricted and full-split figures are statistically indistinguishable (**[spread across cells]**)" | `enemy_aliveCount`, min-to-max across the 5 cells, restricted and full split |
| C | "the advantage rises to +0.091 on the **[X]%** of samples where the plant is unobservable, against **[+0.0XX]** on the remainder" | the fraction is already known, see below. The remainder value is the `plant not before window` row |
| D | "the two condition-2 forecast rows at median n of 45 and 117 are our thinnest subsets, so we report their per-cell spread (**[spread]**)" | `enemy_location_5s` and `enemy_location_10s`, min-to-max across the 5 cells on the `target changed region` row |

**C's fraction is already resolved** from the local flag CSVs and needs nothing
from you: 522/1023 dust2, 368/916 inferno, 348/823 mirage x3, pooled
**1934 / 4408 = 43.9%**, median cell 42.3%. Only the remainder delta is missing.

The five cells throughout are (dust2, siglip2), (inferno, siglip2),
(mirage, dinov3), (mirage, siglip2), (mirage, vjepa2).

## 2. Run it

```bash
cd ~/projects/x-ego
git pull                       # needs commit 5f46a9d or later
export OUTPUT_BASE_PATH=/project2/ustun_1726/x-ego/output

# 0. Confirm the dumps are still there. Expect 120.
find $OUTPUT_BASE_PATH -name 'test_predictions_last.parquet' | wc -l

# 1. Check A must pass before any subset number is meaningful.
python -m src.scripts.eval_subsets.analyze_subsets --verify

# 2. The report. --per-cell is the whole point of this run.
python -m src.scripts.eval_subsets.analyze_subsets --per-cell \
    --out reviews/subset-results-percell.md
```

Expect check A at median |diff| ~0.0002, max ~0.0033, and `multi_cls` at exactly
0.00000. If check A regresses above 1e-2, stop and report; the metric definition
no longer matches the training code and every slice below is noise.

Copy back **only** `reviews/subset-results-percell.md`. Nothing else changed.

Every condition row now ends in a `min to max` column, which alone answers A, B
and D. The per-cell tables give the (map, encoder) identity of each dissenting
cell, which the draft may want to name.

## 3. Two conditions mean less than the draft claims. Read this.

### 3a. The bomb restricted subset is entirely positive-class

`plant_before_window` is defined as `planted_at_pred and plant < start_tick`
(`build_eval_subsets.py:161`), and `planted_at_pred` **is** the label. Verified
against the label CSVs: agreement exactly 1.0000 on all three maps. So the
subset behind `+0.091` contains only label==1 rows, and its "accuracy" is
positive-class recall, not balanced accuracy. Its complement is 85-88%
negatives. A CECL arm merely biased toward predicting "planted" would reproduce
this pattern exactly.

Consequences:

- The draft sentence *"together with matched samples in which no plant has
  occurred, so that both classes remain present and threshold-free metrics stay
  computable"* describes a subset that **was never run**. Do not ship it.
- Three new conditions now sit beside the original one, and the report prints
  all four:

  | condition | what it is | n per map (d2/inf/mir) |
  |---|---|---|
  | `plant before window` | the original. All positives | 522 / 368 / 348 |
  | `plant not before window` | its complement. Placeholder C | 501 / 548 / 475 |
  | `plant inside window` | the directly-visible regime | 75 / 91 / 75 |
  | `plant before window or never planted` | unobservable positives plus never-planted negatives | 961 / 830 / 753 |

  The last row is the honest version of the claim the draft wants to make: both
  classes present, and no retained row contains the plant event. **Report it.**
  If it holds there, the paragraph can be rebuilt around it with `+0.091` as the
  positives-only decomposition rather than as the headline.

### 3b. The alive conditions delete a class, and macro recall charges for it

Requiring an earlier death removes the "nobody has died yet" class entirely
(enemy class 5, teammate class 4 are absent from every restricted slice). The
published metric is macro-averaged recall, which scores an empty class as 0 for
both arms, so the restricted delta is divided by 6 instead of 5 on
`enemy_aliveCount` and by 5 instead of 4 on `teammate_aliveCount`. Corrected,
roughly +0.031 and +0.018 rather than +0.026 and +0.014.

The report now prints a `classes present only` row beside each restricted row.
**That row is the fair comparison.** Report both so the draft can choose, and
note that this makes `enemy_aliveCount`'s "retains 90% of its full-split value"
line stronger, not weaker.

## 4. What to paste back

1. The `--verify` output, verbatim.
2. The whole of `reviews/subset-results-percell.md`.
3. Four one-liners answering A, B, C, D directly, so nobody has to re-read the
   report to find them.
4. The `plant before window or never planted` row, called out separately, with
   its `min to max` and cells-positive. This is the one that decides whether
   section 3a forces a rewrite or not.
5. Anything you changed, and why.

## 5. Two claims in the draft to fix regardless of what comes back

- *"reproduces our reported values to within 0.0006 across all runs"*. The
  0.0006 was the max across the **six task-level deltas** (dumps vs published
  JSONs), not across runs. Per-run agreement was median 0.0002, max 0.0033.
  Either say "across all six tasks" or quote the per-run figures.
- *"statistically indistinguishable"*. A min-to-max spread is not a test. With
  five cells and two seeds a paired comparison across cells is possible once
  `--per-cell` reports both columns; until someone runs it, say the restricted
  value sits inside the per-cell spread of the full split, and drop
  "statistically".

## 6. Known limits, to state rather than fix

- `n` in the summary rows is a **median across cells**, not a total. The
  per-cell tables print a summed n so the two are not confused.
- The reference table's `which` column is meaningless; all 1133 rows are
  duplicated across best and last. Dumps are `last`, the reference holds `best`,
  so a small check-B gap is expected and is not a failure.
- The Jaccard gradient did not replicate and is deliberately absent from the
  draft. Do not resurrect it.
