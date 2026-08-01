# AC Q1: the four placeholder values, resolved

Run on CARC login node, 2026-08-01. `OUTPUT_BASE_PATH=/project2/ustun_1726/x-ego/output`,
120/120 dumps loaded, 0 missing. No retraining, no re-dumping, no checkpoint touched.
Full report: `reviews/subset-results-percell.md`.

## 1. Check A, verbatim

```
output base: /project2/ustun_1726/x-ego/output
loaded 120 runs, 0 missing

check A, recompute vs this run's own torchmetrics value
  120 runs, median |diff| 0.00020, max 0.00328  (tolerance 1e-2; small gaps are mixed-precision)
    OK  binary_cls       n= 20 median 0.00049  max 0.00328
    OK  multi_cls        n= 40 median 0.00000  max 0.00000
    OK  multi_label_cls  n= 60 median 0.00088  max 0.00287

check B, this run vs metrics recorded before the rerun
  120 runs, median |diff| 0.00025, max 0.00443  (reference holds best-checkpoint values, dumps are last)
    OK  binary_cls       n= 20 median 0.00000  max 0.00328
    OK  multi_cls        n= 40 median 0.00000  max 0.00443
    OK  multi_label_cls  n= 60 median 0.00090  max 0.00287
```

Matches the handout's prediction exactly (median ~0.0002, max ~0.0033, `multi_cls`
at 0.00000). Nothing regressed; the slices below are trustworthy.

## 2. A, B, C, D as one-liners

**A — "that cell's magnitude is under [X]" → under 0.029.**
The single dissenting cell in each of the four 4/5 rows:

| row | dissenting cell | delta |
|---|---|---|
| `global_bombPlanted` / plant before window | inferno, siglip2 | **-0.0027** |
| `teammate_aliveCount` / deaths precede window | mirage, dinov3 | **-0.0157** |
| `enemy_location_10s` / target changed region | mirage, dinov3 | **-0.0020** |
| `teammate_location_10s` / target changed region | mirage, dinov3 | **-0.0288** |

Largest magnitude 0.0288, so "under 0.029" is the tight bound. The bound is
unchanged under the 3b correction (teammate's cell becomes -0.0196 classes-present-only).
Worth naming in the prose: **three of the four dissenters are the same cell,
(mirage, dinov3)** — it is the weakest cell on the full split too (+0.0045 to
+0.0237 across tasks), so this is one soft cell, not four independent failures.

**B — enemy_aliveCount spread → restricted +0.0260 sits inside the full split's
+0.0191 to +0.0381.**
Full split +0.0290, per-cell **+0.0191 to +0.0381**, 5/5.
Restricted +0.0260, per-cell **+0.0063 to +0.0356**, 5/5.
Containment is mutual (the full-split mean also sits inside the restricted spread).
Per section 5 of the handout, drop "statistically indistinguishable" and use the
containment phrasing — a min-to-max spread is not a test.

**C — 43.9% unobservable, remainder = +0.0278.**
+0.0911 on the 43.9% (1934/4408; per map 51.0 / 40.2 / 42.3%) where the plant is
unobservable, against **+0.0278** on the remainder (`plant not before window`,
4/5 cells, median n 475, summed n 2474, spread -0.0484 to +0.0789).
Decomposition is arithmetically consistent: (1934x0.0911 + 2474x0.0278)/4408 =
+0.0556 vs the +0.0559 full split.

**D — thinnest-subset spreads.**
`enemy_location_5s`, median n 45: +0.0490, 5/5, **+0.0138 to +0.0879**.
`enemy_location_10s`, median n 117: +0.0425, 4/5, **-0.0020 to +0.0768**.
The thinner of the two is the better-behaved one, which is worth saying explicitly:
thin n is not what drives the dissent.

## 3. The row that decides section 3a — it holds

| condition | mean delta | cells positive | median n | min to max |
|---|---|---|---|---|
| full test split | +0.0559 | 5/5 | 823 | +0.0237 to +0.1081 |
| plant before window (all positives) | +0.0911 | 4/5 | 348 | -0.0027 to +0.1480 |
| plant not before window | +0.0278 | 4/5 | 475 | -0.0484 to +0.0789 |
| plant inside window | +0.0421 | 4/5 | 75 | -0.0267 to +0.1067 |
| **plant before window or never planted** | **+0.0562** | **5/5** | 753 | +0.0226 to +0.1082 |
| plant before window or never planted, no in-window plant | +0.0570 | 5/5 | 748 | +0.0247 to +0.1083 |

**Cells positive: 5/5. Both classes robustly present** — pooled 1238 positive vs
1306 negative over 2544 unique rows, 48.7% positive (per map 522/439, 368/462,
348/405). So section 3a does **not** force the finding to be withdrawn, and the
paragraph can be rebuilt around this row.

It does still force the framing to change, in two ways:

1. **This subset is 92% of the full split** (4050 of 4408 summed n), so its delta
   is close to the full-split value by construction, not as independent
   corroboration. Requiring "no plant event in the retained window" removes only
   the ~358 in-window-plant rows. State it as "the effect survives restricting to
   a plant-event-free, both-classes subset", not as "the effect is larger there".
2. **+0.0911 is positive-class recall and must be labelled as such.** The honest
   structure is: headline +0.0562 on the both-classes restricted subset, then
   +0.0911 / +0.0278 as the positives / remainder decomposition inside it.

## 4. 3b corrections, reported beside the originals as instructed

| task | restricted, as published | classes present only |
|---|---|---|
| `enemy_aliveCount` | +0.0260, 5/5, +0.0063 to +0.0356 | **+0.0313**, 5/5, +0.0075 to +0.0427 |
| `teammate_aliveCount` | +0.0144, 4/5, -0.0157 to +0.0485 | **+0.0180**, 4/5, -0.0196 to +0.0606 |

Both land where the handout predicted (~+0.031, ~+0.018). Note for the draft:
`enemy_aliveCount`'s "retains 90% of its full-split value" line is understated —
corrected, the restricted delta (+0.0313) slightly **exceeds** the full split
(+0.0290), i.e. 108%. Rephrase rather than keep the 90%.

## 5. What I changed

One addition to `analyze_subsets.py`: a fifth bomb condition, `plant before window
or never planted, no in-window plant`.

Why: the `plant before window or never planted` condition retains 23 of 2544 rows
(0.9%) where the plant lands inside the observation window while the label at
prediction time is still 0, so the handout's phrase "no retained row contains the
plant event" was not literally true of it. The strict variant excludes them and
makes the claim exact. It changes nothing material (+0.0570 vs +0.0562, still 5/5),
which is itself the useful result — either row can be quoted safely. Use the strict
row if the prose asserts plant-event-freeness; otherwise the shorter condition is fine.

Nothing else changed. `--verify` was run before the report, as instructed.
