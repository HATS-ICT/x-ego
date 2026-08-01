# Subset analysis

Delta is CECL minus baseline, seeds averaged within each arm before
differencing. `n` is the number of test rows in the condition.

## enemy_location_10s

| condition | mean delta | cells positive | median n |
|---|---|---|---|
| full test split | +0.0394 | 5/5 | 735 |
| target changed region | +0.0425 | 4/5 | 117 |

Delta by how much the present answer overlaps the future one. Lower jaccard means less can be carried forward.

| jaccard | mean delta | cells positive | median n |
|---|---|---|---|
| -0.01-0.00 | +0.0425 | 4/5 | 117 |
| 0.00-0.20 | +0.0403 | 5/5 | 194 |
| 0.20-0.40 | +0.0314 | 5/5 | 215 |
| 0.40-0.60 | +0.0328 | 5/5 | 133 |
| 0.60-0.80 | +0.0418 | 5/5 | 49 |
| 0.80-1.01 | +0.0190 | 5/5 | 27 |

## enemy_location_5s

| condition | mean delta | cells positive | median n |
|---|---|---|---|
| full test split | +0.0638 | 5/5 | 748 |
| target changed region | +0.0490 | 5/5 | 45 |

Delta by how much the present answer overlaps the future one. Lower jaccard means less can be carried forward.

| jaccard | mean delta | cells positive | median n |
|---|---|---|---|
| -0.01-0.00 | +0.0490 | 5/5 | 45 |
| 0.00-0.20 | +0.0605 | 5/5 | 117 |
| 0.20-0.40 | +0.0595 | 5/5 | 209 |
| 0.40-0.60 | +0.0624 | 5/5 | 177 |
| 0.60-0.80 | +0.0626 | 5/5 | 108 |
| 0.80-1.01 | +0.0546 | 5/5 | 92 |

## teammate_location_10s

| condition | mean delta | cells positive | median n |
|---|---|---|---|
| full test split | +0.0329 | 5/5 | 740 |
| target changed region | +0.0230 | 4/5 | 146 |

Delta by how much the present answer overlaps the future one. Lower jaccard means less can be carried forward.

| jaccard | mean delta | cells positive | median n |
|---|---|---|---|
| -0.01-0.00 | +0.0230 | 4/5 | 146 |
| 0.00-0.20 | +0.0299 | 5/5 | 182 |
| 0.20-0.40 | +0.0328 | 5/5 | 219 |
| 0.40-0.60 | +0.0333 | 5/5 | 103 |
| 0.60-0.80 | +0.0344 | 5/5 | 50 |
| 0.80-1.01 | +0.0129 | 3/5 | 40 |

## global_bombPlanted

| condition | mean delta | cells positive | median n |
|---|---|---|---|
| full test split | +0.0559 | 5/5 | 823 |
| plant before window | +0.0911 | 4/5 | 348 |

## enemy_aliveCount

| condition | mean delta | cells positive | median n |
|---|---|---|---|
| full test split | +0.0290 | 5/5 | 698 |
| deaths precede window | +0.0260 | 5/5 | 393 |

## teammate_aliveCount

| condition | mean delta | cells positive | median n |
|---|---|---|---|
| full test split | +0.0260 | 5/5 | 792 |
| deaths precede window | +0.0144 | 4/5 | 430 |
