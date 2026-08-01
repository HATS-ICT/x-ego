# Subset analysis

Delta is CECL minus baseline, seeds averaged within each arm before
differencing. `n` is the number of test rows in the condition.

## enemy_location_10s

| condition | mean delta | cells positive | median n | min to max |
|---|---|---|---|---|
| full test split | +0.0394 | 5/5 | 735 | +0.0090 to +0.0652 |
| target changed region | +0.0425 | 4/5 | 117 | -0.0020 to +0.0768 |


Per cell, full test split:

| map | encoder | baseline | CECL | delta | n |
|---|---|---|---|---|---|
| dust2 | siglip2 | 0.0392 | 0.0760 | +0.0368 | 802 |
| inferno | siglip2 | 0.0854 | 0.1071 | +0.0216 | 874 |
| mirage | dinov3 | 0.0643 | 0.0733 | +0.0090 | 735 |
| mirage | siglip2 | 0.0336 | 0.0979 | +0.0643 | 735 |
| mirage | vjepa2 | 0.0176 | 0.0828 | +0.0652 | 735 |

Summed n across cells: 3881


Per cell, target changed region:

| map | encoder | baseline | CECL | delta | n |
|---|---|---|---|---|---|
| dust2 | siglip2 | 0.0081 | 0.0491 | +0.0410 | 168 |
| inferno | siglip2 | 0.1312 | 0.1529 | +0.0217 | 160 |
| mirage | dinov3 | 0.0698 | 0.0678 | -0.0020 | 117 |
| mirage | siglip2 | 0.0359 | 0.1111 | +0.0752 | 117 |
| mirage | vjepa2 | 0.0158 | 0.0926 | +0.0768 | 117 |

Summed n across cells: 679

Delta by how much the present answer overlaps the future one. Lower jaccard means less can be carried forward.

| jaccard | mean delta | cells positive | median n | min to max |
|---|---|---|---|---|
| -0.01-0.00 | +0.0425 | 4/5 | 117 | -0.0020 to +0.0768 |
| 0.00-0.20 | +0.0403 | 5/5 | 194 | +0.0044 to +0.0809 |
| 0.20-0.40 | +0.0314 | 5/5 | 215 | +0.0080 to +0.0466 |
| 0.40-0.60 | +0.0328 | 5/5 | 133 | +0.0137 to +0.0544 |
| 0.60-0.80 | +0.0418 | 5/5 | 49 | +0.0230 to +0.0660 |
| 0.80-1.01 | +0.0190 | 5/5 | 27 | +0.0062 to +0.0295 |

## enemy_location_5s

| condition | mean delta | cells positive | median n | min to max |
|---|---|---|---|---|
| full test split | +0.0638 | 5/5 | 748 | +0.0280 to +0.1157 |
| target changed region | +0.0490 | 5/5 | 45 | +0.0138 to +0.0879 |


Per cell, full test split:

| map | encoder | baseline | CECL | delta | n |
|---|---|---|---|---|---|
| dust2 | siglip2 | 0.0560 | 0.1112 | +0.0552 | 819 |
| inferno | siglip2 | 0.1097 | 0.1378 | +0.0280 | 880 |
| mirage | dinov3 | 0.1329 | 0.1610 | +0.0281 | 748 |
| mirage | siglip2 | 0.0805 | 0.1724 | +0.0919 | 748 |
| mirage | vjepa2 | 0.0502 | 0.1659 | +0.1157 | 748 |

Summed n across cells: 3943


Per cell, target changed region:

| map | encoder | baseline | CECL | delta | n |
|---|---|---|---|---|---|
| dust2 | siglip2 | 0.0408 | 0.0703 | +0.0295 | 75 |
| inferno | siglip2 | 0.1644 | 0.1782 | +0.0138 | 90 |
| mirage | dinov3 | 0.1655 | 0.1933 | +0.0278 | 45 |
| mirage | siglip2 | 0.1193 | 0.2072 | +0.0879 | 45 |
| mirage | vjepa2 | 0.0813 | 0.1671 | +0.0858 | 45 |

Summed n across cells: 300

Delta by how much the present answer overlaps the future one. Lower jaccard means less can be carried forward.

| jaccard | mean delta | cells positive | median n | min to max |
|---|---|---|---|---|
| -0.01-0.00 | +0.0490 | 5/5 | 45 | +0.0138 to +0.0879 |
| 0.00-0.20 | +0.0605 | 5/5 | 117 | +0.0214 to +0.1296 |
| 0.20-0.40 | +0.0595 | 5/5 | 209 | +0.0227 to +0.1147 |
| 0.40-0.60 | +0.0624 | 5/5 | 177 | +0.0320 to +0.1110 |
| 0.60-0.80 | +0.0626 | 5/5 | 108 | +0.0264 to +0.1225 |
| 0.80-1.01 | +0.0546 | 5/5 | 92 | +0.0213 to +0.0816 |

## teammate_location_10s

| condition | mean delta | cells positive | median n | min to max |
|---|---|---|---|---|
| full test split | +0.0329 | 5/5 | 740 | +0.0045 to +0.0651 |
| target changed region | +0.0230 | 4/5 | 146 | -0.0288 to +0.0672 |


Per cell, full test split:

| map | encoder | baseline | CECL | delta | n |
|---|---|---|---|---|---|
| dust2 | siglip2 | 0.0172 | 0.0347 | +0.0175 | 807 |
| inferno | siglip2 | 0.0716 | 0.0916 | +0.0199 | 895 |
| mirage | dinov3 | 0.0642 | 0.0687 | +0.0045 | 740 |
| mirage | siglip2 | 0.0161 | 0.0738 | +0.0576 | 740 |
| mirage | vjepa2 | 0.0025 | 0.0676 | +0.0651 | 740 |

Summed n across cells: 3922


Per cell, target changed region:

| map | encoder | baseline | CECL | delta | n |
|---|---|---|---|---|---|
| dust2 | siglip2 | 0.0118 | 0.0278 | +0.0160 | 183 |
| inferno | siglip2 | 0.0915 | 0.1132 | +0.0216 | 189 |
| mirage | dinov3 | 0.0681 | 0.0393 | -0.0288 | 146 |
| mirage | siglip2 | 0.0168 | 0.0560 | +0.0392 | 146 |
| mirage | vjepa2 | 0.0000 | 0.0672 | +0.0672 | 146 |

Summed n across cells: 810

Delta by how much the present answer overlaps the future one. Lower jaccard means less can be carried forward.

| jaccard | mean delta | cells positive | median n | min to max |
|---|---|---|---|---|
| -0.01-0.00 | +0.0230 | 4/5 | 146 | -0.0288 to +0.0672 |
| 0.00-0.20 | +0.0299 | 5/5 | 182 | +0.0007 to +0.0721 |
| 0.20-0.40 | +0.0328 | 5/5 | 219 | +0.0117 to +0.0619 |
| 0.40-0.60 | +0.0333 | 5/5 | 103 | +0.0036 to +0.0678 |
| 0.60-0.80 | +0.0344 | 5/5 | 50 | +0.0067 to +0.0674 |
| 0.80-1.01 | +0.0129 | 3/5 | 40 | -0.0377 to +0.0491 |

## global_bombPlanted

| condition | mean delta | cells positive | median n | min to max |
|---|---|---|---|---|
| full test split | +0.0559 | 5/5 | 823 | +0.0237 to +0.1081 |
| plant before window | +0.0911 | 4/5 | 348 | -0.0027 to +0.1480 |
| plant not before window | +0.0278 | 4/5 | 475 | -0.0484 to +0.0789 |
| plant inside window | +0.0421 | 4/5 | 75 | -0.0267 to +0.1067 |
| plant before window or never planted | +0.0562 | 5/5 | 753 | +0.0226 to +0.1082 |
| plant before window or never planted, no in-window plant | +0.0570 | 5/5 | 748 | +0.0247 to +0.1083 |


Per cell, full test split:

| map | encoder | baseline | CECL | delta | n |
|---|---|---|---|---|---|
| dust2 | siglip2 | 0.6281 | 0.6593 | +0.0313 | 1023 |
| inferno | siglip2 | 0.6316 | 0.6621 | +0.0306 | 916 |
| mirage | dinov3 | 0.6962 | 0.7199 | +0.0237 | 823 |
| mirage | siglip2 | 0.6446 | 0.7527 | +0.1081 | 823 |
| mirage | vjepa2 | 0.6166 | 0.7023 | +0.0857 | 823 |

Summed n across cells: 4408


Per cell, plant before window:

| map | encoder | baseline | CECL | delta | n |
|---|---|---|---|---|---|
| dust2 | siglip2 | 0.5383 | 0.5987 | +0.0603 | 522 |
| inferno | siglip2 | 0.6101 | 0.6073 | -0.0027 | 368 |
| mirage | dinov3 | 0.5733 | 0.6954 | +0.1221 | 348 |
| mirage | siglip2 | 0.5632 | 0.7112 | +0.1480 | 348 |
| mirage | vjepa2 | 0.5086 | 0.6365 | +0.1279 | 348 |

Summed n across cells: 1934


Per cell, plant not before window:

| map | encoder | baseline | CECL | delta | n |
|---|---|---|---|---|---|
| dust2 | siglip2 | 0.7216 | 0.7226 | +0.0010 | 501 |
| inferno | siglip2 | 0.6460 | 0.6989 | +0.0529 | 548 |
| mirage | dinov3 | 0.7863 | 0.7379 | -0.0484 | 475 |
| mirage | siglip2 | 0.7042 | 0.7832 | +0.0789 | 475 |
| mirage | vjepa2 | 0.6958 | 0.7505 | +0.0547 | 475 |

Summed n across cells: 2474


Per cell, plant inside window:

| map | encoder | baseline | CECL | delta | n |
|---|---|---|---|---|---|
| dust2 | siglip2 | 0.6067 | 0.5800 | -0.0267 | 75 |
| inferno | siglip2 | 0.6868 | 0.7308 | +0.0440 | 91 |
| mirage | dinov3 | 0.7733 | 0.7867 | +0.0133 | 75 |
| mirage | siglip2 | 0.7067 | 0.8133 | +0.1067 | 75 |
| mirage | vjepa2 | 0.6400 | 0.7133 | +0.0733 | 75 |

Summed n across cells: 391


Per cell, plant before window or never planted:

| map | encoder | baseline | CECL | delta | n |
|---|---|---|---|---|---|
| dust2 | siglip2 | 0.6259 | 0.6608 | +0.0349 | 961 |
| inferno | siglip2 | 0.6253 | 0.6542 | +0.0289 | 830 |
| mirage | dinov3 | 0.6892 | 0.7118 | +0.0226 | 753 |
| mirage | siglip2 | 0.6375 | 0.7457 | +0.1082 | 753 |
| mirage | vjepa2 | 0.6155 | 0.7019 | +0.0863 | 753 |

Summed n across cells: 4050


Per cell, plant before window or never planted, no in-window plant:

| map | encoder | baseline | CECL | delta | n |
|---|---|---|---|---|---|
| dust2 | siglip2 | 0.6297 | 0.6656 | +0.0359 | 948 |
| inferno | siglip2 | 0.6255 | 0.6545 | +0.0291 | 825 |
| mirage | dinov3 | 0.6885 | 0.7132 | +0.0247 | 748 |
| mirage | siglip2 | 0.6384 | 0.7467 | +0.1083 | 748 |
| mirage | vjepa2 | 0.6143 | 0.7012 | +0.0869 | 748 |

Summed n across cells: 4017

## enemy_aliveCount

| condition | mean delta | cells positive | median n | min to max |
|---|---|---|---|---|
| full test split | +0.0290 | 5/5 | 698 | +0.0191 to +0.0381 |
| deaths precede window | +0.0260 | 5/5 | 393 | +0.0063 to +0.0356 |
| deaths precede window (classes present only) | +0.0313 | 5/5 | 393 | +0.0075 to +0.0427 |


Per cell, full test split:

| map | encoder | baseline | CECL | delta | n |
|---|---|---|---|---|---|
| dust2 | siglip2 | 0.3669 | 0.4037 | +0.0367 | 767 |
| inferno | siglip2 | 0.3253 | 0.3459 | +0.0206 | 810 |
| mirage | dinov3 | 0.4118 | 0.4309 | +0.0191 | 698 |
| mirage | siglip2 | 0.3658 | 0.4039 | +0.0381 | 698 |
| mirage | vjepa2 | 0.3941 | 0.4243 | +0.0302 | 698 |

Summed n across cells: 3671


Per cell, deaths precede window:

| map | encoder | baseline | CECL | delta | n |
|---|---|---|---|---|---|
| dust2 | siglip2 | 0.2946 | 0.3302 | +0.0356 | 436 |
| inferno | siglip2 | 0.3054 | 0.3403 | +0.0350 | 514 |
| mirage | dinov3 | 0.3376 | 0.3439 | +0.0063 | 393 |
| mirage | siglip2 | 0.2925 | 0.3187 | +0.0262 | 393 |
| mirage | vjepa2 | 0.3228 | 0.3500 | +0.0272 | 393 |

Summed n across cells: 2129

## teammate_aliveCount

| condition | mean delta | cells positive | median n | min to max |
|---|---|---|---|---|
| full test split | +0.0260 | 5/5 | 792 | +0.0104 to +0.0404 |
| deaths precede window | +0.0144 | 4/5 | 430 | -0.0157 to +0.0485 |
| deaths precede window (classes present only) | +0.0180 | 4/5 | 430 | -0.0196 to +0.0606 |


Per cell, full test split:

| map | encoder | baseline | CECL | delta | n |
|---|---|---|---|---|---|
| dust2 | siglip2 | 0.2866 | 0.3270 | +0.0404 | 868 |
| inferno | siglip2 | 0.2941 | 0.3058 | +0.0118 | 879 |
| mirage | dinov3 | 0.3241 | 0.3346 | +0.0104 | 792 |
| mirage | siglip2 | 0.2988 | 0.3331 | +0.0343 | 792 |
| mirage | vjepa2 | 0.2700 | 0.3032 | +0.0332 | 792 |

Summed n across cells: 4123


Per cell, deaths precede window:

| map | encoder | baseline | CECL | delta | n |
|---|---|---|---|---|---|
| dust2 | siglip2 | 0.1594 | 0.2079 | +0.0485 | 499 |
| inferno | siglip2 | 0.1897 | 0.2024 | +0.0126 | 565 |
| mirage | dinov3 | 0.2276 | 0.2119 | -0.0157 | 430 |
| mirage | siglip2 | 0.2020 | 0.2064 | +0.0044 | 430 |
| mirage | vjepa2 | 0.1876 | 0.2098 | +0.0222 | 430 |

Summed n across cells: 2354

## enemy_location_10s, relay conditions

C1 the POV agent sees an enemy. C2 it does not, but a teammate does and the agent
can see that teammate. C3 the same without visual contact. C4 nobody on the team
sees any enemy. C2 versus C3 is the discriminator.

| condition | mean delta | cells positive | median n | min to max |
|---|---|---|---|---|
| C1 | +0.0151 | 5/5 | 14 | +0.0005 to +0.0284 |
| C2 | no data | 0/0 | 0 | - |
| C3 | +0.0342 | 4/5 | 24 | -0.0112 to +0.1188 |
| C4 | +0.0396 | 5/5 | 695 | +0.0087 to +0.0649 |

Source: dust2-enemy_location_10s-relay-mask-los.csv, inferno-enemy_location_10s-relay-mask-los.csv, mirage-enemy_location_10s-relay-mask-los.csv

## enemy_location_5s, relay conditions

C1 the POV agent sees an enemy. C2 it does not, but a teammate does and the agent
can see that teammate. C3 the same without visual contact. C4 nobody on the team
sees any enemy. C2 versus C3 is the discriminator.

| condition | mean delta | cells positive | median n | min to max |
|---|---|---|---|---|
| C1 | +0.0303 | 3/4 | 11 | -0.0211 to +0.0725 |
| C2 | no data | 0/0 | 0 | - |
| C3 | +0.0345 | 5/5 | 22 | +0.0133 to +0.0814 |
| C4 | +0.0646 | 5/5 | 713 | +0.0277 to +0.1182 |

Source: dust2-enemy_location_5s-relay-mask-los.csv, inferno-enemy_location_5s-relay-mask-los.csv, mirage-enemy_location_5s-relay-mask-los.csv
