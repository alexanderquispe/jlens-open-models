# Experiment 12, olmo-3-32b

## 1. Switch time

EMA rate fitted to the model's jump curves: alpha = 0.070 (half-life 9.5 numbers; RMSE 48.3).

e50 of the predicted mean (95% bootstrap interval over streams):

| arm | model | observer | EMA |
|---|---|---|---|
| up100 | 44 [24, 45] | 7 [3, 13] | 9 [7, 11] |
| down100 | 49 [29, 55] | 7 [4, 12] | 9 [7, 12] |
| up200 | 10 [7, 15] | 3 [2, 4] | 9 [8, 10] |
| down200 | 13 [11, 15] | 2 [1, 3] | 9 [8, 10] |
| up400 | 3 [3, 4] | 1 [1, 1] | 9 [9, 10] |
| down400 | 10 [8, 12] | 1 [1, 1] | 9 [9, 10] |

## 2. Mixture or slide

Mass near the old level / halfway / new level (each within DELTA/4) of the predicted distribution, mean over streams; 'single' = halfway mass of a Gaussian with the model's own mean and sd; observer = its predictive distribution.


**DELTA = 400** (up and down pooled)

| e | model old | model halfway | model new | single halfway | observer old | observer halfway | observer new |
|---|---|---|---|---|---|---|---|
| 0 | 0.66 | 0.16 | 0.01 | 0.20 | 0.48 | 0.19 | 0.14 |
| 1 | 0.50 | 0.21 | 0.08 | 0.30 | 0.11 | 0.22 | 0.43 |
| 2 | 0.35 | 0.25 | 0.17 | 0.34 | 0.02 | 0.22 | 0.54 |
| 3 | 0.28 | 0.25 | 0.23 | 0.34 | 0.01 | 0.21 | 0.57 |
| 4 | 0.21 | 0.26 | 0.30 | 0.35 | 0.01 | 0.20 | 0.59 |
| 6 | 0.16 | 0.22 | 0.39 | 0.33 | 0.01 | 0.19 | 0.62 |
| 8 | 0.12 | 0.20 | 0.45 | 0.32 | 0.01 | 0.18 | 0.63 |
| 12 | 0.09 | 0.19 | 0.52 | 0.31 | 0.01 | 0.17 | 0.66 |
| 16 | 0.08 | 0.18 | 0.56 | 0.30 | 0.01 | 0.16 | 0.67 |
| 24 | 0.03 | 0.14 | 0.68 | 0.26 | 0.01 | 0.16 | 0.68 |
| 30 | 0.04 | 0.14 | 0.67 | 0.25 | 0.01 | 0.16 | 0.67 |

**DELTA = 200** (up and down pooled)

| e | model old | model halfway | model new | single halfway | observer old | observer halfway | observer new |
|---|---|---|---|---|---|---|---|
| 0 | 0.44 | 0.26 | 0.03 | 0.25 | 0.36 | 0.23 | 0.08 |
| 1 | 0.42 | 0.24 | 0.05 | 0.25 | 0.28 | 0.22 | 0.12 |
| 2 | 0.38 | 0.25 | 0.06 | 0.25 | 0.23 | 0.21 | 0.17 |
| 3 | 0.35 | 0.25 | 0.07 | 0.25 | 0.18 | 0.21 | 0.22 |
| 4 | 0.35 | 0.26 | 0.08 | 0.27 | 0.14 | 0.22 | 0.26 |
| 6 | 0.33 | 0.26 | 0.13 | 0.27 | 0.10 | 0.23 | 0.32 |
| 8 | 0.27 | 0.26 | 0.17 | 0.27 | 0.08 | 0.24 | 0.34 |
| 12 | 0.24 | 0.24 | 0.21 | 0.27 | 0.07 | 0.24 | 0.37 |
| 16 | 0.18 | 0.27 | 0.26 | 0.28 | 0.07 | 0.24 | 0.38 |
| 24 | 0.14 | 0.26 | 0.33 | 0.29 | 0.06 | 0.24 | 0.39 |
| 30 | 0.12 | 0.25 | 0.33 | 0.27 | 0.06 | 0.24 | 0.38 |

## 3. Hysteresis on the ramps

True level at which the mean crosses 500 on the way up (mu_up) and down (mu_down); loop width W = mu_up - mu_down; 95% bootstrap intervals over streams. The observer and the EMA (rate fitted on the jumps) read the same numbers.

| ramp | reader | mu_up | mu_down | W | first-ramp lag - second-ramp lag (numbers) |
|---|---|---|---|---|---|
| updown fast | model | 587 [576, 596] | 454 [439, 471] | 132 [114, 152] | +5.2 [+2.9, +7.8] |
| updown fast | observer | 566 [550, 580] | 443 [426, 460] | 123 [95, 146] | +1.1 [-1.3, +3.4] |
| updown fast | EMA | 600 [594, 606] | 393 [385, 402] | 207 [196, 217] | -0.8 [-2.1, +0.3] |
| downup fast | model | 578 [560, 595] | 360 [342, 378] | 218 [187, 248] | +8.3 [+5.5, +11.2] |
| downup fast | observer | 553 [531, 574] | 450 [434, 466] | 104 [75, 131] | -0.3 [-3.2, +2.8] |
| downup fast | EMA | 592 [583, 600] | 392 [386, 399] | 200 [185, 214] | +1.9 [+1.1, +2.8] |
| updown slow | model | 549 [541, 557] | 491 [478, 505] | 58 [44, 73] | +19.9 [+11.3, +28.1] |
| updown slow | observer | 527 [519, 536] | 487 [471, 504] | 40 [24, 57] | +6.8 [-3.6, +16.9] |
| updown slow | EMA | 524 [519, 529] | 479 [471, 488] | 45 [36, 53] | +1.6 [-4.0, +7.1] |
| downup slow | model | 534 [518, 550] | 437 [425, 450] | 98 [79, 116] | +14.6 [+3.7, +25.2] |
| downup slow | observer | 517 [497, 531] | 479 [466, 492] | 38 [16, 59] | +2.1 [-6.3, +13.2] |
| downup slow | EMA | 521 [512, 529] | 475 [466, 484] | 46 [35, 57] | +1.9 [-4.3, +8.5] |

The test: the model's W against the readers' W (positive = the model's loop is wider).


## 4. Which layers switch first

e50 of the lens's integer mean (95% interval). Only layers whose mean moves by at least DELTA / 4 are timed.


**logit** (output e50 in the header)

| layer | up200 (out 10) | down200 (out 13) | up400 (out 3) | down400 (out 10) |
|---|---|---|---|---|
| 24 | - | - | - | 20 [12, 28] |
| 28 | 37 [19, 55] | - | 13 [10, 18] | - |
| 32 | 70 [24, 72] | - | 13 [9, 15] | 32 [25, 50] |
| 36 | - | - | - | 21 [16, 27] |
| 40 | 7 [6, 11] | 6 [5, 11] | 2 [2, 2] | 9 [6, 10] |
| 44 | - | 7 [5, 11] | - | 12 [9, 19] |
| 48 | 8 [6, 10] | 7 [6, 11] | 2 [2, 3] | 10 [6, 13] |
| 52 | 8 [6, 13] | 7 [6, 11] | 3 [2, 3] | 5 [4, 6] |
| 56 | 8 [6, 14] | 8 [6, 11] | 2 [2, 2] | 4 [3, 5] |
| 60 | 9 [7, 14] | 12 [10, 14] | 3 [2, 3] | 9 [6, 11] |
| 63 | 10 [7, 15] | 13 [11, 15] | 3 [3, 4] | 10 [8, 12] |

**jlens** (output e50 in the header)

| layer | up200 (out 10) | down200 (out 13) | up400 (out 3) | down400 (out 10) |
|---|---|---|---|---|
| 28 | - | 16 [10, 18] | - | 24 [17, 30] |
| 32 | - | 14 [11, 17] | - | 17 [14, 22] |
| 36 | 10 [7, 14] | 10 [8, 13] | 2 [2, 2] | 12 [9, 16] |
| 40 | 10 [8, 13] | 7 [5, 11] | 2 [2, 3] | 8 [5, 10] |
| 44 | 10 [8, 13] | 9 [6, 13] | 2 [2, 2] | 10 [8, 15] |
| 48 | 9 [7, 12] | 9 [7, 12] | 2 [2, 2] | 11 [8, 15] |
| 52 | 9 [7, 14] | 7 [6, 11] | 2 [2, 3] | 5 [4, 9] |
| 56 | 10 [8, 15] | 10 [6, 12] | 2 [2, 3] | 6 [5, 10] |
| 60 | 9 [7, 14] | 13 [11, 15] | 3 [2, 3] | 12 [10, 15] |
| 63 | 10 [7, 15] | 13 [11, 15] | 3 [3, 4] | 10 [8, 12] |

**jpp** (output e50 in the header)

| layer | up200 (out 10) | down200 (out 13) | up400 (out 3) | down400 (out 10) |
|---|---|---|---|---|
| 20 | - | 66 [0, 82] | - | 59 [0, 81] |
| 24 | - | 1 [0, 7] | - | 29 [17, 57] |
| 36 | - | - | - | 32 [21, 43] |
| 40 | - | - | 2 [2, 2] | 15 [10, 20] |
| 44 | - | - | - | 32 [21, 40] |
| 48 | 6 [5, 8] | 11 [5, 15] | 2 [2, 2] | 16 [14, 21] |
| 52 | 9 [7, 14] | 6 [5, 10] | 3 [2, 3] | 4 [4, 6] |
| 56 | 8 [6, 13] | 12 [8, 14] | 2 [2, 2] | 10 [6, 11] |
| 60 | 10 [7, 14] | 12 [11, 15] | 3 [3, 3] | 10 [7, 12] |
| 63 | 10 [7, 15] | 13 [11, 15] | 3 [3, 4] | 10 [8, 12] |

Discrete-time hazard of the switch (J-lens if present, else logit lens; all timed layers; bins of e [0, 1, 2, 3, 4, 8, 16, 32]): coefficients on the log-odds of switching at e given not yet, 95% bootstrap interval over streams. Baseline: early third, DELTA = 200, up.

| term | coefficient | 95% interval |
|---|---|---|
| e[0,1) | -5.76 | [-21.44, -4.76] |
| e[4,8) | +0.45 | [+0.07, +2.43] |
| e[32,300) | +0.84 | [+0.24, +3.76] |
| middle third | -2.41 | [-4.35, -2.00] |
| late third | -2.19 | [-4.13, -1.79] |
| log2(DELTA/200) | +0.92 | [+0.70, +1.16] |
| down | -1.05 | [-1.48, -0.73] |

## 5. Magnitude words after the jump

Excess of log P(magnitude) - log P(control) over its pre-jump level: peak (e 0..20) and half-life (first e after the peak where the 3-number running mean is below half the peak), J-lens by layer third; the output sd the same way.

| arm | output sd peak, half-life | early peak, half-life | middle peak, half-life | late peak, half-life |
|---|---|---|---|---|
| up100 | 6.1, 25 | +0.21, 4 | +0.16, 7 | +0.08, 1 |
| down100 | 7.3, 2 | +0.00, 1 | +0.03, 1 | -0.00, 1 |
| up200 | 39.3, 17 | +0.41, 28 | +0.54, 42 | +0.43, 30 |
| down200 | 58.2, 12 | +0.10, 3 | +0.25, 4 | +0.24, 3 |
| up400 | 115.8, 11 | +0.77, 16 | +0.92, 13 | +0.95, 51 |
| down400 | 144.1, 15 | +0.39, 4 | +1.13, 13 | +0.84, 5 |
