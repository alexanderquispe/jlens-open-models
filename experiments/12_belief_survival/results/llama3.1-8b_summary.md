# Experiment 12, llama3.1-8b

## 1. Switch time

EMA rate fitted to the model's jump curves: alpha = 0.109 (half-life 6.0 numbers; RMSE 30.5).

e50 of the predicted mean (95% bootstrap interval over streams):

| arm | model | observer | EMA |
|---|---|---|---|
| up100 | 37 [26, 45] | 7 [3, 13] | 5 [4, 7] |
| down100 | 29 [16, 34] | 7 [4, 12] | 6 [4, 8] |
| up200 | 10 [8, 13] | 3 [2, 4] | 5 [5, 6] |
| down200 | 7 [5, 9] | 2 [1, 3] | 6 [5, 7] |
| up400 | 4 [3, 4] | 1 [1, 1] | 6 [5, 6] |
| down400 | 4 [4, 5] | 1 [1, 1] | 6 [5, 6] |

## 2. Mixture or slide

Mass near the old level / halfway / new level (each within DELTA/4) of the predicted distribution, mean over streams; 'single' = halfway mass of a Gaussian with the model's own mean and sd; observer = its predictive distribution.


**DELTA = 400** (up and down pooled)

| e | model old | model halfway | model new | single halfway | observer old | observer halfway | observer new |
|---|---|---|---|---|---|---|---|
| 0 | 0.56 | 0.23 | 0.02 | 0.24 | 0.48 | 0.19 | 0.14 |
| 1 | 0.43 | 0.26 | 0.10 | 0.31 | 0.11 | 0.22 | 0.43 |
| 2 | 0.32 | 0.27 | 0.19 | 0.34 | 0.02 | 0.22 | 0.54 |
| 3 | 0.24 | 0.27 | 0.26 | 0.35 | 0.01 | 0.21 | 0.57 |
| 4 | 0.19 | 0.25 | 0.33 | 0.35 | 0.01 | 0.20 | 0.59 |
| 6 | 0.13 | 0.21 | 0.47 | 0.34 | 0.01 | 0.19 | 0.62 |
| 8 | 0.09 | 0.22 | 0.48 | 0.33 | 0.01 | 0.18 | 0.63 |
| 12 | 0.06 | 0.20 | 0.54 | 0.30 | 0.01 | 0.17 | 0.66 |
| 16 | 0.04 | 0.21 | 0.57 | 0.29 | 0.01 | 0.16 | 0.67 |
| 24 | 0.02 | 0.18 | 0.62 | 0.24 | 0.01 | 0.16 | 0.68 |
| 30 | 0.02 | 0.19 | 0.61 | 0.25 | 0.01 | 0.16 | 0.67 |

**DELTA = 200** (up and down pooled)

| e | model old | model halfway | model new | single halfway | observer old | observer halfway | observer new |
|---|---|---|---|---|---|---|---|
| 0 | 0.34 | 0.27 | 0.09 | 0.26 | 0.36 | 0.23 | 0.08 |
| 1 | 0.32 | 0.26 | 0.09 | 0.26 | 0.28 | 0.22 | 0.12 |
| 2 | 0.30 | 0.26 | 0.11 | 0.26 | 0.23 | 0.21 | 0.17 |
| 3 | 0.29 | 0.25 | 0.12 | 0.26 | 0.18 | 0.21 | 0.22 |
| 4 | 0.27 | 0.26 | 0.15 | 0.26 | 0.14 | 0.22 | 0.26 |
| 6 | 0.23 | 0.25 | 0.21 | 0.27 | 0.10 | 0.23 | 0.32 |
| 8 | 0.20 | 0.25 | 0.23 | 0.27 | 0.08 | 0.24 | 0.34 |
| 12 | 0.16 | 0.24 | 0.28 | 0.26 | 0.07 | 0.24 | 0.37 |
| 16 | 0.13 | 0.24 | 0.29 | 0.25 | 0.07 | 0.24 | 0.38 |
| 24 | 0.11 | 0.24 | 0.33 | 0.26 | 0.06 | 0.24 | 0.39 |
| 30 | 0.10 | 0.23 | 0.32 | 0.25 | 0.06 | 0.24 | 0.38 |

## 3. Hysteresis on the ramps

True level at which the mean crosses 500 on the way up (mu_up) and down (mu_down); loop width W = mu_up - mu_down; 95% bootstrap intervals over streams. The observer and the EMA (rate fitted on the jumps) read the same numbers.

| ramp | reader | mu_up | mu_down | W | first-ramp lag - second-ramp lag (numbers) |
|---|---|---|---|---|---|
| updown fast | model | 574 [564, 583] | 475 [454, 497] | 99 [72, 127] | +6.0 [+3.9, +8.1] |
| updown fast | observer | 566 [550, 580] | 443 [426, 460] | 123 [95, 146] | +1.1 [-1.3, +3.4] |
| updown fast | EMA | 567 [559, 574] | 427 [418, 435] | 140 [128, 152] | -0.8 [-2.2, +0.3] |
| downup fast | model | 543 [528, 558] | 432 [419, 445] | 111 [89, 132] | +3.1 [+1.2, +5.2] |
| downup fast | observer | 553 [531, 574] | 450 [434, 466] | 104 [75, 131] | -0.3 [-3.2, +2.8] |
| downup fast | EMA | 563 [552, 574] | 437 [426, 448] | 126 [108, 143] | -0.1 [-1.4, +1.4] |
| updown slow | model | 519 [513, 524] | 499 [488, 512] | 20 [9, 29] | +9.2 [+0.6, +17.2] |
| updown slow | observer | 527 [519, 536] | 487 [471, 504] | 40 [24, 57] | +6.8 [-3.6, +16.9] |
| updown slow | EMA | 512 [504, 520] | 493 [484, 502] | 19 [10, 28] | +2.3 [-5.3, +9.7] |
| downup slow | model | 509 [499, 520] | 489 [478, 500] | 21 [5, 34] | +1.1 [-7.9, +9.6] |
| downup slow | observer | 517 [497, 531] | 479 [466, 492] | 38 [16, 59] | +2.1 [-6.3, +13.2] |
| downup slow | EMA | 503 [494, 513] | 498 [488, 508] | 5 [-6, 16] | -0.9 [-8.3, +6.2] |

The test: the model's W against the readers' W (positive = the model's loop is wider).


## 4. Which layers switch first

e50 of the lens's integer mean (95% interval). Only layers whose mean moves by at least DELTA / 4 are timed.


**logit** (output e50 in the header)

| layer | up200 (out 10) | down200 (out 7) | up400 (out 4) | down400 (out 4) |
|---|---|---|---|---|
| 18 | 8 [2, 11] | - | 2 [1, 3] | 4 [4, 6] |
| 20 | 5 [3, 7] | 4 [3, 5] | 2 [2, 3] | 2 [2, 3] |
| 22 | 9 [6, 11] | 4 [3, 5] | 2 [2, 3] | 2 [2, 3] |
| 24 | 9 [7, 11] | 5 [4, 6] | 2 [2, 3] | 2 [2, 3] |
| 26 | 7 [5, 10] | 5 [4, 6] | 2 [2, 2] | 3 [2, 3] |
| 28 | 10 [7, 13] | 6 [5, 9] | 3 [3, 3] | 4 [3, 4] |
| 30 | 5 [4, 7] | 10 [6, 12] | 2 [2, 3] | 5 [4, 6] |
| 31 | 10 [8, 13] | 7 [5, 10] | 4 [3, 4] | 4 [4, 5] |

**jlens** (output e50 in the header)

| layer | up200 (out 10) | down200 (out 7) | up400 (out 4) | down400 (out 4) |
|---|---|---|---|---|
| 20 | 22 [5, 25] | 1 [1, 2] | 3 [3, 3] | 1 [0, 1] |
| 22 | 18 [7, 22] | 2 [1, 3] | 3 [2, 4] | 1 [1, 1] |
| 24 | 10 [6, 13] | 4 [3, 6] | 2 [2, 3] | 2 [1, 2] |
| 26 | 7 [5, 9] | 5 [4, 7] | 2 [2, 2] | 3 [2, 3] |
| 28 | 11 [7, 14] | 6 [5, 8] | 4 [3, 4] | 3 [3, 4] |
| 30 | 8 [6, 10] | 9 [6, 12] | 3 [2, 3] | 5 [5, 6] |
| 31 | 10 [8, 13] | 7 [5, 10] | 4 [3, 4] | 4 [4, 5] |

Discrete-time hazard of the switch (J-lens if present, else logit lens; all timed layers; bins of e [0, 1, 2, 3, 4, 8, 16, 32]): coefficients on the log-odds of switching at e given not yet, 95% bootstrap interval over streams. Baseline: early third, DELTA = 200, up.

| term | coefficient | 95% interval |
|---|---|---|
| e[0,1) | -2.74 | [-3.53, -1.97] |
| e[4,8) | -0.62 | [-1.16, +0.22] |
| e[32,300) | -1.22 | [-1.74, +0.27] |
| middle third | -1.39 | [-2.08, -0.90] |
| late third | -1.75 | [-2.57, -1.27] |
| log2(DELTA/200) | +1.60 | [+1.45, +1.78] |
| down | +0.88 | [+0.39, +1.39] |

## 5. Magnitude words after the jump

Excess of log P(magnitude) - log P(control) over its pre-jump level: peak (e 0..20) and half-life (first e after the peak where the 3-number running mean is below half the peak), J-lens by layer third; the output sd the same way.

| arm | output sd peak, half-life | early peak, half-life | middle peak, half-life | late peak, half-life |
|---|---|---|---|---|
| up100 | 7.3, 46 | +0.20, 7 | +0.39, 135 | +0.13, 143 |
| down100 | 13.0, 49 | +0.01, 1 | +0.05, 1 | +0.04, 1 |
| up200 | 32.0, 15 | +0.39, 27 | +1.06, 44 | +0.48, 49 |
| down200 | 43.8, 12 | +0.06, 2 | +0.31, 2 | +0.10, 3 |
| up400 | 110.3, 12 | +0.64, 34 | +1.71, 34 | +1.17, 65 |
| down400 | 125.2, 9 | +0.20, 5 | +0.92, 12 | +0.42, 7 |

## 6. History length: does the old belief die more slowly the longer it held?

Switch time = first e >= 0 where the belief (predicted mean) crosses the midpoint 500 (3-number running mean; up and down pooled; 95% bootstrap interval over streams; '>199' = never within the 200 numbers). EMA rate fitted to the model on these runs; running mean = cumulative mean.

EMA alpha = 0.077 (half-life 8.7 numbers).


**DELTA = 200**

| tau | model | observer | EMA | running mean | model pre-jump sd | observer pre-jump sd |
|---|---|---|---|---|---|---|
| 25 | 6 [5, 11] | 3 [2, 3] | 8 [8, 8] | 25 [25, 25] | 245.9 | 112.6 |
| 50 | 12 [8, 16] | 3 [2, 3] | 8 [8, 8] | 50 [50, 50] | 188.8 | 109.9 |
| 100 | 12 [11, 14] | 2 [2, 3] | 8 [8, 8] | 100 [100, 100] | 132.6 | 107.1 |
| 200 | 10 [9, 10] | 2 [2, 3] | 8 [8, 8] | >199 | 116.9 | 107.5 |
| 400 | 9 [9, 10] | 2 [2, 3] | 8 [8, 8] | >199 | 106.6 | 106.5 |
| 800 | 9 [9, 10] | 2 [2, 3] | 8 [8, 8] | >199 | 105.5 | 105.7 |

Slope of log switch time on log tau (taus where it is finite; 95% bootstrap interval):

| reader | slope | interval | taus used |
|---|---|---|---|
| model | +0.04 | [-0.09, +0.13] | 6 |
| observer | -0.13 | [-0.15, +0.00] | 6 |
| EMA | +0.00 | [+0.00, +0.00] | 6 |
| running mean | +1.00 | [+1.00, +1.00] | 3 |

**DELTA = 400**

| tau | model | observer | EMA | running mean | model pre-jump sd | observer pre-jump sd |
|---|---|---|---|---|---|---|
| 25 | 5 [3, 7] | 1 [1, 1] | 8 [8, 8] | 25 [25, 25] | 242.3 | 114.5 |
| 50 | 6 [5, 8] | 1 [1, 1] | 8 [8, 8] | 50 [50, 50] | 179.2 | 111.8 |
| 100 | 5 [4, 6] | 1 [1, 1] | 8 [8, 8] | 100 [100, 100] | 143.3 | 109.2 |
| 200 | 5 [4, 5] | 1 [1, 1] | 8 [8, 8] | >199 | 124.0 | 109.5 |
| 400 | 4 [4, 5] | 1 [1, 1] | 8 [8, 8] | >199 | 112.2 | 108.7 |
| 800 | 4 [4, 4] | 1 [1, 1] | 8 [8, 8] | >199 | 110.8 | 108.0 |

Slope of log switch time on log tau (taus where it is finite; 95% bootstrap interval):

| reader | slope | interval | taus used |
|---|---|---|---|
| model | -0.10 | [-0.19, +0.03] | 6 |
| observer | +0.00 | [+0.00, +0.00] | 6 |
| EMA | +0.00 | [+0.00, +0.00] | 6 |
| running mean | +1.00 | [+1.00, +1.00] | 3 |

**logit, DELTA = 200**: switch time of the lens's integer mean by layer (crossing 500; only layers whose mean moves by at least DELTA / 4 at tau = 400)

| layer | tau 25 | tau 50 | tau 100 | tau 200 | tau 400 | tau 800 |
|---|---|---|---|---|---|---|
| 18 | 48 [2, 67] | 11 [2, 24] | 10 [5, 15] | 10 [8, 13] | 10 [9, 12] | 10 [9, 11] |
| 20 | 27 [5, 35] | 7 [4, 13] | 6 [5, 9] | 6 [5, 8] | 6 [5, 7] | 6 [5, 7] |
| 22 | 29 [5, 54] | 6 [3, 22] | 5 [3, 8] | 4 [3, 5] | 4 [3, 5] | 5 [4, 5] |
| 24 | 6 [4, 12] | 6 [4, 11] | 6 [5, 8] | 6 [5, 6] | 6 [5, 7] | 6 [6, 8] |
| 26 | 6 [2, 16] | 6 [4, 9] | 6 [5, 8] | 5 [5, 6] | 6 [5, 7] | 6 [6, 8] |
| 28 | 5 [4, 13] | 8 [5, 11] | 9 [7, 11] | 8 [7, 9] | 8 [7, 10] | 9 [8, 10] |
| 30 | 11 [3, 22] | 11 [7, 19] | 9 [7, 11] | 7 [6, 9] | 8 [7, 10] | 8 [7, 10] |
| 31 | 6 [5, 11] | 12 [9, 16] | 12 [10, 14] | 10 [9, 11] | 9 [9, 10] | 9 [9, 10] |

**logit, DELTA = 400**: switch time of the lens's integer mean by layer (crossing 500; only layers whose mean moves by at least DELTA / 4 at tau = 400)

| layer | tau 25 | tau 50 | tau 100 | tau 200 | tau 400 | tau 800 |
|---|---|---|---|---|---|---|
| 18 | 53 [34, 68] | 47 [28, 66] | 29 [13, 40] | 10 [8, 14] | 7 [5, 9] | 4 [3, 6] |
| 20 | 30 [14, 36] | 11 [6, 16] | 6 [4, 9] | 4 [3, 4] | 3 [2, 3] | 3 [2, 3] |
| 22 | 29 [14, 41] | 12 [9, 16] | 5 [3, 8] | 3 [2, 3] | 2 [2, 2] | 2 [2, 2] |
| 24 | 9 [4, 16] | 5 [3, 8] | 3 [3, 4] | 3 [2, 3] | 2 [2, 3] | 2 [2, 3] |
| 26 | 4 [3, 9] | 3 [3, 4] | 3 [2, 3] | 2 [2, 2] | 2 [2, 2] | 2 [2, 3] |
| 28 | 4 [3, 7] | 4 [3, 5] | 4 [3, 5] | 3 [3, 4] | 3 [3, 4] | 3 [3, 4] |
| 30 | 7 [4, 15] | 5 [4, 6] | 4 [3, 4] | 3 [3, 4] | 3 [3, 4] | 3 [3, 4] |
| 31 | 5 [3, 7] | 6 [5, 8] | 5 [4, 6] | 5 [4, 5] | 4 [4, 5] | 4 [4, 4] |

**jlens, DELTA = 200**: switch time of the lens's integer mean by layer (crossing 500; only layers whose mean moves by at least DELTA / 4 at tau = 400)

| layer | tau 25 | tau 50 | tau 100 | tau 200 | tau 400 | tau 800 |
|---|---|---|---|---|---|---|
| 20 | 60 [1, 79] | 24 [1, 42] | 14 [5, 25] | 6 [2, 22] | 3 [2, 4] | 3 [2, 4] |
| 22 | 26 [3, 60] | 11 [2, 25] | 6 [3, 11] | 4 [2, 6] | 3 [2, 4] | 4 [3, 5] |
| 24 | 16 [3, 28] | 7 [4, 18] | 7 [6, 9] | 5 [5, 7] | 5 [5, 6] | 6 [5, 7] |
| 26 | 4 [2, 11] | 7 [4, 8] | 7 [6, 8] | 6 [5, 6] | 6 [6, 7] | 7 [6, 8] |
| 28 | 6 [1, 19] | 9 [5, 16] | 9 [8, 11] | 8 [7, 9] | 8 [7, 9] | 8 [7, 9] |
| 30 | 10 [5, 15] | 15 [9, 22] | 12 [10, 15] | 9 [8, 11] | 9 [8, 11] | 9 [8, 10] |
| 31 | 6 [5, 10] | 12 [8, 16] | 12 [10, 14] | 10 [9, 11] | 9 [9, 10] | 9 [9, 10] |

**jlens, DELTA = 400**: switch time of the lens's integer mean by layer (crossing 500; only layers whose mean moves by at least DELTA / 4 at tau = 400)

| layer | tau 25 | tau 50 | tau 100 | tau 200 | tau 400 | tau 800 |
|---|---|---|---|---|---|---|
| 20 | 46 [36, 60] | 31 [13, 44] | 11 [6, 23] | 4 [3, 5] | 2 [2, 2] | 2 [2, 2] |
| 22 | 35 [16, 44] | 14 [8, 27] | 6 [4, 9] | 2 [2, 3] | 1 [1, 2] | 2 [1, 2] |
| 24 | 21 [6, 33] | 8 [5, 13] | 4 [3, 5] | 2 [2, 3] | 2 [2, 2] | 2 [2, 2] |
| 26 | 6 [3, 11] | 3 [3, 5] | 3 [3, 3] | 2 [2, 2] | 2 [2, 2] | 2 [2, 3] |
| 28 | 8 [5, 14] | 6 [4, 11] | 5 [4, 6] | 4 [3, 4] | 4 [3, 4] | 3 [3, 4] |
| 30 | 7 [5, 12] | 7 [6, 11] | 6 [5, 7] | 5 [4, 5] | 4 [4, 5] | 4 [4, 4] |
| 31 | 5 [3, 7] | 6 [5, 8] | 5 [4, 6] | 5 [4, 5] | 4 [4, 5] | 4 [4, 4] |
