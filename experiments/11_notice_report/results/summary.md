# Experiment 11: the model notices, but does it say?


## llama3.1-8b

### A. What the model says (output, last prompt token)

Change score = log P(answer meaning 'changed') - log P(other answer); wait: surprise - normal; comma: predicted mean. Jump - control (sign-flip p). Pooled = e in {1, 2, 4, 8, 32}. AUC: jump vs control, pooled cuts. Outlier: extreme - typical last number.

| format | control level | e=0 | e=1 | e=4 | e=32 | pooled | AUC | outlier extreme | outlier moderate |
|---|---|---|---|---|---|---|---|---|---|
| comma | +310.03 | +6.93* | +43.31* | +213.96* | +378.63* | +211.23 (p=0.000) | 0.99 | -13.62 (p=0.022) | -0.98 (p=0.652) |
| wait | +4.63 | -0.11* | -0.11* | -0.06* | -0.16* | -0.07 (p=0.000) | 0.35 | +0.11 (p=0.002) | -0.04 (p=0.052) |
| q_surprise | +0.98 | +0.06* | +0.07* | +0.04* | +0.05* | +0.06 (p=0.000) | 0.76 | +0.06 (p=0.000) | -0.02 (p=0.329) |
| q_change | +0.18 | +0.03* | +0.01 | +0.02 | +0.04* | +0.02 (p=0.000) | 0.66 | +0.02 (p=0.223) | -0.01 (p=0.537) |
| q_same | -0.42 | +0.02 | +0.05* | +0.09* | +0.12* | +0.07 (p=0.000) | 0.67 | -0.04 (p=0.133) | +0.03 (p=0.156) |
| q_ab | +1.25 | +0.01 | -0.05 | -0.03 | +0.22* | +0.03 (p=0.191) | 0.56 | +0.19 (p=0.001) | +0.05 (p=0.225) |
| q_ba | +0.42 | +0.00 | +0.05* | +0.08* | +0.02 | +0.06 (p=0.000) | 0.68 | -0.03 (p=0.172) | +0.00 (p=1.000) |
| fs_change | +0.52 | -0.01 | +0.01 | +0.04* | +0.08* | +0.05 (p=0.000) | 0.59 | +0.04 (p=0.001) | +0.01 (p=0.548) |
| fs_change_r | +0.82 | +0.02 | +0.04* | +0.03 | +0.02 | +0.03 (p=0.007) | 0.54 | +0.02 (p=0.072) | -0.00 (p=0.660) |

### A. The same score read inside (jump - control, pooled cuts)

Mean over the layers of each third, and the layer with the largest |difference| (its value). The last block is the output.

| lens | format | early | middle | late | max layer (value, p) |
|---|---|---|---|---|---|
| logit | wait | -0.01 | -0.02 | -0.15 | L24 (-0.40, p=0.000) |
| logit | q_surprise | +0.00 | +0.15 | +0.19 | L17 (+0.39, p=0.000) |
| logit | q_change | +0.04 | +0.12 | +0.18 | L24 (+0.31, p=0.000) |
| logit | q_same | -0.03 | +0.21 | +0.29 | L24 (+0.47, p=0.000) |
| logit | q_ab | -0.04 | -0.05 | +0.03 | L13 (-0.13, p=0.000) |
| logit | q_ba | +0.03 | +0.06 | +0.11 | L29 (+0.33, p=0.000) |
| logit | fs_change | +0.02 | +0.19 | +0.15 | L21 (+0.62, p=0.000) |
| logit | fs_change_r | +0.01 | +0.14 | +0.07 | L21 (+0.57, p=0.000) |
| jlens | wait | +0.03 | +0.02 | -0.27 | L24 (-0.54, p=0.000) |
| jlens | q_surprise | +0.06 | +0.13 | +0.16 | L17 (+0.29, p=0.000) |
| jlens | q_change | +0.05 | +0.15 | +0.19 | L21 (+0.36, p=0.000) |
| jlens | q_same | -0.04 | +0.05 | +0.18 | L26 (+0.28, p=0.000) |
| jlens | q_ab | -0.01 | -0.01 | +0.01 | L27 (+0.05, p=0.105) |
| jlens | q_ba | +0.01 | +0.02 | +0.10 | L29 (+0.21, p=0.000) |
| jlens | fs_change | +0.03 | +0.14 | +0.13 | L21 (+0.41, p=0.000) |
| jlens | fs_change_r | +0.03 | +0.14 | +0.11 | L21 (+0.50, p=0.000) |

### A. Probes: is the jump decodable at the last prompt token?

Difference-of-means direction fitted on the other 19 streams; AUC of held-out scores (jump vs control, e in {1, 2, 4, 8, 32}), and trained on those cuts and tested at e = 0. Middle probe layer and best.

| format | AUC middle layer | best layer (AUC) | e=0 AUC, best layer | output AUC |
|---|---|---|---|---|
| comma | 1.00 (L16) | L12 (1.00) | L2 (1.00) | 0.99 |
| wait | 1.00 (L16) | L8 (1.00) | L10 (0.99) | 0.35 |
| q_surprise | 1.00 (L16) | L10 (1.00) | L8 (0.98) | 0.76 |
| q_change | 1.00 (L16) | L8 (1.00) | L8 (0.98) | 0.66 |
| q_same | 1.00 (L16) | L14 (1.00) | L10 (0.98) | 0.67 |
| q_ab | 1.00 (L16) | L16 (1.00) | L10 (0.97) | 0.56 |
| q_ba | 1.00 (L16) | L14 (1.00) | L16 (0.97) | 0.68 |
| fs_change | 1.00 (L16) | L12 (1.00) | L12 (1.00) | 0.59 |
| fs_change_r | 1.00 (L16) | L12 (1.00) | L12 (1.00) | 0.54 |

### D. The 2 x 2 change test (history low / high x recent numbers low / high)

Cells: LL = low throughout (the up control), LH = up jump, HL = down jump, HH = high throughout. Change = (LH + HL - LL - HH) / 2, the part of the score a reader of the recent numbers alone cannot produce; recent = (LH + HH - LL - HL) / 2; history = (HL + HH - LL - LH) / 2. Output score at the last prompt token, pooled cuts (e = 0 separately); sign-flip p over streams. AUC: change cells vs the others; same-tail AUC: up jump vs high throughout (identical recent numbers, different history).

| format | change (pooled) | change e=0 | recent | history | AUC change | same-tail AUC (LH vs HH) |
|---|---|---|---|---|---|---|
| comma | -1.71 (p=0.769) | +1.67 (p=0.275) | +212.95 (p=0.000) | +173.17 (p=0.000) | 0.50 | 0.11 |
| wait | +0.00 (p=0.769) | -0.03 (p=0.039) | -0.08 (p=0.000) | -0.23 (p=0.000) | 0.51 | 0.91 |
| q_surprise | +0.01 (p=0.009) | +0.02 (p=0.112) | +0.04 (p=0.000) | -0.10 (p=0.000) | 0.53 | 0.92 |
| q_change | +0.02 (p=0.001) | +0.02 (p=0.114) | +0.00 (p=0.415) | +0.12 (p=0.000) | 0.58 | 0.09 |
| q_same | +0.02 (p=0.003) | -0.01 (p=0.672) | +0.05 (p=0.000) | +0.01 (p=0.308) | 0.56 | 0.48 |
| q_ab | +0.00 (p=0.524) | -0.01 (p=0.645) | +0.02 (p=0.217) | +1.01 (p=0.000) | 0.50 | 0.00 |
| q_ba | +0.04 (p=0.000) | +0.02 (p=0.049) | +0.01 (p=0.283) | -0.20 (p=0.000) | 0.62 | 0.97 |
| fs_change | +0.02 (p=0.001) | +0.00 (p=1.000) | +0.03 (p=0.001) | +0.02 (p=0.059) | 0.55 | 0.51 |
| fs_change_r | -0.01 (p=0.369) | -0.01 (p=0.186) | +0.04 (p=0.000) | -0.15 (p=0.000) | 0.45 | 0.78 |

**Inside** (change effect by layer third, pooled cuts; max layer and its p)

| lens | format | early | middle | late | max layer (value, p) |
|---|---|---|---|---|---|
| logit | wait | +0.00 | -0.05 | -0.15 | L24 (-0.31, p=0.000) |
| logit | q_surprise | +0.00 | +0.03 | +0.02 | L17 (+0.11, p=0.000) |
| logit | q_change | +0.02 | +0.10 | +0.14 | L24 (+0.26, p=0.000) |
| logit | q_same | -0.00 | +0.06 | +0.11 | L26 (+0.17, p=0.000) |
| logit | q_ab | -0.01 | -0.02 | +0.05 | L26 (+0.11, p=0.000) |
| logit | q_ba | +0.01 | +0.02 | +0.06 | L29 (+0.22, p=0.000) |
| logit | fs_change | +0.00 | +0.06 | +0.06 | L21 (+0.36, p=0.000) |
| logit | fs_change_r | -0.00 | +0.00 | -0.07 | L21 (+0.27, p=0.000) |
| jlens | wait | +0.02 | +0.10 | -0.11 | L24 (-0.28, p=0.000) |
| jlens | q_surprise | +0.02 | +0.05 | +0.04 | L17 (+0.10, p=0.000) |
| jlens | q_change | +0.02 | +0.09 | +0.12 | L24 (+0.20, p=0.000) |
| jlens | q_same | -0.02 | -0.02 | +0.04 | L9 (-0.12, p=0.000) |
| jlens | q_ab | -0.00 | -0.00 | +0.03 | L28 (+0.06, p=0.000) |
| jlens | q_ba | -0.00 | +0.01 | +0.06 | L29 (+0.13, p=0.000) |
| jlens | fs_change | +0.02 | +0.05 | +0.06 | L21 (+0.23, p=0.000) |
| jlens | fs_change_r | +0.02 | +0.05 | -0.01 | L21 (+0.26, p=0.000) |

**Probes at the last prompt token** (difference of means on 19 streams, held-out stream scored)

Change probe: change cells vs the others (pooled cuts; e = 0 tested with the same direction). Up-to-down transfer: the up direction (up jump - up control) scored on the down pair (down jump vs high throughout): near 1 = it reads change, near 0 = it reads the recent numbers. Same-tail: the up direction on up jump vs high throughout (above 0.5 = it uses the history).

| format | change AUC, middle layer | change AUC, best layer | change e=0, best | up-to-down transfer, middle (min, max over layers) | same-tail, middle |
|---|---|---|---|---|---|
| comma | 1.00 (L16) | L14 (1.00) | L4 (1.00) | 1.00 (0.00, 1.00) | 1.00 |
| wait | 0.92 (L16) | L10 (1.00) | L6 (0.99) | 0.59 (0.02, 0.76) | 0.01 |
| q_surprise | 0.89 (L16) | L6 (0.99) | L6 (0.92) | 0.77 (0.04, 0.80) | 0.04 |
| q_change | 0.83 (L16) | L10 (1.00) | L10 (0.92) | 0.76 (0.03, 0.80) | 0.00 |
| q_same | 0.97 (L16) | L10 (1.00) | L10 (0.94) | 0.65 (0.03, 0.77) | 0.13 |
| q_ab | 0.79 (L16) | L6 (0.90) | L6 (0.79) | 0.75 (0.06, 0.75) | 0.00 |
| q_ba | 0.83 (L16) | L6 (0.91) | L6 (0.82) | 0.73 (0.06, 0.73) | 0.00 |
| fs_change | 1.00 (L16) | L16 (1.00) | L10 (0.99) | 0.94 (0.06, 0.98) | 1.00 |
| fs_change_r | 0.99 (L16) | L12 (1.00) | L10 (1.00) | 0.96 (0.05, 1.00) | 1.00 |

### B. Injection at one layer, every tail token (e = 4)

G = natural gap (jump - control, no change). Shares of G: transplant = diff_pos +1 on control; removal = diff_pos -1 on jump; proj = setting the diff direction's component to the other arm's mean (on control, on jump). x4 = effect of diff_pos +4 on control, against a random direction of the same norm (randm x4). steer = surprise-word lens direction at c = +1 (control words in brackets). Effects in the score's units, mean over streams; * p < 0.05 (sign-flip). Where |G| is small (the question formats) the shares are noisy ratios; read the x4 column against randm x4 there.


**comma** (score = predicted mean; the sd is in the csv)

| layer | G | transplant | removal | proj control | proj jump | x4 | randm x4 | steer surprise (ctrlw) |
|---|---|---|---|---|---|---|---|---|
| 8 | +214.70 | -0.01* | +0.17* | -0.01* | +0.17* | +17.64* | -2.18* | -6.06* (-2.17*) |
| 12 | +214.70 | +0.01 | +0.47* | +0.01 | +0.49* | +24.01* | -2.96 | -10.45* (-3.18*) |
| 16 | +214.70 | +0.11* | +0.81* | +0.11* | +0.83* | +405.97* | -35.65* | -18.37* (-5.54*) |
| 18 | +214.70 | +0.35* | +0.85* | +0.35* | +0.87* | +299.12* | +31.90* | -24.06* (+2.43*) |
| 20 | +214.70 | +0.50* | +0.87* | +0.50* | +0.88* | +218.74* | -21.13* | -28.91* (+1.35*) |
| 22 | +214.70 | +0.75* | +0.92* | +0.74* | +0.94* | +227.51* | +23.03* | -20.04* (-7.71*) |
| 24 | +214.70 | +0.80* | +0.93* | +0.79* | +0.95* | +376.52* | -18.65* | -5.66* (-15.37*) |
| 28 | +214.70 | +0.91* | +0.98* | +0.91* | +0.99* | +354.63* | -27.21* | -19.29* (-6.20*) |

**wait**

| layer | G | transplant | removal | proj control | proj jump | x4 | randm x4 | steer surprise (ctrlw) |
|---|---|---|---|---|---|---|---|---|
| 8 | -0.07 | +0.26 | +0.23 | +0.34* | +0.36* | -0.08* | +0.25* | -1.56* (-1.10*) |
| 12 | -0.07 | +0.86* | +0.78* | +0.71* | +0.84* | -0.15* | -0.09* | -2.54* (-2.25*) |
| 16 | -0.07 | +0.71* | +0.76* | +0.76* | +0.71* | -0.26* | -0.17* | -1.77* (-3.68*) |
| 18 | -0.07 | +0.62* | +0.70* | +0.66* | +0.68* | -0.24* | +0.16* | -0.58* (-2.67*) |
| 20 | -0.07 | +0.74* | +0.96* | +0.77* | +0.88* | -0.25* | +0.08* | +0.55* (-2.02*) |
| 22 | -0.07 | +0.83* | +0.96* | +0.82* | +0.89* | -0.23* | +0.14* | -0.05* (-1.95*) |
| 24 | -0.07 | +0.98* | +0.89* | +0.91* | +0.93* | -0.26* | +0.54* | -0.09* (-1.65*) |
| 28 | -0.07 | +0.72* | +0.93* | +0.91* | +0.91* | -0.21* | +0.17* | +0.77* (-1.78*) |

**q_surprise**

| layer | G | transplant | removal | proj control | proj jump | x4 | randm x4 | steer surprise (ctrlw) |
|---|---|---|---|---|---|---|---|---|
| 8 | +0.05 | +1.12* | +0.53* | +1.05* | +1.13* | +0.16* | -0.24* | -0.83* (-0.51*) |
| 12 | +0.05 | +1.42* | +1.25* | +1.31* | +1.14* | +0.24* | -0.01 | -1.03* (-0.68*) |
| 16 | +0.05 | +1.07* | +1.55* | +1.19* | +1.32* | +0.27* | +0.21* | -1.36* (+0.05*) |
| 18 | +0.05 | +1.29* | +1.02* | +1.24* | +1.38* | +0.27* | -0.05* | -0.64* (+0.15*) |
| 20 | +0.05 | +1.12* | +1.13* | +1.12* | +1.25* | +0.26* | -0.03* | -0.29* (+0.10*) |
| 22 | +0.05 | +1.06* | +1.26* | +1.41* | +1.32* | +0.27* | +0.03 | -0.20* (+0.46*) |
| 24 | +0.05 | +1.23* | +1.20* | +1.41* | +1.32* | +0.26* | +0.07* | -0.86* (-0.30*) |
| 28 | +0.05 | +1.36* | +1.32* | +1.32* | +1.15* | +0.26* | -0.04* | -0.79* (+0.35*) |

**q_change**

| layer | G | transplant | removal | proj control | proj jump | x4 | randm x4 | steer surprise (ctrlw) |
|---|---|---|---|---|---|---|---|---|
| 8 | +0.02 | +1.45* | +1.86* | +1.18* | +2.00* | +0.15* | +0.02* | +0.07* (+0.45*) |
| 12 | +0.02 | +0.89* | +1.59* | +0.73 | +1.71* | +0.13* | -0.07* | -0.52* (+0.61*) |
| 16 | +0.02 | +0.60 | +1.43* | +0.58 | +0.85 | +0.07* | +0.06* | -0.61* (+0.37*) |
| 18 | +0.02 | +0.72 | +1.00* | +0.73 | +1.28* | +0.09* | -0.06* | -0.10* (+0.35*) |
| 20 | +0.02 | +0.72 | +0.73* | +0.73 | +0.86* | +0.11* | -0.03* | -0.03* (+0.25*) |
| 22 | +0.02 | +0.60 | +1.27* | +0.45 | +0.73* | +0.11* | +0.00 | +0.16* (+0.68*) |
| 24 | +0.02 | +0.99* | +0.87 | +0.59 | +1.16* | +0.08* | +0.05* | -0.76* (-0.58*) |
| 28 | +0.02 | +0.44 | +1.14* | +0.46 | +0.58 | +0.09* | -0.05* | -0.63* (+0.38*) |

### C. Ablation of the magnitude-word direction (layers [16, 17, 18, 19, 20, 21, 22])

Predicted sd over numbers 200..399, mean over 10 streams, by sigma; slope of log sd on log sigma (95% bootstrap interval over streams) and its change from no ablation.

| variant | s=10 | s=20 | s=30 | s=50 | s=80 | s=100 | s=120 | s=150 | s=200 | slope | change |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 13.8 | 24.4 | 35.6 | 54.3 | 81.8 | 109.6 | 131.8 | 182.3 | 244.0 | 0.955 [0.938, 0.971] | +0.000 [+0.000, +0.000] |
| magnitude | 14.1 | 24.7 | 35.9 | 54.7 | 81.9 | 109.6 | 131.4 | 181.5 | 243.4 | 0.947 [0.930, 0.964] | -0.008 [-0.009, -0.007] |
| magnitude_L19 | 14.0 | 24.6 | 35.8 | 54.6 | 81.9 | 109.6 | 131.6 | 181.9 | 243.8 | 0.948 [0.932, 0.965] | -0.006 [-0.007, -0.006] |
| control | 13.8 | 24.4 | 35.6 | 54.4 | 81.8 | 109.2 | 131.4 | 181.6 | 243.5 | 0.953 [0.937, 0.969] | -0.002 [-0.003, -0.001] |
| wide | 13.8 | 24.4 | 35.5 | 54.3 | 81.9 | 109.5 | 131.7 | 182.0 | 243.5 | 0.954 [0.938, 0.971] | -0.000 [-0.001, -0.000] |
| random0 | 13.8 | 24.4 | 35.6 | 54.3 | 81.8 | 109.6 | 131.9 | 182.2 | 244.0 | 0.954 [0.938, 0.971] | -0.001 [-0.001, -0.000] |
| random1 | 13.8 | 24.4 | 35.6 | 54.3 | 81.8 | 109.5 | 131.6 | 182.1 | 243.7 | 0.954 [0.938, 0.971] | -0.000 [-0.001, -0.000] |
| random2 | 13.7 | 24.4 | 35.5 | 54.2 | 81.5 | 109.3 | 131.6 | 182.1 | 244.0 | 0.955 [0.938, 0.971] | +0.000 [-0.000, +0.000] |

## llama3.1-8b-it

### A. What the model says (output, last prompt token)

Change score = log P(answer meaning 'changed') - log P(other answer); wait: surprise - normal; comma: predicted mean. Jump - control (sign-flip p). Pooled = e in {1, 2, 4, 8, 32}. AUC: jump vs control, pooled cuts. Outlier: extreme - typical last number.

| format | control level | e=0 | e=1 | e=4 | e=32 | pooled | AUC | outlier extreme | outlier moderate |
|---|---|---|---|---|---|---|---|---|---|
| q_surprise | -0.17 | +0.25* | +0.26* | +0.21* | +0.03 | +0.18 (p=0.000) | 0.74 | +0.14 (p=0.004) | +0.09 (p=0.005) |
| q_change | +0.67 | -0.01 | +0.03 | +0.13* | -0.03 | +0.09 (p=0.000) | 0.58 | +0.21 (p=0.000) | -0.04 (p=0.062) |
| q_same | +1.58 | +0.33* | +0.60* | +1.03* | +0.58* | +0.78 (p=0.000) | 0.97 | +0.58 (p=0.000) | +0.09 (p=0.003) |
| q_ab | +0.13 | +0.13* | +0.35* | +0.58* | +0.40* | +0.47 (p=0.000) | 0.98 | +0.38 (p=0.000) | +0.06 (p=0.009) |
| q_ba | +0.79 | +0.21* | +0.32* | +0.40* | +0.26* | +0.35 (p=0.000) | 0.92 | +0.27 (p=0.000) | +0.06 (p=0.055) |

### A. The same score read inside (jump - control, pooled cuts)

Mean over the layers of each third, and the layer with the largest |difference| (its value). The last block is the output.

| lens | format | early | middle | late | max layer (value, p) |
|---|---|---|---|---|---|
| logit | q_surprise | +0.01 | +0.31 | +0.57 | L24 (+1.54, p=0.000) |
| logit | q_change | +0.01 | +0.17 | +0.39 | L24 (+0.83, p=0.000) |
| logit | q_same | -0.01 | +0.72 | +1.61 | L21 (+2.54, p=0.000) |
| logit | q_ab | -0.01 | +0.14 | +0.74 | L29 (+1.54, p=0.000) |
| logit | q_ba | +0.01 | +0.13 | +0.63 | L29 (+1.13, p=0.000) |
| jlens | q_surprise | -0.00 | +0.25 | +0.42 | L24 (+1.28, p=0.000) |
| jlens | q_change | -0.00 | +0.11 | +0.29 | L24 (+0.67, p=0.000) |
| jlens | q_same | +0.01 | +0.64 | +1.75 | L21 (+2.48, p=0.000) |
| jlens | q_ab | +0.00 | +0.09 | +0.59 | L29 (+1.35, p=0.000) |
| jlens | q_ba | -0.00 | +0.06 | +0.44 | L29 (+0.89, p=0.000) |

### A. Probes: is the jump decodable at the last prompt token?

Difference-of-means direction fitted on the other 19 streams; AUC of held-out scores (jump vs control, e in {1, 2, 4, 8, 32}), and trained on those cuts and tested at e = 0. Middle probe layer and best.

| format | AUC middle layer | best layer (AUC) | e=0 AUC, best layer | output AUC |
|---|---|---|---|---|
| q_surprise | 1.00 (L16) | L8 (1.00) | L12 (0.99) | 0.74 |
| q_change | 0.99 (L16) | L8 (1.00) | L12 (1.00) | 0.58 |
| q_same | 0.99 (L16) | L8 (1.00) | L8 (1.00) | 0.97 |
| q_ab | 1.00 (L16) | L8 (1.00) | L8 (1.00) | 0.98 |
| q_ba | 1.00 (L16) | L8 (1.00) | L8 (0.99) | 0.92 |

### D. The 2 x 2 change test (history low / high x recent numbers low / high)

Cells: LL = low throughout (the up control), LH = up jump, HL = down jump, HH = high throughout. Change = (LH + HL - LL - HH) / 2, the part of the score a reader of the recent numbers alone cannot produce; recent = (LH + HH - LL - HL) / 2; history = (HL + HH - LL - LH) / 2. Output score at the last prompt token, pooled cuts (e = 0 separately); sign-flip p over streams. AUC: change cells vs the others; same-tail AUC: up jump vs high throughout (identical recent numbers, different history).

| format | change (pooled) | change e=0 | recent | history | AUC change | same-tail AUC (LH vs HH) |
|---|---|---|---|---|---|---|
| q_surprise | +0.10 (p=0.000) | +0.16 (p=0.000) | +0.08 (p=0.000) | -0.51 (p=0.000) | 0.59 | 0.99 |
| q_change | +0.14 (p=0.000) | +0.00 (p=0.998) | -0.05 (p=0.002) | -0.62 (p=0.000) | 0.59 | 0.99 |
| q_same | +0.71 (p=0.000) | +0.31 (p=0.000) | +0.08 (p=0.015) | -0.24 (p=0.003) | 0.95 | 0.99 |
| q_ab | +0.54 (p=0.000) | +0.11 (p=0.000) | -0.07 (p=0.028) | +0.47 (p=0.000) | 0.87 | 0.52 |
| q_ba | +0.34 (p=0.000) | +0.17 (p=0.000) | +0.01 (p=0.496) | +0.31 (p=0.000) | 0.88 | 0.64 |

**Inside** (change effect by layer third, pooled cuts; max layer and its p)

| lens | format | early | middle | late | max layer (value, p) |
|---|---|---|---|---|---|
| logit | q_surprise | +0.01 | +0.27 | +0.37 | L24 (+0.95, p=0.000) |
| logit | q_change | +0.01 | +0.22 | +0.50 | L24 (+0.91, p=0.000) |
| logit | q_same | -0.01 | +0.63 | +1.52 | L21 (+2.23, p=0.000) |
| logit | q_ab | +0.00 | +0.09 | +0.83 | L29 (+1.64, p=0.000) |
| logit | q_ba | -0.00 | +0.16 | +0.61 | L29 (+1.09, p=0.000) |
| jlens | q_surprise | -0.00 | +0.19 | +0.27 | L24 (+0.78, p=0.000) |
| jlens | q_change | -0.01 | +0.14 | +0.41 | L24 (+0.79, p=0.000) |
| jlens | q_same | +0.00 | +0.58 | +1.63 | L24 (+2.21, p=0.000) |
| jlens | q_ab | -0.00 | +0.09 | +0.65 | L29 (+1.47, p=0.000) |
| jlens | q_ba | +0.00 | +0.06 | +0.44 | L29 (+0.87, p=0.000) |

**Probes at the last prompt token** (difference of means on 19 streams, held-out stream scored)

Change probe: change cells vs the others (pooled cuts; e = 0 tested with the same direction). Up-to-down transfer: the up direction (up jump - up control) scored on the down pair (down jump vs high throughout): near 1 = it reads change, near 0 = it reads the recent numbers. Same-tail: the up direction on up jump vs high throughout (above 0.5 = it uses the history).

| format | change AUC, middle layer | change AUC, best layer | change e=0, best | up-to-down transfer, middle (min, max over layers) | same-tail, middle |
|---|---|---|---|---|---|
| q_surprise | 1.00 (L16) | L31 (1.00) | L6 (0.98) | 0.98 (0.07, 0.99) | 1.00 |
| q_change | 0.93 (L16) | L10 (1.00) | L10 (1.00) | 0.95 (0.07, 0.99) | 1.00 |
| q_same | 0.99 (L16) | L10 (1.00) | L6 (0.99) | 0.99 (0.05, 1.00) | 0.99 |
| q_ab | 1.00 (L16) | L16 (1.00) | L8 (0.96) | 0.99 (0.08, 1.00) | 0.99 |
| q_ba | 1.00 (L16) | L16 (1.00) | L8 (0.98) | 0.99 (0.09, 0.99) | 0.99 |

### B. Injection at one layer, every tail token (e = 4)

G = natural gap (jump - control, no change). Shares of G: transplant = diff_pos +1 on control; removal = diff_pos -1 on jump; proj = setting the diff direction's component to the other arm's mean (on control, on jump). x4 = effect of diff_pos +4 on control, against a random direction of the same norm (randm x4). steer = surprise-word lens direction at c = +1 (control words in brackets). Effects in the score's units, mean over streams; * p < 0.05 (sign-flip). Where |G| is small (the question formats) the shares are noisy ratios; read the x4 column against randm x4 there.


**q_surprise**

| layer | G | transplant | removal | proj control | proj jump | x4 | randm x4 | steer surprise (ctrlw) |
|---|---|---|---|---|---|---|---|---|
| 8 | +0.24 | +0.21* | +0.24* | +0.24* | +0.29* | +0.21* | -0.01 | -0.31* (+0.82*) |
| 12 | +0.24 | +1.00* | +1.11* | +0.97* | +1.05* | +1.31* | +0.11* | +0.88* (+2.15*) |
| 16 | +0.24 | +1.08* | +1.00* | +1.08* | +0.92* | +1.35* | -0.13* | -0.55* (+0.59*) |
| 18 | +0.24 | +1.16* | +1.05* | +1.11* | +1.03* | +1.24* | -0.19* | -0.50* (+0.16*) |
| 20 | +0.24 | +1.13* | +1.08* | +1.16* | +1.08* | +1.20* | -0.27* | -1.08* (-0.26*) |
| 22 | +0.24 | +1.11* | +0.97* | +1.03* | +0.95* | +1.17* | -0.17* | -0.26* (+0.47*) |
| 24 | +0.24 | +1.03* | +0.97* | +1.05* | +0.97* | +1.24* | +0.22* | -1.60* (-0.79*) |
| 28 | +0.24 | +1.11* | +0.92* | +1.05* | +1.03* | +1.19* | -0.34* | -0.89* (-0.56*) |

**q_change**

| layer | G | transplant | removal | proj control | proj jump | x4 | randm x4 | steer surprise (ctrlw) |
|---|---|---|---|---|---|---|---|---|
| 8 | +0.15 | -0.79* | -0.54* | -1.08* | -0.54* | -0.41* | -0.08* | -0.94* (+0.13*) |
| 12 | +0.15 | +0.50* | +0.71* | +0.63* | +0.79* | +0.83* | +0.61* | +0.24* (+0.63*) |
| 16 | +0.15 | +0.79* | +0.87* | +0.67* | +0.96* | +0.57* | -0.08* | -0.90* (-0.05*) |
| 18 | +0.15 | +1.04* | +1.00* | +0.75* | +1.12* | +0.53* | -0.13* | -1.07* (-0.39*) |
| 20 | +0.15 | +0.87* | +1.08* | +0.83* | +1.04* | +0.52* | -0.14* | -1.69* (-0.87*) |
| 22 | +0.15 | +0.96* | +1.04* | +0.75* | +1.08* | +0.52* | -0.09* | -0.89* (-0.25*) |
| 24 | +0.15 | +0.96* | +1.00* | +0.83* | +1.04* | +0.53* | +0.14* | -2.29* (-1.60*) |
| 28 | +0.15 | +0.83* | +1.04* | +0.75* | +1.08* | +0.51* | -0.16* | -1.52* (-1.28*) |

## olmo-3-7b

### A. What the model says (output, last prompt token)

Change score = log P(answer meaning 'changed') - log P(other answer); wait: surprise - normal; comma: predicted mean. Jump - control (sign-flip p). Pooled = e in {1, 2, 4, 8, 32}. AUC: jump vs control, pooled cuts. Outlier: extreme - typical last number.

| format | control level | e=0 | e=1 | e=4 | e=32 | pooled | AUC | outlier extreme | outlier moderate |
|---|---|---|---|---|---|---|---|---|---|
| comma | +310.50 | -1.61* | +2.07* | +64.66* | +228.88* | +91.99 (p=0.000) | 0.87 | -11.44 (p=0.000) | -0.62 (p=0.435) |
| wait | +1.71 | +0.10* | +0.27* | +0.48* | +0.88* | +0.56 (p=0.000) | 0.67 | +0.08 (p=0.002) | +0.02 (p=0.490) |
| q_surprise | +0.55 | +0.02* | +0.02* | +0.05* | +0.15* | +0.06 (p=0.000) | 0.58 | +0.01 (p=0.339) | -0.01 (p=0.402) |
| q_change | +0.68 | -0.00 | -0.01 | +0.00 | +0.01 | -0.00 (p=0.784) | 0.48 | +0.05 (p=0.004) | +0.02 (p=0.100) |
| q_same | -0.31 | -0.02* | -0.02 | +0.03* | +0.00 | +0.00 (p=0.595) | 0.52 | -0.01 (p=0.181) | +0.00 (p=0.619) |
| q_ab | +1.58 | -0.03* | +0.01 | -0.02 | -0.16* | -0.05 (p=0.084) | 0.47 | -0.05 (p=0.025) | -0.16 (p=0.000) |
| q_ba | -1.16 | +0.01 | +0.01 | +0.01 | +0.14* | +0.05 (p=0.101) | 0.53 | +0.01 (p=0.467) | +0.12 (p=0.000) |
| fs_change | +0.75 | +0.01 | -0.03 | +0.02 | -0.11 | -0.01 (p=0.668) | 0.49 | +0.21 (p=0.000) | +0.10 (p=0.000) |
| fs_change_r | +0.76 | -0.02 | -0.07* | -0.00 | -0.21* | -0.07 (p=0.039) | 0.45 | +0.15 (p=0.000) | +0.10 (p=0.000) |

### A. The same score read inside (jump - control, pooled cuts)

Mean over the layers of each third, and the layer with the largest |difference| (its value). The last block is the output.

| lens | format | early | middle | late | max layer (value, p) |
|---|---|---|---|---|---|
| logit | wait | +0.00 | +0.08 | +0.80 | L27 (+1.39, p=0.000) |
| logit | q_surprise | +0.01 | +0.08 | +0.14 | L22 (+0.22, p=0.000) |
| logit | q_change | +0.01 | +0.05 | -0.01 | L21 (+0.12, p=0.038) |
| logit | q_same | -0.00 | -0.02 | +0.01 | L22 (-0.18, p=0.001) |
| logit | q_ab | +0.00 | +0.06 | -0.29 | L27 (-0.47, p=0.009) |
| logit | q_ba | -0.00 | -0.09 | +0.19 | L26 (+0.31, p=0.040) |
| logit | fs_change | +0.00 | +0.10 | +0.31 | L22 (+0.65, p=0.000) |
| logit | fs_change_r | +0.00 | +0.09 | +0.24 | L22 (+0.64, p=0.000) |
| jlens | wait | -0.00 | +0.11 | +0.47 | L27 (+0.85, p=0.000) |
| jlens | q_surprise | -0.01 | +0.01 | +0.04 | L20 (+0.07, p=0.000) |
| jlens | q_change | -0.02 | +0.01 | -0.01 | L10 (-0.05, p=0.000) |
| jlens | q_same | +0.01 | -0.03 | -0.03 | L22 (-0.12, p=0.000) |
| jlens | q_ab | -0.00 | +0.01 | -0.11 | L27 (-0.22, p=0.003) |
| jlens | q_ba | +0.00 | -0.02 | +0.06 | L27 (+0.13, p=0.037) |
| jlens | fs_change | -0.01 | -0.03 | +0.11 | L22 (+0.21, p=0.000) |
| jlens | fs_change_r | -0.01 | -0.06 | -0.01 | L20 (-0.14, p=0.002) |

### A. Probes: is the jump decodable at the last prompt token?

Difference-of-means direction fitted on the other 19 streams; AUC of held-out scores (jump vs control, e in {1, 2, 4, 8, 32}), and trained on those cuts and tested at e = 0. Middle probe layer and best.

| format | AUC middle layer | best layer (AUC) | e=0 AUC, best layer | output AUC |
|---|---|---|---|---|
| comma | 0.95 (L16) | L10 (1.00) | L6 (1.00) | 0.87 |
| wait | 0.93 (L16) | L10 (0.94) | L10 (0.73) | 0.67 |
| q_surprise | 0.75 (L16) | L28 (0.90) | L28 (0.67) | 0.58 |
| q_change | 0.80 (L16) | L28 (0.90) | L26 (0.65) | 0.48 |
| q_same | 0.77 (L16) | L28 (0.93) | L28 (0.65) | 0.52 |
| q_ab | 0.73 (L16) | L31 (0.81) | L30 (0.61) | 0.47 |
| q_ba | 0.80 (L16) | L31 (0.87) | L14 (0.62) | 0.53 |
| fs_change | 0.91 (L16) | L16 (0.91) | L16 (0.78) | 0.49 |
| fs_change_r | 0.91 (L16) | L16 (0.91) | L16 (0.75) | 0.45 |

### D. The 2 x 2 change test (history low / high x recent numbers low / high)

Cells: LL = low throughout (the up control), LH = up jump, HL = down jump, HH = high throughout. Change = (LH + HL - LL - HH) / 2, the part of the score a reader of the recent numbers alone cannot produce; recent = (LH + HH - LL - HL) / 2; history = (HL + HH - LL - LH) / 2. Output score at the last prompt token, pooled cuts (e = 0 separately); sign-flip p over streams. AUC: change cells vs the others; same-tail AUC: up jump vs high throughout (identical recent numbers, different history).

| format | change (pooled) | change e=0 | recent | history | AUC change | same-tail AUC (LH vs HH) |
|---|---|---|---|---|---|---|
| comma | +22.30 (p=0.000) | -4.66 (p=0.045) | +69.69 (p=0.000) | +314.07 (p=0.000) | 0.52 | 0.00 |
| wait | +0.38 (p=0.000) | +0.08 (p=0.023) | +0.17 (p=0.003) | +1.43 (p=0.000) | 0.59 | 0.18 |
| q_surprise | +0.01 (p=0.251) | -0.00 (p=0.693) | +0.05 (p=0.000) | +0.35 (p=0.000) | 0.50 | 0.07 |
| q_change | -0.04 (p=0.000) | -0.02 (p=0.004) | +0.03 (p=0.000) | +0.30 (p=0.000) | 0.45 | 0.04 |
| q_same | +0.03 (p=0.000) | -0.01 (p=0.276) | -0.03 (p=0.000) | -0.14 (p=0.000) | 0.55 | 0.80 |
| q_ab | -0.02 (p=0.307) | -0.00 (p=0.861) | -0.04 (p=0.029) | -0.21 (p=0.022) | 0.49 | 0.63 |
| q_ba | +0.02 (p=0.055) | +0.00 (p=0.573) | +0.02 (p=0.186) | +0.38 (p=0.000) | 0.52 | 0.22 |
| fs_change | -0.01 (p=0.405) | +0.01 (p=0.340) | +0.00 (p=0.999) | -0.08 (p=0.123) | 0.49 | 0.57 |
| fs_change_r | -0.05 (p=0.011) | +0.01 (p=0.567) | -0.02 (p=0.285) | -0.27 (p=0.000) | 0.47 | 0.65 |

**Inside** (change effect by layer third, pooled cuts; max layer and its p)

| lens | format | early | middle | late | max layer (value, p) |
|---|---|---|---|---|---|
| logit | wait | -0.00 | +0.11 | +0.70 | L26 (+1.08, p=0.000) |
| logit | q_surprise | +0.01 | +0.06 | +0.01 | L14 (+0.11, p=0.000) |
| logit | q_change | +0.01 | +0.06 | -0.06 | L27 (-0.13, p=0.000) |
| logit | q_same | -0.00 | -0.00 | +0.07 | L27 (+0.14, p=0.001) |
| logit | q_ab | -0.00 | +0.04 | -0.18 | L25 (-0.31, p=0.001) |
| logit | q_ba | +0.00 | -0.06 | +0.17 | L25 (+0.31, p=0.000) |
| logit | fs_change | -0.00 | +0.04 | +0.14 | L22 (+0.32, p=0.000) |
| logit | fs_change_r | -0.00 | +0.03 | +0.09 | L22 (+0.31, p=0.000) |
| jlens | wait | -0.01 | +0.09 | +0.37 | L27 (+0.58, p=0.000) |
| jlens | q_surprise | -0.02 | +0.00 | -0.00 | L10 (-0.07, p=0.000) |
| jlens | q_change | -0.02 | +0.01 | -0.03 | L10 (-0.06, p=0.000) |
| jlens | q_same | +0.01 | -0.01 | +0.02 | L10 (+0.05, p=0.000) |
| jlens | q_ab | -0.01 | +0.00 | -0.08 | L27 (-0.15, p=0.000) |
| jlens | q_ba | +0.01 | -0.01 | +0.07 | L26 (+0.14, p=0.000) |
| jlens | fs_change | -0.01 | -0.02 | +0.06 | L22 (+0.11, p=0.000) |
| jlens | fs_change_r | -0.00 | -0.04 | -0.02 | L15 (-0.10, p=0.000) |

**Probes at the last prompt token** (difference of means on 19 streams, held-out stream scored)

Change probe: change cells vs the others (pooled cuts; e = 0 tested with the same direction). Up-to-down transfer: the up direction (up jump - up control) scored on the down pair (down jump vs high throughout): near 1 = it reads change, near 0 = it reads the recent numbers. Same-tail: the up direction on up jump vs high throughout (above 0.5 = it uses the history).

| format | change AUC, middle layer | change AUC, best layer | change e=0, best | up-to-down transfer, middle (min, max over layers) | same-tail, middle |
|---|---|---|---|---|---|
| comma | 0.87 (L16) | L12 (0.97) | L2 (0.94) | 0.76 (0.43, 0.94) | 0.66 |
| wait | 0.85 (L16) | L14 (0.90) | L14 (0.64) | 0.77 (0.42, 0.81) | 0.39 |
| q_surprise | 0.64 (L16) | L14 (0.66) | L14 (0.57) | 0.67 (0.41, 0.70) | 0.13 |
| q_change | 0.72 (L16) | L16 (0.72) | L14 (0.59) | 0.70 (0.39, 0.73) | 0.32 |
| q_same | 0.67 (L16) | L22 (0.69) | L16 (0.55) | 0.61 (0.42, 0.64) | 0.12 |
| q_ab | 0.61 (L16) | L26 (0.62) | L14 (0.54) | 0.54 (0.43, 0.65) | 0.17 |
| q_ba | 0.65 (L16) | L14 (0.68) | L14 (0.55) | 0.56 (0.43, 0.65) | 0.26 |
| fs_change | 0.76 (L16) | L16 (0.76) | L16 (0.67) | 0.72 (0.44, 0.72) | 0.45 |
| fs_change_r | 0.77 (L16) | L16 (0.77) | L16 (0.68) | 0.69 (0.45, 0.69) | 0.43 |

## olmo-3-7b-it

### A. What the model says (output, last prompt token)

Change score = log P(answer meaning 'changed') - log P(other answer); wait: surprise - normal; comma: predicted mean. Jump - control (sign-flip p). Pooled = e in {1, 2, 4, 8, 32}. AUC: jump vs control, pooled cuts. Outlier: extreme - typical last number.

| format | control level | e=0 | e=1 | e=4 | e=32 | pooled | AUC | outlier extreme | outlier moderate |
|---|---|---|---|---|---|---|---|---|---|
| q_surprise | -1.83 | -0.08* | -0.08* | -0.10* | -0.08* | -0.10 (p=0.001) | 0.46 | -0.01 (p=0.728) | -0.08 (p=0.007) |
| q_change | -6.68 | +0.05 | -0.13* | -0.11* | -0.25* | -0.15 (p=0.000) | 0.46 | +0.18 (p=0.000) | -0.04 (p=0.378) |
| q_same | +6.53 | +0.07 | +0.12* | +0.08 | +0.26* | +0.16 (p=0.000) | 0.55 | +0.11 (p=0.071) | +0.18 (p=0.000) |
| q_ab | -2.14 | +0.01 | -0.03 | -0.06* | +0.05 | -0.02 (p=0.022) | 0.48 | +0.19 (p=0.000) | +0.16 (p=0.000) |
| q_ba | +1.91 | -0.01 | -0.00 | +0.04 | -0.01 | +0.02 (p=0.044) | 0.52 | -0.13 (p=0.141) | -0.08 (p=0.047) |

### A. The same score read inside (jump - control, pooled cuts)

Mean over the layers of each third, and the layer with the largest |difference| (its value). The last block is the output.

| lens | format | early | middle | late | max layer (value, p) |
|---|---|---|---|---|---|
| logit | q_surprise | -0.00 | -0.03 | -0.12 | L23 (-0.14, p=0.000) |
| logit | q_change | -0.00 | -0.04 | -0.16 | L28 (-0.25, p=0.000) |
| logit | q_same | +0.00 | +0.02 | +0.20 | L26 (+0.25, p=0.000) |
| logit | q_ab | -0.00 | +0.02 | -0.00 | L16 (+0.06, p=0.000) |
| logit | q_ba | +0.00 | -0.04 | -0.01 | L16 (-0.08, p=0.000) |

### A. Probes: is the jump decodable at the last prompt token?

Difference-of-means direction fitted on the other 19 streams; AUC of held-out scores (jump vs control, e in {1, 2, 4, 8, 32}), and trained on those cuts and tested at e = 0. Middle probe layer and best.

| format | AUC middle layer | best layer (AUC) | e=0 AUC, best layer | output AUC |
|---|---|---|---|---|
| q_surprise | 0.73 (L16) | L12 (0.78) | L10 (0.69) | 0.46 |
| q_change | 0.76 (L16) | L16 (0.76) | L18 (0.66) | 0.46 |
| q_same | 0.69 (L16) | L12 (0.73) | L8 (0.63) | 0.55 |
| q_ab | 0.66 (L16) | L10 (0.76) | L10 (0.67) | 0.48 |
| q_ba | 0.67 (L16) | L10 (0.75) | L10 (0.69) | 0.52 |

### D. The 2 x 2 change test (history low / high x recent numbers low / high)

Cells: LL = low throughout (the up control), LH = up jump, HL = down jump, HH = high throughout. Change = (LH + HL - LL - HH) / 2, the part of the score a reader of the recent numbers alone cannot produce; recent = (LH + HH - LL - HL) / 2; history = (HL + HH - LL - LH) / 2. Output score at the last prompt token, pooled cuts (e = 0 separately); sign-flip p over streams. AUC: change cells vs the others; same-tail AUC: up jump vs high throughout (identical recent numbers, different history).

| format | change (pooled) | change e=0 | recent | history | AUC change | same-tail AUC (LH vs HH) |
|---|---|---|---|---|---|---|
| q_surprise | -0.12 (p=0.000) | -0.11 (p=0.000) | +0.02 (p=0.293) | +1.88 (p=0.000) | 0.47 | 0.01 |
| q_change | -0.27 (p=0.000) | -0.10 (p=0.021) | +0.12 (p=0.000) | +2.44 (p=0.000) | 0.46 | 0.05 |
| q_same | +0.21 (p=0.000) | +0.14 (p=0.000) | -0.04 (p=0.018) | -2.48 (p=0.000) | 0.54 | 0.96 |
| q_ab | -0.03 (p=0.000) | -0.01 (p=0.180) | +0.01 (p=0.513) | -0.36 (p=0.001) | 0.48 | 0.73 |
| q_ba | +0.05 (p=0.000) | +0.01 (p=0.261) | -0.02 (p=0.012) | +0.33 (p=0.003) | 0.53 | 0.30 |

**Inside** (change effect by layer third, pooled cuts; max layer and its p)

| lens | format | early | middle | late | max layer (value, p) |
|---|---|---|---|---|---|
| logit | q_surprise | -0.00 | -0.04 | -0.12 | L23 (-0.14, p=0.000) |
| logit | q_change | -0.01 | -0.04 | -0.30 | L28 (-0.40, p=0.000) |
| logit | q_same | +0.00 | +0.02 | +0.25 | L26 (+0.31, p=0.000) |
| logit | q_ab | -0.00 | +0.02 | -0.01 | L16 (+0.04, p=0.000) |
| logit | q_ba | -0.00 | -0.03 | +0.02 | L16 (-0.05, p=0.000) |

**Probes at the last prompt token** (difference of means on 19 streams, held-out stream scored)

Change probe: change cells vs the others (pooled cuts; e = 0 tested with the same direction). Up-to-down transfer: the up direction (up jump - up control) scored on the down pair (down jump vs high throughout): near 1 = it reads change, near 0 = it reads the recent numbers. Same-tail: the up direction on up jump vs high throughout (above 0.5 = it uses the history).

| format | change AUC, middle layer | change AUC, best layer | change e=0, best | up-to-down transfer, middle (min, max over layers) | same-tail, middle |
|---|---|---|---|---|---|
| q_surprise | 0.64 (L16) | L16 (0.64) | L30 (0.60) | 0.56 (0.45, 0.58) | 0.01 |
| q_change | 0.71 (L16) | L16 (0.71) | L16 (0.62) | 0.58 (0.42, 0.58) | 0.15 |
| q_same | 0.70 (L16) | L16 (0.70) | L16 (0.61) | 0.53 (0.45, 0.54) | 0.26 |
| q_ab | 0.64 (L16) | L10 (0.65) | L16 (0.59) | 0.56 (0.45, 0.56) | 0.10 |
| q_ba | 0.64 (L16) | L28 (0.65) | L16 (0.59) | 0.55 (0.45, 0.55) | 0.22 |

### B. Injection at one layer, every tail token (e = 4)

G = natural gap (jump - control, no change). Shares of G: transplant = diff_pos +1 on control; removal = diff_pos -1 on jump; proj = setting the diff direction's component to the other arm's mean (on control, on jump). x4 = effect of diff_pos +4 on control, against a random direction of the same norm (randm x4). steer = surprise-word lens direction at c = +1 (control words in brackets). Effects in the score's units, mean over streams; * p < 0.05 (sign-flip). Where |G| is small (the question formats) the shares are noisy ratios; read the x4 column against randm x4 there.


**q_surprise**

| layer | G | transplant | removal | proj control | proj jump | x4 | randm x4 | steer surprise (ctrlw) |
|---|---|---|---|---|---|---|---|---|
| 8 | -0.05 | +0.41 | -0.18 | +0.12 | -0.88 | +0.08* | -0.10* | - (-) |
| 12 | -0.05 | +0.17 | -0.35 | -0.35 | +0.29 | +0.02 | -0.09* | - (-) |
| 16 | -0.05 | +1.41* | +1.35* | +1.71* | +1.24* | -0.29* | -0.02 | - (-) |
| 18 | -0.05 | +2.18* | +1.53* | +2.59* | +0.70 | -0.35* | +0.12* | - (-) |
| 20 | -0.05 | +2.06* | +1.71* | +2.88* | +0.64 | -0.36* | +0.27* | - (-) |
| 22 | -0.05 | +1.82* | +1.59* | +2.18* | +0.99 | -0.36* | -0.15* | - (-) |
| 24 | -0.05 | +1.82* | +1.24* | +2.12* | +0.76 | -0.34* | -0.30* | - (-) |
| 28 | -0.05 | +1.47* | +1.47* | +2.06* | +0.59 | -0.29* | -0.10* | - (-) |

**q_change**

| layer | G | transplant | removal | proj control | proj jump | x4 | randm x4 | steer surprise (ctrlw) |
|---|---|---|---|---|---|---|---|---|
| 8 | -0.12 | +1.02* | +1.13* | +1.05* | +0.49 | -0.40* | +0.23* | - (-) |
| 12 | -0.12 | +0.23 | +0.26 | +0.33 | +0.54 | -0.13* | -0.18* | - (-) |
| 16 | -0.12 | +1.13* | +1.13* | +0.90* | +1.03* | -0.39* | -0.23* | - (-) |
| 18 | -0.12 | +1.00* | +1.05* | +1.18* | +0.92* | -0.44* | +0.44* | - (-) |
| 20 | -0.12 | +1.08* | +1.26* | +1.15* | +1.08* | -0.49* | +0.37* | - (-) |
| 22 | -0.12 | +1.08* | +1.05* | +1.10* | +0.93* | -0.40* | -0.03 | - (-) |
| 24 | -0.12 | +0.87* | +0.80* | +0.90* | +0.52* | -0.35* | -0.36* | - (-) |
| 28 | -0.12 | +0.90* | +0.82* | +0.90* | +0.74* | -0.34* | +0.02 | - (-) |

## olmo-3-32b

### A. What the model says (output, last prompt token)

Change score = log P(answer meaning 'changed') - log P(other answer); wait: surprise - normal; comma: predicted mean. Jump - control (sign-flip p). Pooled = e in {1, 2, 4, 8, 32}. AUC: jump vs control, pooled cuts. Outlier: extreme - typical last number.

| format | control level | e=0 | e=1 | e=4 | e=32 | pooled | AUC | outlier extreme | outlier moderate |
|---|---|---|---|---|---|---|---|---|---|
| comma | +304.25 | -11.50* | +46.24* | +273.88* | +386.69* | +241.15 (p=0.000) | 0.97 | -11.92 (p=0.038) | -9.28 (p=0.034) |
| wait | +2.24 | +0.15* | +0.23* | +0.30* | +0.07* | +0.23 (p=0.000) | 0.80 | +0.07 (p=0.179) | +0.09 (p=0.019) |
| q_surprise | +0.82 | -0.01 | +0.04* | +0.08* | +0.04* | +0.06 (p=0.000) | 0.64 | +0.04 (p=0.047) | +0.04 (p=0.011) |
| q_change | +0.91 | +0.01 | -0.01 | -0.02* | +0.00 | -0.01 (p=0.005) | 0.47 | +0.03 (p=0.003) | -0.00 (p=0.968) |
| q_same | -0.53 | +0.03* | +0.11* | +0.12* | +0.08* | +0.12 (p=0.000) | 0.70 | -0.02 (p=0.363) | +0.03 (p=0.009) |
| q_ab | +0.67 | -0.01 | -0.01 | +0.04* | +0.05* | +0.03 (p=0.004) | 0.56 | +0.03 (p=0.008) | -0.01 (p=0.322) |
| q_ba | -0.38 | +0.03* | +0.03* | +0.03* | +0.01 | +0.02 (p=0.007) | 0.55 | +0.02 (p=0.312) | +0.03 (p=0.004) |
| fs_change | +0.65 | +0.13* | +0.20* | +0.13* | +0.17* | +0.17 (p=0.000) | 0.89 | -0.07 (p=0.007) | +0.05 (p=0.000) |
| fs_change_r | +0.77 | +0.17* | +0.20* | +0.23* | +0.20* | +0.22 (p=0.000) | 0.87 | -0.02 (p=0.341) | +0.10 (p=0.000) |

### A. The same score read inside (jump - control, pooled cuts)

Mean over the layers of each third, and the layer with the largest |difference| (its value). The last block is the output.

| lens | format | early | middle | late | max layer (value, p) |
|---|---|---|---|---|---|
| logit | wait | +0.00 | +0.25 | +0.48 | L50 (+0.60, p=0.000) |
| logit | q_surprise | +0.00 | +0.16 | +0.12 | L25 (+0.32, p=0.000) |
| logit | q_change | +0.02 | +0.11 | -0.07 | L26 (+0.28, p=0.000) |
| logit | q_same | +0.00 | +0.16 | +0.41 | L42 (+0.53, p=0.000) |
| logit | q_ab | -0.01 | +0.03 | +0.05 | L52 (+0.11, p=0.008) |
| logit | q_ba | +0.01 | +0.10 | +0.12 | L58 (+0.23, p=0.000) |
| logit | fs_change | +0.00 | +0.71 | +0.69 | L39 (+1.10, p=0.000) |
| logit | fs_change_r | +0.01 | +0.70 | +0.77 | L39 (+1.05, p=0.000) |
| jlens | wait | +0.02 | +0.25 | +0.39 | L39 (+0.60, p=0.000) |
| jlens | q_surprise | -0.01 | +0.10 | +0.11 | L60 (+0.20, p=0.000) |
| jlens | q_change | +0.01 | +0.07 | -0.06 | L26 (+0.17, p=0.000) |
| jlens | q_same | +0.01 | +0.13 | +0.33 | L42 (+0.40, p=0.000) |
| jlens | q_ab | +0.00 | +0.00 | +0.05 | L50 (+0.10, p=0.001) |
| jlens | q_ba | -0.00 | +0.06 | +0.06 | L41 (+0.14, p=0.000) |
| jlens | fs_change | +0.01 | +0.37 | +0.46 | L41 (+0.73, p=0.000) |
| jlens | fs_change_r | +0.01 | +0.33 | +0.53 | L41 (+0.66, p=0.000) |

### A. Probes: is the jump decodable at the last prompt token?

Difference-of-means direction fitted on the other 19 streams; AUC of held-out scores (jump vs control, e in {1, 2, 4, 8, 32}), and trained on those cuts and tested at e = 0. Middle probe layer and best.

| format | AUC middle layer | best layer (AUC) | e=0 AUC, best layer | output AUC |
|---|---|---|---|---|
| comma | 1.00 (L32) | L20 (1.00) | L8 (1.00) | 0.97 |
| wait | 0.89 (L32) | L20 (1.00) | L16 (0.98) | 0.80 |
| q_surprise | 0.99 (L32) | L24 (1.00) | L24 (0.93) | 0.64 |
| q_change | 0.96 (L32) | L24 (0.98) | L24 (0.83) | 0.47 |
| q_same | 0.95 (L32) | L24 (1.00) | L24 (0.91) | 0.70 |
| q_ab | 0.89 (L32) | L28 (0.90) | L28 (0.73) | 0.56 |
| q_ba | 0.90 (L32) | L63 (0.95) | L28 (0.76) | 0.55 |
| fs_change | 1.00 (L32) | L16 (1.00) | L16 (1.00) | 0.89 |
| fs_change_r | 1.00 (L32) | L24 (1.00) | L16 (1.00) | 0.87 |

### D. The 2 x 2 change test (history low / high x recent numbers low / high)

Cells: LL = low throughout (the up control), LH = up jump, HL = down jump, HH = high throughout. Change = (LH + HL - LL - HH) / 2, the part of the score a reader of the recent numbers alone cannot produce; recent = (LH + HH - LL - HL) / 2; history = (HL + HH - LL - LH) / 2. Output score at the last prompt token, pooled cuts (e = 0 separately); sign-flip p over streams. AUC: change cells vs the others; same-tail AUC: up jump vs high throughout (identical recent numbers, different history).

| format | change (pooled) | change e=0 | recent | history | AUC change | same-tail AUC (LH vs HH) |
|---|---|---|---|---|---|---|
| comma | +35.05 (p=0.000) | -19.94 (p=0.000) | +206.11 (p=0.000) | +182.85 (p=0.000) | 0.52 | 0.15 |
| wait | +0.22 (p=0.000) | +0.12 (p=0.000) | +0.01 (p=0.398) | +0.06 (p=0.433) | 0.76 | 0.68 |
| q_surprise | +0.05 (p=0.000) | -0.01 (p=0.413) | +0.01 (p=0.368) | +0.25 (p=0.000) | 0.58 | 0.11 |
| q_change | +0.01 (p=0.040) | +0.01 (p=0.134) | -0.02 (p=0.000) | +0.02 (p=0.215) | 0.52 | 0.45 |
| q_same | +0.06 (p=0.000) | +0.01 (p=0.049) | +0.05 (p=0.000) | -0.10 (p=0.003) | 0.63 | 0.78 |
| q_ab | +0.04 (p=0.000) | +0.01 (p=0.151) | -0.01 (p=0.376) | +0.08 (p=0.003) | 0.57 | 0.40 |
| q_ba | +0.04 (p=0.000) | +0.02 (p=0.004) | -0.01 (p=0.014) | +0.17 (p=0.000) | 0.56 | 0.27 |
| fs_change | +0.05 (p=0.000) | +0.03 (p=0.062) | +0.12 (p=0.000) | +0.03 (p=0.055) | 0.61 | 0.53 |
| fs_change_r | +0.09 (p=0.000) | +0.09 (p=0.000) | +0.12 (p=0.000) | +0.11 (p=0.000) | 0.65 | 0.48 |

**Inside** (change effect by layer third, pooled cuts; max layer and its p)

| lens | format | early | middle | late | max layer (value, p) |
|---|---|---|---|---|---|
| logit | wait | +0.01 | +0.32 | +0.43 | L39 (+0.58, p=0.000) |
| logit | q_surprise | -0.01 | +0.10 | +0.07 | L27 (+0.26, p=0.000) |
| logit | q_change | +0.01 | +0.10 | -0.02 | L26 (+0.23, p=0.000) |
| logit | q_same | +0.02 | +0.10 | +0.25 | L42 (+0.33, p=0.000) |
| logit | q_ab | -0.01 | +0.07 | +0.10 | L38 (+0.17, p=0.000) |
| logit | q_ba | +0.01 | +0.11 | +0.10 | L58 (+0.17, p=0.000) |
| logit | fs_change | -0.00 | +0.30 | +0.08 | L27 (+0.65, p=0.000) |
| logit | fs_change_r | -0.01 | +0.36 | +0.21 | L27 (+0.71, p=0.000) |
| jlens | wait | +0.02 | +0.25 | +0.33 | L39 (+0.55, p=0.000) |
| jlens | q_surprise | -0.01 | +0.11 | +0.07 | L27 (+0.22, p=0.000) |
| jlens | q_change | +0.01 | +0.09 | -0.01 | L26 (+0.17, p=0.000) |
| jlens | q_same | +0.01 | +0.04 | +0.20 | L58 (+0.25, p=0.000) |
| jlens | q_ab | +0.00 | +0.03 | +0.07 | L52 (+0.11, p=0.000) |
| jlens | q_ba | +0.00 | +0.05 | +0.05 | L58 (+0.09, p=0.000) |
| jlens | fs_change | +0.01 | +0.18 | +0.07 | L36 (+0.35, p=0.000) |
| jlens | fs_change_r | +0.01 | +0.24 | +0.18 | L27 (+0.38, p=0.000) |

**Probes at the last prompt token** (difference of means on 19 streams, held-out stream scored)

Change probe: change cells vs the others (pooled cuts; e = 0 tested with the same direction). Up-to-down transfer: the up direction (up jump - up control) scored on the down pair (down jump vs high throughout): near 1 = it reads change, near 0 = it reads the recent numbers. Same-tail: the up direction on up jump vs high throughout (above 0.5 = it uses the history).

| format | change AUC, middle layer | change AUC, best layer | change e=0, best | up-to-down transfer, middle (min, max over layers) | same-tail, middle |
|---|---|---|---|---|---|
| comma | 0.99 (L32) | L20 (1.00) | L8 (1.00) | 0.96 (0.16, 1.00) | 0.88 |
| wait | 0.87 (L32) | L20 (1.00) | L12 (0.98) | 0.77 (0.36, 0.99) | 0.21 |
| q_surprise | 0.93 (L32) | L20 (0.96) | L28 (0.84) | 0.77 (0.36, 0.93) | 0.28 |
| q_change | 0.93 (L32) | L28 (0.97) | L20 (0.80) | 0.86 (0.36, 0.92) | 0.26 |
| q_same | 0.80 (L32) | L20 (0.99) | L20 (0.87) | 0.79 (0.35, 0.92) | 0.23 |
| q_ab | 0.81 (L32) | L24 (0.92) | L24 (0.77) | 0.73 (0.27, 0.75) | 0.52 |
| q_ba | 0.88 (L32) | L24 (0.90) | L28 (0.74) | 0.67 (0.26, 0.70) | 0.40 |
| fs_change | 1.00 (L32) | L40 (1.00) | L24 (1.00) | 1.00 (0.35, 1.00) | 1.00 |
| fs_change_r | 1.00 (L32) | L28 (1.00) | L16 (1.00) | 1.00 (0.36, 1.00) | 1.00 |

### B. Injection at one layer, every tail token (e = 4)

G = natural gap (jump - control, no change). Shares of G: transplant = diff_pos +1 on control; removal = diff_pos -1 on jump; proj = setting the diff direction's component to the other arm's mean (on control, on jump). x4 = effect of diff_pos +4 on control, against a random direction of the same norm (randm x4). steer = surprise-word lens direction at c = +1 (control words in brackets). Effects in the score's units, mean over streams; * p < 0.05 (sign-flip). Where |G| is small (the question formats) the shares are noisy ratios; read the x4 column against randm x4 there.


**comma** (score = predicted mean; the sd is in the csv)

| layer | G | transplant | removal | proj control | proj jump | x4 | randm x4 | steer surprise (ctrlw) |
|---|---|---|---|---|---|---|---|---|
| 16 | +273.94 | +0.00 | +0.09* | +0.01 | +0.09* | +6.18 | -3.63 | -5.77* (-4.68*) |
| 24 | +273.94 | +0.09* | +0.93* | +0.09* | +0.94* | +344.42* | -12.48* | -1.35 (-0.25) |
| 32 | +273.94 | +0.34* | +0.96* | +0.34* | +0.96* | +270.87* | -23.87* | +1.02 (-0.68) |
| 36 | +273.94 | +0.51* | +0.96* | +0.51* | +0.96* | +397.58* | -4.36 | +3.40* (+1.59) |
| 40 | +273.94 | +0.64* | +0.97* | +0.64* | +0.97* | +420.64* | +16.70* | +5.44* (+4.03*) |
| 44 | +273.94 | +0.71* | +0.98* | +0.71* | +0.97* | +415.54* | +18.00* | +6.02* (+4.71*) |
| 48 | +273.94 | +0.87* | +0.98* | +0.86* | +0.98* | +431.42* | -8.68* | +9.81* (+7.14*) |
| 56 | +273.94 | +0.89* | +0.98* | +0.89* | +0.98* | +436.30* | -60.79* | +25.99* (+23.73*) |

**wait**

| layer | G | transplant | removal | proj control | proj jump | x4 | randm x4 | steer surprise (ctrlw) |
|---|---|---|---|---|---|---|---|---|
| 16 | +0.29 | +0.59* | +0.61* | +0.58* | +0.59* | +0.79* | -0.08* | -0.28* (-0.77*) |
| 24 | +0.29 | +0.90* | +0.84* | +0.86* | +0.83* | +1.26* | +0.16* | +0.45* (-1.07*) |
| 32 | +0.29 | +1.01* | +0.98* | +0.94* | +0.99* | +1.26* | +0.10* | -0.59* (-2.55*) |
| 36 | +0.29 | +1.01* | +0.99* | +0.96* | +1.00* | +1.23* | +0.51* | -0.80* (-2.81*) |
| 40 | +0.29 | +1.01* | +0.99* | +0.97* | +1.00* | +1.22* | -0.03 | -0.55* (-3.14*) |
| 44 | +0.29 | +1.00* | +1.01* | +1.00* | +1.02* | +1.17* | +0.14* | +0.18* (-2.78*) |
| 48 | +0.29 | +0.96* | +0.98* | +0.93* | +0.97* | +1.14* | -0.38* | +0.56* (-2.74*) |
| 56 | +0.29 | +0.95* | +0.98* | +0.94* | +0.96* | +0.88* | -0.07* | +2.68* (-2.19*) |

**q_surprise**

| layer | G | transplant | removal | proj control | proj jump | x4 | randm x4 | steer surprise (ctrlw) |
|---|---|---|---|---|---|---|---|---|
| 16 | +0.08 | +0.36* | +0.25* | +0.24* | +0.52* | -0.04 | -0.18* | -2.12* (-2.56*) |
| 24 | +0.08 | +0.96* | +0.65* | +0.76* | +0.88* | +0.26* | -0.08* | -0.77* (-1.04*) |
| 32 | +0.08 | +0.88* | +0.73* | +0.80* | +0.69* | +0.23* | -0.07* | -1.55* (-1.42*) |
| 36 | +0.08 | +0.76* | +0.81* | +0.80* | +0.80* | +0.25* | +0.20* | -1.45* (-1.38*) |
| 40 | +0.08 | +1.00* | +0.81* | +1.07* | +0.73* | +0.26* | +0.06* | -1.74* (-1.64*) |
| 44 | +0.08 | +1.04* | +0.96* | +0.96* | +0.85* | +0.26* | -0.06* | -0.87* (-0.67*) |
| 48 | +0.08 | +0.84* | +0.77* | +0.92* | +0.77* | +0.25* | +0.12* | -0.55* (-0.38*) |
| 56 | +0.08 | +0.88* | +0.77* | +1.00* | +0.77* | +0.26* | -0.02 | -0.84* (-0.52*) |

**q_change**

| layer | G | transplant | removal | proj control | proj jump | x4 | randm x4 | steer surprise (ctrlw) |
|---|---|---|---|---|---|---|---|---|
| 16 | -0.02 | +0.61 | +0.80 | +0.41 | -0.21 | -0.09* | -0.08* | -0.22* (+0.27*) |
| 24 | -0.02 | +1.62* | +0.80 | +1.22* | +1.81* | -0.08* | -0.00 | -0.33* (-0.97*) |
| 32 | -0.02 | +1.62* | +1.01 | +1.42* | +1.41* | -0.06* | -0.07* | -0.79* (-0.64*) |
| 36 | -0.02 | +1.60* | +1.40* | +1.82* | -0.01 | -0.06* | -0.06* | -0.72* (-0.63*) |
| 40 | -0.02 | +1.41* | +0.80 | +1.61* | -0.02 | -0.06* | -0.03* | -0.91* (-1.33*) |
| 44 | -0.02 | +1.21* | +0.99 | +1.21* | +0.79 | -0.08* | -0.05* | -0.79* (-1.03*) |
| 48 | -0.02 | +1.41* | +0.60 | +1.82* | +0.19 | -0.07* | -0.01 | -0.58* (-0.96*) |
| 56 | -0.02 | +0.60 | +0.19 | +1.21* | +0.79 | -0.07* | -0.04* | -0.69* (-0.41*) |

## olmo-3.1-32b-it

### A. What the model says (output, last prompt token)

Change score = log P(answer meaning 'changed') - log P(other answer); wait: surprise - normal; comma: predicted mean. Jump - control (sign-flip p). Pooled = e in {1, 2, 4, 8, 32}. AUC: jump vs control, pooled cuts. Outlier: extreme - typical last number.

| format | control level | e=0 | e=1 | e=4 | e=32 | pooled | AUC | outlier extreme | outlier moderate |
|---|---|---|---|---|---|---|---|---|---|
| q_surprise | -1.03 | +0.30* | +0.39* | +0.47* | +0.41* | +0.42 (p=0.000) | 0.97 | +0.31 (p=0.000) | +0.09 (p=0.003) |
| q_change | -0.11 | +0.11* | +0.22* | +0.18* | +0.15* | +0.20 (p=0.000) | 0.71 | +0.17 (p=0.001) | +0.05 (p=0.026) |
| q_same | -0.47 | +0.57* | +0.61* | +0.71* | +0.54* | +0.65 (p=0.000) | 0.97 | +0.40 (p=0.000) | +0.20 (p=0.000) |
| q_ab | -1.13 | +0.29* | +0.42* | +0.66* | +0.66* | +0.61 (p=0.000) | 0.99 | +0.32 (p=0.000) | +0.11 (p=0.000) |
| q_ba | +0.12 | +0.23* | +0.23* | +0.27* | +0.20* | +0.23 (p=0.000) | 0.93 | +0.21 (p=0.000) | +0.08 (p=0.002) |

### A. The same score read inside (jump - control, pooled cuts)

Mean over the layers of each third, and the layer with the largest |difference| (its value). The last block is the output.

| lens | format | early | middle | late | max layer (value, p) |
|---|---|---|---|---|---|
| logit | q_surprise | -0.01 | +1.22 | +1.71 | L42 (+2.29, p=0.000) |
| logit | q_change | +0.02 | +0.78 | +0.81 | L27 (+1.41, p=0.000) |
| logit | q_same | +0.02 | +0.63 | +1.90 | L53 (+2.85, p=0.000) |
| logit | q_ab | +0.01 | +0.27 | +1.63 | L53 (+2.13, p=0.000) |
| logit | q_ba | -0.01 | +0.44 | +0.15 | L32 (+1.03, p=0.000) |

### A. Probes: is the jump decodable at the last prompt token?

Difference-of-means direction fitted on the other 19 streams; AUC of held-out scores (jump vs control, e in {1, 2, 4, 8, 32}), and trained on those cuts and tested at e = 0. Middle probe layer and best.

| format | AUC middle layer | best layer (AUC) | e=0 AUC, best layer | output AUC |
|---|---|---|---|---|
| q_surprise | 1.00 (L32) | L20 (1.00) | L20 (1.00) | 0.97 |
| q_change | 0.99 (L32) | L20 (1.00) | L20 (0.99) | 0.71 |
| q_same | 1.00 (L32) | L16 (1.00) | L20 (1.00) | 0.97 |
| q_ab | 1.00 (L32) | L20 (1.00) | L20 (0.99) | 0.99 |
| q_ba | 1.00 (L32) | L20 (1.00) | L20 (0.99) | 0.93 |

### D. The 2 x 2 change test (history low / high x recent numbers low / high)

Cells: LL = low throughout (the up control), LH = up jump, HL = down jump, HH = high throughout. Change = (LH + HL - LL - HH) / 2, the part of the score a reader of the recent numbers alone cannot produce; recent = (LH + HH - LL - HL) / 2; history = (HL + HH - LL - LH) / 2. Output score at the last prompt token, pooled cuts (e = 0 separately); sign-flip p over streams. AUC: change cells vs the others; same-tail AUC: up jump vs high throughout (identical recent numbers, different history).

| format | change (pooled) | change e=0 | recent | history | AUC change | same-tail AUC (LH vs HH) |
|---|---|---|---|---|---|---|
| q_surprise | +0.27 (p=0.000) | +0.24 (p=0.000) | +0.15 (p=0.000) | +0.11 (p=0.000) | 0.85 | 0.69 |
| q_change | +0.08 (p=0.000) | +0.07 (p=0.001) | +0.12 (p=0.000) | +0.08 (p=0.057) | 0.57 | 0.42 |
| q_same | +0.63 (p=0.000) | +0.55 (p=0.000) | +0.03 (p=0.248) | -0.07 (p=0.152) | 0.98 | 0.98 |
| q_ab | +0.65 (p=0.000) | +0.38 (p=0.000) | -0.04 (p=0.026) | -0.15 (p=0.002) | 0.98 | 1.00 |
| q_ba | +0.14 (p=0.000) | +0.22 (p=0.000) | +0.09 (p=0.000) | +0.01 (p=0.682) | 0.82 | 0.82 |

**Inside** (change effect by layer third, pooled cuts; max layer and its p)

| lens | format | early | middle | late | max layer (value, p) |
|---|---|---|---|---|---|
| logit | q_surprise | -0.01 | +0.86 | +1.14 | L42 (+1.53, p=0.000) |
| logit | q_change | +0.00 | +0.37 | +0.29 | L27 (+0.98, p=0.000) |
| logit | q_same | +0.02 | +0.53 | +1.77 | L54 (+2.69, p=0.000) |
| logit | q_ab | +0.01 | +0.26 | +1.65 | L48 (+2.14, p=0.000) |
| logit | q_ba | -0.00 | +0.35 | +0.05 | L32 (+0.79, p=0.000) |

**Probes at the last prompt token** (difference of means on 19 streams, held-out stream scored)

Change probe: change cells vs the others (pooled cuts; e = 0 tested with the same direction). Up-to-down transfer: the up direction (up jump - up control) scored on the down pair (down jump vs high throughout): near 1 = it reads change, near 0 = it reads the recent numbers. Same-tail: the up direction on up jump vs high throughout (above 0.5 = it uses the history).

| format | change AUC, middle layer | change AUC, best layer | change e=0, best | up-to-down transfer, middle (min, max over layers) | same-tail, middle |
|---|---|---|---|---|---|
| q_surprise | 0.99 (L32) | L20 (1.00) | L20 (0.99) | 0.95 (0.42, 1.00) | 0.95 |
| q_change | 1.00 (L32) | L20 (1.00) | L16 (0.98) | 0.99 (0.42, 1.00) | 0.99 |
| q_same | 1.00 (L32) | L20 (1.00) | L16 (0.98) | 1.00 (0.43, 1.00) | 0.99 |
| q_ab | 1.00 (L32) | L20 (1.00) | L32 (0.99) | 1.00 (0.40, 1.00) | 1.00 |
| q_ba | 1.00 (L32) | L20 (1.00) | L20 (0.98) | 1.00 (0.40, 1.00) | 1.00 |

### B. Injection at one layer, every tail token (e = 4)

G = natural gap (jump - control, no change). Shares of G: transplant = diff_pos +1 on control; removal = diff_pos -1 on jump; proj = setting the diff direction's component to the other arm's mean (on control, on jump). x4 = effect of diff_pos +4 on control, against a random direction of the same norm (randm x4). steer = surprise-word lens direction at c = +1 (control words in brackets). Effects in the score's units, mean over streams; * p < 0.05 (sign-flip). Where |G| is small (the question formats) the shares are noisy ratios; read the x4 column against randm x4 there.


**q_surprise**

| layer | G | transplant | removal | proj control | proj jump | x4 | randm x4 | steer surprise (ctrlw) |
|---|---|---|---|---|---|---|---|---|
| 16 | +0.47 | +0.57* | +0.65* | +0.61* | +0.63* | +0.81* | -0.24* | - (-) |
| 24 | +0.47 | +1.11* | +1.13* | +1.08* | +1.08* | +2.13* | -0.02 | - (-) |
| 32 | +0.47 | +0.97* | +0.97* | +1.03* | +0.95* | +1.67* | -0.36* | - (-) |
| 36 | +0.47 | +1.01* | +0.99* | +1.01* | +0.93* | +1.50* | -0.04 | - (-) |
| 40 | +0.47 | +1.00* | +1.03* | +1.00* | +0.96* | +1.50* | +0.33* | - (-) |
| 44 | +0.47 | +1.00* | +0.97* | +1.04* | +0.96* | +1.53* | -0.13* | - (-) |
| 48 | +0.47 | +1.04* | +0.97* | +1.04* | +1.01* | +1.54* | -0.37* | - (-) |
| 56 | +0.47 | +1.05* | +1.01* | +1.01* | +1.01* | +1.50* | +0.03 | - (-) |

**q_change**

| layer | G | transplant | removal | proj control | proj jump | x4 | randm x4 | steer surprise (ctrlw) |
|---|---|---|---|---|---|---|---|---|
| 16 | +0.16 | +1.00* | +1.44* | +0.88* | +1.36* | +0.70* | -0.39* | - (-) |
| 24 | +0.16 | +1.24* | +1.08* | +1.04* | +1.16* | +0.98* | -0.01 | - (-) |
| 32 | +0.16 | +1.04* | +1.00* | +1.00* | +0.88* | +0.66* | -0.40* | - (-) |
| 36 | +0.16 | +1.04* | +0.96* | +0.96* | +0.92* | +0.75* | +0.28* | - (-) |
| 40 | +0.16 | +1.08* | +0.88* | +1.00* | +0.88* | +0.84* | +0.34* | - (-) |
| 44 | +0.16 | +1.12* | +1.04* | +1.04* | +0.96* | +0.96* | -0.15* | - (-) |
| 48 | +0.16 | +1.04* | +0.96* | +1.00* | +0.92* | +0.93* | -0.23* | - (-) |
| 56 | +0.16 | +1.00* | +0.96* | +0.92* | +0.92* | +0.79* | -0.11* | - (-) |

(* p < 0.05, sign-flip over the 20 streams, not corrected for the number of cells.)
