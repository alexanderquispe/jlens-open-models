# Experiment 11, step 3: injecting the pure change direction

Cells: LL low throughout, LH up jump, HL down jump, HH high throughout (cut e = 4). The change direction is (LH + HL - LL - HH) / 2 of the residual stream at each tail token, fitted without the test stream. Report score = log P(changed answer) - log P(other answer).

## llama3.1-8b

Natural effects on the report score (no injection, nats; change = (LH + HL - LL - HH) / 2):

| question | change (se) | recent | history |
|---|---|---|---|
| q_surprise | +0.01 (0.01) | +0.03 | -0.10 |
| q_change | +0.02 (0.01) | -0.00 | +0.13 |
| q_same | +0.04 (0.01) | +0.07 | +0.02 |
| q_ab | +0.02 (0.02) | -0.02 | +1.02 |
| q_ba | +0.05 (0.01) | +0.02 | -0.22 |

**+1 x change direction on the unchanged cells (LL, HH): the transplant.** Effect in nats (t over streams), by layer:

| question | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|
| q_surprise | +0.02 (+3.1) | +0.03 (+3.4) | +0.02 (+3.4) | +0.02 (+2.4) | +0.03 (+3.9) | +0.02 (+2.0) |
| q_change | +0.02 (+2.3) | +0.02 (+2.5) | +0.02 (+1.9) | +0.01 (+2.0) | +0.02 (+2.4) | +0.02 (+2.3) |
| q_same | -0.06 (-4.5) | -0.02 (-1.4) | +0.04 (+2.7) | +0.01 (+1.3) | +0.02 (+2.4) | +0.02 (+2.4) |
| q_ab | +0.04 (+3.9) | +0.02 (+1.6) | +0.03 (+3.7) | +0.01 (+1.5) | +0.01 (+1.3) | -0.01 (-1.3) |
| q_ba | -0.01 (-0.8) | -0.01 (-0.5) | +0.01 (+0.5) | +0.03 (+2.5) | +0.05 (+4.7) | +0.06 (+5.0) |

Layers where all 5 questions move toward "changed": L16, L20, L24.

**+4 x change direction on the unchanged cells.** Effect in nats (t over streams), by layer:

| question | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|
| q_surprise | +0.07 (+7.2) | +0.10 (+10.5) | +0.09 (+14.6) | +0.09 (+11.8) | +0.09 (+9.3) | +0.08 (+9.8) |
| q_change | +0.11 (+12.0) | +0.14 (+21.6) | +0.10 (+10.0) | +0.10 (+12.8) | +0.09 (+10.0) | +0.08 (+7.3) |
| q_same | -0.14 (-10.3) | +0.02 (+1.9) | +0.08 (+5.6) | +0.07 (+5.6) | +0.05 (+4.1) | +0.07 (+4.8) |
| q_ab | +0.12 (+9.5) | +0.01 (+0.5) | +0.04 (+3.5) | -0.07 (-6.8) | -0.04 (-3.7) | -0.03 (-2.8) |
| q_ba | +0.02 (+1.6) | +0.04 (+3.2) | +0.08 (+5.6) | +0.14 (+10.5) | +0.13 (+9.5) | +0.18 (+13.8) |

Layers where all 5 questions move toward "changed": L12, L16.

**-1 x change direction on the changed cells (LH, HL): the removal.** Effect in nats (t over streams), by layer:

| question | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|
| q_surprise | -0.02 (-2.4) | -0.02 (-2.8) | -0.01 (-1.5) | -0.01 (-2.0) | -0.01 (-1.8) | -0.01 (-2.7) |
| q_change | -0.03 (-3.9) | -0.04 (-4.4) | -0.04 (-6.6) | -0.03 (-3.8) | -0.03 (-3.9) | -0.02 (-2.7) |
| q_same | +0.03 (+3.5) | -0.02 (-2.1) | -0.04 (-3.9) | -0.02 (-2.2) | -0.04 (-3.9) | -0.02 (-2.1) |
| q_ab | -0.05 (-4.4) | -0.03 (-1.8) | -0.03 (-2.9) | +0.00 (+0.6) | -0.00 (-0.5) | -0.00 (-0.2) |
| q_ba | -0.01 (-1.0) | -0.00 (-0.3) | -0.01 (-0.7) | -0.04 (-4.0) | -0.03 (-2.6) | -0.04 (-3.8) |

Layers where all 5 questions move toward "unchanged": L12, L16, L24, L28.

**Projection to the other class's mean, changed cells.** Effect in nats (t over streams), by layer:

| question | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|
| q_surprise | -0.02 (-2.6) | -0.02 (-2.8) | -0.03 (-5.4) | -0.02 (-2.3) | -0.02 (-2.9) | -0.02 (-2.7) |
| q_change | -0.03 (-3.3) | -0.04 (-4.2) | -0.04 (-4.1) | -0.04 (-4.9) | -0.02 (-3.0) | -0.03 (-4.0) |
| q_same | +0.03 (+3.5) | -0.02 (-2.2) | -0.03 (-2.8) | -0.04 (-3.6) | -0.03 (-3.3) | -0.04 (-4.0) |
| q_ab | -0.02 (-1.9) | -0.02 (-1.5) | -0.03 (-2.9) | +0.01 (+1.6) | -0.00 (-0.4) | -0.01 (-0.8) |
| q_ba | -0.01 (-0.6) | -0.01 (-0.7) | -0.00 (-0.3) | -0.03 (-2.1) | -0.03 (-2.2) | -0.04 (-4.1) |

Layers where all 5 questions move toward "changed": none.

**Controls, all four cells** (mean effect on the report score over the questions, and the largest absolute effect on one question; nats):

| condition | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|
| +1 change | +0.01 (max 0.06) | +0.01 (max 0.03) | +0.02 (max 0.02) | +0.01 (max 0.03) | +0.02 (max 0.04) | +0.02 (max 0.05) |
| +4 change | +0.04 (max 0.13) | +0.06 (max 0.14) | +0.07 (max 0.11) | +0.06 (max 0.13) | +0.07 (max 0.13) | +0.08 (max 0.16) |
| +4 random 0 (norm of 4 change) | +0.09 (max 0.56) | +0.07 (max 0.14) | +0.00 (max 0.18) | -0.03 (max 0.14) | -0.02 (max 0.05) | +0.02 (max 0.07) |
| +4 random 1 | -0.02 (max 0.16) | -0.01 (max 0.13) | -0.01 (max 0.14) | +0.00 (max 0.07) | -0.01 (max 0.08) | +0.01 (max 0.11) |
| +1 recent | +0.00 (max 0.04) | +0.01 (max 0.06) | +0.01 (max 0.05) | +0.01 (max 0.04) | +0.01 (max 0.05) | +0.01 (max 0.05) |
| +4 recent | +0.03 (max 0.15) | +0.06 (max 0.22) | +0.04 (max 0.22) | +0.05 (max 0.20) | +0.05 (max 0.21) | +0.05 (max 0.21) |
| +1 history | +0.14 (max 0.87) | +0.16 (max 0.78) | +0.14 (max 0.89) | +0.15 (max 0.98) | +0.15 (max 1.01) | +0.16 (max 1.04) |
| +4 history | -0.34 (max 1.31) | +0.21 (max 0.65) | +0.33 (max 2.18) | +0.39 (max 2.75) | +0.43 (max 3.11) | +0.58 (max 3.73) |

A direction that only shifts the answer bias moves q_ab and q_ba, or q_surprise and q_same, in opposite directions on this score, so its mean over the questions is near 0 while its largest effect is not.

**surprise_words after "wait"** (nats; natural change +0.01, recent -0.09, history -0.25). Effect on all cells (t):

| condition | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|
| +1 change | +0.07 (+15.8) | +0.03 (+7.6) | +0.02 (+5.4) | +0.02 (+4.9) | +0.02 (+4.0) | +0.01 (+3.6) |
| +4 change | +0.21 (+20.9) | +0.13 (+18.1) | +0.05 (+8.7) | +0.06 (+13.4) | +0.06 (+12.2) | +0.11 (+18.7) |
| +1 recent | -0.08 (-16.3) | -0.09 (-29.9) | -0.08 (-18.5) | -0.08 (-19.1) | -0.07 (-14.6) | -0.08 (-19.3) |
| +1 history | -0.30 (-34.1) | -0.25 (-44.1) | -0.26 (-47.8) | -0.21 (-43.5) | -0.19 (-40.8) | -0.21 (-53.1) |
| +4 random 0 | +0.08 (+13.2) | -0.00 (-1.0) | -0.37 (-75.1) | -0.08 (-18.7) | +0.00 (+0.3) | -0.16 (-43.5) |

**int_mean after "comma"** (numbers; natural change -3.03, recent +217.41, history +169.56). Effect on all cells (t):

| condition | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|
| +1 change | -3.50 (-1.9) | -11.31 (-2.8) | -13.25 (-5.0) | -11.25 (-7.2) | -6.41 (-5.8) | +1.34 (+0.9) |
| +4 change | -24.88 (-6.0) | -98.04 (-13.5) | -77.36 (-13.2) | -55.49 (-10.7) | -31.09 (-7.5) | -4.36 (-1.1) |
| +1 recent | +4.05 (+7.2) | +2.48 (+4.7) | +22.63 (+38.5) | +52.37 (+37.5) | +92.07 (+47.3) | +123.88 (+50.6) |
| +1 history | +0.46 (+0.7) | +0.87 (+0.9) | +2.06 (+2.7) | +10.50 (+10.7) | +27.51 (+31.1) | +95.54 (+46.0) |
| +4 random 0 | +4.62 (+3.7) | -15.05 (-7.9) | +6.05 (+3.5) | +2.84 (+3.4) | -13.96 (-12.6) | -40.26 (-34.8) |

**int_sd after "comma"** (numbers; natural change +115.53, recent -6.68, history +1.32). Effect on all cells (t):

| condition | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|
| +1 change | +3.52 (+7.3) | +5.52 (+4.8) | +8.48 (+5.3) | +21.31 (+25.7) | +24.25 (+39.4) | +32.04 (+38.0) |
| +4 change | +33.73 (+26.0) | +59.86 (+21.6) | +53.75 (+15.9) | +113.97 (+35.3) | +95.38 (+28.7) | +98.81 (+65.9) |
| +1 recent | -0.04 (-0.1) | -0.25 (-1.0) | -1.43 (-2.2) | +0.91 (+0.6) | -1.29 (-0.6) | -11.83 (-3.8) |
| +1 history | +0.33 (+1.5) | +3.31 (+12.3) | +0.99 (+3.1) | +2.40 (+5.9) | +0.03 (+0.0) | -3.20 (-1.3) |
| +4 random 0 | +10.88 (+15.1) | +3.13 (+3.2) | -18.01 (-21.1) | -3.43 (-7.8) | +3.43 (+6.1) | +11.31 (+8.5) |

Directions at the tail of q_surprise (mean over tail tokens): norm of change / recent / history, and cosine(change, recent): L8: 0.3 / 0.2 / 1.4, +0.21; L12: 0.4 / 0.3 / 1.7, +0.32; L16: 0.6 / 0.5 / 2.4, +0.32; L20: 0.9 / 0.7 / 3.4, +0.37; L24: 1.3 / 0.9 / 4.8, +0.40; L28: 1.8 / 1.4 / 6.8, +0.40.

20 streams; 216 s on NVIDIA H200.

## llama3.1-8b-it

Natural effects on the report score (no injection, nats; change = (LH + HL - LL - HH) / 2):

| question | change (se) | recent | history |
|---|---|---|---|
| q_surprise | +0.12 (0.02) | +0.12 | -0.57 |
| q_change | +0.17 (0.02) | -0.02 | -0.66 |
| q_same | +0.86 (0.03) | +0.17 | -0.30 |
| q_ab | +0.64 (0.03) | -0.08 | +0.45 |
| q_ba | +0.41 (0.02) | -0.00 | +0.30 |

**+1 x change direction on the unchanged cells (LL, HH): the transplant.** Effect in nats (t over streams), by layer:

| question | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|
| q_surprise | +0.06 (+3.8) | +0.17 (+12.7) | +0.12 (+10.4) | +0.15 (+12.2) | +0.11 (+7.7) | +0.12 (+13.8) |
| q_change | -0.06 (-3.9) | +0.12 (+9.3) | +0.14 (+14.0) | +0.17 (+12.1) | +0.18 (+12.9) | +0.17 (+16.5) |
| q_same | -0.01 (-0.7) | +0.62 (+45.9) | +0.85 (+81.9) | +0.85 (+43.7) | +0.86 (+62.6) | +0.84 (+45.8) |
| q_ab | +0.02 (+1.3) | +0.33 (+25.8) | +0.65 (+44.5) | +0.67 (+55.4) | +0.65 (+61.7) | +0.66 (+57.1) |
| q_ba | -0.03 (-2.8) | +0.20 (+12.9) | +0.41 (+35.1) | +0.38 (+27.1) | +0.39 (+35.5) | +0.37 (+26.6) |

Layers where all 5 questions move toward "changed": L12, L16, L20, L24, L28.

**+4 x change direction on the unchanged cells.** Effect in nats (t over streams), by layer:

| question | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|
| q_surprise | +0.29 (+18.6) | +1.03 (+46.9) | +0.67 (+44.9) | +0.60 (+40.7) | +0.52 (+45.9) | +0.51 (+36.5) |
| q_change | -0.11 (-6.5) | +1.00 (+40.5) | +0.67 (+61.1) | +0.67 (+48.9) | +0.63 (+51.9) | +0.65 (+49.6) |
| q_same | -0.05 (-4.0) | +2.96 (+79.0) | +3.45 (+173.2) | +4.49 (+99.2) | +4.34 (+124.1) | +4.27 (+108.6) |
| q_ab | +0.04 (+2.6) | +1.69 (+64.4) | +1.69 (+45.1) | +2.69 (+138.1) | +2.89 (+151.8) | +2.84 (+159.5) |
| q_ba | -0.15 (-9.2) | +0.97 (+46.3) | +1.53 (+65.8) | +1.95 (+74.4) | +1.70 (+98.1) | +1.65 (+92.7) |

Layers where all 5 questions move toward "changed": L12, L16, L20, L24, L28.

**-1 x change direction on the changed cells (LH, HL): the removal.** Effect in nats (t over streams), by layer:

| question | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|
| q_surprise | -0.08 (-6.3) | -0.18 (-18.1) | -0.10 (-12.6) | -0.12 (-11.8) | -0.10 (-9.1) | -0.10 (-9.1) |
| q_change | -0.00 (-0.3) | -0.17 (-9.3) | -0.19 (-12.5) | -0.20 (-18.9) | -0.19 (-13.6) | -0.19 (-15.6) |
| q_same | -0.05 (-3.6) | -0.67 (-52.8) | -0.86 (-74.1) | -0.88 (-72.3) | -0.87 (-62.7) | -0.86 (-60.3) |
| q_ab | -0.04 (-4.8) | -0.32 (-18.9) | -0.65 (-56.6) | -0.63 (-39.9) | -0.62 (-50.2) | -0.63 (-45.0) |
| q_ba | +0.00 (+0.0) | -0.22 (-14.9) | -0.42 (-45.6) | -0.40 (-42.1) | -0.41 (-26.9) | -0.40 (-30.4) |

Layers where all 5 questions move toward "unchanged": L12, L16, L20, L24, L28.

**Projection to the other class's mean, changed cells.** Effect in nats (t over streams), by layer:

| question | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|
| q_surprise | -0.07 (-6.0) | -0.18 (-15.2) | -0.12 (-9.5) | -0.12 (-9.2) | -0.10 (-8.4) | -0.10 (-9.9) |
| q_change | +0.00 (+0.0) | -0.15 (-10.6) | -0.20 (-14.3) | -0.18 (-11.6) | -0.19 (-10.4) | -0.17 (-8.9) |
| q_same | -0.07 (-7.8) | -0.67 (-39.3) | -0.87 (-22.3) | -0.88 (-21.2) | -0.88 (-20.5) | -0.89 (-19.2) |
| q_ab | -0.01 (-0.9) | -0.33 (-21.9) | -0.65 (-25.6) | -0.63 (-21.0) | -0.64 (-18.3) | -0.63 (-19.1) |
| q_ba | -0.01 (-0.5) | -0.22 (-16.2) | -0.41 (-21.6) | -0.39 (-18.5) | -0.41 (-21.6) | -0.40 (-20.0) |

Layers where all 5 questions move toward "changed": none.

**Controls, all four cells** (mean effect on the report score over the questions, and the largest absolute effect on one question; nats):

| condition | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|
| +1 change | +0.01 (max 0.08) | +0.32 (max 0.69) | +0.44 (max 0.87) | +0.47 (max 0.94) | +0.46 (max 0.94) | +0.46 (max 0.92) |
| +4 change | +0.09 (max 0.39) | +1.54 (max 2.97) | +1.45 (max 3.35) | +2.14 (max 4.80) | +2.08 (max 4.57) | +2.04 (max 4.54) |
| +4 random 0 (norm of 4 change) | +0.03 (max 0.21) | +0.08 (max 0.31) | -0.01 (max 0.45) | -0.01 (max 0.18) | +0.04 (max 0.31) | -0.01 (max 0.22) |
| +4 random 1 | -0.09 (max 0.23) | -0.07 (max 0.22) | -0.08 (max 0.62) | +0.06 (max 0.40) | +0.01 (max 0.39) | -0.05 (max 0.47) |
| +1 recent | -0.02 (max 0.04) | +0.02 (max 0.08) | +0.03 (max 0.13) | +0.03 (max 0.12) | +0.04 (max 0.14) | +0.04 (max 0.14) |
| +4 recent | -0.05 (max 0.20) | +0.07 (max 0.27) | +0.15 (max 0.58) | +0.17 (max 0.49) | +0.18 (max 0.58) | +0.18 (max 0.57) |
| +1 history | +0.15 (max 0.66) | +0.03 (max 0.51) | -0.14 (max 0.66) | -0.15 (max 0.71) | -0.16 (max 0.68) | -0.15 (max 0.68) |
| +4 history | +0.17 (max 1.59) | -0.02 (max 1.12) | -0.58 (max 2.76) | -0.74 (max 3.45) | -0.65 (max 2.83) | -0.54 (max 2.70) |

A direction that only shifts the answer bias moves q_ab and q_ba, or q_surprise and q_same, in opposite directions on this score, so its mean over the questions is near 0 while its largest effect is not.

Directions at the tail of q_surprise (mean over tail tokens): norm of change / recent / history, and cosine(change, recent): L8: 0.1 / 0.1 / 0.6, +0.18; L12: 0.2 / 0.1 / 0.8, +0.13; L16: 0.4 / 0.2 / 1.1, +0.11; L20: 0.6 / 0.4 / 1.4, +0.11; L24: 0.9 / 0.5 / 2.1, +0.09; L28: 1.2 / 0.7 / 2.9, +0.09.

20 streams; 186 s on NVIDIA H200.

## olmo-3-32b

Natural effects on the report score (no injection, nats; change = (LH + HL - LL - HH) / 2):

| question | change (se) | recent | history |
|---|---|---|---|
| q_surprise | +0.07 (0.01) | +0.01 | +0.24 |
| q_change | +0.01 (0.01) | -0.03 | +0.02 |
| q_same | +0.07 (0.01) | +0.05 | -0.11 |
| q_ab | +0.05 (0.01) | -0.00 | +0.10 |
| q_ba | +0.04 (0.01) | -0.02 | +0.16 |

**+1 x change direction on the unchanged cells (LL, HH): the transplant.** Effect in nats (t over streams), by layer:

| question | L16 | L24 | L32 | L40 | L48 | L56 |
|---|---|---|---|---|---|---|
| q_surprise | +0.00 (+0.3) | +0.06 (+9.9) | +0.05 (+8.4) | +0.05 (+7.9) | +0.05 (+7.4) | +0.05 (+7.4) |
| q_change | -0.01 (-2.5) | -0.01 (-1.1) | -0.01 (-2.4) | -0.00 (-0.7) | -0.01 (-1.3) | -0.00 (-0.6) |
| q_same | +0.03 (+5.8) | +0.09 (+19.6) | +0.08 (+13.6) | +0.09 (+15.7) | +0.09 (+15.6) | +0.09 (+16.1) |
| q_ab | +0.01 (+2.9) | +0.04 (+13.6) | +0.05 (+16.5) | +0.04 (+14.6) | +0.05 (+14.9) | +0.05 (+15.2) |
| q_ba | +0.05 (+10.5) | +0.05 (+16.1) | +0.05 (+14.3) | +0.05 (+14.6) | +0.05 (+17.4) | +0.05 (+16.0) |

Layers where all 5 questions move toward "changed": none.

**+4 x change direction on the unchanged cells.** Effect in nats (t over streams), by layer:

| question | L16 | L24 | L32 | L40 | L48 | L56 |
|---|---|---|---|---|---|---|
| q_surprise | -0.05 (-4.1) | +0.24 (+26.9) | +0.19 (+29.9) | +0.19 (+21.3) | +0.19 (+28.9) | +0.19 (+24.7) |
| q_change | -0.09 (-15.4) | -0.03 (-4.5) | -0.03 (-3.9) | -0.01 (-0.8) | -0.02 (-2.4) | -0.01 (-1.2) |
| q_same | +0.13 (+11.6) | +0.34 (+48.9) | +0.33 (+50.5) | +0.34 (+55.1) | +0.33 (+55.4) | +0.35 (+44.7) |
| q_ab | -0.01 (-1.0) | +0.17 (+51.4) | +0.18 (+41.9) | +0.17 (+45.8) | +0.18 (+43.7) | +0.19 (+59.2) |
| q_ba | +0.21 (+40.1) | +0.20 (+44.3) | +0.20 (+46.9) | +0.21 (+58.2) | +0.20 (+56.8) | +0.20 (+62.7) |

Layers where all 5 questions move toward "changed": none.

**-1 x change direction on the changed cells (LH, HL): the removal.** Effect in nats (t over streams), by layer:

| question | L16 | L24 | L32 | L40 | L48 | L56 |
|---|---|---|---|---|---|---|
| q_surprise | -0.00 (-0.4) | -0.06 (-10.8) | -0.05 (-9.0) | -0.05 (-9.3) | -0.06 (-9.7) | -0.06 (-9.2) |
| q_change | +0.01 (+2.2) | +0.00 (+0.0) | -0.00 (-0.0) | +0.00 (+0.3) | +0.01 (+2.6) | +0.00 (+0.2) |
| q_same | -0.03 (-5.5) | -0.08 (-14.2) | -0.09 (-17.6) | -0.08 (-13.3) | -0.09 (-14.0) | -0.08 (-13.6) |
| q_ab | -0.01 (-3.0) | -0.04 (-15.7) | -0.04 (-12.7) | -0.04 (-12.8) | -0.04 (-15.1) | -0.04 (-13.1) |
| q_ba | -0.04 (-11.2) | -0.05 (-14.3) | -0.04 (-11.7) | -0.05 (-19.3) | -0.05 (-14.0) | -0.05 (-15.1) |

Layers where all 5 questions move toward "unchanged": L32.

**Projection to the other class's mean, changed cells.** Effect in nats (t over streams), by layer:

| question | L16 | L24 | L32 | L40 | L48 | L56 |
|---|---|---|---|---|---|---|
| q_surprise | -0.00 (-0.2) | -0.05 (-6.9) | -0.05 (-6.5) | -0.05 (-6.4) | -0.05 (-8.2) | -0.05 (-5.6) |
| q_change | +0.01 (+1.9) | +0.00 (+0.5) | -0.00 (-0.8) | -0.00 (-1.0) | +0.00 (+0.2) | +0.00 (+0.5) |
| q_same | -0.03 (-5.9) | -0.08 (-9.3) | -0.08 (-7.9) | -0.08 (-6.9) | -0.08 (-7.2) | -0.08 (-6.0) |
| q_ab | -0.01 (-3.0) | -0.04 (-11.7) | -0.04 (-7.6) | -0.04 (-7.0) | -0.04 (-8.7) | -0.04 (-9.2) |
| q_ba | -0.05 (-8.8) | -0.04 (-10.9) | -0.05 (-9.7) | -0.05 (-9.2) | -0.05 (-8.6) | -0.04 (-7.4) |

Layers where all 5 questions move toward "changed": none.

**Controls, all four cells** (mean effect on the report score over the questions, and the largest absolute effect on one question; nats):

| condition | L16 | L24 | L32 | L40 | L48 | L56 |
|---|---|---|---|---|---|---|
| +1 change | +0.01 (max 0.05) | +0.05 (max 0.09) | +0.05 (max 0.09) | +0.05 (max 0.09) | +0.05 (max 0.09) | +0.05 (max 0.09) |
| +4 change | +0.04 (max 0.21) | +0.18 (max 0.34) | +0.17 (max 0.32) | +0.18 (max 0.33) | +0.18 (max 0.34) | +0.18 (max 0.35) |
| +4 random 0 (norm of 4 change) | -0.01 (max 0.13) | -0.00 (max 0.04) | -0.02 (max 0.11) | -0.01 (max 0.07) | -0.01 (max 0.12) | +0.03 (max 0.13) |
| +4 random 1 | -0.07 (max 0.19) | -0.00 (max 0.08) | -0.01 (max 0.08) | -0.00 (max 0.10) | +0.03 (max 0.21) | -0.02 (max 0.09) |
| +1 recent | +0.00 (max 0.02) | +0.00 (max 0.04) | +0.01 (max 0.05) | +0.01 (max 0.05) | +0.01 (max 0.05) | +0.01 (max 0.05) |
| +4 recent | +0.00 (max 0.08) | +0.01 (max 0.16) | +0.02 (max 0.17) | +0.02 (max 0.20) | +0.02 (max 0.19) | +0.03 (max 0.21) |
| +1 history | +0.02 (max 0.15) | +0.05 (max 0.25) | +0.06 (max 0.20) | +0.06 (max 0.17) | +0.06 (max 0.17) | +0.06 (max 0.17) |
| +4 history | -0.14 (max 0.54) | +0.34 (max 0.72) | +0.25 (max 0.78) | +0.22 (max 0.70) | +0.22 (max 0.67) | +0.19 (max 0.61) |

A direction that only shifts the answer bias moves q_ab and q_ba, or q_surprise and q_same, in opposite directions on this score, so its mean over the questions is near 0 while its largest effect is not.

**surprise_words after "wait"** (nats; natural change +0.26, recent +0.03, history +0.06). Effect on all cells (t):

| condition | L16 | L24 | L32 | L40 | L48 | L56 |
|---|---|---|---|---|---|---|
| +1 change | +0.16 (+23.3) | +0.25 (+33.3) | +0.28 (+47.4) | +0.28 (+58.7) | +0.26 (+80.6) | +0.25 (+68.3) |
| +4 change | +0.65 (+34.1) | +1.01 (+41.2) | +1.13 (+56.5) | +1.09 (+94.7) | +1.04 (+95.6) | +0.82 (+84.9) |
| +1 recent | +0.00 (+1.0) | +0.03 (+6.8) | +0.04 (+8.8) | +0.03 (+13.6) | +0.01 (+3.1) | +0.01 (+2.7) |
| +1 history | -0.15 (-9.3) | +0.17 (+9.2) | +0.17 (+9.8) | +0.25 (+21.8) | +0.28 (+36.6) | +0.06 (+9.5) |
| +4 random 0 | -0.05 (-3.6) | -0.33 (-33.4) | +0.41 (+46.1) | -0.34 (-27.1) | -0.31 (-32.2) | -0.16 (-19.2) |

**int_mean after "comma"** (numbers; natural change +56.37, recent +217.62, history +171.99). Effect on all cells (t):

| condition | L16 | L24 | L32 | L40 | L48 | L56 |
|---|---|---|---|---|---|---|
| +1 change | -2.69 (-2.5) | -7.73 (-4.3) | -2.66 (-0.9) | +5.51 (+2.0) | +16.90 (+6.1) | +19.51 (+8.2) |
| +4 change | -12.22 (-5.0) | +2.75 (+0.9) | +27.02 (+5.0) | +80.88 (+16.3) | +96.57 (+23.2) | +80.57 (+22.1) |
| +1 recent | +0.83 (+3.1) | +17.97 (+13.7) | +39.25 (+22.0) | +57.83 (+29.2) | +73.05 (+26.5) | +97.61 (+33.5) |
| +1 history | -1.82 (-0.5) | -0.32 (-0.1) | +12.16 (+8.3) | +26.24 (+17.6) | +25.49 (+28.0) | +42.71 (+20.6) |
| +4 random 0 | -7.48 (-4.6) | +22.26 (+6.5) | +28.63 (+8.0) | +9.96 (+3.2) | -10.78 (-1.6) | -5.79 (-1.3) |

**int_sd after "comma"** (numbers; natural change +133.85, recent -23.62, history +13.77). Effect on all cells (t):

| condition | L16 | L24 | L32 | L40 | L48 | L56 |
|---|---|---|---|---|---|---|
| +1 change | +7.47 (+4.0) | +27.24 (+13.2) | +30.13 (+15.2) | +36.33 (+17.4) | +38.18 (+18.6) | +36.03 (+18.9) |
| +4 change | +31.02 (+18.8) | +91.12 (+37.8) | +97.44 (+34.5) | +96.72 (+40.4) | +76.68 (+39.1) | +96.55 (+48.5) |
| +1 recent | +0.20 (+0.7) | -6.23 (-6.1) | -8.65 (-4.3) | -12.91 (-5.4) | -17.38 (-6.6) | -11.01 (-4.0) |
| +1 history | +8.06 (+13.1) | +10.15 (+10.1) | +4.11 (+3.2) | -1.31 (-0.8) | -5.51 (-3.3) | -6.38 (-3.1) |
| +4 random 0 | +1.73 (+1.5) | -7.76 (-6.3) | +16.19 (+10.3) | +8.62 (+6.3) | +4.61 (+1.5) | +13.97 (+12.9) |

Directions at the tail of q_surprise (mean over tail tokens): norm of change / recent / history, and cosine(change, recent): L16: 1.0 / 0.6 / 5.3, +0.37; L24: 1.9 / 1.1 / 7.1, +0.27; L32: 3.0 / 1.7 / 12.0, +0.10; L40: 3.5 / 2.0 / 14.6, +0.07; L48: 4.1 / 2.4 / 17.2, +0.04; L56: 4.7 / 2.8 / 20.5, +0.04.

20 streams; 538 s on NVIDIA H200.

## olmo-3.1-32b-it

Natural effects on the report score (no injection, nats; change = (LH + HL - LL - HH) / 2):

| question | change (se) | recent | history |
|---|---|---|---|
| q_surprise | +0.33 (0.02) | +0.15 | +0.09 |
| q_change | +0.06 (0.03) | +0.11 | +0.08 |
| q_same | +0.70 (0.03) | +0.03 | -0.06 |
| q_ab | +0.69 (0.03) | -0.03 | -0.20 |
| q_ba | +0.15 (0.02) | +0.12 | +0.02 |

**+1 x change direction on the unchanged cells (LL, HH): the transplant.** Effect in nats (t over streams), by layer:

| question | L16 | L24 | L32 | L40 | L48 | L56 |
|---|---|---|---|---|---|---|
| q_surprise | +0.18 (+11.6) | +0.34 (+26.0) | +0.32 (+17.6) | +0.32 (+22.1) | +0.32 (+23.1) | +0.32 (+28.9) |
| q_change | +0.00 (+0.2) | +0.06 (+5.2) | +0.07 (+5.3) | +0.06 (+4.9) | +0.06 (+4.6) | +0.08 (+7.4) |
| q_same | +0.33 (+18.8) | +0.70 (+47.0) | +0.71 (+51.4) | +0.70 (+47.9) | +0.69 (+47.1) | +0.69 (+52.3) |
| q_ab | +0.25 (+19.9) | +0.69 (+49.5) | +0.69 (+45.7) | +0.70 (+47.9) | +0.68 (+51.8) | +0.68 (+64.5) |
| q_ba | +0.09 (+4.9) | +0.13 (+8.3) | +0.14 (+14.1) | +0.16 (+18.4) | +0.13 (+11.9) | +0.14 (+14.0) |

Layers where all 5 questions move toward "changed": L16, L24, L32, L40, L48, L56.

**+4 x change direction on the unchanged cells.** Effect in nats (t over streams), by layer:

| question | L16 | L24 | L32 | L40 | L48 | L56 |
|---|---|---|---|---|---|---|
| q_surprise | +0.56 (+37.2) | +1.43 (+74.6) | +1.19 (+70.0) | +1.07 (+66.5) | +1.07 (+49.3) | +1.01 (+52.5) |
| q_change | +0.35 (+16.9) | +0.63 (+30.3) | +0.35 (+23.6) | +0.35 (+20.8) | +0.39 (+20.4) | +0.30 (+20.3) |
| q_same | +1.77 (+43.0) | +2.06 (+76.8) | +2.02 (+88.9) | +2.18 (+102.8) | +2.01 (+104.6) | +2.06 (+91.8) |
| q_ab | +1.26 (+42.8) | +2.03 (+68.7) | +1.48 (+56.5) | +1.42 (+44.9) | +1.64 (+68.7) | +1.71 (+85.9) |
| q_ba | +0.55 (+20.0) | +0.73 (+39.0) | +0.69 (+44.4) | +0.45 (+23.7) | +0.68 (+43.6) | +0.62 (+59.1) |

Layers where all 5 questions move toward "changed": L16, L24, L32, L40, L48, L56.

**-1 x change direction on the changed cells (LH, HL): the removal.** Effect in nats (t over streams), by layer:

| question | L16 | L24 | L32 | L40 | L48 | L56 |
|---|---|---|---|---|---|---|
| q_surprise | -0.22 (-17.6) | -0.35 (-21.9) | -0.33 (-21.9) | -0.32 (-16.7) | -0.32 (-22.6) | -0.33 (-21.0) |
| q_change | -0.10 (-6.9) | -0.10 (-7.8) | -0.06 (-4.7) | -0.05 (-3.7) | -0.04 (-3.1) | -0.07 (-5.1) |
| q_same | -0.43 (-24.2) | -0.65 (-32.5) | -0.66 (-50.4) | -0.67 (-36.8) | -0.67 (-47.2) | -0.68 (-61.9) |
| q_ab | -0.32 (-18.9) | -0.67 (-42.4) | -0.68 (-57.2) | -0.69 (-59.9) | -0.69 (-57.3) | -0.68 (-42.7) |
| q_ba | -0.11 (-8.6) | -0.15 (-9.8) | -0.14 (-12.8) | -0.15 (-14.1) | -0.14 (-15.8) | -0.14 (-11.1) |

Layers where all 5 questions move toward "unchanged": L16, L24, L32, L40, L48, L56.

**Projection to the other class's mean, changed cells.** Effect in nats (t over streams), by layer:

| question | L16 | L24 | L32 | L40 | L48 | L56 |
|---|---|---|---|---|---|---|
| q_surprise | -0.21 (-13.7) | -0.35 (-18.8) | -0.32 (-13.8) | -0.32 (-15.0) | -0.32 (-15.8) | -0.32 (-15.4) |
| q_change | -0.10 (-6.0) | -0.07 (-4.9) | -0.05 (-3.8) | -0.06 (-4.4) | -0.05 (-3.8) | -0.07 (-5.2) |
| q_same | -0.42 (-21.7) | -0.64 (-28.3) | -0.67 (-29.4) | -0.67 (-28.9) | -0.68 (-30.7) | -0.67 (-35.0) |
| q_ab | -0.29 (-19.4) | -0.69 (-28.0) | -0.70 (-23.7) | -0.68 (-26.2) | -0.70 (-27.9) | -0.68 (-31.4) |
| q_ba | -0.12 (-7.1) | -0.14 (-12.8) | -0.14 (-11.0) | -0.15 (-12.4) | -0.14 (-10.3) | -0.12 (-11.1) |

Layers where all 5 questions move toward "changed": none.

**Controls, all four cells** (mean effect on the report score over the questions, and the largest absolute effect on one question; nats):

| condition | L16 | L24 | L32 | L40 | L48 | L56 |
|---|---|---|---|---|---|---|
| +1 change | +0.23 (max 0.45) | +0.37 (max 0.61) | +0.34 (max 0.61) | +0.33 (max 0.60) | +0.33 (max 0.58) | +0.34 (max 0.60) |
| +4 change | +0.93 (max 2.08) | +1.33 (max 1.88) | +1.07 (max 1.88) | +1.04 (max 2.14) | +1.09 (max 1.92) | +1.07 (max 1.99) |
| +4 random 0 (norm of 4 change) | +0.10 (max 0.37) | +0.07 (max 0.67) | -0.12 (max 2.00) | +0.07 (max 0.85) | +0.08 (max 0.79) | +0.02 (max 0.56) |
| +4 random 1 | -0.01 (max 0.23) | -0.09 (max 0.40) | -0.02 (max 0.52) | -0.02 (max 0.83) | -0.12 (max 1.17) | -0.02 (max 0.38) |
| +1 recent | +0.05 (max 0.08) | +0.08 (max 0.12) | +0.08 (max 0.15) | +0.07 (max 0.13) | +0.08 (max 0.13) | +0.08 (max 0.14) |
| +4 recent | +0.24 (max 0.36) | +0.32 (max 0.59) | +0.29 (max 0.56) | +0.29 (max 0.53) | +0.29 (max 0.52) | +0.29 (max 0.50) |
| +1 history | +0.05 (max 0.20) | -0.05 (max 0.37) | -0.04 (max 0.32) | -0.03 (max 0.25) | -0.03 (max 0.27) | -0.03 (max 0.22) |
| +4 history | +0.20 (max 0.57) | -0.16 (max 2.22) | -0.21 (max 1.53) | -0.14 (max 1.09) | -0.15 (max 1.21) | -0.20 (max 1.14) |

A direction that only shifts the answer bias moves q_ab and q_ba, or q_surprise and q_same, in opposite directions on this score, so its mean over the questions is near 0 while its largest effect is not.

Directions at the tail of q_surprise (mean over tail tokens): norm of change / recent / history, and cosine(change, recent): L16: 0.9 / 0.4 / 2.2, +0.29; L24: 2.5 / 1.2 / 3.3, +0.17; L32: 4.5 / 2.0 / 5.9, +0.18; L40: 5.7 / 2.5 / 8.1, +0.18; L48: 7.1 / 3.1 / 10.3, +0.19; L56: 9.8 / 4.4 / 14.7, +0.19.

20 streams; 486 s on NVIDIA H200.
