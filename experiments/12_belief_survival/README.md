# 12 · How the old belief dies

**Question.** After a jump, how long does the old belief survive in each layer, and is the switch a smooth average
or a sticky flip? Experiment 09: the guess takes about 30 numbers to settle after 300 → 700; the magnitude words and
the middle-layer integer means decay with half-lives of 18–25 numbers. The J-lens paper (§9.3) notes that hysteresis
(the switch point depending on the direction of approach) is impossible inside one forward pass, so finding it
along the sequence would be evidence of attractor-like dynamics; ignition has only been reported with transition
widths (report project 2).

**Stimuli** (`stimuli.py`; noise sd 100, 20 streams per arm):
- *jumps*: the level jumps at number 500 by 100, 200 or 400 (1, 2, 4 sd), up or down, centred on 500, plus a flat
  control (7 arms, 800 numbers);
- *ramps*: flat at 300 for 300 numbers, slide to 700 over 50 (fast) or 200 (slow) numbers, stay 200, slide back,
  stay 200; and the mirror image (start at 700).
The ideal observer of experiment 10 (default settings) reads the same numbers. The model runs reuse experiment
10's `run_streams.py`.

```bash
python experiments/12_belief_survival/go_nogo.py     # CPU, on experiment 09's arrays
python experiments/12_belief_survival/stimuli.py
sbatch --qos=debug --time=00:30:00 --gres=gpu:nvidia_h200:1 hpc/gpu.sbatch .venv-tf5/bin/python experiments/10_unlikely_vs_update/run_streams.py \
    --model llama3.1-8b --stimuli experiments/12_belief_survival/data/jumps.jsonl --name jumps --raw $JLENS_SHARED/experiments/12_belief_survival
# (and ramps_fast, ramps_slow with --batch 2)
python experiments/12_belief_survival/analyze.py --model llama3.1-8b
```

## Go / no-go (done, CPU, on experiment 09's jumps)

Rule fixed before running: go if the switch time of the output's mean and of at least half of the middle and late
layers whose integer mean moves by ≥ 100 can be timed with a 95% bootstrap interval no wider than e50 itself (or than
4 numbers). Result (`results/go_nogo.md`): **GO** for both models. The output switches half-way at e50 = 3 numbers
in both; 24 of 26 timed layers in Llama-3.1-8B and 67 of 72 (three lenses) in OLMo-3-32B pass, with intervals of
1–2 numbers. The late layers switch at e50 = 3–4 like the output; in OLMo-3-32B the J-lens middle layers (32–48)
at 2–3.

## Outcomes, fixed before the run

1. **Switch time** of the predicted mean per jump arm (e50, bootstrap intervals), against the observer and an
   exponential moving average (EMA) of the numbers whose rate is fitted to the model's jump curves.
2. **Mixture or slide.** A change-point reader predicts a two-bump distribution right after the jump, a moving
   average one sliding bump. Bimodal = the model's mass halfway between the levels is below that of a single
   Gaussian with the model's own mean and sd, when the mean is halfway.
3. **Hysteresis.** On the ramps, the true level at which the predicted mean crosses 500 on the way up and on the way
   down; loop width W. The test is the model's W against the EMA's and the observer's (any reader with memory lags):
   a wider loop = a sticky belief; up/down asymmetry = a prior toward some numbers.
4. **Layers.** e50 of the lens integer mean per layer (intervals), and a discrete-time hazard model of the switch
   (event-time bins, layer third, jump size, direction; censored at e = 299; bootstrap over streams). The lens
   readouts are continuous, so the top-k detection censoring the report warns about does not arise here; only right
   censoring does.
5. **Magnitude words** (uncertainty): peak and half-life after the jump by layer third, against the output's sd.

**Added after the first results** (not part of the plan above): the first-ramp lag exceeded the second-ramp lag in
every ramp, but the level before the first ramp had held for 300 numbers and before the second for 200. To separate
order from exposure, a history-length test (`stimuli.py --history`, Llama-3.1-8B only): one jump of 200 or 400 after
τ = 25, 50, 100, 200, 400 or 800 numbers at the old level, with the same draws just before and after the jump for
every τ. Switch time = when the predicted mean crosses the midpoint (500). A running mean predicts a switch time
proportional to τ (slope 1 on log–log), a moving average and the change-point observer a constant one (slope 0).

## Results

Llama-3.1-8B (logit lens, J-lens) and OLMo-3-32B (logit, J-lens, J++), 20 streams per arm; full tables in
[results/llama3.1-8b_summary.md](results/llama3.1-8b_summary.md) and
[results/olmo-3-32b_summary.md](results/olmo-3-32b_summary.md).

**1. Switch time scales with the jump, as a change-point reader's does, but 3–10 times slower.** e50 (numbers after
the jump until the predicted mean is half-way; 95% intervals in the summaries):

| arm | Llama | OLMo-32B | observer | EMA (Llama / OLMo) |
|---|---|---|---|---|
| up 100 | 37 | 44 | 7 | 5 / 9 |
| down 100 | 29 | 49 | 7 | 6 / 9 |
| up 200 | 10 | 10 | 3 | 5 / 9 |
| down 200 | 7 | 13 | 2 | 6 / 9 |
| up 400 | 4 | 3 | 1 | 6 / 9 |
| down 400 | 4 | 10 | 1 | 6 / 9 |

An EMA fitted to the model's own curves (half-life 6.0 numbers for Llama, 9.5 for OLMo) switches at the same time
whatever the jump; the models switch 5–15 times faster after a 4 sd jump than after a 1 sd jump, as the observer does,
but each is several times slower than the observer at every size.

**2. Partly a mixture.** After a 4 sd jump the predicted distribution splits: at e = 3 the mass half-way between the
levels is 0.27 (Llama) and 0.25 (OLMo), below the 0.35 and 0.34 a single bump with the same mean and sd would have;
the observer has 0.21. But the model keeps much more mass on the old level (0.24 and 0.28 at e = 3; observer 0.01).
After a 2 sd jump there is no split (half-way mass 0.25 against 0.26 / 0.25): a slide.

**3. No sticky loop in Llama; a wider one in OLMo, mostly an order effect.** Loop width W (true level at the upward
crossing of 500 minus the level at the downward crossing):

| ramp | Llama | OLMo-32B | observer | EMA (Llama / OLMo) |
|---|---|---|---|---|
| up-down fast | 99 [72, 127] | 132 [114, 152] | 123 | 140 / 207 |
| down-up fast | 111 [89, 132] | 218 [187, 248] | 104 | 126 / 200 |
| up-down slow | 20 [9, 29] | 58 [44, 73] | 40 | 19 / 45 |
| down-up slow | 21 [5, 34] | 98 [79, 116] | 38 | 5 / 46 |

Llama's loop is no wider than either reader's. OLMo's is wider than both on the slow ramps and than the observer's
on the fast down-up ramp. In both models the lag on the first ramp exceeds the lag on the second in all four ramps
(Llama +6.0, +3.1, +9.2, +1.1 numbers; OLMo +5.2, +8.3, +19.9, +14.6; intervals above 0 except Llama's slow down-up),
while the observer's and the EMA's differences are within ±2 numbers or not distinguishable from 0. **Caveat:** the old level held for 300 numbers before the
first ramp and 200 before the second, so the order effect may be an exposure effect; section 6 tests this.

**4. Middle layers flip a few numbers before the output.** Llama: logit-lens layers 20–26 reach half-way 2 numbers
after a 4 sd jump, the output 4; J-lens layers 20–22 after down jumps at 1–2 (output 4–7). OLMo: J-lens layers 52–56
at 5–6 after the 4 sd down jump (output 10). In the hazard model, switching is faster for larger jumps (log2 Δ:
+1.60 Llama, +0.92 OLMo) and in the few timed early layers than in the middle and late thirds; direction goes
opposite ways in the two models (down: +0.88 and −1.05). Few early layers move enough to be timed, so the early
baseline rests on a handful of noisy layers.

**5. Magnitude words rise with the jump but are confounded with number size.** Peak excess (J-lens, middle third)
grows with the jump (Llama up 100 / 200 / 400: +0.39 / +1.06 / +1.71), but up jumps give larger peaks than down jumps
(Llama +1.71 vs +0.92 at 400; OLMo +0.54 vs +0.25 at 200), while the output's sd peaks higher after down jumps (125
vs 110). Up jumps land on larger numbers, so the words partly read the size of the numbers, not only the uncertainty.

**6. History length (Llama-3.1-8B): the old belief does not get stronger with age.** Switch time (numbers after the
jump until the predicted mean crosses the midpoint), by how long the old level had held:

| τ (numbers at the old level) | 25 | 50 | 100 | 200 | 400 | 800 | slope on log τ |
|---|---|---|---|---|---|---|---|
| jump of 200: model | 6 | 12 | 12 | 10 | 9 | 9 | +0.04 [−0.09, +0.13] |
| jump of 400: model | 5 | 6 | 5 | 5 | 4 | 4 | −0.10 [−0.19, +0.03] |
| observer (200 / 400) | 3 / 1 | 3 / 1 | 2 / 1 | 2 / 1 | 2 / 1 | 2 / 1 | |
| running mean | 25 | 50 | 100 | > 199 | > 199 | > 199 | +1 |

A running mean would switch after about τ numbers; the model's switch time is flat from τ = 50 to 800 (and a little
faster at τ = 25, where its predicted sd before the jump is still 246, against 106 at τ = 800). So the order effect
of section 3 is not an exposure effect: 300 against 200 numbers of history cannot explain it, since 200, 400 and
800 numbers give the same switch time. What differs between the two ramps is that the second returns to a level the
context has already seen; the observer has no memory of past levels and shows no order effect. The middle layers
need a long history to read the level: the J-lens integer mean at layer 20 crosses after 46 numbers at τ = 25,
11 at τ = 100 and 2 at τ ≥ 400 (jump of 400); the output's switch time does not change.

**Reading.** The models behave like change-point readers in where the switch time comes from (jump size, not
history length), but they are several times slower than the ideal observer, keep a separate bump at the old level
for longer, and switch back to a level seen earlier faster than to a new one. **Caveats:** 20 streams; the ramp
explanation (a familiar level) is inferred, not tested directly; the history test ran on Llama only.
