# 10 · Unlikely, or belief-changing?

**Question.** When a model reading a number stream meets something unexpected, does its inside react to *how
unlikely the number was* (surprisal), to *how far its belief moved* (Bayesian surprise), to *how far the belief now
is from where it was* (deviation), or to *how uncertain it has become*? Experiment 08 could not tell the first two
apart: in its stimuli they moved together (r = 0.917). Experiment 09 found hints that the "Wait," reaction
follows real jumps and not single extreme numbers, but had no ground truth.

Here every stream comes in five versions that share the same random draws and differ only from number TAU = 400 on
(`common.py`; 20 streams of 600 numbers, noise sd 60 around 500; up for even streams, down for odd):

| arm | what happens at TAU | surprisal | belief change | deviation | uncertainty |
|---|---|---|---|---|---|
| control | nothing | low | none | none | low |
| jump | the level moves 300 (5 sd) and stays | high, then back | large | large, lasting | brief rise |
| outlier | one number moves 300 (the jump arm's number), then back | high once | small | none | small |
| drift | the level slides 300 over 100 numbers (3 per number) | low | small per number | grows to 5 sd | low |
| noise | the sd triples (60 → 180), same level | higher, lasting | moderate | none | large, lasting |

At TAU the jump and outlier arms are **the same number**, so they only differ in what comes after.

**Ground truth.** An exact Bayesian observer (`observer.py`) that knows streams can jump (probability h_mu = 1/200
per number), change their noise (h_sig = 1/500), and contain one-off outliers (rho = 1/100, uniform on 10..990),
the model used by O'Reilly et al. (2013) and Nassar et al. (2010) to separate surprise from belief updating. It is
an exact filter on a (sigma, mu) grid and gives, at every number, surprisal, Bayesian surprise (KL of the posterior
from the prior over the state), change-point and outlier probabilities, and predictive mean and sd. Sensitivity
settings: rho in {0, 1/1000, 1/20}, h_mu in {1/1000, 1/50}.

```bash
python experiments/10_unlikely_vs_update/stimuli.py      # data/: streams, cue contexts, observer values; prints the design check
sbatch --qos=debug --time=00:30:00 --gres=gpu:nvidia_h200:1 hpc/gpu.sbatch .venv-tf5/bin/python \
    experiments/10_unlikely_vs_update/run_streams.py --model llama3.1-8b --stimuli experiments/10_unlikely_vs_update/data/streams.jsonl --name streams
sbatch ... experiments/10_unlikely_vs_update/run_cues.py --model llama3.1-8b --contexts experiments/10_unlikely_vs_update/data/cue_contexts.jsonl
python experiments/10_unlikely_vs_update/analyze.py --model llama3.1-8b   # CPU; results/, figures/
```

`run_streams.py` (also used by experiment 12) reads every stream once and records, at every comma, the output's
distribution over the integers, and per lens (logit, J-lens, J++ lens) and layer the lens's integer statistics and
the log-probabilities of word sets (surprise, change, normal, magnitude, wide, narrow, high, low, decades, control)
at the comma and at the number's own token. `run_cues.py` cuts every stream after number TAU + e (e in −1, 0, 1, 2,
4, 8, 32, 128) and reads the lenses at "Wait," and "The last number is" (experiment 09's cues).

## Design check (on paper, before any model run)

The rule fixed in the proposal: the design is usable if surprisal and Bayesian surprise correlate below 0.5 after
removing (stream, number) fixed effects, over numbers TAU − 5 .. TAU + 40. Result (`stimuli.py`):

| observer | rho | h_mu | r(surprisal, KL) |
|---|---|---|---|
| default | 1/100 | 1/200 | **0.57** |
| outlier-prone | 1/20 | 1/200 | 0.39 |
| rare change points | 1/100 | 1/1000 | 0.44 |
| rare outliers | 1/1000 | 1/200 | 0.68 |
| frequent change points | 1/100 | 1/50 | 0.64 |
| no outliers | 0 | 1/200 | 0.70 |

The default is just above the threshold (0.57; experiment 08 had 0.917). **Deviation from the plan:** the experiment
went ahead, with the observer-free design contrasts below as the primary tests and the regression demoted to
secondary. Deviation and uncertainty are nearly uncorrelated with both surprises (|r| ≤ 0.2 for deviation).

## Outcomes, fixed before the run

**Go / no-go: behaviour first.** One number after the outlier, the model's predicted mean has moved less than 20%
of the way to the outlier's level; during the drift, the slope of its mean on the true level is at least 0.5. The
observer setting that best predicts the model's mean (RMSE over numbers TAU − 5 .. TAU + 149) is chosen on
behaviour alone, before any internal signal is looked at.

**Primary (design contrasts, no observer needed).** Signals: lens log-ratios surprise − normal, change − normal,
magnitude − control, wide − narrow, at the comma after each number (and at the number token), averaged over the
middle third of the layers (late third secondary; per layer in the csv).
- *Lasting*: jump − control and drift − control over e = 100..199. Nothing is surprising there, but the belief is
  5 sd from where it was. A non-zero difference means the signal tracks deviation.
- *Noise*: noise − control over e = 20..199. More uncertainty, same level; the magnitude words should rise if they
  read uncertainty (experiment 09).
- *Profile*: over e = 0..10, the correlation across the four arms of the signal's arm − control curve with the
  observer's surprisal curve and with its Bayesian-surprise curve, at lag 0 and lag 1 (experiment 08's surprise words
  lagged by one number).

**Secondary.** The regression of each signal on z(surprisal), z(Bayesian surprise), z(deviation), z(uncertainty)
with (stream, number) fixed effects, numbers TAU − 5 .. TAU + 40, clustered by stream, with and without the
previous number's surprisal and Bayesian surprise. Cue readouts: arm − control by cut, and the same regression across
arms and cuts with (stream, cut) fixed effects.

## Results

Llama-3.1-8B (logit lens, J-lens) and OLMo-3-32B (logit, J-lens, J++), 20 streams per arm. Full tables in
[results/llama3.1-8b_summary.md](results/llama3.1-8b_summary.md) and
[results/olmo-3-32b_summary.md](results/olmo-3-32b_summary.md); csv files alongside.

**Added after the first run: a level-matched arm.** The contrasts between arms that read different current numbers
(jump − control, noise − control) mix a reaction to the change with a reading of the numbers themselves. A sixth
arm, always at the jump's new level (`stimuli.py --level-only`, same draws), gives "jump − level" and
"drift − level" at e = 100..199: same current level, different history. The clean short-run contrast is
"after outlier" (outlier − control at e ≥ 1: the two streams differ in one past number only).

**Go / no-go: GO in both.**

| | Llama | OLMo-32B |
|---|---|---|
| mean moved toward the outlier, one number later | −0.00 of the way | 0.01 |
| drift slope (mean on true level) | 0.96 | 0.94 |
| jump: share of the way at e = 0 / 1 / 2 / 4 / 8 | 0.05 / 0.28 / 0.52 / 0.77 / 0.92 | 0.01 / 0.08 / 0.22 / 0.43 / 0.76 |
| noise arm: predicted sd / control's (true 3) | 2.80 | 2.36 |
| best observer (RMSE of the mean) | rare change points (h_mu = 1/1000) | the same |

Both models ignore a single outlier and follow a jump, OLMo more slowly; both are best described by an observer
that expects change points to be rarer than they are in the stimuli.

**At the unexpected number every signal fires** (e = 0, jump − control, comma): surprise words +0.55 (Llama, logit
lens, middle third) and +0.43 (OLMo); magnitude words +0.86 (Llama J-lens, middle) and +0.73 (OLMo).

**After an outlier (clean contrast), the magnitude words stay up; surprise words barely.** Magnitude − control,
J-lens middle third, outlier − control: Llama +0.110 (e 1..3) and +0.094 (e 4..10); OLMo +0.292 and +0.131 (all
p < 0.001). Surprise words: Llama +0.043 (p = 0.09) and +0.024; OLMo −0.060 and −0.090. One deviant number, which
moves no belief about the mean, leaves an uncertainty readout raised for about ten numbers.

**Lasting (jump − level and drift − level, e = 100..199).** Nothing is surprising there and the current level is the
same; only the history differs.
- Magnitude words (J-lens, middle / late): Llama +0.132 / +0.206 after the jump and +0.416 / +0.440 after the drift
  (p < 0.001); the predicted sd is the same as the level stream's after the jump (+1.0, se 0.7) and higher after the
  drift (+8.4). OLMo: −0.088 / +0.066 after the jump, +0.006 / +0.155 after the drift; its sd is lower than the
  level stream's (−4.8 and −6.1). So in Llama the magnitude words keep a trace of the history that the predicted sd
  does not show; in OLMo they do not.
- Change words, late third: small positive traces in both (Llama logit +0.092 and +0.201; OLMo J-lens +0.103 and
  +0.151, J++ +0.103 and +0.117).
- Surprise words: at or below the level stream (Llama J-lens middle −0.105, late −0.161).

**Which surprise? Per-number regression (secondary; default observer, both surprises together).** Surprise words
(J-lens, middle, comma) load on surprisal (Llama +0.058, t 10.2; OLMo +0.044, t 2.9), not on Bayesian surprise
(Llama −0.027; OLMo +0.029). Magnitude words load on the observer's uncertainty (Llama +0.208, t 15.2; OLMo +0.218,
t 7.6; J++ +0.274, t 8.9) and on the deviation of the belief from where it was (Llama +0.190).

**"Wait," cue.** Surprise − normal words at the last cue token:

| J-lens, middle third | e = 0 | 1 | 2 | 4 | 8 | 32 |
|---|---|---|---|---|---|---|
| Llama jump − control | +0.14 | +0.19 | +0.24 | +0.15 | +0.09 | −0.06 |
| Llama outlier − control | +0.14 | +0.10 | +0.09 | +0.07 | +0.03 | +0.02 |
| Llama jump − level | −0.01 | +0.05 | +0.06 | −0.00 | −0.09 | −0.18 |
| OLMo jump − control | +0.20 | +0.41 | +0.39 | +0.31 | +0.15 | +0.08 |
| OLMo outlier − control | +0.20 | +0.19 | +0.13 | +0.07 | +0.02 | +0.01 |
| OLMo jump − level | −0.22 | −0.10 | −0.07 | −0.18 | −0.22 | −0.34 |

In both models a jump raises the "Wait," response two to four times as much as an outlier at e = 1..4, and for
longer. But a stream
that was always at the new level also raises it (level − control at e = −1: Llama +0.16, OLMo +0.45), so part of
the jump response is a reaction to extreme numbers rather than to the change; against the level stream the
J-lens difference is near zero or negative in both. The logit lens in Llama's middle layers keeps a change-specific
part (jump − level +0.22, +0.22, +0.18 at e = 0, 1, 2; p < 0.05). Across arms and cuts (regression, observer with
rare change points), the "Wait," response in the middle layers loads on surprisal (J-lens: Llama t 2.3, OLMo
t 4.4) and uncertainty (t 4.1, 3.6), in the late layers on Bayesian surprise (Llama t 6.2, OLMo t 4.1), in both
models.

**Reading.** The two surprises separate by layer more than by word set: middle layers carry "this number is
unlikely" and "the spread is large", late layers "the belief moved". Magnitude words are the most robust readout:
they rise at the event, stay up after a lone outlier and follow the observer's uncertainty number by number.
**Caveats:** the design check failed narrowly (r = 0.57), so the regression separates the two surprises only
partly; the "Wait," contrasts depend on the current level of the numbers (see the level-matched rows); 20 streams.
