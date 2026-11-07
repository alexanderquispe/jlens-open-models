# 08 · Surprise with a known right answer

**Question.** When a model reads something unexpected, does its workspace (the J-lens reading of the middle layers)
register it, and which kind of surprise: how unlikely the token was (*surprisal*, −log p), or how far the reader's
beliefs had to move (*Bayesian surprise*, the KL divergence from the belief before to the belief after; Itti and
Baldi 2009)? The two usually move together. Here the stimuli come with an ideal observer that computes both
exactly, and in twins that differ in one place only, so the model's internal signals can be regressed on each
while the other is held fixed.

- **Part A, change points.** Sequences of sensor readings whose level jumps once, or that show one wild value and
  return (an outlier). At the wild value the two are identical; one number later, the jump is confirmed and the
  outlier is not. Surprisal and Bayesian surprise come apart here.
- **Part B, planted arithmetic errors.** GSM8K solutions in which one result *a op b = r* is replaced by a close
  or a distant wrong value, and a control twin in which a and b are swapped (still correct, a harmless edit of the
  same size). The right answer is known, so the question is whether the workspace flags the error.

```bash
python experiments/08_surprise/stimuli.py                          # data/changepoint.csv, data/arith.jsonl (seconds)
python experiments/08_surprise/tests.py --tokenizer llama3.1-8b    # CPU, a few minutes; must print "9 passed"
python experiments/08_surprise/run_model.py --model llama3.1-8b    # one H100; raw outputs to $JLENS_SHARED
python experiments/08_surprise/analyze.py --model llama3.1-8b      # CPU; results/ and figures/
```

On the cluster, `sbatch hpc/gpu.sbatch python …` for run_model.py and `sbatch hpc/cpu.sbatch python …` for the
rest. run_model.py writes its arrays to `$JLENS_SHARED/experiments/08_surprise/<model>/` (`--raw` to change; about
1 GB for Llama-3.1-8B, 2 GB more with `--save-hidden`; not for git); analyze.py reads them from the same place.

## Design

### Part A: change points (720 items × 3 arms)

Each sequence is `The following are readings from a sensor:\n` followed by 30 integers joined by `, `. An item is
one draw of Gaussian noise around a level μ₀, shown in three arms:

| arm | numbers |
|---|---|
| control | x_t = round(μ₀ + σ ε_t) |
| jump | the control numbers + a shift of ±δσ from number τ on |
| outlier | the control numbers, except number τ, which is the jump arm's |

The three arms are identical before τ, and jump and outlier are identical through τ. σ ∈ {4, 8, 16} and
δ ∈ {2, 4, 8}, both directions, 40 items per cell, τ uniform in 12..22, numbers in 10..989 (one token each in
Llama-3). The same jump is very surprising in a calm sequence and ordinary in a noisy one.

**The ideal observer** (`bayes.py`) is exact Bayesian online change-point detection (Adams and MacKay 2007) with
the noise level learned: σ has a uniform prior on a grid from 1 to 40, a new segment starts before each number with
probability h = 1/30, and a segment's level is drawn from N(500, 250²). The belief about (σ, run length, level) is
a finite mixture of Gaussians, updated exactly. An integer k gets the probability of [k − ½, k + ½]. For each
number it returns the surprisal, the Bayesian surprise (the KL over the whole belief; exact, because the mixtures
before and after share their components), the change-point probability (Nassar et al. 2010) and its prediction
of the next number. `tests.py` checks it against brute-force enumeration of every change pattern.

**Positions read** at each number x_t: *num* (the token of x_t), *sep* (the comma after it) and *pred* (the token
just before x_{t+1}, where the model predicts it; for Llama the space). The sep slot is the same token in every arm,
so a difference between arms there is context alone. The num slot also differs in token identity, so it is not in
the headline numbers.

### Part B: arithmetic twins (602 GSM8K test problems × 4 versions)

From each solution of the GSM8K test split (Cobbe et al. 2021; calculator annotations removed), one visible
equation `a op b = r` (op + or ×, r < 1000, checked against the annotation and the arithmetic), in four versions
that differ only inside it:

| twin | edit |
|---|---|
| clean | as written |
| near | r → a wrong value close to it (±1 or ±2 in the last digit, same number of digits) |
| far | r → a wrong value far from it (leading digit moved by 3 or more) |
| swap | a and b swapped: correct, a harmless edit that changes tokens |

The texts have the same length, and every token before the edit and every token after it is the same in the four
versions. Event time k counts tokens after each twin's own last edited token. The model reads the text (teacher
forcing); it never writes the error. Items are kept where the model *knows* r: in the clean text, its top-1
prediction at r is r. Knowing is decided before the edit, so this selection does not depend on the outcome. The
next standalone occurrence of r later in the solution (the *reuse*; every item has one, since each result is used
in a later step or is the final answer) is the same token in every twin, with a different past.

### Why the twins: difference-in-differences with zero pre-periods

Within an item, the twins see identical tokens up to the edit, so with causal attention their activations there
are identical, not just similar: the "pre-trends" of a difference-in-differences are zero by construction, and
`analyze.py` checks it on every signal (it stops if a twin differs before the edit by more than 10⁻³ of the
signal's sd). After the edit, the twins differ by the edit alone. Comparing them at the same item and position
(fixed effects for (item, t) in Part A) removes everything about the sequence and the position. Part B's
triple difference, (wrong − clean) − (swap − clean), removes what any edit of the same size does.

## Signals

At every layer, at the positions read:

- **jump**: ‖h_t − h_{t−1}‖ in units of the layer's median ‖h‖ (Part A: the same slot of the previous number;
  Part B: the previous token).
- **lex_surprise / lex_error**: the J-lens readout of a word list (`common.LEXICON`) minus that of neutral
  control words. The readout is the mean z-score, over the vocabulary, of the lens logits of the list's
  one-token words. Part A uses the surprise and change words; Part B the error and surprise words.
- **lex_dir** (Part A): up words minus down words, times the sign of the jump. Does the workspace say which way?
- **turnover10**: Jaccard distance between the sets of the first 10 J-space atoms at consecutive positions
  (nonnegative matching pursuit, k = 25, of the centered activation on the J-lens vectors, at 8 layers).
- **div**: distance to the control arm or the clean twin, ‖h − h₀‖ / ‖h₀‖ (event studies only).

Headline numbers average a signal over the middle third of the layers (`common.band`; Llama-3.1-8B: 10–21).

## Outcomes, fixed before the run

**Go / no-go: behaviour first.** If the model does not track the task, its internals have nothing to track.

- Part A: the model's predicted mean of the next number (from its distribution over the 1000 integers) moves like
  the observer's (slope of the change on the observer's change ≥ 0.5), its surprisal correlates with the
  observer's (≥ 0.5), and after a jump it has moved at least half way to the new level by the third number.
- Part B: the model knows r in at least half the items, and the near error costs it at least 2 nats.

**Primary, Part A.** At the sep slot, band-averaged, the regression with (item, t) fixed effects of each signal on
z(Bayesian surprise) and z(surprisal) together, clustered by item. "The workspace carries Bayesian surprise"
needs b_KL > 0 with surprisal in the regression; "it carries surprisal" the reverse. The identifying variation is
printed first: the within-(item, t) correlation of the two regressors.

**Primary, Part B.** Among items the model knows, lex_error and jump averaged over k = 1..5, near − swap, with a
sign-flip randomization test over items.

**Secondary.** Part A: the pooled specification (fixed effects for t only), the buffer specification (adding the
model's own change of prediction; see the caveats), per-layer curves, the event study around τ (jump − control,
outlier − control, jump − outlier). Part B: far − swap and far − near (does a bigger error move it more?), the
reuse of r, event studies by layer (`--by-layer`).

## Outputs

`results/<model>_…`: `changepoint_behaviour.csv`, `changepoint_adaptation.csv`, `changepoint_regression.csv`
(every signal × slot × layer × specification), `changepoint_event.csv`, `arith_behaviour.csv`, `arith_event.csv`,
`arith_tests.csv`, `headline.csv`, `checks.csv` (the pre-period checks), `verdict.json` (go / no-go and the key
numbers). `figures/<model>_…`: `changepoint_behaviour`, `changepoint_regression`, `changepoint_event`,
`arith_event`. `run_info.json` (next to the raw arrays) records the items dropped by the tokenization checks, the
lexicon words kept, and the layers.

## Checks of the pipeline

- `tests.py`:
  - the observer against brute-force enumeration;
  - the integer probability against an independent formula to 10⁻⁹, deep in the tails;
  - fixed-effects OLS and its clustered errors against a regression with dummies;
  - the stimuli's invariants;
  - the whole of `analyze.py` on synthetic outputs with planted effects of known size. It recovers them exactly,
    passes its pre-period checks, and fails them when a difference is planted before the edit.
- With `--tokenizer`, `tests.py` runs `prep.py` (the token alignment that `run_model.py` uses) on the real stimuli.
  With Llama-3.1-8B, all 1000 integers are one token, all 720 change-point items and all 602 arithmetic items pass,
  and every token's character span decodes to the token. The Llama-3 tokenizer under tokenizers 0.19 returns empty
  character spans for whole-word tokens; `prep.py` repairs them, and without the repair 368 of the 602 arithmetic
  items would be dropped.
- `run_model.py` drops items whose twins are not aligned token by token. The counts are in `run_info.json`.
- Smoke test: GPT-2 small on CPU, on a small stimulus set (`stimuli.py --sigmas 4 --deltas 2 --lo 10 --hi 99
  --n-rep 2 --max-arith 40`), the whole pipeline: run_model.py (30 min on 8 cores, almost all of it the J-space
  pursuit), then analyze.py. On real activations every pre-period difference is exactly 0 (27 checks). GPT-2
  fails both gates, as a 124M model should: its predictions barely follow the observer (surprisal correlation
  0.34), and it never has r as its top choice (mean log p(r) = −4.5, above the near value in 71% of the items).

## Caveats

- **The observer is the ground truth by assumption.** It has no outlier process: it reads a one-off wild value as
  a change followed by a second change to a new level that happens to sit near the old one. A model with a notion
  of outliers may be surprised differently at τ + 1 in the outlier arm. Its hazard (1/30, one change per sequence
  of 30) and level prior are fixed; `analyze.py --hazard` reruns it with another hazard.
- **Buffering.** The model's own change of prediction is a consequence of the surprise (a mediator). The buffer
  specification asks what is left beyond "the next output changed"; it is not a total effect.
- **GSM8K is public and may be in the training data.** A model that remembers a solution knows r by memory, and the
  planted error contradicts the memorized text as well as the arithmetic. The swap twin contradicts the memorized
  text too, which is one more reason the headline is near − swap.
- **Positions.** The lenses were fit without the first 16 positions. Part A's regressions start at the fourth
  number (position ≥ 18 in Llama-3), and every twin difference comes after τ ≥ 12 (position ≥ 42). The centering
  mean of the J-space uses positions ≥ 16.
- **Word lists.** Words that are not one token are dropped (`run_info.json`). The lists are short and in English.
  A readout of "surprise" words is what the lens says, not proof of a surprise representation.
- **Many signals and layers.** The primary outcomes are few and fixed above; per-layer results are descriptive.
