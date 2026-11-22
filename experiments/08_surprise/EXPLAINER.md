# Experiment 08, explained: surprise with a known right answer

*A plain-language guide to what we asked, what we built, what we measured and what we found. Model:
Llama-3.1-8B. Run of 7 November 2026. The technical description is in [README.md](README.md); the numbers quoted
here come from `results/llama3.1-8b_*.csv` and `figures/llama3.1-8b_*.png`.*

---

## The idea in one paragraph

Read "She makes 9 × 2 = $17 every day": something in you says *wait, that's wrong*. Watch a thermometer that has
read 537, 534, 540 for a while suddenly read 568: you think *something changed*. Language models notice these things
too, in the sense that they predict the wrong number with very low probability. This experiment asks whether the
"wait" also shows up **inside** the model, in the part of its internal state that the J-lens can read as words, and
**which kind** of surprise it is: "that was unlikely" or "I have to change my mind".

---

## 1. The research question

### 1.1 Two kinds of surprise

| Name | Plain meaning | Formula | Unit |
|---|---|---|---|
| **Surprisal** | How unlikely was what I just saw? | −log p(observation) | nats (a 1-in-1000 event is 6.9 nats) |
| **Bayesian surprise** | How much did it change what I believe? | KL(belief after ‖ belief before) | nats |

They usually move together, but not always. Real numbers from our ideal observer (section 3.4) on two of our
sequences: a calm sensor (noise sd 4, level about 537) and a noisy one (noise sd 16, level about 609).

| Situation | Reading | Surprisal (nats) | Bayesian surprise (nats) | What the observer concludes |
|---|---|---|---|---|
| Calm sensor, an ordinary reading | 536 | 2.6 | 0.05 | nothing new |
| Calm sensor, a sudden reading | **568** | 9.8 | **6.65** | the level changed (96% sure) |
| Calm sensor, the reading after 568 goes back | **538** | 6.5 | **3.27** | no change after all: the sensor is noisier than it thought (90%) |
| Noisy sensor, an ordinary reading | 606 | 4.0 | 0.05 | nothing new |
| Noisy sensor, the same +32 jump | **638** | 5.1 | **0.12** | probably just noise (92%) |
| Noisy sensor, a second high reading | **663** | 6.9 | **0.60** | still mostly noise (84%); starts to suspect a change |

The 538 and 663 rows are almost equally unlikely (6.5 vs 6.9 nats), but one moves the beliefs five times more than
the other (3.27 vs 0.60). That is the kind of difference we hoped to use to tell the two surprises apart. Section 3.5
shows, step by step, how every number in this table is computed.

### 1.2 The two parts of the experiment

- **Part A, change points: which kind of surprise?** Sequences of sensor readings that jump to a new level, or show
  one wild value and come back. An exact ideal observer gives both surprises for every reading.
- **Part B, planted arithmetic errors: is a known error flagged?** Math solutions in which one result is replaced
  by a wrong value. The right answer is known, and we check that the model knows it.

### 1.3 Why it matters

- **Monitoring.** If a model's internal "something is off" can be read, it can be watched: for instance a model that
  notices an error in a tool's output but carries on anyway. Anthropic's emotions paper (Transformer Circuits, April
  2026) found a "surprised" direction in Claude that fires when it notices that an attachment the user mentioned is
  missing.
- **The J-lens workspace.** Gurnee et al. (2026) argue that the middle layers hold a "global workspace" of content
  the model could put into words. Is surprise part of that content?

---

## 2. What we expected: hypotheses fixed before the run

All of these were written into the README before the model ran.

**Gates (behaviour first).** If the model doesn't do the task, its internals have nothing to show.

| Gate | Threshold |
|---|---|
| A: the model's prediction moves like the observer's (slope of model change on observer change) | ≥ 0.5 |
| A: the model's surprisal correlates with the observer's | ≥ 0.5 |
| A: after a jump, the model is at least halfway to the new level by the third reading | ≥ 0.5 |
| B: the model *knows* the result (its top prediction at the result is the correct number) | ≥ 50% of items |
| B: the near-miss error costs the model at least 2 nats more than the right answer | ≥ 2 nats |

**Primary hypotheses.**

- **H-A.** At the comma after each reading, averaged over the middle layers, a signal "carries Bayesian surprise"
  if it rises with Bayesian surprise *holding surprisal fixed*, and "carries surprisal" if the reverse.
- **H-B.** After a planted error, the J-lens *error words* and the *size of the state change* rise more than after a
  harmless edit of the same size (averaged over the 5 tokens after the edit).

**Secondary.** Other specifications and slots, layer-by-layer curves, the event studies, whether a bigger error
gives a bigger signal, and what happens when the result is used again later in the solution.

---

## 3. The model and the tools

### 3.1 The model

**Llama-3.1-8B** (base, not chat): 32 layers; the internal state at each token and layer is a vector of 4,096
numbers (the *residual stream*). The model never writes anything in this experiment: we feed it the text and read
what it computes at every token (*teacher forcing*). At each position we get

- its probability for every possible next token (so: how surprised it is by the token that actually comes), and
- its internal states at all 32 layers.

### 3.2 The J-lens: reading an internal state as words

For every vocabulary token *w* (128,256 of them) and every layer ℓ there is a direction d_ℓ(w) in the 4,096-number
space. The score ⟨d_ℓ(w), h⟩ says how strongly the state h at layer ℓ pushes the model, through all the remaining
layers, toward eventually saying *w*.

- The direction is d_ℓ(w) = J̄_ℓᵀ Γ u_w: the output row of *w* (u_w) pulled back to layer ℓ by the **average
  Jacobian** J̄_ℓ, which records how a small change at layer ℓ changes the final output. Our lens averages it over
  539 Wikipedia passages.
- The older **logit lens** applies the output row directly to a middle-layer state, as if the remaining layers did
  nothing. The J-lens accounts for them (linearly, on average), which is why it works in the middle layers.

*Example.* At the comma right after the wild reading 568, we ask: do "higher", "suddenly" or "wait" score above where
they usually are?

### 3.3 The J-space dictionary

A state can be approximated by a few word directions added with positive weights. We pick 25 of them greedily
(*nonnegative matching pursuit*: take the word direction that best matches what is left, subtract it, repeat). The
10 largest are "the 10 words the state is mostly about right now". This is used for the *turnover* signal below.

### 3.4 The ideal observer (Part A's ground truth)

Exact Bayesian online change-point detection (Adams and MacKay 2007), in `bayes.py`. It assumes:

- each reading = current level + Gaussian noise;
- the noise size is unknown (somewhere between 1 and 40) and is learned from the readings;
- before each reading there is a 1-in-30 chance that a new level starts (the *hazard*);
- a new level is drawn from a wide range (normal, mean 500, sd 250).

For each reading it gives the probability of that reading (→ surprisal), its full belief before and after (→
Bayesian surprise, computed exactly), and its prediction of the next reading. `tests.py` checks it against brute-force
enumeration of every possible pattern of changes.

Part B needs no observer: the correct result is known.

### 3.5 How surprisal and Bayesian surprise are computed, step by step

**The observer's belief is a list of stories with weights.** A story says where the current level started (at which
reading the last change happened), what the level is now, and how noisy the sensor is. The weight says how much the
observer believes the story. (The code keeps 25 possible noise sizes × every possible start; below, the stories are
grouped by start and the noise size is averaged within each story.)

Worked example: the calm sensor of item 160, just before reading 17, which will be **568**. The readings so far are
533, 532, 536, 535, 546, 538, 546, 531, 534, 542, 534, 534, 539, 534, 534, 540.

| Story | Weight before | Level | Noise sd | Predicts the next reading as |
|---|---|---|---|---|
| A: "no change since reading 1" | 95.8% | 536.7 | 4.9 | about 537 ± 5 |
| B: "a new level starts right now" | 3.3% (= 1/30) | unknown: anywhere, N(500, 250²) | 4.9 | anything from about 0 to 1000 |
| Others: "a change happened at reading 3, 5, 16, …" | 0.9% together | | | |

Story B always gets exactly 1/30 before each reading: that is the hazard, the built-in chance that a new level starts.

**Step 1: how likely is the reading under each story?** Each story predicts a bell curve for the next reading. The
probability of an integer k is the area under the curve between k − ½ and k + ½, roughly the height of the curve
at k.

- Under story A, 568 is 31 above the level, more than 6 noise sds: P(568 | A) = 0.0000018. Most of even this tiny
  number comes from the possibility, inside story A, that the sensor is noisier than it looked (see step 3).
- Under story B, the new level could be anywhere in a range about 1000 wide, so any particular number gets about
  1/650: P(568 | B) = 0.00154. The formula is the height of N(500, 250²) at 568, which is
  (1/(250·√(2π)))·exp(−½·(68/250)²).

**Step 2: surprisal is −ln of the total probability.** The total probability adds up every story's prediction,
weighted by the story's weight:

  P(568) = 0.958 × 0.0000018 + 0.0333 × 0.00154 + (others)
         = 0.0000018 + 0.0000513 + (small) = 0.0000532

  **surprisal = −ln(0.0000532) = 9.84 nats**, odds of about 1 in 19,000.

Almost all of P(568) comes from story B. This puts a cap on surprisal in this model: a reading can never be much
less likely than "1/30 chance of a new level × 1/650 for this particular number", so no reading costs much more than
ln(30 × 650) ≈ 10 nats (a bit more for numbers far from 500).

**Step 3: Bayes' rule gives the new weights.** For each story, new weight = old weight × P(reading | story) /
P(reading):

| Story | Weight before | × P(568 given story) | Weight after |
|---|---|---|---|
| B: "the level changed at reading 17" | 3.3% | × 0.00154 | **96.4%** |
| A: "no change since reading 1" | 95.8% | × 0.0000018 | **3.3%** (and its noise estimate rises from 4.9 to 9.3) |

Inside each story, the belief about the level updates too, by the usual normal–normal rule: the new level is an
average of the old level and the reading, each weighted by how precise it is. In story B the level goes from
"anywhere, sd 250" to "568, sd 4.9". In story A, the only way "no change" survives 568 is a noisier sensor, so the
noise sizes that can explain it (about 9) gain weight.

**Step 4: Bayesian surprise is KL(after ‖ before), and it splits into two parts.** The code computes both exactly
(`kl_discrete` and the rest of `bayes_kl` in `bayes.py`).

- **Which story** = Σ weight_after × ln(weight_after / weight_before)
  ≈ 0.964 × ln(0.964 / 0.033) + 0.033 × ln(0.033 / 0.958) + … = 3.25 − 0.11 + … = **3.28** (exact, including the
  noise-size weights). In words: "I went from 3% to 96% belief in a change", about ln(1/0.033) ≈ 3.4.
- **Where the level is** = Σ weight_after × KL(level after ‖ level before). For story B,
  KL(N(568, 4.9²) ‖ N(500, 250²)) = ½ [ln(250²/4.9²) + (4.9² + 68²)/250² − 1] ≈ 3.47, × weight 0.964 = 3.34.
  Adding the small contributions of the other stories gives exactly **3.38**. In words: "I went from not knowing
  the level at all (± 250) to knowing it to ± 5", about ln(250/4.9) − ½ ≈ 3.4.
- **Total: 6.65 nats** (3.28 + 3.38; the parts are rounded).

**The other examples, computed the same way** (main stories only; weights in %; the two parts of the Bayesian
surprise are rounded, so they may not add up exactly):

| Example | Main stories before | P(reading) → surprisal | Main stories after | Bayesian surprise (which story + level) |
|---|---|---|---|---|
| Calm, ordinary reading **536** | "no change" 96 | 0.078 → **2.55** | "no change" 99 | **0.05** (0.05 + 0.00) |
| Calm, after 568, next is **570** (change confirmed) | "changed at 17, level 568" 93 | 0.053 → **2.94** | "changed at 17" 99.8 | **0.20** (0.08 + 0.12) |
| Calm, after 568, next is **538** (outlier: back to normal) | "changed at 17" 93; "no change, noisier sensor" 3.2; "new level now" 3.3 | 0.0015 → **6.49** | "no change, noisier sensor" 90; "changed at 17" 3.6; "new level now" 3.5 | **3.27** (3.05 + 0.21) |
| Noisy, ordinary reading **606** | "no change, level 608.5, noise ≈ 20" 86 | 0.019 → **3.95** | "no change" 87 | **0.05** (0.04 + 0.01) |
| Noisy, the +32 jump **638** | "no change" 86 | 0.0062 → **5.09** | "no change" 92 | **0.12** (0.05 + 0.08) |
| Noisy, second high reading **663** | "no change" 89; "changed at 19" 0.7 | 0.0010 → **6.90** | "no change" 84; "changed at 19" 6.5; "new level now" 4.3 | **0.60** (0.36 + 0.24) |

What the table teaches:

1. **Surprisal has a floor.** Even an ordinary reading costs 2.5–3 nats on the calm sensor and 4–4.5 on the noisy
   one, because the observer cannot predict the exact integer: it spreads its probability over about ±5 or ±20
   numbers. On average an ordinary reading costs about ln(noise sd × 4.13) nats.
2. **The two surprises answer different questions.**
   - Surprisal: "how much probability did my stories, together, give this reading?"
   - Bayesian surprise: "how much did the weights of my stories, and their levels, move?"

   At 568 both are large, because the stories flip. At 663 the reading is about as unlikely as 538 (6.9 vs 6.5),
   but the observer can still blame noise, so its stories barely move (0.60 vs 3.27).
3. **How this observer explains an outlier.** It has no story "one wild value, then back to normal". When 538
   follows 568, it mostly concludes that nothing changed and the sensor is noisier than it thought: its noise
   estimate goes from 5.1 to 8.8. This is one place where the model behaves differently (section 8.1).

`bayes.py` computes all of this exactly for every reading, over all 25 noise sizes and every possible start, and
`tests.py` checks it against brute-force enumeration of every possible pattern of changes.

---

## 4. The data

### 4.1 Part A: sensor readings (720 items × 3 versions = 2,160 sequences)

Every sequence is

```
The following are readings from a sensor:
533, 532, 536, 535, 546, 538, 546, 531, 534, 542, 534, 534, 539, 534, 534, 540, 568, 570, 574, ...
```

Each item comes in three versions ("arms"). Item 160 (noise 4, change of +32 at reading 17):

| Reading number | … 15 | 16 | **17 (τ)** | 18 | 19 | 20 … |
|---|---|---|---|---|---|---|
| control | 534 | 540 | 536 | 538 | 542 | 542 |
| jump (stays at the new level) | 534 | 540 | **568** | 570 | 574 | 574 |
| outlier (one wild value, then back) | 534 | 540 | **568** | 538 | 542 | 542 |

- The three versions are **identical before τ**, so any later difference is caused by the change.
- **Jump and outlier are identical up to and including τ**: at the wild value they are the same. They differ only
  from the next reading, when the change is confirmed (jump) or not (outlier).
- Grid: noise σ ∈ {4, 8, 16} × change size δ ∈ {2, 4, 8} noise units × up or down × 40 items per cell = 720 items.
  τ is between 12 and 22; every number is between 10 and 989, so it is a single token for Llama-3.
- The same jump can be shocking or ordinary: +32 is 8 noise units in a calm sequence and 2 in a noisy one (table
  in 1.1).

### 4.2 Part B: arithmetic twins (602 GSM8K problems × 4 versions)

GSM8K is a public set of grade-school word problems with worked solutions. From each solution we pick one visible
equation `a op b = r` (op is + or ×, r < 1000) and make four versions that differ only inside it. The first test
problem:

> Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends
> every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in
> dollars does she make every day at the farmers' market?
>
> Answer: Janet sells 16 - 3 - 4 = 9 duck eggs a day.
> She makes **9 * 2 = $18** every day at the farmer's market.
> \#### 18

| Version | The equation | What it is |
|---|---|---|
| clean | 9 * 2 = $**18** | as written |
| near | 9 * 2 = $**17** | wrong, close (last digit off by 1 or 2) |
| far | 9 * 2 = $**48** | wrong, far (leading digit moved by 3 or more) |
| swap | **2 * 9** = $18 | correct; a harmless edit that still changes tokens |

- Every token before and after the edit is the same in the four versions.
- **"The model knows r"**: in the clean text, its top prediction at the position of 18 is 18. Decided before any
  edit, so the selection cannot depend on the outcome. True for 595 of 602 problems (98.8%).
- **The reuse**: the next time r appears on its own later on, here the final `#### 18`. It is the same token in
  every version, but it comes after a different past.
- **Why the swap?** Any edit changes the text, and the model's state reacts to changed tokens whether or not they
  are wrong. The swap is an edit of the same size that is *not* an error, so "near minus swap" isolates "wrong" from
  "edited".

### 4.3 Where we read the model

**Slots (Part A).** Llama-3 splits `626, 630` into the tokens `626` `,` ` ` `630`. For each reading we look at:

| Slot | Token | Why |
|---|---|---|
| num | `626` (the number itself) | differs between versions in its token, so not used for the headline |
| **sep** | `,` (the comma after it) | **headline**: the same token in every version, so any difference is context alone |
| pred | ` ` (the space before the next number) | where the model predicts the next number |

**Event time.**

- Part A: k = (reading number) − τ. k = 0 is the wild value, k = 1 the next reading, and so on.
- Part B: k counts tokens after each version's own last edited token.
  - near: k = 0 is `17`, then k = 1 ` every`, 2 ` day`, 3 ` at`, 4 ` the`, 5 ` farmer`.
  - swap: the last edited token is `9`, then k = 1 ` =`, 2 ` $`, 3 `18`, 4 ` every`, 5 ` day`.

**Layers.** Headline numbers average over the middle third of the model, layers 10–21 (the *band*), where the J-lens
reads best.

---

## 5. The signals (what we measure inside the model)

| Signal | Plain meaning | How it is computed | Example |
|---|---|---|---|
| **jump** | How much did the internal state move? | ‖h_t − h_{t−1}‖ / (typical size of h at that layer). Part A compares the same slot one reading back; Part B the previous token | 0.4 = moved 40% of a typical state's length |
| **lex_surprise** (A) | Do surprise words rise in the J-lens? | Mean J-lens score of *wait, actually, suddenly, unexpected, strange, hmm, however, …* and *change, jump, shift, spike, anomaly, outlier, …* minus neutral words (*table, river, green, coffee, …*) | see below |
| **lex_error** (B) | Do error words rise? | The same with *wrong, error, mistake, incorrect, oops, typo, fix, …* and the surprise words | |
| **lex_dir** (A) | Does it say which way? | (*higher, increase, rise, up, above, …* minus *lower, decrease, drop, down, below, …*) × the sign of the change | positive = it "says" the right direction |
| **turnover10** | Did the topic change? | Jaccard distance between the 10 main dictionary words at consecutive readings | 8 of 10 shared → 1 − 8/12 = 0.33 |
| **div** | How far is this version from its twin? | ‖h − h_twin‖ / ‖h_twin‖ at the same position | 0.6 = differs by 60% of the state's length |

**How a word-list score works (made-up numbers).** At one position the J-lens gives every one of the 128,256 tokens
a score. Convert the scores to z-scores (how many standard deviations above the average token). Say *wrong* is at
+2.5, *error* at +1.5 and *mistake* at +2.0, so the list averages +2.0, and the neutral words average +0.3. Then
lex_error = 2.0 − 0.3 = 1.7. Subtracting neutral words removes changes that lift all ordinary words at once.

Words that are not a single token are dropped. Kept: 17 surprise, 9 error, 10 change, 10 up, 10 down, 20 neutral.

---

## 6. The statistics

### 6.1 Twins = difference-in-differences with zero pre-periods

In a difference-in-differences you compare a treated and a control group before and after a treatment, and you have
to hope their trends were parallel before. Here the twins see **exactly the same tokens before the edit**, and a
language model only looks backwards, so their internal states before the edit are **identical, not just parallel**.

`analyze.py` checks this on every signal and stops if any twin differs before the edit by more than 10⁻³ of the
signal's sd. **Result: 27 of 27 checks pass, with a largest difference of exactly 0.0.**

### 6.2 Part A: regression with twin fixed effects

For each signal *y* (in sd units) at the sep slot:

  y = b_KL · z(Bayesian surprise) + b_S · z(surprisal) + fixed effect(item, reading number) + error

- **Fixed effects for (item, reading number)** mean we only compare the three versions of the *same* sequence at the
  *same* reading: everything about the sequence and the position cancels. With 720 items × 3 versions × 26 readings
  there are 56,160 rows.
- Both surprises are in **z-scores**, so b = "sd of the signal per sd of surprise".
- Standard errors are **clustered by item** (the 78 rows of one item are not independent).
- Three specifications:
  - *twin*, the headline, as above;
  - *pooled*, which compares different sequences at the same reading number; it is open to confounding and serves as
    a contrast;
  - *buffer*, which adds the model's own change of prediction, and asks what is left beyond "the next prediction
    changed".
- **The catch, measured before any result:** within (item, reading), the two surprises correlate at **0.917**. When
  two causes almost always move together, a regression can barely tell them apart. It is like asking whether a plant
  grows from sunlight or from warmth when sunny days are almost always warm.

### 6.3 Part B: triple difference and a sign-flip test

- For each item and signal, average over k = 1…5: (near − clean) − (swap − clean) = **near − swap**.
- **Sign-flip test.** If errors had no effect, each item's near − swap would be as likely negative as positive.
  Flip the signs at random 10,000 times and see how often the average is as extreme as the real one. Never → p <
  0.0001, the smallest value 10,000 flips can show.
- 95% confidence intervals by bootstrap over items.

---

## 7. The pipeline, step by step

| Step | Script | What it does | Where it runs |
|---|---|---|---|
| 1 | `stimuli.py` | Builds `data/changepoint.csv` (2,160 sequences) and `data/arith.jsonl` (602 × 4 texts from `data/gsm8k_test.jsonl`) | seconds, CPU |
| 2 | `tests.py --tokenizer llama3.1-8b` | 9 tests: observer vs brute force, integer probabilities to 10⁻⁹, fixed-effects regression vs dummies, the stimuli's rules, the whole analysis on fake data with planted effects, and Llama's tokenization of every item | minutes, CPU |
| 3 | `run_model.py` | Runs Llama on every text. Saves at every layer: state norms and jumps, J-lens word-list scores, the model's predictions over the 1,000 integers, its log-probabilities, the dictionary decomposition at 8 layers, and hidden states for later probes | **9 minutes on one H200 GPU** |
| 4 | `analyze.py` | Runs the observer, checks the gates, runs the pre-period checks, regressions, event studies and tests, and writes results and figures | under 1 minute, CPU |

**Two bugs the checks caught before any result was trusted:**

1. **Tokenizer.** The Llama-3 tokenizer in our library version returned empty character spans for whole-word tokens.
   Without the repair, 368 of the 602 math items would have been silently dropped. `prep.py` repairs it, and the
   test now checks every span.
2. **GPU arithmetic.** The first full run failed the pre-period check: twins that should have been identical
   differed slightly. The GPU picks a different internal routine when the last batch is shorter, which changes the
   last decimals. Padding every batch to the same size fixed it; now the twins agree exactly.

---

## 8. Results

### 8.1 Behaviour: the model does both tasks (all gates pass)

| Gate | Result | Threshold |
|---|---|---|
| A: update slope (model on observer) | **0.62** (se 0.02) | ≥ 0.5 |
| A: surprisal correlation, model vs observer | **0.87** | ≥ 0.5 |
| A: share of the way to the new level by the third reading | **0.78** (observer 0.75) | ≥ 0.5 |
| B: model knows r | **98.8%** (595 of 602) | ≥ 50% |
| B: extra cost of the near error | **9.2 nats** (far 11.2; swap 1.4) | ≥ 2 |

- **It adapts like the observer.** After a jump, the model's prediction is 45% → 70% → 78% → 83% of the way to the
  new level; the observer's is 51% → 66% → 75% → 81%. After an outlier both return to the old level.
- **It half-expects outliers.** When the reading comes back after an outlier, the model is much less surprised than
  the observer: 0.57 vs 1.97 nats more than at the same reading in the control sequence. Our observer has no notion
  of one-off outliers: it had believed the change (93% in the example of 3.5) and has to fall back on "the sensor
  is noisier than I thought". The model seems to have such a notion.
- **It is pulled toward the planted error.** At the later `#### 18`, the model gives the correct value 3.4 nats less
  log-probability after a near error (2.7 after a far one, 0.02 after a swap). Note that it is *reading*, not
  writing: in a document that contains a typo, the typo often recurs, so this is a reasonable prediction rather than
  a failure.

### 8.2 Part A: which surprise?

Headline (sep slot, layers 10–21, twin fixed effects; sd of signal per sd of surprise; t-statistics in brackets):

| Signal | On Bayesian surprise | On surprisal |
|---|---|---|
| jump | +0.060 (4.3) | −0.191 (−10.4) |
| turnover10 | +0.084 (4.6) | −0.196 (−8.6) |
| lex_surprise | +0.015 (1.4) | +0.002 (0.1) |
| lex_dir | −0.076 (−6.0) | +0.216 (12.3) |

What this means:

1. **On paper**, jump and turnover10 meet the "carries Bayesian surprise" rule (b_KL > 0 with surprisal held fixed),
   and lex_dir meets the "carries surprisal" rule. **We do not claim either.** With the two surprises correlated at
   0.917, every pair of coefficients splits into opposite signs, the usual symptom of two regressors that cannot be
   separated.
2. **At the comma, the state moves *less* after a surprise.** One reading after the change, jump is 0.17 (jump arm)
   to 0.21 (outlier arm) below control, and turnover10 dips too. At the **pred** slot (the space before the next
   number) it is the other way round: both rise with both surprises (jump +0.10 and +0.11, t ≈ 10 and 9). We don't
   yet know why the comma behaves this way.
3. **The surprise words rise, but one reading late.**
   - Jump arm, against control: +0.03 at the wild value (k = 0), **+0.08 at the next reading (k = 1)**, then about
     +0.04 through k = 8.
   - Outlier arm: the peak is at k = 2 (+0.08), one reading after the value comes back.

   The observer's surprise peaks one step earlier in both cases. A regression that pairs each reading with its own
   surprise misses a delayed response, so the twin regression shows nothing. The *pooled* regression shows a large
   effect (t = 22), but that comes from differences between sequences, which is exactly what the twins remove.
4. **The direction words track the belief.** At the wild value, "up minus down" (signed by the change) rises by
   +0.20 in both arms, immediately. In the jump arm it stays up (+0.13 at k = 2, +0.09 at k = 8); in the outlier arm
   it falls back to +0.02 by k = 2. That is what a belief about the new level should do: hold when the change is real,
   drop when it was a one-off.

### 8.3 Part B: is the error flagged?

Primary tests over k = 1…5, 595 items (95% CI in brackets):

| Signal | near − swap | far − swap | far − near |
|---|---|---|---|
| **lex_error** | **+0.0094 [0.0054, 0.0139], p < 0.0001** | +0.0075, p = 0.003 | −0.0018, p = 0.39 |
| **jump** | **−0.0011 [−0.0022, −0.00002], p = 0.065** | +0.0010, p = 0.12 | +0.0021, p < 0.0001 |
| div | −0.081, p < 0.0001 | −0.024, p < 0.0001 | +0.057, p < 0.0001 |
| turnover10 | +0.0001, p = 0.84 | +0.0007, p = 0.018 | +0.0006, p = 0.008 |

What this means:

1. **Yes, the error words rise, more than for a harmless edit** (p < 0.0001). But the effect is small and brief. Error
   words, near minus clean, by token:

   | k | 0 (`17`) | 1 | 2 | 3 | 4 | 5 |
   |---|---|---|---|---|---|---|
   | near − clean | 0.174 | 0.031 | 0.018 | 0.012 | 0.004 | −0.002 |

   It peaks on the wrong number itself (k = 0, outside the test window because the token differs) and is gone by
   about the fourth token. A bigger error does not give a bigger rise (far − near, p = 0.39).
2. **The size of the state change does not flag errors** (jump: p = 0.065, and the estimate is negative). Its curve
   has a spike one token after the edit (near +0.021, far +0.031, swap +0.009) followed by a dip at k = 4–6, and the
   two cancel in the 1…5 average. The spike alone is an exploratory observation.
3. **A harmless edit moves the state more than an error.** div is 0.17 for swap and 0.09 for near. So *how much* the
   state changes says "something was edited", not "something is wrong". Only *where it points*, toward error words,
   separates the two.
4. **When the right value returns later** (`#### 18`), the error words rise again after a near error (+0.055,
   p < 0.0001) but not after a far one.

---

## 9. What we conclude so far

1. **Behaviourally, Llama-3.1-8B tracks surprise very well.** It updates like an exact Bayesian observer, ignores
   one-off outliers even a little better than that observer, and knows 99% of the arithmetic results.
2. **Inside, the J-lens shows matching but weak signals.** Error words rise briefly after a planted error, beyond a
   harmless edit. Direction words ("higher", "lower") follow the model's belief about the new level.
3. **Surprise words appear one step late** in the number sequences.
4. **This design cannot say which surprise the workspace carries.** The two surprises move together too closely in
   our stimuli (r = 0.917).
5. **How much the state moves is not an error detector.** Harmless edits move it more. The information is in the
   direction, not the size.

In one sentence: the model clearly notices errors and changes, and its workspace shows matching words, but the
signals are weak or delayed, and the current stimuli cannot separate the two kinds of surprise.

---

## 10. Caveats and open puzzles

- **Word lists are a readout, not proof.** A rise in "wrong" or "error" is what the lens says; it does not show that
  a dedicated error representation exists.
- **One model**, and a base model.
- **The observer is the ground truth by assumption.** It has no outlier process: it explains a one-off wild value
  mostly as a noisier sensor (3.5), while the model behaves as if it expected outliers (8.1).
- **GSM8K is public**, so Llama may have seen these solutions. The swap also contradicts a memorized text, which is
  one more reason to compare against it.
- **Swap alignment.** Because k starts after each version's own last edit, the swap's window (` =`, ` $`, `18`, …)
  covers reading the result, while near's window covers the words after the wrong result. This was fixed in advance,
  but the two windows are not the same moment.
- **Open puzzle 1:** why the comma state moves *less* after a surprise while the next slot moves *more*.
- **Open puzzle 2:** why the surprise words lag by one reading while the direction words respond immediately.

---

## 11. Next steps

1. **Separate the two surprises by design** (the main fix). Give the observer a notion of one-off outliers, as in
   neuroscience (O'Reilly et al. 2013, PNAS, "Dissociable effects of surprise and model update in parietal and
   anterior cingulate cortex"). Then an outlier is very unlikely (high surprisal) but changes no beliefs (low
   Bayesian surprise), and the correlation between the two drops. We can check on paper how far it drops before using
   a GPU.
2. **Let the model write after the error.** Plant the error, let the model continue the solution, and check whether
   it corrects itself. Then ask whether the internal error signal *predicts* the correction. This is the
   monitoring-relevant version.
3. **Lagged regression** for Part A (each signal against the surprise one reading earlier), clearly labelled as
   exploratory. CPU only, minutes.
4. **More models**, and a comparison of readouts: J-lens words vs a supervised "surprise" direction built from
   labelled examples (as in the emotions paper). This would tell whether surprise is in the verbalizable workspace or
   somewhere else.

---

## 12. Glossary

- **Residual stream / internal state (h):** the 4,096 numbers the model keeps at each token and layer.
- **Layer band:** layers 10–21 of 32, the middle third.
- **J-lens:** reads a middle-layer state as scores over words, using the average Jacobian to account for the
  remaining layers.
- **Logit lens:** the same idea without accounting for the remaining layers; works only near the end of the model.
- **J-space dictionary / atoms:** the J-lens word directions, used as building blocks to decompose a state.
- **Teacher forcing:** feeding the model a fixed text and reading its predictions at each token, without letting it
  generate.
- **Twin / arm / version:** copies of the same text that differ in one place only.
- **τ (tau):** the reading at which the change (or the outlier) happens.
- **Event time k:** steps after the change or the edit.
- **Pre-period check:** confirms the twins are identical before the edit.
- **Fixed effects:** compare only within a group (here: the versions of one item at one position).
- **Clustered standard errors:** uncertainty that allows rows from the same item to be related.
- **Sign-flip test:** a test that randomly flips the sign of each item's difference to see what "no effect" looks
  like.
- **nat:** unit of surprise using natural logarithms (1 nat ≈ 1.44 bits).

---

## 13. Where the files are

- Code: `stimuli.py`, `bayes.py`, `prep.py`, `run_model.py`, `analyze.py`, `tests.py`, `common.py` (this folder).
- Stimuli: `data/`.
- Raw model outputs (2.7 GB, not in git): `$JLENS_SHARED/experiments/08_surprise/llama3.1-8b/`.
- Results: `results/llama3.1-8b_*.csv`.
  - `verdict.json`: gates and key numbers.
  - `checks.csv`: the 27 pre-period checks.
  - `headline.csv`: the primary numbers.
- Figures: `figures/llama3.1-8b_changepoint_behaviour.png`, `_changepoint_regression.png`, `_changepoint_event.png`,
  `_arith_event.png`.
