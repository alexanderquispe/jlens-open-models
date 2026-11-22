# 09 · Words of beliefs

**Question.** Sarfati et al. (2026, *The Shape of Beliefs*, arXiv:2602.02315; code
[raphael-goodfire/shape-of-beliefs](https://github.com/raphael-goodfire/shape-of-beliefs)) feed a base model streams
of integers drawn from N(μ, σ). The model's next-number distribution converges to N(μ, σ) within about 100 numbers,
and its activations form curved manifolds indexed by (μ, σ). When the stream switches from N(300, 100) to N(700, 100),
the predicted mean moves at once and the spread widens, then relaxes. The belief is thus known and measurable at
every position. This experiment asks what the workspace *says* while the belief forms, layer by layer. Which tokens
and multi-token concepts do the logit lens, the J-lens, the J++ lens and the Concept Lens surface? Which of them move
with μ, with σ, with time, and at the switch? It is a first, descriptive step: no causal claim.

```bash
python experiments/09_beliefs/stimuli.py                                  # data/sequences.jsonl (seconds; optional)
.venv-tf5/bin/python experiments/09_beliefs/tests.py --tokenizer llama3.1-8b   # CPU, ~1 min; must print "7 passed"
sbatch hpc/gpu.sbatch .venv-tf5/bin/python experiments/09_beliefs/run_model.py --model llama3.1-8b
sbatch hpc/gpu.sbatch .venv-tf5/bin/python experiments/09_beliefs/concept_lens.py --model llama3.1-8b
sbatch --gres=gpu:nvidia_h200:1 --mem=160G hpc/gpu.sbatch .venv-tf5/bin/python experiments/09_beliefs/run_model.py --model olmo-3-32b
sbatch hpc/cpu.sbatch .venv-tf5/bin/python experiments/09_beliefs/analyze.py --model llama3.1-8b   # then olmo-3-32b
# follow-ups (see "Follow-ups"); GPU jobs fit in --qos=debug --time=00:30:00 (OLMo-3-32B: an H200)
sbatch hpc/gpu.sbatch .venv-tf5/bin/python experiments/09_beliefs/cue_lens.py --model llama3.1-8b   # words after a cue
sbatch hpc/gpu.sbatch .venv-tf5/bin/python experiments/09_beliefs/steer.py --model llama3.1-8b      # steering
sbatch hpc/gpu.sbatch .venv-tf5/bin/python experiments/09_beliefs/wordsets.py --model llama3.1-8b   # word sets
python experiments/09_beliefs/analyze_cues.py --model llama3.1-8b    # CPU, each; also analyze_steer.py,
python experiments/09_beliefs/checks.py --model llama3.1-8b          # checks_words.py, surprise_words.py
```

Everything runs in `.venv-tf5` (transformers 5.11; see the top-level README). `run_model.py` and `concept_lens.py`
write to `$JLENS_SHARED/experiments/09_beliefs/<model>/` (`--raw` to change; not for git), and `analyze.py` reads from
there and writes `results/` and `figures/` here.

## Stimuli: the paper's, exactly

`stimuli.py` re-implements the paper's `generate_sequences.py`. Integers come from N(μ, σ) with seed 18, rounded,
clipped to [10, 990] and written `533,460,689,...,` with no spaces and a comma after the last number. There are 1000
numbers per sequence and 10 sequences per condition. `tests.py` checks that the 17 files the paper's code writes are
reproduced byte for byte (SHA-256).

| set | sequences |
|---|---|
| stationary | 17 conditions × 10: μ ∈ {300, 350, …, 700} at σ = 100, and σ ∈ {10, 20, 30, 50, 80, 100, 120, 150, 200} at μ = 500 |
| switch | 10: the N(300, 100) sequence followed by the N(700, 100) one, 2000 numbers, the change at number 1000 (the paper's Fig. 2) |

Every condition uses the same seed, so sequence k of every condition has the same standard normal draws. Conditions
differ only by (μ, σ), which makes comparisons across conditions matched, like twins. With BOS, number t is token
2t + 1 ("num2com", it predicts the comma), and the comma after it is token 2t + 2 ("com2num", it predicts number
t + 1). com2num is the paper's main position and ours.

## Models and lenses

| model | logit lens | J-lens | J++ lens | Concept Lens |
|---|---|---|---|---|
| Llama-3.1-8B (one H100) | ✓ | ✓ (Neuronpedia) | none exists | ✓, with the J-lens (the authors' main model) |
| OLMo-3-32B (one H200) | ✓ | ✓ (Neuronpedia) | ✓ (Neuronpedia) | ✓, with the J++ lens (`--lens jpp`) |

The paper's main model is Llama-3.2-1B. It has no lens, and fitting one (`jlens_open/fit.py`) would come with its own
caveats. Llama-3.1-8B has the same tokenizer, and the paper's App. B shows the same manifolds across Llama models.
The design needs every integer 0–999 to be one token. That holds for Llama 3 and OLMo 3, but not for Qwen3 (digits
are split), so Qwen3-4B's J++ lens cannot be used here. OLMo-3-32B is therefore the model where the J-lens and the
J++ lens can be compared on the same activations.

- **Logit, J-lens, J++** (`run_model.py`) are read at every token and every layer. The J++ lens has the same file
  format and readout as the J-lens (`unembed(J_l h)`; see the top-level README).
- **Concept Lens** (`jlens_open/concept.py`; Gong, Huang and Wang 2026, arXiv:2608.31084) is reimplemented from the
  authors' code ([XijieGo/c-lens](https://github.com/XijieGo/c-lens) at `f1a6811`), which is not imported or run.
  - **Steps.** The module docstring lists the steps and constants.
    - Proposals: the model completes the top J-lens tokens after a scaffold that contains the prompt.
    - Each proposal's direction v_C comes from a fixed carrier sentence at layer 27, through a reader fitted in
      closed form on the authors' 300 calibration concepts.
    - v_C is transported to each source layer (10–30) by ridge regression on J.
    - Proposals are ranked against the whole vocabulary by cosine with the normalized J-lens query.
  - **Our deviations.** Transport maps in float32 rather than float64, batched decoding, and v_C cached by string.
  - **Not available.** The R-lens variant: the authors' repository has no code to fit it.

## What is recorded

`run_model.py` (teacher forcing; the model never generates):

- **behaviour:** at every com2num position, the model's distribution over the 1000 integer tokens: mean, sd,
  entropy, mass on integers, and KL(model ‖ N(μ, σ)) as in the paper's Fig. 7 (for the switch, the Gaussian of the
  number being predicted). The whole distribution is kept at 12 times.
- **per lens and layer, at every token:**
  - the lens's softmax mass on integer tokens, and the mean and sd of its distribution renormalized over them:
    *the belief in number words*;
  - the readout of seven word groups (`common.LEXICON`: statistics, change, surprise, high, low, wide, narrow, and
    control words);
  - the top 20 tokens;
  - at com2num positions, the mean log-softmax over the whole vocabulary in cells of (condition × bin of t) for the
    stationary set and of bins of event time for the switch.

`concept_lens.py`: at the com2num position of every condition × 10 sequences at t = 127 and 999, and of the switch at
e = −1, 0, 1, 2, 4, 8, 16, 64, 256, 999. For each source layer it records the candidate concepts with their joint rank
(among the candidates and all 128,256 tokens) and the 10 best vocabulary tokens.

## Analysis (`analyze.py`, CPU)

1. **Go / no-go: does this model hold the belief?** It is a precondition, as in experiment 08. Late in the stream
   (t ≥ 512), the predicted mean must track μ (r ≥ 0.95 across the σ = 100 conditions) and the log sd must track
   log σ (r ≥ 0.9 across the μ = 500 conditions). The figure repeats the paper's Fig. 7A (KL against t) and Fig. 2B
   (the switch).
2. **The belief in number words, layer by layer.** For each lens:
   - how much its top mass is on integers;
   - whether its integer distribution's mean tracks μ (correlation and slope) and its spread tracks σ;
   - at the switch, a layer × event-time map of its integer mean against the output.
3. **Words.**
   - the words most often in each lens's top 20, by layer third;
   - the top words by mean log-prob per condition, early and late;
   - *contrasts*: the tokens whose mean log-prob rises or falls most with μ (700 vs 300), with σ (200 vs 10, on log
     σ), from early (8 ≤ t < 32) to late (t ≥ 512), and at the switch (e in [1, 8), and e = 0, vs e in [−100, −10)).
     Candidates are tokens in the top 200 of at least one compared cell, so the list is about salient tokens. Words
     and integers are listed separately.
4. **Concept Lens.** The concepts in the joint top 10 (the authors' Rank@10), as a fraction of readouts, by condition
   or event time and by layer third.

`results/<model>_words.md` collects all of it in one readable page.

## Checks

- **`tests.py`:**
  - the stimuli match the paper's files byte for byte, and the switch sequences are the two segments end to end;
  - the ridge transport solves its normal equations, and is exact without ridge;
  - the positive-pool reader finds the suffix position that carries the target;
  - the nucleus rule and prefix dominance on hand-made cases;
  - `analyze.py` on synthetic outputs finds planted words that rise with μ, with σ, late and after the switch, and
    a planted lens belief that tracks μ from a given layer on;
  - with `--tokenizer`: every integer is one token, every stream (and every cut of it after a comma) tokenizes as
    expected, and the Concept Lens carrier aligns for all 300 calibration concepts. For both Llama-3.1-8B and
    OLMo-3-32B, 119 of the 300 are one token in the carrier ("Concept:anger" has no space); the reader is fitted on
    those, as in the authors' code.
- **At run time.** `run_model.py` asserts that every stream tokenizes as `[BOS] n , n , …` and that every integer is
  one token. `concept_lens.py` asserts that the cut prompt tokenizes as a prefix of the stream.
- **CPU smoke test (2026-11-07).** All three scripts ran end to end on Llama-3.1-8B, on a CPU node with 32 cores:
  - `run_model.py`: 1 sequence per condition, 40 numbers, 758 s;
  - `concept_lens.py`: 3 readouts, source layers 16 and 27, 75 s per readout;
  - `analyze.py`: 34 s.

  Two observations, after only 40 numbers:
  - **Behaviour.** The predicted mean already rises with μ (409 at μ = 300, 589 at μ = 700), and the sd follows σ
    (36 at σ = 10, about 250 at σ ≥ 100).
  - **Lens words.** In the middle layers, the J-lens's most frequent words are number words and mathematics
    ("twenty", "seventy", "sixty", "trillion", "mathematical"). The logit lens shows junk sub-words there.

## Cost

Measured on 2026-11-07 (at $0.012 per compute unit: one H100 with 8 cores ≈ $1.54 per hour, one H200 ≈ $1.97 per
hour):

| job | GPU | time | cost |
|---|---|---|---|
| `run_model.py`, Llama-3.1-8B (logit + J-lens) | H100 | 3 min | $0.08 |
| `concept_lens.py`, Llama-3.1-8B (440 readouts, 21 source layers, 1.0–1.2 s each) | H100 | 8.5 min | $0.22 |
| `run_model.py`, OLMo-3-32B (logit + J-lens + J++) | H200 | 12 min | $0.40 |
| `analyze.py`, each model | CPU (32 cores) | 1–1.5 min | < $0.01 |
| `concept_lens.py --lens jpp`, OLMo-3-32B (440 readouts, 41 source layers, 5.6–6.2 s each) | H200 | 46 min | $1.5 |
| `cue_lens.py` (250 contexts x 6 cues) | H100 / H200 | 1.7 / 4 min | $0.04 / $0.13 |
| `steer.py` (20 contexts x 2 bases x 8 layers x 17 directions x 6 strengths) | H100 / H200 | 1.2 / 2.8 min | $0.03 / $0.09 |
| `wordsets.py` | H100 / H200 | 1.3 / 4.9 min | $0.03 / $0.16 |
| `checks.py`, `checks_words.py`, `analyze_cues.py`, `analyze_steer.py` | CPU | < 2 min each | < $0.02 each |

All of them except the OLMo-3-32B Concept Lens fit in the debug QOS (`--qos=debug`, at most 30 min, one job at a
time), which usually starts at once. That one needs the normal QOS (`--time=01:00:00`).
Raw outputs: 4.1 GB for Llama-3.1-8B and about 11 GB for OLMo-3-32B.

## First results (2026-11-07)

`results/<model>_words.md` has everything. In short:

1. **Both models hold the belief.** Late in the stream, the predicted mean tracks μ:
   - Llama-3.1-8B: r = 1.000, slope 0.96;
   - OLMo-3-32B: r = 1.000, slope 0.97.

   The predicted sd tracks σ on a log scale:
   - Llama-3.1-8B: r = 0.998, slope 0.95;
   - OLMo-3-32B: r = 0.987, slope 0.74.

   KL to the target Gaussian is 0.02–0.24 for Llama-3.1-8B.
2. **The belief in number tokens appears only in the last third of the layers, μ before σ.**
   - **Llama-3.1-8B, J-lens:**

     | layer | integer mass | slope on μ | slope on log σ |
     |---|---|---|---|
     | 20 | 0.26 | 0.29 | 0.02 |
     | 24 | 0.46 | 0.69 | 0.28 |
     | 28 | 0.89 | 0.81 | 0.32 |

     The full spread (slope 0.95) is only at the output.
   - **OLMo-3-32B.** Here the logit lens puts more mass on integers than the J-lens in layers 40–56, and it tracks σ
     better there. The J++ lens is the noisiest of the three across layers.
   - **Layers 0–4.** The r(mean, μ) of 0.9 or more there comes with a slope of 0.03–0.15: the copying baseline (see
     the caveats).
3. **Magnitude words fade as the belief settles.** This is the most consistent word-level result: the same contrast,
   early (8 ≤ t < 32) vs late (t ≥ 512), in every J-type lens of both models. The words that drop are:
   - Llama-3.1-8B, J-lens, middle layers: "trillion", "billions", "millennia", "millions", "thousand", "infinite",
     "mathematic";
   - OLMo-3-32B, J-lens, late layers: "billions", "trillion", "million";
   - OLMo-3-32B, J++ lens, middle layers: "billions", "BigInteger", "zeroes", "divisor", "hexadecimal";
   - OLMo-3-32B, J++ lens, late layers: "numbers", "number".

   The logit lens shows sub-word junk in both models.
4. **μ shows up as number words for the leading digit.**
   - **OLMo-3-32B, middle layers.** "sixty" and "seventy" rise from μ = 300 to μ = 700, in the J-lens and the J++
     lens.
   - **Llama-3.1-8B, Concept Lens, late layers, t = 127.** The concepts follow the hundreds digit of μ:
     - "forty-two" to "forty-eight" at μ = 400–450;
     - "fifty-five" at μ = 500;
     - "sixty-three" to "sixty-seven" at μ = 550–650;
     - "seventy-five" to "seventy-nine" at μ = 700.

     They are in up to 17% of the readouts. By t = 999 they have mostly gone.
5. **Surprise words: absent from the top, small and specific movements below it** (`surprise_words.py`, CPU; ranks
   are the best over layers, out of about 100,000–128,000 tokens).
   - **The tops.** No word for surprise or change reaches the top 20 or top 200 of any lens, in either model.
   - **OLMo-3-32B at the switch.** The surprise group ("wait", "actually", "unexpected", "surprise", "strange",
     "however", …) rises more than the other word groups, in all three lenses:
     - by 0.2–0.8 vocabulary sd, with t up to 23 over the 10 sequences;
     - at layers 14–23, and in the J-lens also at layer 59;
     - at the first new number and the 3 after it.

     Single words climb but stay far from the top:
     - " surprise", logit lens: rank 5272 → 291;
     - " anomaly", J-lens: rank 626 → 192;
     - " error", J++ lens: rank 1747 → 521 → 737.

     The change group ("change", "shift", "jump", "switch", …) does not rise.
   - **Llama-3.1-8B at the switch.** The surprise group shows no specific rise. Instead the "high" words rise (J-lens,
     layers 9–12), and " new" (J-lens: rank 571 → 226).
   - **Single improbable numbers in stationary streams, both models.** Two measures disagree:
     - correlated with a number's surprisal, every word group moves together (Llama-3.1-8B: |r| = 0.4–0.6;
       OLMo-3-32B: |r| = 0.1–0.2);
     - locked to the outlier and compared with the other groups (follow-up 4, C), the surprise group does rise
       specifically, for 1–3 numbers.
   - **Selection.** The best layer is chosen among 32 or 64, so the t values are optimistic.

6. **The switch.** In the 7 numbers after N(300, 100) switches to N(700, 100):
   - **Llama-3.1-8B, J-lens, middle layers.** Leading-digit words rise: "seven", "eight", "nine", "seventh", "VII",
     "VIII".
   - **OLMo-3-32B.** The magnitude words that had faded return: "trillion", "billion", "decimal", "numbers", together
     with "sixty", "seventy", "eighty".
   - **OLMo-3-32B, integer mean of the logit and J++ lenses, layers 21–42.** It jumps at the switch, then decays
     back over about 100 numbers, a transient rather than an update. The J-lens's integer mean there stays up.
   - **Concept Lens (Llama-3.1-8B).** It echoes its own scaffold ("the concept is", "the word") at e = 0–8 (from 0 to
     10% of readouts), and "seven hundred" and "seventy-five" appear in the late layers from e = 2–4.

   The middle-layer concepts are mostly the same junk in every condition ("pitifully", "coca cola", "london").

## Follow-ups (night of 2026-11-07)

At the comma, the model always predicts a number, so what the lenses show there is mostly numbers. These follow-ups
look below the top of the lists, read the lenses after a word cue, and test causally.

1. **Words after a cue** (`cue_lens.py`, `analyze_cues.py`; `results/<model>_cues.md`). Each stream is cut, then a
   cue is appended ("Wait,", "The last number is", "Q: Was the last number surprising? A:", …). The lenses are read
   at each cue token. There are two kinds of matched pairs:
   - switch vs control: the same 1000 numbers of N(300, 100), then the same standard normal draws from N(700, 100)
     or from N(300, 100);
   - a single extreme number vs a typical one.

   Findings:
   - **OLMo-3-32B, "Wait,".** After the switch, surprise words gain on normal ones:
     - in all three lenses, at layers 38–47, by +0.4 to +0.67 nats, and in the output by +0.2 to +0.3 nats;
     - the time course is that of a surprise: at its largest from the first new number to e = 8, half gone by
       e = 32, almost gone by e = 128.

     The top movers in the middle layers (J-lens, J++ lens) are exclamations: "WTF", "wow", "HUGE", "REALLY", "Nope",
     "kidding". The control rises on "error", "forgot", "inconsistency". The late layers rise on "those" and "that".
   - **Llama-3.1-8B, "Wait,".** It does not show this; the sign is the opposite.
   - **"The last number is", after the switch (e ≤ 4).** Both models' lenses judge the new number, but differently:
     - OLMo-3-32B describes its size: "huge", "gigantic", "massive", "HUGE", "seven", "seventy" (+1 to +1.9 nats,
       J-lens and J++ lens, middle and late layers). The change set gains on the normal set by +0.46 to +0.93 nats
       in the lenses (p = 0.002) and by +0.13 nats in the output.
     - Llama-3.1-8B's J-lens, late layers, calls it wrong: "fake", "nonsense", "incorrect", "wrong", "ridiculous"
       (+0.4 to +0.5 nats). These words are not in the fixed surprise set, which loses on the normal set here.
     - After a single extreme number, neither model shows this. OLMo-3-32B gains "unclear", "incomplete",
       "ambiguous", "rounded".
   - **Direct questions.** Neither model answers them differently: P(Yes) to "Was the last number surprising?" moves
     by at most 0.013 (OLMo-3-32B: 0.71 either way).
   - **A single extreme number.** It does not trigger the "Wait," response either.
2. **Steering with the lens's word directions** (`steer.py`, `analyze_steer.py`; `results/<model>_steer.md`). At the
   last comma, c‖h‖ times a word set's lens direction is added, and the output's distribution over the integers is
   read.
   - **The integer tokens move the mean in both models** (the positive control):
     - Llama-3.1-8B, 300–399 or 700–799: down by 51, up by 57;
     - OLMo-3-32B, layers 32–56: up to −40 and +38.
   - **Number words.** "thirty" lowers the mean in Llama-3.1-8B (−12 to −25 at layers 22–28; "seventy": +6 to +10).
     In OLMo-3-32B, neither moves it beyond the control words.
   - **Llama-3.1-8B, magnitude words** ("trillion", "billion", "thousand", …):
     - At layers 16–22 they widen the output distribution with a clean dose-response. The sd changes by:

       | c | −1 | −0.5 | +0.5 | +1 |
       |---|---|---|---|---|
       | sd change | −13 to −23 | −5 to −12 | +17 to +24 | +41 to +118 |

     - Control words change it by at most about 11, and random directions by about 0 on average.
     - The effect has the same sign in all 10 sequences.
   - **OLMo-3-32B.** Every word direction, control words included, widens the distribution by a similar amount.
3. **Word sets at every comma** (`wordsets.py`, `checks_words.py`; `results/<model>_checks_words.md`).
   - **Magnitude words track uncertainty in both models.** In the J-lens, at layers 10–28 (Llama-3.1-8B) and 24–56
     (OLMo-3-32B), their readout follows the output's sd three ways:
     - position by position, with condition and time held fixed: partial r = +0.2 to +0.4, against −0.04 to −0.48
       for control words;
     - across σ;
     - after the switch, in time: r = +0.81 to +0.96.

     With the steering, this makes them a readout of the model's uncertainty, causal in Llama-3.1-8B.
   - **Number words ("thirty" … "seventy") are mostly belief, partly echo.** Late layers raise the decade of μ by
     0.7–1.7 nats, and the decade of the last number read by 0.3–0.6 nats.
4. **Checks** (`checks.py`; `results/<model>_checks.md`).
   - **B: null for the surprise-word rise at the switch** (pseudo-switch points). In OLMo-3-32B, J++ beats it at the
     comma after the first new number and over the next 3 numbers, and the logit lens at that comma. The J-lens does
     not.
   - **C: single outliers.**
     - Surprise words rise specifically at the outlier and the comma after it, in both models (layers 8–23). They
       are back to baseline within 1–3 numbers.
     - In OLMo-3-32B, the integer mean of the middle layers overshoots the output by 2–4×, then decays within about
       10 numbers.
   - **D: the lenses' integer means follow the long-run mean, not the last number.** The coefficient on the last
     number is at most about 0.05, against up to 0.8 on μ.
   - **E: after the switch, the middle layers follow the output's uncertainty.** In OLMo-3-32B, the integer mean of
     layers 21–42 follows the output's sd, not its mean:
     - r = 0.87 for the logit lens and 0.78 for the J++ lens;
     - half-lives of 18–25 numbers, against 16 for the sd itself.

     The late layers follow the mean.
5. **Concept Lens with the J++ lens, OLMo-3-32B** (`concept_lens.py --lens jpp`; `results/olmo-3-32b_words.md`,
   last section, and `results/olmo-3-32b_concepts.csv`). Most readouts are generic: "the number" and "the answer"
   in the middle layers, and in the early layers unrelated concepts ("chessboard", "hexadecimal number",
   "city or country"). Two exceptions:
   - **"a large number" names the spread, not the size.** In the middle layers at t = 999 its share of readouts
     grows with σ (at μ = 500), from 0.01–0.04 at σ ≤ 50 to 0.11, 0.15, 0.23, 0.35 and 0.29 at σ = 80–200. It stays
     flat across μ (0.15–0.18 for μ = 300–700 at σ = 100). This is the concept-level counterpart of the magnitude
     words (follow-up 3). At t = 127, "numbers in the sequence" and "numbers in a row" grow with σ in the same way
     (from 0 to 0.23–0.27).
   - **At the switch, "prediction" and "predictions".** These appear in the middle and late layers from the first
     new number to e = 16 (0.06–0.16 of readouts, against 0.00–0.01 before the switch and at e = 999). The
     scaffold echo ("x is a concept", "x is y") peaks at the same time (up to 0.46 at e = 2), and "seven is a
     number" appears in the late layers (up to 0.09). No concept for surprise or change appears; the closest is
     "right or wrong", in under 0.5% of readouts.

## Caveats

- **Mostly descriptive.** A word in the lens is what the lens says, not proof that the model uses that concept.
  The steering (follow-up 2) is the only causal test so far. It is causal for magnitude words in Llama-3.1-8B,
  and non-specific in OLMo-3-32B.
- **Number words echo the last numbers read, but mostly follow the belief** (follow-up 3). Late layers raise the
  decade of μ 2–3× more than the decade of the last number.
- **A lens mean that tracks μ is not yet a computed belief.** The smoke test already shows r(mean, μ) ≥ 0.9 from
  layer 0–2. The residual stream at a comma holds a mix of the numbers seen so far, and the unembedding of that mix
  points to numbers near their average. A lens can therefore track μ by copying, before any inference. The early
  layers are the baseline to compare the middle layers against, and σ (which copying does not give as directly) is
  the stronger test.
- **Out of distribution for the lenses.** They were fitted on WikiText, and number streams are far from it. The
  first 8 numbers (tokens < 16) are before the positions the lenses were fitted on; the analysis bins them apart.
- **Concept Lens on streams.**
  - The authors built it for cloze prompts. Here its proposal scaffold contains up to 2000 numbers, far from that
    setting.
  - Proposals are the bottleneck in their paper (a concept must be proposed to be ranked), and could be more so here.
  - It reads one position per prompt, so it is run at a few times only.
- **J++ on one model only.** Neuronpedia has no J++ lens for Llama-3.1-8B. The J-lens / J++ comparison is on
  OLMo-3-32B, where the J++ lens was fitted on 96 prompts (`config.yaml` beside the lens).
- **The paper's model is Llama-3.2-1B, not 8B.** The go / no-go checks that the 8B model holds the belief before
  any reading of its workspace.
