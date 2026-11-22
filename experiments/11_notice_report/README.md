# 11 · The model notices, but does it say?

**Question.** Experiment 09 found a gap. After a jump in the number stream the inside registers it (OLMo-3-32B's
lenses after "Wait,": +0.41 to +0.57 nats of surprise over normal words; Llama-3.1-8B's logit lens at the comma:
+1.17 at layer 19), but asked "Was the last number surprising?", P(Yes) moves by at most 0.013. Is this *access
without report*, and does the inside signal cause the report at all?

**Contexts** (`stimuli.py`, `data/contexts.jsonl`): experiment 09's cue contexts, extended to 20 streams (streams
0..9 are experiment 09's, checked). Stream k of N(300, 100), 1000 numbers, then e + 1 numbers from N(700, 100)
("jump") or N(300, 100) ("control"), same draws, e in −1, 0, 1, 2, 4, 8, 32. Plus single last numbers at 2.5 sd
and 4.5 sd ("moderate", "extreme") against a typical one.

**Models.** Two families, base and instruct: Llama-3.1-8B and Llama-3.1-8B-Instruct (both with a J-lens),
OLMo-3-7B (J-lens) and OLMo-3-7B-Instruct, OLMo-3-32B (J-lens and J++ lens) and OLMo-3.1-32B-Instruct. The
instruct OLMo models have no lens: behaviour, probes and the logit lens only.

## Steps

0. **Ask** (`ask.py`). Base models: experiment 09's cues ("Wait,", the comma) and five questions, including one
   with reversed polarity ("Do the last numbers look like the earlier ones?") and a two-option question in both
   orders, plus a two-example few-shot frame in both orders of the examples (`prompts.py`). Instruct models: the
   stream in a user message, the same questions, "Answer with Yes or No only." Read at the last prompt token: the
   answer's log-probabilities in the output and under the lenses at every layer, and the residual stream for
   probes.
1. **Inject** (`inject.py`). At cut e = 4, the tail of each prompt (everything after the stream's last number) is
   re-run with one change at one layer, at every tail token. Changes use directions fitted on the other 19 streams:
   add α times the mean jump − control difference at that token (α in −2, −1, 1, 2, 4; +1 on a control stream
   transplants the average jump state at that layer), or the difference at the comma; set the component along that
   direction to the other arm's mean; steer along the J-lens direction of surprise words or control words; random
   directions, including ones with the same norm as the difference. Readouts: the answers, the surprise/normal sets
   after "Wait,", and the predicted mean and sd at the comma.
2. **Ablate** (`ablate.py`, Llama-3.1-8B). Experiment 09 showed that adding the J-lens direction of magnitude words
   at layers 16–22 widens the predicted distribution (sufficiency). Here the direction is mean-ablated at every
   token and every layer 16–22, over streams of N(500, σ) for nine σ: does the predicted sd follow σ less
   (necessity)? Controls: control words, "wide" words, random directions, one layer only.

```bash
python experiments/11_notice_report/stimuli.py
sbatch --qos=debug --time=00:30:00 --gres=gpu:nvidia_h200:1 hpc/gpu.sbatch .venv-tf5/bin/python experiments/11_notice_report/ask.py --model llama3.1-8b
sbatch ... inject.py --model llama3.1-8b ;  sbatch ... ablate.py --model llama3.1-8b
python experiments/11_notice_report/analyze.py     # CPU, every model found; results/summary.md
```

## Outcomes, fixed before the run

- **Report.** Change score = log P(the answer meaning "changed / surprising") − log P(the other), jump − control,
  pooled over e in {1, 2, 4, 8, 32}, sign-flip test over streams; AUC of jump vs control. A report needs the same
  sign in the reversed-polarity question and in both orders of the two-option question (otherwise it is an answer
  bias).
- **Access without report** = the jump is decodable at the answer token (held-out probe AUC well above 0.5) or the
  answer read inside (lens, middle layers) moves, while the output answer does not.
- **Causal.** Transplant share = effect of +1 × difference on control / natural gap; removal share = −effect of
  −1 × difference on jump / gap; dose-response up to 4× against norm-matched random directions. If even 4× the jump
  difference does not move the answer while it moves the number guess, the representation is used but not routed
  to the report ("can't"). If the injected difference moves the answer, the route exists but the natural jump is
  too weak to drive it.
- **Ablation.** Necessity if the slope of log sd on log σ falls with the magnitude direction removed, beyond the
  control and random directions (bootstrap over streams).

## Results

All six models; full tables in [results/summary.md](results/summary.md) (csv: `report_output.csv`,
`change_2x2.csv`).

**Added after the first results: a 2 × 2 control.** Jump − control mixes two things: the stream changed, and its
recent numbers are higher (700 vs 300). A probe or an answer that reads "the numbers are high" separates the arms
perfectly. The mirror block (`stimuli.py --down`, `data/contexts_down.jsonl`) adds down jumps (700 → 300) and
streams that are high throughout, with the same draws. Crossing history (low / high) with recent numbers (low /
high), **change = (LH + HL − LL − HH) / 2** is the interaction: what "changed?" should track, and what a reader of
the recent numbers or of the history alone cannot produce. Probes are refitted on it, and the up-jump direction is
tested on the down pair (transfer: near 1 = it reads change, near 0 = it reads the recent numbers).

**What the models say.** AUC of the output answer, jump vs control (first number) and change cells vs the others
(second number), pooled cuts:

| model | q_surprise | q_same (reversed) | q_ab | q_ba | best other format |
|---|---|---|---|---|---|
| Llama-3.1-8B | 0.76 / 0.53 | 0.67 / 0.56 | 0.56 / 0.50 | 0.68 / 0.62 | few-shot 0.59 / 0.55 |
| Llama-3.1-8B-Instruct | 0.74 / 0.59 | **0.97 / 0.95** | **0.98 / 0.87** | **0.92 / 0.88** | – |
| OLMo-3-7B | 0.58 / 0.50 | 0.52 / 0.55 | 0.47 / 0.49 | 0.53 / 0.52 | "Wait," 0.67 / 0.59 |
| OLMo-3-7B-Instruct | 0.46 / 0.47 | 0.55 / 0.54 | 0.48 / 0.48 | 0.52 / 0.53 | – |
| OLMo-3-32B | 0.64 / 0.58 | 0.70 / 0.63 | 0.56 / 0.57 | 0.55 / 0.56 | "Wait," 0.80 / 0.76; few-shot 0.87 / 0.65 |
| OLMo-3.1-32B-Instruct | **0.97 / 0.85** | **0.97 / 0.98** | **0.99 / 0.98** | **0.93 / 0.82** | – |

- Two instruct models report the change itself, with the same sign under reversed polarity and in both option
  orders: Llama-3.1-8B-Instruct (change effect on "Do the last numbers look like the earlier ones?" +0.71 nats,
  already +0.31 after one new number) and OLMo-3.1-32B-Instruct (+0.63 and +0.55). Their answers depend little on
  the level of the numbers once change is accounted for.
- The base models and OLMo-3-7B-Instruct barely report it: on the questions the base models' change effects are at
  most 0.09 nats and OLMo-3-7B-Instruct's go both ways (−0.27 to +0.21), AUC ≤ 0.65. What moves their answers is how high the numbers are (e.g. "history" main effect on q_ab +1.01 in
  Llama-3.1-8B; +1.88 to −2.48 across OLMo-3-7B-Instruct's questions). OLMo-3-32B's few-shot "reports" (0.87
  against control) shrink to 0.65 in the 2 × 2: mostly a reaction to high numbers. Its "Wait," cue is the one
  base-model readout that tracks change (0.76).
- The question "Did the numbers change?" is weak everywhere (≤ 0.71): the numbers always change.

**Access without report.** Change probe at the last prompt token (difference of means on 19 streams, held-out
stream), over the five questions:

| model | change AUC, best layer | up → down transfer, middle layer | output change AUC |
|---|---|---|---|
| Llama-3.1-8B | 0.90–1.00 | 0.65–0.77 | 0.50–0.62 |
| Llama-3.1-8B-Instruct | 1.00 | 0.95–0.99 | 0.59–0.95 |
| OLMo-3-7B | 0.62–0.72 | 0.54–0.70 | 0.45–0.55 |
| OLMo-3-7B-Instruct | 0.64–0.71 | 0.53–0.58 | 0.46–0.54 |
| OLMo-3-32B | 0.90–0.99 | 0.67–0.86 | 0.52–0.63 |
| OLMo-3.1-32B-Instruct | 1.00 | 0.95–1.00 | 0.57–0.98 |

In Llama-3.1-8B and OLMo-3-32B the change is linearly decodable where the model answers, but the answer does not
follow it: access without report, now with the recent-numbers confound removed. In the two instruct models that
report, the direction found on up jumps also separates down jumps from high streams (transfer ≥ 0.95): one
direction-free change code. OLMo-3-7B (base and instruct) carries little of it at the answer token. The answer read
inside with the lenses moves more in the middle and late layers than in the output (e.g. Llama-3.1-8B, "Did the
numbers change?": change effect +0.10 in the middle third, +0.14 late, +0.02 at the output), but lens log-odds at
inner layers are not on the output's scale.

**Causal (injection at e = 4, jump − control direction).** Adding the difference to the tail tokens of a control
prompt moves the answer by about the natural gap at every layer from 12 on (transplant shares 0.8–1.4 for
q_surprise in Llama-3.1-8B, Llama-3.1-8B-Instruct, OLMo-3-32B and OLMo-3.1-32B-Instruct), and 4 times the difference
moves it about 5 times as far (Llama-3.1-8B: +0.26 nats against +0.05 natural; random directions of the same norm
mostly within ±0.07, at most ±0.24). By the rule fixed in advance, the route to the answer exists in the base models and the natural signal is too
weak to drive it. **Caveat:** the injected direction is jump − control, so it carries the recent numbers' level
too, and the base models' answers react to that level; the injection does not isolate the change component. For
the number guess (comma) the transplant share grows with depth (Llama 0.0 at layer 8 → 0.9 at layer 28): the tail
tokens matter for the guess only from the middle layers on. Steering along the J-lens direction of surprise words
lowers the "surprising" answer about as much as control words do: not specific.

**Ablation (Llama-3.1-8B): necessity not supported.** Mean-ablating the magnitude-word direction at layers 16–22
leaves the predicted sd's dependence on σ intact: slope of log sd on log σ 0.947 against 0.955 without ablation
(change −0.008 [−0.009, −0.007]; control words −0.002, random directions ≤ 0.001). Sufficient (experiment 09) but
not necessary: the uncertainty is carried elsewhere too.

**Reading.** The inside of the base models registers the change, beyond the level of the numbers, but their
answers to direct questions mostly read the level. Instruction tuning turns the inside signal into a report in
Llama-3.1-8B and OLMo-3.1-32B, not in OLMo-3-7B, where the signal is weak at the answer token to begin with.
**Caveats:** 20 streams; the 2 × 2 cells share draws by construction; the instruct OLMo models have no lens; the
"history" main effect mixes the context's overall level with its history.

## Step 3: injecting the pure change direction (2026-11-10)

`inject_change.py` (GPU), `analyze_change.py` (CPU); full tables in
[results/inject_change.md](results/inject_change.md), csv `results/inject_change.csv`. The injection of step 1 adds
jump − control, which mixes "the stream changed" with "the recent numbers are high". Here the direction is the
2 × 2 interaction of the residual stream at each tail token, **C = (LH + HL − LL − HH) / 2**, at cut e = 4, fitted
without the test stream. It is added to the unchanged cells (LL, HH) or subtracted from the changed cells (LH, HL)
at one layer (6 layers from 1/4 to 7/8 depth), with strengths −4 to +4. Controls: the recent-numbers and history
directions of the same design, and random directions with the norm of C. A report of change must move all five
questions toward their "changed" answer (q_same has reversed polarity; q_ab and q_ba swap the options). Llama-3.1-8B
and OLMo-3-32B, base and instruct, 20 streams; 25 min of H200 in total (with the smoke test).

```bash
sbatch --qos=debug --time=00:30:00 --gres=gpu:nvidia_h200:1 hpc/gpu.sbatch .venv-tf5/bin/python experiments/11_notice_report/inject_change.py --model llama3.1-8b
python experiments/11_notice_report/analyze_change.py
```

**Instruct models: the change direction carries the report.** Effect on the report score (nats) of +1 × C on the
unchanged cells, against the natural change effect:

| question | Llama-3.1-8B-Instruct: natural | +1 × C, layers 16–28 | OLMo-3.1-32B-Instruct: natural | +1 × C, layers 24–56 |
|---|---|---|---|---|
| q_surprise | +0.12 | +0.11 to +0.15 | +0.33 | +0.32 to +0.34 |
| q_change | +0.17 | +0.14 to +0.18 | +0.06 | +0.06 to +0.08 |
| q_same (reversed) | +0.86 | +0.84 to +0.86 | +0.70 | +0.69 to +0.71 |
| q_ab | +0.64 | +0.65 to +0.67 | +0.69 | +0.68 to +0.70 |
| q_ba | +0.41 | +0.37 to +0.41 | +0.15 | +0.13 to +0.16 |

- All five questions move toward "changed" at every layer from 12 (Llama) and 16 (OLMo) on; −1 × C on the changed
  cells removes the same amount; 4 × C gives roughly 3 to 6 times the effect (up to +4.5 nats on q_same in Llama).
- In Llama-3.1-8B-Instruct the effect at layer 8 is near 0 and at layer 12 half or more of it: the change reaches the
  question tokens between layers 8 and 16.
- Random directions of the norm of 4 × C average −0.12 to +0.10 over the questions (single questions up to 2.0 in
  OLMo, with no consistent sign); the recent-numbers direction at +1 moves the score by at most 0.15.

**Base models: the same direction moves the number guess, hardly the answer.**

| | Llama-3.1-8B | OLMo-3-32B |
|---|---|---|
| natural change effect on the answers | +0.01 to +0.05 | +0.01 to +0.07 |
| +1 × C on the answers | −0.06 to +0.06 | −0.01 to +0.09 |
| +4 × C on the answers | −0.14 to +0.18 | −0.09 to +0.35 |
| layers where all five questions move toward "changed" at +4 | 12, 16 | none (q_change goes slightly negative) |
| +4 × C on the sd of the number guess (natural change effect) | +34 to +114 (+116) | +31 to +97 (+134) |
| +4 × C on surprise words after "Wait," (natural) | +0.05 to +0.21 (+0.01) | +0.65 to +1.13 (+0.26) |

The norm of C is of the same order in base and instruct models (Llama, layer 16: 0.6 and 0.4), so the base models
do not lack the signal. Four times the natural signal moves their answers by at most 0.35 nats, against 0.3 to 4.5
in the instruct models, while it widens their number guess by about the natural amount and, in OLMo-3-32B, raises
the "Wait," response four-fold.

**Reading.** In the outcome terms fixed before the run, the base models are the "can't" case: the change signal is
present and used (for the spread of the prediction, and for "Wait,"), but it is not routed to the answer of a
direct question. In the instruct models one linear direction, free of the level of the numbers, is enough to
produce and to remove the report.
**Caveats:** adding the mean difference at every tail token replaces much of what the later layers of the tail
compute from, so a share near 1 shows that the signal is already at the question tokens and acts linearly, not
which component reads it; the direction is fitted per question format and at e = 4 only; t values over 20 streams
that share one direction overstate the evidence; numbers only.
