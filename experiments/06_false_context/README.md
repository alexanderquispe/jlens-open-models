# 06 · Does the truth reach the workspace when the context pushes a falsehood?

**Question.** Halawi, Denain and Steinhardt (ICLR 2024, *Overthinking the Truth*) show that with wrong few-shot
labels, a model copies the wrong pattern, yet early-exit readouts at middle layers still pick the correct label
until "critical layers" override it. Read with the J-lens: does the true answer become verbalizable before the
model outputs the false one, and does the J-lens see it better or earlier than the logit lens? This is the
situation of an agent receiving adversarial context from another agent.

```bash
python experiments/06_false_context/false_context.py --model llama3.1-8b
```

Outputs: `results/<model>.csv` (task × condition × lens × layer), `results/<model>_final.csv` (model output),
`figures/<model>.{pdf,png}`.

## Method

Six tasks with 12 demonstrations each, true or with labels permuted consistently:

| Task | Format | Permutation |
|---|---|---|
| parity | *Number: 37 / Parity: odd* (10–99) | even ↔ odd |
| continent | *Country: France / Continent: Europe* (48 countries) | Europe → Asia → Africa → Europe |
| color | *Object: banana / Color: yellow* (30 objects) | red → yellow → green → blue → white → black → red |
| animal | *Animal: eagle / Class: bird* (40 animals) | mammal → bird → fish → insect → reptile → mammal |
| language | *Country: Brazil / Language: Portuguese* (26 countries) | French → Spanish → German → Portuguese → Arabic → French |
| sentiment | *Word: lovely / Sentiment: positive* (30 words) | positive ↔ negative |

Three demonstration samples per query. At the query's last token, every layer is read with both lenses,
restricted to the task's labels: *true* = the lens picks the true label, *false* = it picks the label the
demonstrations teach.

`results/<model>_when_output_false.csv` repeats the readout only on false-demonstration prompts where the model
outputs the taught (false) label: did the lens see the truth on the prompts where the model then says the
falsehood?

The tables for each model below cover the first three tasks (run first); the last section covers all six.

## Results (Llama-3.1-8B)

Model output with false demonstrations (`results/llama3.1-8b_final.csv`):

| Task | Outputs the true label | Outputs the taught (false) label |
|---|---|---|
| parity | 0.81 | 0.19 |
| continent | 0.21 | **0.62** |
| color | 0.76 | 0.02 |

Only the continent task makes the model follow the falsehood (with true demonstrations it is 98–100% right on
all three). On the **89 continent prompts where it outputs the false continent**, the lenses at the query's
last token pick the true continent with these frequencies (chance 1/3):

| Layer | 5 | 7 | 8 | 9 | 10 | 11 | 12 | 14 | 16 | 18 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| J-lens | 0.51 | 0.69 | 0.76 | **0.87** | 0.76 | 0.78 | 0.48 | 0.38 | 0.25 | 0.09 | 0.09 |
| logit lens | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 | 0.49 | 0.16 | 0.19 | 0.22 | 0.16 |

![False context, Llama-3.1-8B](figures/llama3.1-8b.png)

- **The truth is in the workspace before the model says the falsehood.** In layers 7–11 the J-lens reads the true
  continent on 69–87% of the prompts the model ends up answering falsely; the logit lens reads it on 24% (below
  chance). The truth fades in layers 12–18, where the taught label takes over: Halawi et al.'s "critical layers",
  seen here with the J-lens and not with the logit lens.
- On all continent prompts with false demonstrations, the J-lens picks the true continent at layers 8–11 about
  as often as with true demonstrations (0.67–0.78 vs 0.63–0.76): the false context does not change the early
  computation, it overrides it later.
- **Parity is not readable** with either lens before layer 27 (they pick the same label for every prompt), so
  this task says nothing about the workspace. Color: only 2 prompts output the false color, too few to condition on.

## Results (Llama-3.1-8B-Instruct)

With false demonstrations, the instruction-tuned model follows the falsehood on continents as often (0.64; true
0.03) and resists it on parity (0.74 true) and colors (0.58 true, 0.09 false). On the **92 continent prompts
where it outputs the false continent**, the J-lens picks the true one:

| Layer | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| J-lens | 0.38 | 0.38 | 0.62 | 0.68 | 0.72 | **0.77** | 0.70 | 0.39 | 0.04 | 0.03 |
| logit lens | 0.26 | 0.27 | 0.26 | 0.51 | 0.27 | 0.45 | 0.45 | 0.43 | 0.07 | 0.01 |

![False context, Llama-3.1-8B-Instruct](figures/llama3.1-8b-it.png)

- **Same picture after instruction tuning**: the truth is in the workspace at layers 8–12 on prompts the model
  will answer falsely, and up to layer 11 the J-lens reads the true continent as often with false as with true
  demonstrations (0.57–0.76 vs 0.62–0.78).
- **The override is sharper**: from 0.70 at layer 12 to 0.04 at layer 14, against a fade over layers 12–18 in
  the base model.

## Six tasks, both Llamas: the truth is there only if it forms before the override

The three tasks added later do not reproduce the continent result as such, and the reason is when the true label
forms. For each task (`results/<model>_final.csv`, `_when_output_false.csv`, `<model>.csv`):

| Model | Task | Follows the falsehood | Prompts answered falsely | J-lens reads the truth (true demos, ≥ 0.6) from layer | J-lens picks the truth on prompts answered falsely, best layer |
|---|---|---|---|---|---|
| base | continent | 0.62 | 89 | 8 | **0.87** (layer 9; chance 0.33) |
| base | sentiment | 0.96 | 86 | 11 | **0.81** (layer 12; chance 0.5) |
| base | language | 0.38 | 30 | 14 | 0.50 (layer 8; chance 0.2) |
| base | animal | 0.27 | 32 | 15 | 0.44 (layer 1; chance 0.2) |
| Instruct | continent | 0.64 | 92 | 8 | **0.77** (layer 11) |
| Instruct | sentiment | 0.97 | 87 | 9 | **0.68** (layer 12) |
| Instruct | language | 0.24 | 19 | 14 | 0.42 (layer 10) |
| Instruct | animal | 0.28 | 34 | 15 | 0.29 (layer 9) |

![False context, six tasks, Llama-3.1-8B](figures/llama3.1-8b.png)

- In both models the falsehood takes over at layers 12–14. When the true label is readable before that
  (continent from layer 8, sentiment from layers 9–11), it is in the workspace on most prompts the model then
  answers falsely. When it only forms at layers 14–15 (language, animal), the context wins first and there is
  little to read: on those prompts the J-lens picks the truth at most 0.29–0.50 of the time (chance 0.2), at
  scattered layers and on 19–34 prompts.
- So the "truth before the lie" is a race between when the model retrieves the answer and when it switches to
  the taught mapping, not a general property of false contexts. For sentiment, both lenses see the truth at
  layer 12 (0.81 each in the base model); for continents only the J-lens does.
- With true demonstrations every task is answered correctly (98–100%), so the late-forming labels are known to
  the model; they are just computed after the override layers.
