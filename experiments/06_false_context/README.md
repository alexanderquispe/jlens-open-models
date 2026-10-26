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

Three tasks with 12 demonstrations each, true or with labels permuted consistently:

| Task | Format | Permutation |
|---|---|---|
| parity | *Number: 37 / Parity: odd* (10–99) | even ↔ odd |
| continent | *Country: France / Continent: Europe* (48 countries) | Europe → Asia → Africa → Europe |
| color | *Object: banana / Color: yellow* (30 objects) | red → yellow → green → blue → white → black → red |

Three demonstration samples per query. At the query's last token, every layer is read with both lenses,
restricted to the task's labels: *true* = the lens picks the true label, *false* = it picks the label the
demonstrations teach.

## Results

Pending (overnight run of 2026-10-25).
