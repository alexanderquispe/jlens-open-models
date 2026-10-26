# 04 · Is the intermediate that the J-lens reads causally used?

**Question.** In *"The official language of the country whose capital is Lima is"*, the model needs the
intermediate "Peru". Gurnee et al. (2026) swap an intermediate's J-lens coordinate for another entity's and the
answer follows in 54–70% of trials on Claude models. Does that hold on an open model, and is it specific to
J-lens directions?

```bash
python experiments/04_multihop_swap/multihop_swap.py --model llama3.1-8b
```

Outputs: `results/<model>_swap.csv` (layer × position × patch), `results/<model>_readout.csv` (rank of the
intermediate country in each lens), `figures/<model>.{pdf,png}`.

## Method

172 prompts: 43 countries × 4 phrasings (`jlens_open/facts.py`). On prompts the model answers correctly (among the
languages), country A is replaced by a country B with another language, after block *l*, at the last token or at
the capital's last token; success = the answer becomes B's language.

| Patch | Displacement |
|---|---|
| J-lens swap | *s (v_B − v_A)*, unit J-lens vectors of " B" and " A" (centered over the vocabulary) |
| logit-lens swap | the same with logit-lens vectors |
| random swap | the same with two random unit vectors |
| coordinate swap (×1, ×4) | *c_A (v_B − v_A)*, *c_A* the coefficient of *v_A* in the k = 16 J-space decomposition (the paper's swap; zero when *v_A* is not among the 16 atoms) |
| centroid swap | *m_B − m_A*, mean activations over the other phrasings of each country's prompt; also its part in span(*v_A*, *v_B*) and the rest |

For the first three, *s = α · mean‖h − mean‖* at that layer and position, α ∈ {0.5, 1, 2}.

## Results

Pending (overnight run of 2026-10-25).
