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

## Results (Llama-3.1-8B)

The model answers 95.3% of the 172 prompts correctly (among 33 languages); 492 swap trials. Success = fraction
of trials whose answer becomes B's language (`results/llama3.1-8b_swap.csv`), as a range over layers:

| Patch | Capital token, layers 1–15 | Last token, layers 15–26 |
|---|---|---|
| J-lens swap, α = 1 | **0.59–0.80** | **0.65–0.94** |
| logit-lens swap, α = 1 | 0.00–0.14 | 0.12–0.74 |
| logit-lens swap, α = 2 | 0.09–0.41 | 0.63–0.82 |
| random swap, α = 1 | 0.00 | 0.00–0.01 |
| centroid swap (full) | 0.82–0.91 | 0.02–0.95 (≥ 0.85 from layer 19) |
| … its part in span(*v_A*, *v_B*) | 0.00–0.44 | 0.00–0.57 |
| … its part outside | 0.20–0.87 | 0.01–0.21 |
| coordinate swap ×1 (paper) | 0.00 | 0.00–0.02 |

![Swap results, Llama-3.1-8B](figures/llama3.1-8b.png)

- **J-lens directions are causal for the intermediate.** At the capital token, replacing the J-lens vector of
  " Peru" by that of " Brazil" turns *Spanish* into *Portuguese* in up to 80% of trials; the same displacement
  along logit-lens vectors does almost nothing (≤ 0.14) and random directions nothing.
- **The country enters the J-lens span at layer 5.** At the capital token, the full centroid difference works
  from layer 0, but in layers 0–4 its part inside span(*v_A*, *v_B*) does nothing (≤ 0.03) and the part outside
  does it all (0.76–0.87). From layer 5 on, the inside part takes over (0.29–0.44 in layers 6–13) and the
  outside part falls (0.20–0.41). Layer 5 is also where the J-lens starts reading the country at the capital
  token (median rank 4, then 2; in the top 10 for 84–92% of prompts in layers 5–10), while the logit lens ranks
  it in the thousands or worse until layer 18.
- **The intermediate moves to the last token at layers 15–17**: swaps at the capital token stop working at
  layer 17, swaps at the last token start at 15 (J-lens) to 17 (centroid). A J-lens swap at the last token
  already works at layer 15 (0.65), before the country itself is there (centroid swap 0.02): downstream layers
  read country identity from J-lens directions.
- **Contrast with the weekday circle (experiment 02)**: there, the day lens span did nothing at the day token.
  Nameable entities live in the J-space; the circle used to compute day arithmetic does not.
- **The paper's coordinate swap fails here** because the country's atom is rarely among the 16 atoms of the
  nonnegative decomposition of the centered activation (≤ 13% of prompts at the capital token before layer 21,
  20–48% at the last token from layer 17), even though the lens ranks the country second. How the swap's
  strength is chosen matters; with a fixed norm (α), the J-lens direction works.
