# 03 · How much of the residual stream is J-space?

**Question.** Gurnee et al. (2026) define the J-space at a layer as the sparse nonnegative combinations of J-lens
vectors (the rows of *W_U diag(g) J_l*, one per token), and report that (i) the J-space component carries
"never more than 10%" of the activation variance and (ii) ablating it in a band of layers destroys multi-hop
reasoning while MMLU stays intact. Is 10% small compared with what any dictionary of 128k directions would
capture? And does the ablation result hold on an open model, against controls?

```bash
python experiments/03_jspace/jspace.py --model llama3.1-8b
```

Outputs: `results/<model>_variance.csv` (layer × dictionary × k), `results/<model>_ablation.csv`
(band × dictionary × k), `figures/<model>.{pdf,png}`.

## Method

- **Decomposition** (`jlens_open/jspace.py`): nonnegative orthogonal matching pursuit. Each step adds the unit
  atom most aligned with the residual and refits all coefficients by nonnegative least squares (the paper uses
  gradient pursuit, a faster approximation of the same).
- **Centering the lens vectors.** Adding one vector to every row of *W_U* does not change the softmax, so that
  shared component carries no readout. Uncentered, it dominates: in GPT-2 small, the unit J-lens vectors have
  cosine ≈ 0.99 with their mean, so they fill a narrow cone and nonnegative combinations of them explain
  almost nothing of the centered activations (< 1%). The main dictionaries are therefore centered over the
  vocabulary; the uncentered ones are reported too.
- **Dictionaries**, all with one atom per vocabulary token: J-lens, logit lens (J = I), random isotropic,
  random with the activations' covariance (the chance level for a dictionary aligned with where activations
  vary), and random with the J-lens vectors' covariance (same second moments as the J-lens dictionary, so the
  same directions of high downstream sensitivity, but no atom tied to a token). Top-k PCA is shown as a
  reference (one fixed subspace, not a per-token choice). `--dicts` recomputes only the named dictionaries and
  keeps the other rows of existing results (a partial run writes the same files as a full one).
- **Data:** mean-centered activations at positions ≥ 16 of 16 wikitext paragraphs (128 tokens each;
  `data/wikitext_sample.json`, wikitext-103 test split, CC BY-SA 3.0).
- **Ablation:** in a band of 4 layers, every position except BOS loses its k-atom component (k = 10, 25). Measured
  on two-hop prompts (*"… the official language of the country whose capital is Lima is"*), their one-hop
  controls (*"… the official language of Peru is"*: same answers, no intermediate entity; prompts in
  `jlens_open/facts.py`) and the wikitext next-token loss.

## Results

Pending (overnight run of 2026-10-25).
