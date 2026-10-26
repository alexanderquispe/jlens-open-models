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

## Results (Llama-3.1-8B)

**Variance** (1,792 wikitext tokens per layer; layers 0–30, `results/llama3.1-8b_variance.csv`):

| Dictionary | k = 10 | k = 25 |
|---|---|---|
| J-lens | 0.07–0.11 | 0.11–0.17 |
| random, J-lens covariance | 0.04–0.08 | 0.09–0.13 |
| logit lens | 0.04–0.07 | 0.08–0.12 |
| random, isotropic | 0.04 | 0.09 |
| top-k PCA | 0.16–0.28 | 0.26–0.39 |
| random, activation covariance | 0.35–0.46 | 0.52–0.62 |

- The paper's "never more than 10%" holds: with k = 10 the J-space explains 7–11%. That is 1.8–2.8 times an
  isotropic dictionary but far below a random dictionary shaped like the activations (35–46%): the J-space is
  a small part of the residual stream, not where its variance is.
- Up to layer 14 the random dictionary with the J-lens covariance explains almost as much (within 4–20%: 0.06–0.08
  vs 0.07–0.08): there, the J-lens's edge over chance comes from its geometry. The gap opens from layer 15 and
  from layer 21 on the token-aligned atoms explain more than twice as much (0.10–0.11 vs 0.04–0.05).
- Llama's lens vectors do not fill a narrow cone (norm of the mean unit atom 0.15–0.34 uncentered, against 0.99 in
  GPT-2 small), so centering changes less here (uncentered J-lens: 0.05–0.10).

**Ablation** of the k = 10 component in a band of 4 layers (`results/llama3.1-8b_ablation.csv`). Unablated:
two-hop 0.95, one-hop 0.98, wikitext loss 2.44. Accuracy is among the 33 languages.

| Band | J-lens: two-hop / one-hop / loss | random, J-lens covariance | random, activation covariance |
|---|---|---|---|
| 0–3 | 0.89 / 0.95 / 3.03 | 0.90 / 0.97 / 2.98 | 0.65 / 0.79 / 8.99 |
| 4–7 | 0.73 / 0.91 / 3.41 | 0.92 / 0.97 / 3.33 | 0.78 / 0.98 / 7.48 |
| 8–11 | 0.67 / 0.87 / 3.01 | 0.94 / 0.97 / 2.99 | 0.91 / 0.98 / 7.16 |
| 12–15 | 0.67 / 0.64 / 2.82 | 0.98 / 0.93 / 2.72 | 0.89 / 0.95 / 6.92 |
| 16–19 | 0.60 / 0.82 / 2.84 | 0.90 / 0.97 / 2.60 | 0.92 / 0.98 / 6.49 |
| 20–23 | 0.32 / 0.59 / 3.16 | 0.87 / 0.92 / 2.61 | 0.95 / 0.98 / 6.39 |
| 24–27 | 0.05 / 0.06 / 3.94 | 0.92 / 0.97 / 2.60 | 0.97 / 0.99 / 6.15 |
| 28–31 | 0.05 / 0.03 / 6.20 | 0.82 / 0.92 / 2.67 | 0.95 / 0.97 / 5.91 |
| removed variance | 0.04–0.06 | 0.04–0.05 | 0.09–0.14 |

The logit-lens and isotropic dictionaries remove about as much variance as the J-lens (0.035–0.05) and change
little up to layer 23 (two-hop ≥ 0.91, loss ≤ 2.70).

![J-space, Llama-3.1-8B](figures/llama3.1-8b.png)

- **The token-aligned atoms carry the facts.** The random dictionary with the J-lens covariance removes the same
  variance and raises the loss as much as the J-lens up to layer 15 (2.72–3.33 vs 2.82–3.41), yet two-hop stays
  at 0.90–0.98 where the J-lens ablation brings it to 0.60–0.73. What breaks recall is not removing variance in
  the directions the output is sensitive to, but removing the specific token directions.
- **Double dissociation with high-variance directions.** Removing 9–11% of the variance along random
  activation-shaped directions wrecks next-token prediction (loss 5.9–9.0) but leaves two-hop at 0.89–0.97 from
  layer 8 on. The J-space ablation does the opposite: small loss increase, recall broken.
- **Not specific to multi-hop.** The paper's result (multi-hop destroyed, MMLU intact) suggests a dissociation
  between multi-hop and direct recall; here one-hop recall also falls (0.64–0.91 in bands 4–19 at k = 10, and
  0.16–0.35 in bands 8–19 at k = 25, against two-hop 0.19–0.28). Two-hop is more fragile in 4 of the 5 middle
  bands, not categorically.
- In the last bands the J-lens approaches the logit lens (J → I), and removing the top tokens removes the
  answer itself.

## Results (Llama-3.1-8B-Instruct)

Unablated: two-hop 0.99, one-hop 0.99, wikitext loss 2.57. Variance explained with k = 10: J-lens 0.07–0.11,
J-lens covariance 0.04–0.08, isotropic 0.04, activation covariance 0.35–0.45 (layers 0–30). Ablation, k = 10
(two-hop / one-hop / loss):

| Band | J-lens | random, J-lens covariance | random, activation covariance |
|---|---|---|---|
| 0–3 | 0.98 / 0.96 / 3.25 | 0.99 / 0.99 / 3.19 | 0.80 / 0.99 / 8.60 |
| 4–7 | 0.79 / 0.94 / 3.64 | 0.97 / 0.99 / 3.61 | 0.91 / 0.98 / 7.43 |
| 8–11 | 0.80 / 0.82 / 3.19 | 0.96 / 0.99 / 3.21 | 0.95 / 0.98 / 7.27 |
| 12–15 | 0.90 / 0.85 / 2.96 | 1.00 / 0.98 / 2.85 | 0.94 / 0.97 / 6.97 |
| 16–19 | 0.54 / 0.48 / 2.93 | 0.97 / 0.81 / 2.72 | 0.96 / 0.98 / 6.66 |
| 20–23 | 0.42 / 0.51 / 3.28 | 0.87 / 0.74 / 2.74 | 0.98 / 0.92 / 6.48 |
| 24–27 | 0.08 / 0.06 / 3.90 | 0.97 / 0.87 / 2.73 | 0.99 / 1.00 / 6.19 |

![J-space, Llama-3.1-8B-Instruct](figures/llama3.1-8b-it.png)

The same dissociation as in the base model: up to layer 15 the J-lens-covariance control matches the J-lens
ablation's loss (2.85–3.61 vs 2.96–3.64) but leaves recall at 0.96–1.00, while the J-lens ablation brings two-hop
to 0.79–0.90 (and to 0.42–0.54 at layers 16–23). The instruction-tuned model is somewhat more robust in the early
bands.

## Results (Qwen3-8B)

Run with `transformers` 5.11 (see the top-level README). Unablated: two-hop 0.96, one-hop 0.98, wikitext loss 2.98.
Variance explained with k = 10 (layers 0–34): J-lens 0.07–0.11, J-lens covariance 0.04–0.11, logit lens
0.03–0.10, isotropic 0.04, activation covariance 0.38–0.55, top-k PCA 0.17–0.38. Ablation, k = 10 (two-hop /
one-hop / loss):

| Band | J-lens | random, J-lens covariance | random, activation covariance |
|---|---|---|---|
| 0–3 | 0.96 / 0.99 / 3.15 | 0.96 / 0.98 / 3.19 | 0.35 / 0.41 / 9.78 |
| 4–7 | 0.97 / 0.98 / 3.75 | 0.95 / 0.98 / 3.74 | 0.75 / 0.91 / 9.04 |
| 8–11 | 0.92 / 0.95 / 3.46 | 0.96 / 0.99 / 3.49 | 0.70 / 0.68 / 8.24 |
| 12–15 | 0.41 / 0.40 / 3.18 | 0.75 / 0.83 / 3.21 | 0.67 / 0.66 / 8.19 |
| 16–19 | 0.56 / 0.63 / 3.14 | 0.84 / 0.91 / 3.10 | 0.66 / 0.85 / 7.96 |
| 20–23 | 0.44 / 0.61 / 3.25 | 0.88 / 0.92 / 3.13 | 0.64 / 0.69 / 7.91 |
| 24–27 | 0.37 / 0.33 / 3.41 | 0.86 / 0.89 / 3.12 | 0.92 / 0.99 / 7.29 |
| 28–31 | 0.12 / 0.13 / 4.35 | 0.85 / 0.90 / 3.19 | 0.95 / 0.98 / 6.78 |
| 32–35 | 0.05 / 0.05 / 7.86 | 0.91 / 0.91 / 3.36 | 0.95 / 0.99 / 6.53 |
| removed variance | 0.04–0.06 | 0.04–0.06 | 0.09–0.20 |

![J-space, Qwen3-8B](figures/qwen3-8b.png)

- **Same direction, weaker dissociation.** In layers 12–23 the J-lens ablation brings two-hop to 0.41–0.56; the
  J-lens-covariance control removes the same variance with the same loss (3.10–3.21 vs 3.14–3.25) and brings it
  to 0.75–0.88. Unlike Llama, the control does hurt recall, so in Qwen part of the damage comes from the geometry
  (where the output is sensitive) and part from the token alignment. Up to layer 11 neither ablation matters
  (≥ 0.92).
- **The J-lens's variance edge is its geometry up to layer 21.** The J-lens-covariance control explains nearly as
  much (within 1–19%); the token-aligned atoms pull ahead from layer 22 and explain about twice as much from
  layer 29 (0.09–0.11 vs 0.04–0.05).
- **High-variance directions hurt recall here too.** The activation-covariance ablation wrecks the loss
  (6.5–9.8) and, unlike in Llama, also lowers two-hop to 0.35–0.75 in bands 0–23.
- Qwen's uncentered lens vectors fill a narrow cone (norm of the mean unit atom up to 0.92); uncentered, the
  J-lens explains only 0.01–0.04.

## Results (OLMo-3-7B)

Run with `transformers` 5.11. Unablated: two-hop 1.00, one-hop 0.99, wikitext loss 2.76. Variance explained with
k = 10 (layers 0–30): J-lens 0.06–0.10, J-lens covariance 0.04–0.07, logit lens 0.04–0.09, isotropic 0.04,
activation covariance 0.34–0.43, top-k PCA 0.16–0.29. Ablation, k = 10 (two-hop / one-hop / loss):

| Band | J-lens | random, J-lens covariance | random, activation covariance |
|---|---|---|---|
| 0–3 | 0.97 / 0.99 / 2.88 | 0.99 / 0.98 / 2.88 | 0.77 / 0.94 / 9.15 |
| 4–7 | 0.80 / 0.94 / 2.91 | 0.98 / 0.99 / 2.91 | 0.95 / 0.99 / 8.49 |
| 8–11 | 0.87 / 0.96 / 2.89 | 0.97 / 0.99 / 2.88 | 0.97 / 0.99 / 7.07 |
| 12–15 | 0.94 / 1.00 / 2.87 | 0.98 / 0.99 / 2.84 | 0.98 / 0.99 / 6.44 |
| 16–19 | 0.91 / 0.98 / 2.99 | 0.99 / 1.00 / 2.85 | 0.95 / 0.99 / 6.48 |
| 20–23 | 0.67 / 0.85 / 3.29 | 0.95 / 0.98 / 2.90 | 0.86 / 0.99 / 6.18 |
| 24–27 | 0.24 / 0.36 / 3.96 | 0.98 / 0.97 / 2.91 | 0.92 / 1.00 / 6.19 |
| 28–31 | 0.05 / 0.07 / 6.25 | 0.97 / 0.98 / 3.07 | 1.00 / 1.00 / 6.28 |
| removed variance | 0.04–0.05 | 0.04 | 0.07–0.10 |

![J-space, OLMo-3-7B](figures/olmo-3-7b.png)

- **The Llama dissociation, with smaller effects.** In layers 4–23 the J-lens ablation brings two-hop to 0.67–0.94
  while the J-lens-covariance control, with the same removed variance and the same loss up to layer 19
  (2.84–2.91 vs 2.87–2.99), keeps it at 0.95–0.99. The activation-covariance ablation wrecks the loss (6.2–9.2)
  and leaves two-hop at 0.86–1.00 from layer 4.
- **Closer to the paper's multi-hop pattern.** Here one-hop barely moves (0.94–1.00 in bands 4–19, against
  two-hop 0.80–0.94), unlike Llama and Qwen.
- The J-lens explains 12–18% more than the J-lens-covariance control up to layer 14, and about twice as much
  from layer 28 (0.09–0.10 vs 0.04–0.05). Uncentered, the lens vectors fill a narrow cone (norm of the mean unit
  atom up to 0.93) and explain only 0.01–0.02.

## Caveats

- Accuracy is the argmax among the 33 languages, which can survive a broken full distribution (the activation
  covariance ablation).
- Nonnegative OMP with 150 projected-gradient steps approximates the paper's gradient pursuit; the components
  removed are the ones this algorithm finds.
