# 07 · Can the J-space reading be trusted?

**Question.** Gurnee et al. (2026) read an activation as a nonnegative combination of at most k ≈ 25 J-lens vectors
(the rows of *W_U diag(g) J_l*, one per token), found by a greedy pursuit; experiment 03 does the same. With far
more vectors than dimensions (Llama-3.1-8B: N = 128,256 in d = 4096), three things can go wrong: two different
sets of words can explain an activation equally well, a small change of the activation can change the words, and
the pursuit can miss the best set. Is the reading stable and found by the pursuit, and is the occupancy of ≈ 25 a
property of the model or of the measurement?

```bash
python experiments/07_jspace_recovery/recovery.py --model llama3.1-8b
python experiments/07_jspace_recovery/recovery.py --model llama3.1-8b --skip-geometry \
    --noise "activation covariance,activations"                                     # plant into real activations
python experiments/07_jspace_recovery/recovery.py --model llama3.1-8b --skip-geometry \
    --noise "activation covariance,activations" --native-only                       # the noises alone, nothing planted
python experiments/07_jspace_recovery/recovery.py --model llama3.1-8b --plot-only   # redraw the figure

# Width at a fixed vocabulary. Qwen3 and OLMo-3 need the transformers 5 environment (.venv-tf5, top-level
# README); the 32B models one H200 with --mem=192G (hpc/README.md)
for m in qwen3-4b qwen3-8b qwen3-32b olmo-3-7b olmo-3-32b; do
  python experiments/07_jspace_recovery/recovery.py --model $m --skip-geometry \
      --noise "activation covariance,activations" --native-only
  python experiments/07_jspace_recovery/recovery.py --model $m --skip-geometry \
      --noise isotropic --ks 25,50,100,200,400 --shares 0.05,0.1,0.2
done
python experiments/07_jspace_recovery/width.py                                      # all models: d, occupancy
```

With the default isotropic noise only the unembedding, the final norm's gain and the lens are used: the model is
loaded for its weights but never run. The two other noises run it on 64 paragraphs of experiment 03's wikitext
sample. Outputs: `results/<model>_geometry.csv` (one row per dictionary and layer),
`results/<model>_distributions.csv` (quantiles of the nearest-neighbour cosines and of the spectrum),
`results/<model>_duplicates.csv` (the largest clusters of near-duplicate atoms and the closest pairs, with their
tokens), `results/<model>_recovery.csv` (noise × dictionary × weights × s × k; a run adds or replaces only the rows
of its own noises, and `--native-only` only their k = 0 rows), `figures/<model>.{pdf,png}` (only for models with
part A). The activations' covariances (64–100 MB per layer) go to the group's storage,
`$JLENS_SHARED/experiments/07_jspace_recovery/`, not to the repository. `width.py` collects the models into
`results/width.csv` (one row per model and layer) and `figures/width.{pdf,png}`.

## Method

The atoms are those of the decoder in experiment 03 (`jlens_open/jspace.py`): lens vectors centered over the
vocabulary and normalized to length 1. Three dictionaries of N atoms: J-lens (per layer), logit lens (J = I) and
random isotropic.

- **A. Geometry**, at every layer with a lens:
  - each atom's nearest neighbour (largest cosine with another atom); the coherence (largest |cosine|), against
    the Welch bound, the smallest coherence any N unit vectors in d dimensions can have; the number of pairs
    above 0.9, 0.95 and 0.99, and the clusters that pairs above 0.9 form;
  - the spectrum of *(d / N) VᵀV* (mean 1), against the Marchenko–Pastur interval *(1 ± √(d/N))²* that N random
    unit vectors fill;
  - the atoms' norms before normalization (an atom much shorter than the rest was mostly the shared mean that
    centering removed, so its direction is unreliable), and the singular values of *J_l*.
- **B. Plant and recover**, at 1/4, 1/2, 5/8 and 3/4 of the depth (layers 8, 16, 20, 24 for Llama-3.1-8B). In each of
  128 trials, k atoms are drawn uniformly, given positive weights (equal, or lognormal: *e^z*, *z ~ N(0, 1)*) and
  summed; isotropic Gaussian noise is added so that the planted part is a share *s* of *|y|²*. The same trials
  (atom indices, weights, noise) are used for every dictionary. The pursuit runs on *y* for k + max(10, k/4)
  steps. Reported:
  - **recall**: the share of planted atoms among the first k picks; **up to near-duplicates**: a planted atom also
    counts as found when a pick has cosine ≥ 0.9 with it; **all k right**: the share of trials with every
    planted atom found;
  - **occupancy**: the paper's stopping rule (§4.2: "the value of K at which the marginal improvement in
    reconstruction falls below that of a control set of random directions of the same size"). In each trial, the
    first step at which the dictionary's gain in explained *|y|²* falls below the gain of a random isotropic
    dictionary of N atoms (another seed) on the same *y*; reported as the **median over trials**, as the paper
    reports the median over positions. The CSV also has the 25th and 75th percentiles for the noise alone, and
    `occupancy`, the same rule on the gains averaged over trials (the headline of an earlier version of this
    README).
- **B′. Real activations** (`--noise`). The isotropic noise is replaced, with the same trials and the same rescaling
  to a share *s*, by:
  - **activations**: centered block outputs of the same layer, drawn from 7,168 wikitext tokens (64 paragraphs of
    128 tokens, positions ≥ 16). They carry their own J-space words, so *s* is only the planted share;
  - **activation covariance**: Gaussian noise with the covariance of those activations. It has their anisotropy
    but no words, which separates the two effects.

  Every dictionary is run on the activations of each of the four layers (*s* = 1 is skipped: there is no noise).
  The rule is also run on each noise alone, with nothing planted: 1,024 activations, or 1,024 draws of the
  Gaussian, for up to 100 steps (experiment 03 stopped at 25). If the Gaussian, which has no words, gives the
  same occupancy as the activations, the rule cannot tell words from second-order structure.
  **recall within all steps** counts a planted atom found anywhere in the k + max(10, k/4) picks (these two noises
  only).
- **A prediction to test.** With equal weights and isotropic noise, each planted atom carries about *s/k* of
  *|y|²*, while the best of N random directions picks up about *(1 − s) · 2 ln N / d* of the noise. The rule
  therefore stops near *k ≈ s d / ((1 − s) 2 ln N)*, about 19 for Llama-3.1-8B at *s* = 0.1 (the J-lens explains
  7–11% of the centered variance in experiment 03), whatever the number of planted atoms. If that holds, an
  occupancy of ≈ 25 measures the share of the activation that is J-space, not the number of active words.

Grid: k ∈ {5, 10, 25, 50, 100, 200, 400}, *s* ∈ {1, 0.5, 0.2, 0.1, 0.05}, two weight profiles, six dictionaries
(J-lens at four layers, logit lens, random), three noises.

## Results (Llama-3.1-8B)

One H100, 22 minutes (GPT-2 small: 18 minutes). The random dictionary matches the theory, which checks the code:
coherence 0.102 (expected about √(2 ln(N²/2) / d) = 0.106) and eigenvalues 0.676–1.39 (Marchenko–Pastur
0.675–1.389).

**Geometry** (`results/llama3.1-8b_geometry.csv`, `results/llama3.1-8b_duplicates.csv`):

| Dictionary | Atoms with a near-duplicate (≥ 0.9) | Largest cluster | Largest eigenvalue (random: 1.39) | Effective rank / d | Condition number of J |
|---|---|---|---|---|---|
| J-lens, layer 0 | 17,794 (13.9%) | 12,665 | 881 | 0.003 | 7.2 × 10⁶ |
| J-lens, layer 8 | 12,009 (9.4%) | 4,156 | 975 | 0.003 | 4.2 × 10⁶ |
| J-lens, layer 16 | 1,282 (1.0%) | 595 | 267 | 0.032 | 2.1 × 10⁵ |
| J-lens, layer 20 | 670 (0.5%) | 479 | 95 | 0.144 | 2.2 × 10⁴ |
| J-lens, layer 24 | 579 (0.5%) | 464 | 61 | 0.247 | 2.2 × 10⁴ |
| logit lens | 570 (0.4%) | 446 | 83 | 0.353 | – |
| random, isotropic | 0 | 1 | 1.39 | 0.969 | – |

- **J squeezes the early layers.** Up to layer 8 the J-lens atoms live in a handful of directions (effective rank
  0.3% of d; J's condition number 10⁶–10⁷): 7–14% of them have a near-duplicate, and the largest cluster at layer 8
  holds 4,156 word-start fragments (" t", " a", " th", " c", …), with clusters of single letters, digits and capitals.
  From layer 20 the J-lens dictionary looks like the logit lens.
- **Exact duplicates.** In the J-lens and logit-lens dictionaries the never-trained `<|reserved_special_token_*|>`
  rows coincide with byte tokens (cosine 1.0): the largest cluster at layers 16 and 24 (595 and 464 atoms) and in the
  logit lens (446) is these and the byte tokens `�`. They should be dropped from the decoder's dictionary.
- Real near-duplicates are semantic families, not only case variants: number words (" twenty", " thirty", …) and
  digits at layer 16; in the logit lens, pairs such as " Prince"/"Prince".

**Plant and recover, isotropic noise** (`results/llama3.1-8b_recovery.csv`), equal weights, recall among the first
k picks:

| Dictionary | s = 1: k = 25 / 100 / 400 | s = 0.1: k = 5 / 10 / 25 / 50 / 100 |
|---|---|---|
| J-lens, layer 8 | 0.39 / 0.13 / 0.07 | 0.57 / 0.22 / 0.04 / 0.01 / 0.01 |
| J-lens, layer 16 | 0.96 / 0.86 / 0.23 | 0.99 / 0.90 / 0.30 / 0.09 / 0.04 |
| J-lens, layer 20 | 0.99 / 0.95 / 0.38 | 0.99 / 0.96 / 0.45 / 0.15 / 0.05 |
| J-lens, layer 24 | 0.99 / 0.97 / 0.52 | 1.00 / 0.96 / 0.50 / 0.17 / 0.06 |
| logit lens | 1.00 / 1.00 / 0.93 | 1.00 / 0.98 / 0.57 / 0.22 / 0.09 |
| random, isotropic | 1.00 / 1.00 / 0.99 | 1.00 / 0.99 / 0.63 / 0.24 / 0.10 |

- **Without noise the pursuit finds many words** from layer 16 on (86–97% of 100), but fewer than in a random or
  logit-lens dictionary (93–99% of 400), and few at layer 8 (39% of 25).
- **At a 10% share, noise is the limit.** About 10 planted words are found (90–96% in layers 16–24); of 25, only
  30–50%, and no trial gets all 25 right in any dictionary. Lognormal weights are harder (recall 0.21–0.25 at k = 25
  in layers 16–24).
- **Misses are not near-duplicates.** Counting a pick within cosine 0.9 of a planted atom as a hit changes recall by
  at most 0.013: the pursuit loses weak words to noise, not to their twins.
- **The occupancy rule reports the signal share, not the number of words.** With equal weights at *s* = 0.1 the
  median occupancy is 15–19 at layer 16, 17–25 at layer 20 and 14–26 at layer 24 for every planted k from 25 to
  400; the back-of-envelope gives 19. The same holds at the other shares (layer 16: 9 at *s* = 0.05, predicted 9;
  26–31 at *s* = 0.2, predicted 44) and with lognormal weights (10–16 at layer 16). GPT-2 small (d = 768) gives
  3–5 at *s* = 0.1 (predicted 3.9). An occupancy of ≈ 25 is therefore what the rule returns at a 10% share for any
  number of active words above about 25, on Llama-3.1-8B's dictionary, if the rest of the activation behaves like
  isotropic noise. The pooled version of the rule gives the same picture, a few steps higher (20–22, 26–31 and
  29–36). (With the random dictionary as the truth, the rule compares two random dictionaries and its output is
  noise; those curves are shown only for completeness.)

**Planted into real activations** (same CSV, `noise` = `activations` or `activation covariance`; one H100,
61 minutes for the planted grid, 90 seconds for the noises alone). First the noises alone, with nothing planted
(1,024 samples each; median occupancy over samples, 25th–75th percentiles in brackets):

| | Layer 8 | Layer 16 | Layer 20 | Layer 24 |
|---|---|---|---|---|
| J-lens occupancy, activations | 12 (9–16) | 17 (13–22) | 16 (11–21) | 16 (10–23) |
| J-lens occupancy, Gaussian with their covariance | 13 (10–16) | 15 (11–20) | 10 (6–15) | 7 (4–11) |
| logit-lens occupancy, activations | 1 | 3 | 9 | 15 |
| logit-lens occupancy, Gaussian | 1 | 1 | 1 | 1 |
| J-lens share of *\|y\|²* explained in 25 steps, activations | 0.121 | 0.124 | 0.134 | 0.147 |
| the same, Gaussian | 0.113 | 0.104 | 0.095 | 0.090 |
| random dictionary (the control), either noise | 0.087 | 0.087 | 0.087 | 0.087 |

Equal weights, *s* = 0.1, recall among the first k picks, isotropic / activation covariance / activations:

| Dictionary | k = 5 | k = 10 | k = 25 | All 5 right (k = 5) |
|---|---|---|---|---|
| J-lens, layer 8 | 0.57 / 0.29 / 0.23 | 0.22 / 0.08 / 0.07 | 0.04 / 0.02 / 0.01 | 0.12 / 0.00 / 0.01 |
| J-lens, layer 16 | 0.99 / 0.94 / 0.81 | 0.90 / 0.68 / 0.62 | 0.30 / 0.21 / 0.24 | 0.96 / 0.77 / 0.28 |
| J-lens, layer 20 | 0.99 / 0.98 / 0.80 | 0.96 / 0.86 / 0.69 | 0.45 / 0.41 / 0.40 | 0.97 / 0.91 / 0.25 |
| J-lens, layer 24 | 1.00 / 0.98 / 0.79 | 0.96 / 0.90 / 0.70 | 0.50 / 0.48 / 0.45 | 0.99 / 0.91 / 0.20 |
| logit lens (activations of layer 16) | 1.00 / 1.00 / 0.99 | 0.98 / 0.98 / 0.96 | 0.57 / 0.62 / 0.61 | 0.99 / 0.98 / 0.96 |
| logit lens (activations of layer 24) | 1.00 / 1.00 / 0.94 | 0.98 / 0.98 / 0.85 | 0.57 / 0.61 / 0.54 | 0.99 / 0.99 / 0.71 |
| random (activations of layer 16) | 1.00 / 1.00 / 1.00 | 0.99 / 0.98 / 0.98 | 0.63 / 0.63 / 0.62 | 1.00 / 1.00 / 1.00 |

- **The activations alone give an occupancy of 12–17** for the J-lens at layers 8–24 (the paper: ≈ 25 on a model
  of unknown width; the pooled rule gives 17–22), well below the 100-step cap. In 25 steps the J-lens explains
  12–15% of *|y|²*, against 8.7% for the random dictionary. The logit lens has few words of its own in the middle
  layers (1–3 at layers 8–16) and more near the output (15 at layer 24).
- **A Gaussian with the activations' covariance, which has no words, gives the same occupancy at layers 8–16 but
  not at 20–24.** At layers 8 and 16 it gives 13 and 15 against 12 and 17, with nearly the same spread: there the
  rule cannot tell the activations from their second-order statistics. At layers 20 and 24 it gives 10 and 7
  against 16 and 16: the activations carry 6–9 more above-chance J-lens vectors than their covariance accounts
  for. The variance explained says the same more sharply: of the J-lens's excess over the random dictionary on the
  activations (0.034–0.059 of *|y|²*), the Gaussian's excess over the random dictionary on the Gaussian reproduces
  78% at layer 8, 48% at 16, 17% at 20 and 6% at 24. The logit
  lens finds nothing in the Gaussian (1 at every layer) but 9–15 vectors in the activations of layers 20–24.
- **Planted words compete with the activation's own.** In layers 16–24 the J-lens finds 79–81% of 5 planted words
  among its first 5 picks and all 5 in 20–28% of trials (isotropic: 99–100% and 96–99%). Most of the others come
  a few steps later (94–98% within 15 picks). Noise with the activations' covariance costs less at k = 5
  (94–98%), but in layer 16 it accounts for most of the loss at k = 10 (0.90 → 0.68; activations 0.62). That fits
  J-lens atoms crowded into a few directions (effective rank 0.03–0.25 of d) in which the activations also have
  most of their variance; we have not measured that overlap. The logit lens and the random dictionary barely
  change, except the logit lens in the activations of layers 20–24, where it has words of its own. At k = 25 the three noises give similar recall. With lognormal weights the J-lens
  finds 48% of 5 at layers 16–24 (covariance 52–58%, isotropic 61–64%). Layer 8 stays poor (23% of 5).
- **The occupancy still does not count the planted words.** With equal weights and any planted k from 50 to 400,
  the median occupancy at layer 16 is 22–24 at *s* = 0.05, 26–27 at *s* = 0.1 and 35–38 at *s* = 0.2 (isotropic:
  9, 15–17, 26–31). Layers 20 and 24 give 18–21 at *s* = 0.05 and 23–28 at *s* = 0.1. At low shares the output
  stays at or above the occupancy of the activations alone (12–17) instead of falling with *s* as in isotropic
  noise. At layer 8
  it is 14–20 for every such k and *s* ≤ 0.2.
- **With words planted, a Gaussian with the activations' covariance gives nearly the same occupancy at layers 8–16,
  less at layers 20–24.** At layer 16: 21–23, 26–28 and 35–36 at *s* = 0.05, 0.1 and 0.2 (activations: 22–24,
  26–27, 35–38); at layer 8, 14–21 (activations 14–20). At layers 20–24 and *s* = 0.05 the Gaussian gives 12–17
  against 18–21 for the activations, the same split as with nothing planted. Where the two agree, the rule's
  output is set by the activations' covariance (how much of their variance lies along J-lens directions), which a
  Gaussian without sparse words shares, and is not evidence of how many sparse words are present. The baseline
  that separates words from second-order structure is this Gaussian, not the random isotropic dictionary.

## Width (Qwen3, OLMo-3, Llama-3.1)

Does the occupancy grow with the width d? Within a family the vocabulary is fixed (Qwen3: N = 151,936 at
d = 2560, 4096, 5120; OLMo-3: N = 100,278 at d = 4096, 5120), so d changes, and with it the number of blocks
(36 or 64). For each model: the noises alone at 1/4, 1/2, 5/8 and 3/4 of the depth, and the isotropic grid
(*s* ≤ 0.2, k from 25 to 400). One H100 per model up to 8B, one H200 for the 32B models; 10–12 minutes each.

J-lens occupancy of the noises alone (median over 1,024 samples), activations / Gaussian with their covariance
(`results/width.csv`, `figures/width.png`):

| Model | d | 1/4 | 1/2 | 5/8 | 3/4 |
|---|---|---|---|---|---|
| Qwen3-4B | 2560 | 15 / 16 | 13 / 1\* | 10 / 1\* | 11 / 1\* |
| Qwen3-8B | 4096 | 23 / 24 | 17 / 1\* | 17 / 2\* | 20 / 7\* |
| Qwen3-32B | 5120 | 25 / 27 | 24 / 26 | 27 / 30 | 16 / 2\* |
| OLMo-3-7B | 4096 | 12 / 14 | 18 / 17 | 20 / 12 | 21 / 9 |
| OLMo-3-32B | 5120 | 15 / 15 | 20 / 7 | 21 / 7 | 24 / 6 |
| Llama-3.1-8B | 4096 | 12 / 13 | 17 / 15 | 16 / 10 | 16 / 7 |

\* One dimension carries 24–93% of the activations' variance; the Gaussian is no baseline there (below).

- **The occupancy stays in a narrow band and grows with d about as a detection limit would.** Over six models
  and d from 2560 to 5120 the activations give 10–27. Within a family it grows with d: from Qwen3-4B to Qwen3-32B
  (d × 2) by a factor of 1.5–2.7 at the four depths, from OLMo-3-7B to OLMo-3-32B (d × 1.25) by 1.05–1.25;
  Qwen3-8B to 32B is not monotone at 3/4 depth (20 → 16). The share *s* for which the isotropic back-of-envelope
  gives each measured occupancy is 6–12% in all 24 model-layers: the spread across models is about what the change
  of d alone gives at a constant J-space share, with no change in the number of active words. That reading
  assumes the isotropic picture, which the Gaussian results show is only partly right.
- **The detection limit itself grows with d where the dictionary allows it.** Planted at *s* = 0.1 in isotropic
  noise (k from 50 to 400), the median occupancy at 3/4 depth is 9–11, 14–20 and 16–20 for Qwen3-4B, 8B and 32B
  (predicted 12, 19, 24), 16–24 and 21–37 for OLMo-3-7B and 32B (predicted 20, 25), and 14–24 for Llama-3.1-8B
  (predicted 19). At 1/4 depth it is 7–12 whatever d, and so is Qwen3-32B at 1/2 and 5/8: there the formula
  fails, presumably because the J-lens dictionary is degenerate, as Llama's at layer 8 (part A; the geometry of
  the other models was not computed).
- **At 1/4 depth the Gaussian reproduces the occupancy in all six models** (within 2: 13–27 against 12–25).
  Deeper, where it is a baseline, the activations exceed it by 6–18 in Llama-3.1-8B and OLMo-3 from 5/8 depth on
  (OLMo-3-32B from 1/2), but not in Qwen3-32B up to 5/8 depth (26–30 against 24–27).
- **Massive activations break the Gaussian baseline.** In Qwen3-4B and 8B beyond 1/4 depth and in Qwen3-32B at
  3/4, one dimension carries 24–93% of the activations' variance (elsewhere at most 6%). A Gaussian with this
  covariance puts that dimension in every sample; the J-lens then explains less of it than the random dictionary
  does (0.05–0.09 of *|y|²* against 0.07–0.14) and the rule stops at 1–7. Real
  activations presumably carry the dimension on a few tokens only (not checked token by token), so a baseline
  there needs their heavy tails, not only their covariance.

## Caveats

- Real activations change the picture for the J-lens (above), so the isotropic results are a best case. In real
  activations, a planted word missed among the first k picks may have lost to a word that is truly in the
  activation, so recall there understates what the pursuit gets right.
- The activations are 7,168 tokens from 64 wikitext paragraphs (positions ≥ 16); other kinds of text are not
  covered.
- The back-of-envelope prediction assumes isotropic noise; it does not apply to the two other noises.
- Width: Qwen3 and OLMo-3 run under transformers 5.11 and Llama-3.1-8B under 4.43; the families differ in
  training, and the number of blocks changes with d. Qwen3-4B ties its embedding and unembedding, and Qwen3's
  151,936 unembedding rows include 267 padding rows that no token uses, which are atoms too. Each family has two or
  three widths.
- Planted atoms are drawn uniformly over the vocabulary, including rare and never-trained tokens; real readouts
  are dominated by common tokens.
- The recall measures our pursuit (nonnegative orthogonal matching pursuit with 150 projected-gradient steps per
  refit). The thresholds of the theory (Donoho–Tanner, about 600 nonzeros for a random dictionary of this size)
  are for ℓ1 decoders without noise, and need not apply to it. In the early layers, where the atoms are nearly
  parallel, 150 refit steps may not converge; part of the low recall at layer 8 may be the decoder's.
- The occupancy rule follows §4.2 and Figure 30 ("random directions of the same size", percentiles over
  positions), but the paper does not say whether the J-lens vectors or the activations are centered or
  normalized, and it uses gradient pursuit (a cheaper approximation of the refit at every step that we do). Its
  model (Sonnet 4.5) is closed, so the back-of-envelope cannot be checked against its ≈ 25. The paper finds an
  occupancy near zero in the first third of the layers; we do not find that in Llama-3.1-8B (layer 8 in the
  table of the activations alone).
- The paper leans on ≈ 25: §2.3 sets the J-space's sparsity from it ("no more than 25, which we empirically
  observed to be the number of J-lens vectors that are meaningfully active at a given time"), and its ≤ 10%
  share of variance is computed at that K. It notes that vectors are not concepts, but does not discuss the
  dependence on the signal share or on d.
