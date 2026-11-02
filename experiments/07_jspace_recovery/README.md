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
```

With the default isotropic noise only the unembedding, the final norm's gain and the lens are used: the model is
loaded for its weights but never run. The two other noises run it on 64 paragraphs of experiment 03's wikitext
sample. Outputs: `results/<model>_geometry.csv` (one row per dictionary and layer),
`results/<model>_distributions.csv` (quantiles of the nearest-neighbour cosines and of the spectrum),
`results/<model>_duplicates.csv` (the largest clusters of near-duplicate atoms and the closest pairs, with their
tokens), `results/<model>_recovery.csv` (noise × dictionary × weights × s × k; a run adds or replaces only the rows
of its own noises, and `--native-only` only their k = 0 rows), `figures/<model>.{pdf,png}`. The activations'
covariances (64 MB per layer) go to the group's storage, `$JLENS_SHARED/experiments/07_jspace_recovery/`, not to
the repository.

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
61 minutes). On the activations alone, with nothing planted:

| Dictionary | Layer 8 | Layer 16 | Layer 20 | Layer 24 |
|---|---|---|---|---|
| J-lens: occupancy (median over tokens) | 17 (12) | 22 (17) | 18 (15) | 20 (16) |
| J-lens: share of *\|y\|²* explained in 25 steps | 0.120 | 0.123 | 0.131 | 0.142 |
| logit lens: occupancy | 2 | 5 | 12 | 24 |
| random: share explained in 25 steps | 0.087 | 0.087 | 0.087 | 0.087 |

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

- **The activations alone give an occupancy near the paper's.** The rule reports 17–22 for the J-lens at layers
  8–24, close to the paper's ≈ 25, well below the 100-step cap. But in 25 steps the J-lens explains only 12–14% of
  *|y|²*, against 8.7% for the random dictionary. The logit lens has few words of its own in the middle layers
  (occupancy 2–5 at layers 8–16) and more near the output (24 at layer 24).
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
  stays near the occupancy of the activations alone instead of falling with *s* as in isotropic noise. At layer 8
  it is 14–20 for every such k and *s* ≤ 0.2.
- **With words planted, a Gaussian with the activations' covariance gives nearly the same occupancy at layers 8–16,
  less at layers 20–24.** At layer 16: 21–23, 26–28 and 35–36 at *s* = 0.05, 0.1 and 0.2 (activations: 22–24,
  26–27, 35–38); at layer 8, 14–21 (activations 14–20). At layers 20–24 and *s* = 0.05 the Gaussian gives 12–17
  against 18–21 for the activations: there, the activations carry something the rule picks up beyond their
  covariance. Where the two agree, the rule's output is set by the activations' covariance (how much of their
  variance lies along J-lens directions), which a Gaussian without sparse words shares, and is not evidence of
  how many sparse words are present.

## Caveats

- Real activations change the picture for the J-lens (above), so the isotropic results are a best case. In real
  activations, a planted word missed among the first k picks may have lost to a word that is truly in the
  activation, so recall there understates what the pursuit gets right.
- The activations are 7,168 tokens from 64 wikitext paragraphs (positions ≥ 16); other kinds of text are not
  covered.
- The back-of-envelope prediction assumes isotropic noise; it does not apply to the two other noises.
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
