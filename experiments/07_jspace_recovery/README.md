# 07 · Can the J-space reading be trusted?

**Question.** Gurnee et al. (2026) read an activation as a nonnegative combination of at most k ≈ 25 J-lens vectors
(the rows of *W_U diag(g) J_l*, one per token), found by a greedy pursuit; experiment 03 does the same. With far
more vectors than dimensions (Llama-3.1-8B: N = 128,256 in d = 4096), three things can go wrong: two different
sets of words can explain an activation equally well, a small change of the activation can change the words, and
the pursuit can miss the best set. Is the reading stable and found by the pursuit, and is the occupancy of ≈ 25 a
property of the model or of the measurement?

```bash
python experiments/07_jspace_recovery/recovery.py --model llama3.1-8b
python experiments/07_jspace_recovery/recovery.py --model llama3.1-8b --plot-only   # redraw the figure
```

Only the unembedding, the final norm's gain and the lens are used: the model is loaded for its weights but never
run. Outputs: `results/<model>_geometry.csv` (one row per dictionary and layer), `results/<model>_distributions.csv`
(quantiles of the nearest-neighbour cosines and of the spectrum), `results/<model>_duplicates.csv` (the largest
clusters of near-duplicate atoms and the closest pairs, with their tokens), `results/<model>_recovery.csv`
(dictionary × weights × s × k), `figures/<model>.{pdf,png}`.

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
  - **occupancy**: the paper's stopping rule (§4.2) as we read it: the first step at which the dictionary's gain
    in explained *|y|²*, averaged over trials, falls below the gain of a random isotropic dictionary (another
    seed) on the same *y*. Also the median over trials of the same rule applied trial by trial.
- **A prediction to test.** With equal weights and isotropic noise, each planted atom carries about *s/k* of
  *|y|²*, while the best of N random directions picks up about *(1 − s) · 2 ln N / d* of the noise. The rule
  therefore stops near *k ≈ s d / ((1 − s) 2 ln N)*, about 19 for Llama-3.1-8B at *s* = 0.1 (the J-lens explains
  7–11% of the centered variance in experiment 03), whatever the number of planted atoms. If that holds, an
  occupancy of ≈ 25 measures the share of the activation that is J-space, not the number of active words.

Grid: k ∈ {5, 10, 25, 50, 100, 200, 400}, *s* ∈ {1, 0.5, 0.2, 0.1, 0.05}, two weight profiles, six dictionaries
(J-lens at four layers, logit lens, random).

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

**Plant and recover** (`results/llama3.1-8b_recovery.csv`), equal weights, recall among the first k picks:

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
- **The occupancy rule reports the signal share, not the number of words.** With equal weights at s = 0.1 it reports
  20–22 at layer 16, 26–31 at layer 20 and 29–36 at layer 24 for every planted k from 25 to 400; the
  back-of-envelope gives 19. The same holds at the other shares (layer 16: 11–13 at s = 0.05, predicted 9; 30–38 at
  s = 0.2, predicted 44) and with lognormal weights (10–19 at layer 16). GPT-2 small (d = 768) reports 5–6 at
  s = 0.1 (predicted 3.9). An occupancy of ≈ 25 is therefore what the rule returns at a 10% share for any number of
  active words above about 25, on Llama-3.1-8B's dictionary, if the rest of the activation behaves like isotropic
  noise. (With the random dictionary as the truth, the rule compares two random dictionaries and its output is
  noise; those curves are shown only for completeness.)

## Caveats

- The noise is isotropic. The real non-J-space part of an activation is not: it is concentrated in the
  activations' high-variance directions (experiment 03: a random dictionary shaped like the activations explains
  35–46%). A follow-up should plant words into real activations or use their covariance for the noise.
- Planted atoms are drawn uniformly over the vocabulary, including rare and never-trained tokens; real readouts
  are dominated by common tokens.
- The recall measures our pursuit (nonnegative orthogonal matching pursuit with 150 projected-gradient steps per
  refit). The thresholds of the theory (Donoho–Tanner, about 600 nonzeros for a random dictionary of this size)
  are for ℓ1 decoders without noise, and need not apply to it. In the early layers, where the atoms are nearly
  parallel, 150 refit steps may not converge; part of the low recall at layer 8 may be the decoder's.
- The occupancy rule is our reading of §4.2: the paper does not say which random dictionary it compares against.
