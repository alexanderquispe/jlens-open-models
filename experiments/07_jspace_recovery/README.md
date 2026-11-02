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

## Results

Pending (first run: GPT-2 small as a check, then Llama-3.1-8B, one H100).

## Caveats

- The noise is isotropic. The real non-J-space part of an activation is not: it is concentrated in the
  activations' high-variance directions (experiment 03: a random dictionary shaped like the activations explains
  35–46%). A follow-up should plant words into real activations or use their covariance for the noise.
- Planted atoms are drawn uniformly over the vocabulary, including rare and never-trained tokens; real readouts
  are dominated by common tokens.
- The recall measures our pursuit (nonnegative orthogonal matching pursuit with 150 projected-gradient steps per
  refit). The thresholds of the theory (Donoho–Tanner, about 600 nonzeros for a random dictionary of this size)
  are for ℓ1 decoders without noise, and need not apply to it.
- The occupancy rule is our reading of §4.2: the paper does not say which random dictionary it compares against.
