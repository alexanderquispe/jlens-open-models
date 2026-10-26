# 05 · Which irreducible representations of Z/n carry the answer?

**Question.** Engels et al. (2025) patch the 2-dim circle of the weekday/month representation. The n centroids
of the days (months) span the regular representation of Z/n minus its trivial part, which splits into real
irreps: a plane for each frequency k < n/2, and a line at k = n/2 when n is even. Which of these frequencies
does the model use to compute "two months from January"? And is the k = 1 circle a regular polygon, as an
equivariant (rotation-like) code would give?

```bash
python experiments/05_cyclic_irreps/cyclic_irreps.py --model llama3.1-8b --concept months --position start
```

`--concept days|months`, `--position start|last` (the named day/month, Engels' `a_token`, or the last token).
Outputs: `results/<model>_<concept>_<position>_{layers,rotation}.csv`, `figures/…`.

## Method

Engels et al.'s prompts: *"Let's do some calendar math. Two months from January is"* (12 months × durations
1–12) and *"Let's do some days of the week math. Two days from Monday is"* (7 × 7). Centroids and Fourier bases
come from odd durations; patches are evaluated on even ones.

- **Irrep spectrum:** the share of the centroids' energy in each frequency k.
- **Polygon regularity** in the k = 1 plane: cyclic order, spread of the angle gaps around 2π/n, spread of the radii.
- **Rotation by irrep:** add *P_K (c_{s+r} − c_s)* at the position after block *l*, for K = {k} for each k,
  K = {1, 2}, and all k (the full centroid difference); controls: the span of the n lens directions of the names
  and a random direction with the k = 1 patch's norm. Success = the answer moves from *a* to *a + r*,
  averaged over r = 1, …, n − 1, on held-out prompts the model solves.

## Results

Pending (overnight run of 2026-10-25).
