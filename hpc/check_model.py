"""Quick check of a newly registered model on the GPU, before a full run_all.sh: it loads, predicts a known fact,
its lens has one J per block except the last, and unembedding the last block's output gives the model's own
logits (so the final norm, its gain and any logit softcap are handled right). Exits with an error otherwise.

    sbatch --time=00:30:00 hpc/gpu.sbatch env PYTHONPATH=. python hpc/check_model.py qwen3.6-27b
"""
import sys

import torch

from jlens_open.models import LensModel

lm = LensModel(sys.argv[1])
gain = "weight" if torch.equal(lm.gain, lm.norm.weight.float().to(lm.device)) else "1 + weight"
print(f"{lm.name}: {lm.n_blocks} blocks, d = {lm.d}, final norm gain = {gain}, softcap = {lm.softcap}, "
      f"lens averaged over {lm.n_lens_prompts} prompts")

hs, logits = lm.run(lm.encode("The capital of France is")[None])
top5 = [lm.tok.decode(t) for t in logits[0].topk(5).indices]
print(f"top 5 next tokens after 'The capital of France is': {top5}")
assert any("Paris" in t for t in top5), top5  # base models often say " a" or " the" first

assert sorted(lm.J) == list(range(lm.n_blocks - 1)), f"lens layers {sorted(lm.J)[:3]}...{sorted(lm.J)[-3:]}"
assert all(J.shape == (lm.d, lm.d) for J in lm.J.values())

diff = (lm.unembed(hs[-1][:, -1]) - logits).abs().max().item()
scale = logits.abs().max().item()
print(f"max |unembed(last block) - model logits| = {diff:.3g} (largest logit {scale:.3g})")
assert diff < 0.05 * scale

# The two lenses at three depths, on a two-hop prompt whose intermediate is "France"
hs, _ = lm.run(lm.encode("The Eiffel Tower is in a country whose official language is")[None])
for layer in (lm.n_blocks // 2, 2 * lm.n_blocks // 3, 5 * lm.n_blocks // 6):
    for name, use_j in (("J-lens", True), ("logit lens", False)):
        top10 = lm.unembed(lm.transport(hs[layer][0, -1], layer, use_j)).topk(10).indices
        print(f"  layer {layer:3d} {name:10s}", [lm.tok.decode(t) for t in top10])
print("OK")
