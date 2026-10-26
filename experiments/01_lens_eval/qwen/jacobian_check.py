"""Does the stored J match the model we load? Autograd probes of the Jacobian, against the lens file.

The reference estimator (jlens/fitting.py) sets the cotangent at every valid target position at once, so
row u of J_l is the mean over source positions p of  u^T sum_{p' >= p} d h_final[p'] / d h_l[p]:
the effect of h_l[p] on the current output and on every later one. Probing with K random unit directions
u on wikitext paragraphs the lens was not averaged over (the last 6 of ../../03_jspace/data):
  - sum:  cotangent u at all valid positions (16..SEQ-2), gradient averaged over valid source positions
          (the reference estimator; with enough prompts its rows match u^T J_stored);
  - same: cotangent u at a single position p, gradient read at p (only d h_final[p] / d h_l[p]).

    python experiments/01_lens_eval/qwen/jacobian_check.py --model qwen3-8b
"""
import argparse
import json
import os

import pandas as pd
import torch

from jlens_open import MODELS, LensModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: the first available)")
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_OUT = os.path.join(HERE, f"results/jacobian_check_{args.model}.csv")
K, N_PROMPTS, SEQ, SKIP = 16, 6, 96, 16
SAME_POSITIONS = [40, 60, 80, 94]

torch.manual_seed(0)
lm = LensModel(args.model, args.device)
L = lm.n_blocks
layers = sorted({0, L // 4, L // 2, (3 * L) // 4, L - 2})
with open(os.path.join(HERE, "../../03_jspace/data/wikitext_sample.json")) as f:
    paragraphs = json.load(f)["paragraphs"]
prompts = [lm.encode(p)[:SEQ] for p in paragraphs if len(lm.encode(p)) >= SEQ][-N_PROMPTS:]
U = torch.nn.functional.normalize(torch.randn(K, lm.d), dim=-1).to(lm.device)
valid = torch.arange(SKIP, SEQ - 1)

acc = {(kind, layer): torch.zeros(K, lm.d, device=lm.device) for kind in ("sum", "same") for layer in layers}
for i, ids in enumerate(prompts):
    acts = {}

    def hook(layer):
        def f(module, inputs, output):
            acts[layer] = output[0] if isinstance(output, tuple) else output

        return f

    handles = [lm.blocks[layer].register_forward_hook(hook(layer)) for layer in layers + [L - 1]]
    with torch.enable_grad():
        h_in = lm.model.get_input_embeddings()(ids[None].expand(K, -1).to(lm.device)).detach().requires_grad_(True)
        lm.model(inputs_embeds=h_in, use_cache=False)
        for h in handles:
            h.remove()
        target, sources = acts[L - 1], [acts[layer] for layer in layers]
        cotangent = torch.zeros_like(target)
        cotangent[:, valid] = U[:, None, :].to(cotangent.dtype)
        grads = torch.autograd.grad(target, sources, cotangent, retain_graph=True)
        for layer, g in zip(layers, grads):
            acc["sum", layer] += g[:, valid].float().mean(1) / len(prompts)
        for j, p in enumerate(SAME_POSITIONS):
            cotangent.zero_()
            cotangent[:, p] = U.to(cotangent.dtype)
            grads = torch.autograd.grad(target, sources, cotangent, retain_graph=j < len(SAME_POSITIONS) - 1)
            for layer, g in zip(layers, grads):
                acc["same", layer] += g[:, p].float() / (len(prompts) * len(SAME_POSITIONS))
    del acts, grads
    print(f"prompt {i + 1}/{len(prompts)}", flush=True)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


rows = []
for layer in layers:
    stored = U @ lm.J[layer]  # rows u^T J_stored
    s, m = acc["sum", layer], acc["same", layer]
    rows.append({
        "layer": layer,
        "cos(sum probe, stored J)": cos(s, stored),
        "norm(sum probe) / norm(stored J)": (s.norm(dim=-1) / stored.norm(dim=-1)).mean().item(),
        "cos(same-position probe, stored J)": cos(m, stored),
        "cos(same-position probe, identity)": cos(m, U),
        "cos(stored J, identity)": cos(stored, U),
    })
df = pd.DataFrame(rows)
df.to_csv(CSV_OUT, index=False)
print(df.set_index("layer").round(3).T.to_string())
print(f"\nSaved: {CSV_OUT}")
