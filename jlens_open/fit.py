"""Fit the Jacobian lens of a model in MODELS, written where its "lens" entry points ($JLENS_SHARED/lenses/...).

Same estimator as the reference code (github.com/anthropics/jacobian-lens, jlens/fitting.py) and the recipe
Neuronpedia used for its lenses (the config.yaml beside each lens in neuronpedia/jacobian-lens):

- Prompts: the first WikiText-103 (raw, train) records of at least 600 characters, cut to 2000 characters and then
  to 128 tokens, BOS included.
- Per prompt, one forward pass on the prompt repeated dim_batch times, then ceil(d / dim_batch) backward passes. In
  each, copy b carries a one-hot cotangent at output dimension i + b of h_final (the last block's output), at every
  position from 16 to the second to last. Row i of J_l is the gradient at block l's output, averaged over the same
  positions: the mean over sources p of sum_{p' >= p} d h_final[p', i] / d h_l[p].
- The lens is the mean over prompts. The fit stops once the running mean moves by less than 0.002 per prompt
  (relative, largest over layers, averaged over the last 10 prompts) and at least 100 prompts are in, or at 1000.

The file has the format of Neuronpedia's lenses (J in fp16, source_layers, d_model, n_prompts, provenance). A
checkpoint is written every 25 prompts, and an interrupted fit resumes from it.

    sbatch hpc/gpu.sbatch env PYTHON=.venv-tf5/bin/python PYTHONPATH=. python -m jlens_open.fit mistral-7b
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from .models import MODELS, LensModel, lens_path

SKIP_FIRST = 16  # early positions are attention sinks; the reference code leaves them out

parser = argparse.ArgumentParser(description="Fit a Jacobian lens.")
parser.add_argument("model", choices=[m for m in MODELS if MODELS[m]["lens"].startswith("$")])
parser.add_argument("--dim-batch", type=int, default=64, help="output dimensions per backward pass (memory, not speed)")
parser.add_argument("--max-seq-len", type=int, default=128)
parser.add_argument("--max-prompts", type=int, default=1000)
parser.add_argument("--min-prompts", type=int, default=100)
parser.add_argument("--stop-at", type=float, default=0.002)
parser.add_argument("--window", type=int, default=10)
parser.add_argument("--checkpoint-every", type=int, default=25)
parser.add_argument("--dtype", choices=["bfloat16", "float32"], help="default: the model's dtype in MODELS")
parser.add_argument("--out", help="default: the model's lens path in MODELS")
args = parser.parse_args()


def wikitext_prompts(n, min_chars=600, max_chars=2000):
    shard = hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/train-00000-of-00002.parquet", repo_type="dataset")
    text = pd.read_parquet(shard).text
    return [t[:max_chars] for t in text[text.str.strip().str.len() >= min_chars].iloc[:n]]


def prompt_jacobians(lm, ids, dim_batch):
    """{l: (d, d) float32 J_l} for l = 0..n-2 on one prompt."""
    n, d = lm.n_blocks, lm.d
    positions = torch.arange(SKIP_FIRST, len(ids) - 1, device=lm.device)
    acts = {}

    def hook(layer):
        def f(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if layer == 0:  # the graph starts here: gradients are taken only with respect to activations
                hidden = hidden.detach().requires_grad_(True)
            acts[layer] = hidden
            if isinstance(output, tuple):
                return (hidden,) + tuple(output[1:])
            return hidden

        return f

    handles = [b.register_forward_hook(hook(i)) for i, b in enumerate(lm.blocks)]
    try:
        with torch.enable_grad():
            lm.decoder(input_ids=ids.expand(dim_batch, -1).to(lm.device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    target, sources = acts[n - 1], [acts[layer] for layer in range(n - 1)]
    J = {layer: torch.empty(d, d, device=lm.device) for layer in range(n - 1)}
    copies = torch.arange(dim_batch, device=lm.device)
    cotangent = torch.zeros_like(target)
    n_passes = math.ceil(d / dim_batch)
    for k, start in enumerate(range(0, d, dim_batch)):
        m = min(dim_batch, d - start)
        cotangent.zero_()
        cotangent[copies[:m, None], positions[None, :], start + copies[:m, None]] = 1
        grads = torch.autograd.grad(target, sources, cotangent, retain_graph=k < n_passes - 1)
        for layer, g in enumerate(grads):
            J[layer][start:start + m] = g[:m, positions].float().mean(1)
    return J


def save(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


lm = LensModel(args.model, lens=False)
for p in lm.model.parameters():
    p.requires_grad_(False)
if args.dtype:
    lm.model.to(getattr(torch, args.dtype))
    lm.dtype = getattr(torch, args.dtype)
out = args.out or lens_path(MODELS[args.model])
os.makedirs(os.path.dirname(out), exist_ok=True)
checkpoint = out + ".checkpoint"
layers = range(lm.n_blocks - 1)
prompts = wikitext_prompts(args.max_prompts)
print(f"{args.model}: {lm.n_blocks} blocks, d = {lm.d}, up to {len(prompts)} prompts, dim_batch = {args.dim_batch}")

if os.path.exists(checkpoint):
    state = torch.load(checkpoint, weights_only=True)
    total = {layer: state["total"][layer].to(lm.device) for layer in layers}
    n_done, next_prompt, changes = state["n_done"], state["next_prompt"], state["changes"]
    print(f"resuming from {checkpoint}: {n_done} prompts")
else:
    total = {layer: torch.zeros(lm.d, lm.d, device=lm.device) for layer in layers}
    n_done, next_prompt, changes = 0, 0, []

for i in range(next_prompt, len(prompts)):
    ids = lm.encode(prompts[i])[:args.max_seq_len]
    if len(ids) < SKIP_FIRST + 2:
        continue
    t0 = time.time()
    J = prompt_jacobians(lm, ids, args.dim_batch)
    if n_done:
        # How far this prompt moves the running mean, relative to its norm (the reference code's convergence check)
        changes.append(max(((J[l] - total[l] / n_done).norm() / ((n_done + 1) * (total[l] / n_done).norm())).item()
                           for l in layers))
    for layer in layers:
        total[layer] += J[layer]
    n_done, next_prompt = n_done + 1, i + 1
    recent = np.mean(changes[-args.window:]) if changes else float("nan")
    print(f"prompt {i + 1}: {len(ids)} tokens, {time.time() - t0:.1f} s, change {changes[-1] if changes else float('nan'):.2e}, "
          f"mean of last {args.window} {recent:.2e}")
    done = n_done >= args.min_prompts and len(changes) >= args.window and recent < args.stop_at
    if n_done % args.checkpoint_every == 0 and not done:
        save({"total": {layer: total[layer].cpu() for layer in layers}, "n_done": n_done, "next_prompt": next_prompt,
              "changes": changes}, checkpoint)
    if done:
        break

save({
    "J": {layer: (total[layer] / n_done).half().cpu() for layer in layers},
    "source_layers": list(layers),
    "d_model": lm.d,
    "n_prompts": n_done,
    "provenance": {
        "method": "jlens", "code": "jlens_open/fit.py", "model": MODELS[args.model]["hf"], "dtype": str(lm.dtype),
        "dataset": "Salesforce/wikitext wikitext-103-raw-v1 train, records of >= 600 characters cut to 2000",
        "max_seq_len": args.max_seq_len, "skip_first": SKIP_FIRST, "dim_batch": args.dim_batch,
        "final_mean_rel_change": float(np.mean(changes[-args.window:])), "capture_point": "block_output",
    },
}, out)
if os.path.exists(checkpoint):
    os.remove(checkpoint)
print(f"Saved: {out} ({n_done} prompts)")
