"""Experiment 09, steering with the lens's word directions (GPU): are the words the lenses read used for the belief?

At the last comma of a stream (the model predicts the next number), add c ||h_l|| u to the block-l output, where u is
the unit direction that raises a word set's lens logits (the mean of the unit rows of W_U diag(g) J_l over the set's
words; jlens_open.LensModel.lens_directions), and read the change of the output's distribution over the integers:
its mean, sd, entropy and mass. Word sets: number words (to move the mean: "seventy", "thirty"), magnitude words
(which fade as the belief settles), the surprise and change groups, the integers 700-799 themselves (a positive
control), control words and random directions (negative controls). Only the last position is changed: the context
is read once (KV cache) and every condition runs one token on it.

Contexts: N(500, 100) (the paper's sequences), cut after t + 1 numbers, t in {127, 999}.

    sbatch --qos=debug --time=00:30:00 hpc/gpu.sbatch .venv-tf5/bin/python experiments/09_beliefs/steer.py --model llama3.1-8b
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LEXICON, RAW  # noqa: E402
from jlens_open import MODELS, LensModel  # noqa: E402
from stimuli import draw  # noqa: E402

SETS = {
    "seventy": ["seventy", "seven", "seventh"],
    "thirty": ["thirty", "three", "third"],
    "magnitude": ["trillion", "billion", "billions", "million", "millions", "thousand", "thousands", "infinite"],
    "surprise": LEXICON["surprise"],
    "change": LEXICON["change"],
    "wide": LEXICON["wide"],
    "control": LEXICON["control"],
}
COEF = (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)
N_RANDOM = 8

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--device", default=None)
parser.add_argument("--raw", default=RAW)
parser.add_argument("--seqs", type=int, default=10)
parser.add_argument("--times", default="127,999")
parser.add_argument("--layers", default=None, help="comma list (default: 8 layers spread over 1/4 .. 15/16 depth)")
parser.add_argument("--batch", type=int, default=16)
args = parser.parse_args()

t0 = time.time()
lm = LensModel(args.model, args.device)
L, d, dev = lm.n_blocks, lm.d, lm.device
layers = ([int(x) for x in args.layers.split(",")] if args.layers
          else sorted({round(L * f) for f in (0.25, 0.375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.875)}))
int_id = [lm.tok.encode(str(k), add_special_tokens=False)[0] for k in range(1000)]
INT = torch.tensor(int_id, device=dev)
KV = torch.arange(1000, device=dev, dtype=torch.float64)
comma = lm.tok.encode(",", add_special_tokens=False)[0]
BOS = [lm.tok.bos_token_id] if lm.tok.bos_token_id is not None else []

ids_of = {}
for name, words in SETS.items():
    keep = [t[0] for t in (lm.tok.encode(" " + w, add_special_tokens=False) for w in words) if len(t) == 1]
    ids_of[name] = keep
ids_of["int700"] = int_id[700:800]
ids_of["int300"] = int_id[300:400]


def directions(layer, basis):
    """{name: unit (d,) direction} at `layer`: J-lens rows (basis 'jlens') or plain W_U diag(g) rows ('logit')."""
    out = {}
    for name, ids in ids_of.items():
        w = lm.model.lm_head.weight[ids].float().to(dev) * lm.gain
        if basis == "jlens" and layer in lm.J:
            w = w @ lm.J[layer]
        w = w / w.norm(dim=1, keepdim=True)
        u = w.mean(0)
        out[name] = u / u.norm()
    g = torch.Generator(device="cpu").manual_seed(layer)
    for r in range(N_RANDOM):
        u = torch.randn(d, generator=g).to(dev)
        out[f"random{r}"] = u / u.norm()
    return out


@torch.no_grad()
def int_stats(logits):
    lsm = torch.log_softmax(logits.float(), -1)
    li = lsm[:, INT].double()
    lmass = torch.logsumexp(li, 1)
    q = torch.exp(li - lmass[:, None])
    mean = q @ KV
    sd = (q @ KV**2 - mean**2).clamp(min=0).sqrt()
    ent = -(q * torch.log(q.clamp(min=1e-300))).sum(1)
    return torch.stack([mean, sd, ent, lmass.exp()], 1).float().cpu().numpy()


@torch.no_grad()
def run_last(kv, last_id, layer, deltas):
    """Output logits (B, V) for the last token on the cached context, with deltas (B, d) added at block `layer`."""
    from transformers.cache_utils import DynamicCache, DynamicLayer

    B = len(deltas)
    cache = DynamicCache()
    for K, V in kv:
        lay = DynamicLayer()
        lay.lazy_initialization(K, V)
        lay.keys, lay.values = K.expand(B, -1, -1, -1), V.expand(B, -1, -1, -1)
        cache.layers.append(lay)
    norms = {}

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        norms["h"] = h[:, -1].float().norm(dim=-1)
        h = h.clone()
        h[:, -1] += (deltas * norms["h"][:, None]).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    handle = lm.blocks[layer].register_forward_hook(hook)
    try:
        out = lm.model(torch.full((B, 1), last_id, device=dev), past_key_values=cache, use_cache=True,
                       logits_to_keep=1)
    finally:
        handle.remove()
    return out.logits[:, -1]


rows = []
seqs = draw(500, 100)[: args.seqs]
for t in [int(x) for x in args.times.split(",")]:
    for k in range(args.seqs):
        ids = BOS + [x for n in seqs[k, : t + 1] for x in (int_id[n], comma)]
        out = lm.model(torch.tensor([ids[:-1]], device=dev), use_cache=True, logits_to_keep=1)
        kv = [(lay.keys, lay.values) for lay in out.past_key_values.layers]
        base = int_stats(run_last(kv, ids[-1], layers[0], torch.zeros(1, d, device=dev)))[0]
        for basis in ("jlens", "logit"):
            for layer in layers:
                U = directions(layer, basis)
                conds = [(name, c) for name in U for c in COEF]
                for b0 in range(0, len(conds), args.batch):
                    chunk = conds[b0 : b0 + args.batch]
                    D = torch.stack([c * U[name] for name, c in chunk])
                    st = int_stats(run_last(kv, ids[-1], layer, D))
                    for (name, c), s in zip(chunk, st):
                        rows.append({"t": t, "seq": k, "basis": basis, "layer": layer, "dir": name, "c": c,
                                     "d_mean": float(s[0] - base[0]), "d_sd": float(s[1] - base[1]),
                                     "d_entropy": float(s[2] - base[2]), "d_mass": float(s[3] - base[3]),
                                     "base_mean": float(base[0]), "base_sd": float(base[1])})
        del kv, out
        print(f"t = {t}: {k + 1}/{args.seqs} sequences, {time.time() - t0:.0f} s", flush=True)

OUT = os.path.join(args.raw, args.model)
os.makedirs(OUT, exist_ok=True)
import pandas as pd  # noqa: E402

pd.DataFrame(rows).to_csv(os.path.join(OUT, "steer.csv.gz"), index=False)
json.dump({"model": args.model, "layers": layers, "sets": {k: len(v) for k, v in ids_of.items()}, "coef": COEF,
           "args": vars(args), "seconds": round(time.time() - t0),
           "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev},
          open(os.path.join(OUT, "steer_info.json"), "w"), indent=1)
print(f"Done in {time.time() - t0:.0f} s: {OUT}/steer.csv.gz", flush=True)
