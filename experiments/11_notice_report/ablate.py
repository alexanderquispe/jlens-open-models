"""Experiment 11, step 2 (GPU): is the magnitude-word direction necessary for the model's uncertainty?

Experiment 09 found that adding the J-lens direction of magnitude words ("trillion", "thousand", ...) at the last
comma widens Llama-3.1-8B's predicted distribution with a clean dose-response (layers 16-22): sufficiency. Here the
direction is removed instead. At every token from position 16 on (where the lenses were fit) and at every layer in
LAYERS, the component of the block output along the direction u_l is set to its mean over reference streams, taken
separately for comma tokens and number tokens (mean ablation: h <- h - (h.u - m_l,type) u),
and the output's distribution over the integers is read at every comma. Streams: experiment 09's N(500, sigma),
sigma in SIGMAS, 10 streams each, first T numbers. If the direction carries the uncertainty, the predicted sd should
follow sigma less (a flatter slope of sd on sigma) when it is removed, and not when control directions are removed.

Variants: none; magnitude words at all LAYERS; magnitude at layer 19 only; control words; "wide" words (spread,
scattered, ...); 3 random directions. Directions: the mean of the unit rows of W_U diag(g) J_l over the set's words
(as experiment 09's steer.py). Reference means m_l,type: the sigma = 100 streams, tokens 16 and after, commas and
numbers separately.

    sbatch --qos=debug --time=00:30:00 hpc/gpu.sbatch .venv-tf5/bin/python experiments/11_notice_report/ablate.py --model llama3.1-8b
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from jlens_open import MODELS, LensModel  # noqa: E402

SETS = {
    # experiment 09's steer.py set (the sufficiency test), so this is its direct counterpart
    "magnitude": ["trillion", "billion", "billions", "million", "millions", "thousand", "thousands", "infinite"],
    "control": ["table", "river", "green", "letter", "music", "paper", "window", "garden", "chair", "yellow",
                "bottle", "mountain", "pencil", "coffee", "island", "horse", "cloud", "kitchen", "forest", "shirt"],
    "wide": ["wide", "spread", "scattered", "varied", "variable", "volatile", "noisy", "vary", "fluctuate", "uncertain"],
}
SIGMAS = (10, 20, 30, 50, 80, 100, 120, 150, 200)
N_RANDOM = 3
RAW = (os.path.join(os.environ["JLENS_SHARED"], "experiments", "11_notice_report") if os.environ.get("JLENS_SHARED")
       else os.path.join(HERE, "raw"))

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), default="llama3.1-8b")
parser.add_argument("--raw", default=RAW)
parser.add_argument("--device", default=None)
parser.add_argument("--layers", default="16,17,18,19,20,21,22")
parser.add_argument("--single", type=int, default=19)
parser.add_argument("--numbers", type=int, default=400)
parser.add_argument("--seqs", type=int, default=10)
args = parser.parse_args()

t0 = time.time()
lm = LensModel(args.model, args.device)
L, d, dev, tok = lm.n_blocks, lm.d, lm.device, lm.tok
LAYERS = [int(x) for x in args.layers.split(",")]
int_id = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(1000)]
comma = tok.encode(",", add_special_tokens=False)[0]
INT = torch.tensor(int_id, device=dev)
KV = torch.arange(1000, device=dev, dtype=torch.float64)
BOS = [tok.bos_token_id] if tok.bos_token_id is not None else []
T = args.numbers


def draw(mu, sigma, n_seq=10, length=1000, seed=18):
    """Experiment 09's stimuli.draw (Sarfati et al.'s generator)."""
    rng = np.random.default_rng(seed)
    return np.clip(np.round(rng.normal(loc=mu, scale=sigma, size=(n_seq, length))).astype(int), 10, 990)


def ids_of(nums):
    return BOS + [x for n in nums for x in (int_id[n], comma)]


def direction(words, layer):
    ids = [t[0] for t in (tok.encode(" " + w, add_special_tokens=False) for w in words) if len(t) == 1]
    w = lm.model.lm_head.weight[ids].float().to(dev) * lm.gain
    if layer in lm.J:
        w = w @ lm.J[layer]
    w = w / w.norm(dim=1, keepdim=True)
    u = w.mean(0)
    return u / u.norm()


U = {name: {l: direction(ws, l) for l in LAYERS} for name, ws in SETS.items()}
g = torch.Generator(device="cpu").manual_seed(0)
for r in range(N_RANDOM):
    U[f"random{r}"] = {}
    for l in LAYERS:
        u = torch.randn(d, generator=g).to(dev)
        U[f"random{r}"][l] = u / u.norm()
VARIANTS = {"none": {}, "magnitude": {l: "magnitude" for l in LAYERS}, f"magnitude_L{args.single}": {args.single: "magnitude"},
            "control": {l: "control" for l in LAYERS}, "wide": {l: "wide" for l in LAYERS},
            **{f"random{r}": {l: f"random{r}" for l in LAYERS} for r in range(N_RANDOM)}}
com = torch.tensor([len(BOS) + 2 * t + 1 for t in range(T)], device=dev)
N_TOK = len(BOS) + 2 * T
FIRST = 16  # tokens before this are left untouched (the lenses were fit without them)
is_com = torch.zeros(N_TOK, dtype=torch.bool, device=dev)
is_com[com] = True
is_num = ~is_com
is_num[: len(BOS)] = False
is_com[:FIRST] = False
is_num[:FIRST] = False


@torch.no_grad()
def forward(ids, plan=None, means=None, read=False):
    """Logits at the commas (B, T, vocab) with mean ablation {layer: direction name}; read=True returns instead the
    projections {(name, layer): mean over the batch's tokens >= 16} for every direction at LAYERS."""
    handles, proj = [], {}

    def ablate(layer, name):
        u = U[name][layer]

        def f(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            hf = h.float()
            p = hf @ u  # (B, N_TOK)
            target = torch.where(is_com, means[(name, layer, "comma")], means[(name, layer, "number")])
            mask = (is_com | is_num).float()  # tokens >= FIRST only
            hf = hf - ((p - target) * mask)[..., None] * u
            h = hf.to(h.dtype)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return f

    def reader(layer):
        def f(module, inputs, output):
            h = (output[0] if isinstance(output, tuple) else output).float()
            for name in U:
                p = h @ U[name][layer]  # (B, N_TOK)
                proj[(name, layer, "comma")] = p[:, is_com].mean().item()
                proj[(name, layer, "number")] = p[:, is_num].mean().item()
        return f

    for layer in LAYERS:
        if read:
            handles.append(lm.blocks[layer].register_forward_hook(reader(layer)))
        elif plan and layer in plan:
            handles.append(lm.blocks[layer].register_forward_hook(ablate(layer, plan[layer])))
    try:
        out = lm.model(ids.to(dev), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return proj if read else out.logits[:, com]


with torch.no_grad():
    streams = {s: draw(500, s)[: args.seqs, :T] for s in SIGMAS}
    ids = {s: torch.tensor([ids_of(x) for x in streams[s]]) for s in SIGMAS}
    for s in SIGMAS:  # tokenization check
        assert lm.encode(",".join(map(str, streams[s][0])) + ",").tolist() == ids[s][0].tolist()
    means = forward(ids[100], read=True)
    OUTS = np.zeros((len(VARIANTS), len(SIGMAS), args.seqs, T, 2), np.float32)
    for vi, (vname, plan) in enumerate(VARIANTS.items()):
        for si, s in enumerate(SIGMAS):
            lg = forward(ids[s], plan, means).float()
            li = torch.log_softmax(lg, -1)[..., INT].double()
            q = torch.exp(li - torch.logsumexp(li, -1, keepdim=True))
            mean = q @ KV
            sd = (q @ KV**2 - mean**2).clamp(min=0).sqrt()
            OUTS[vi, si] = torch.stack([mean, sd], -1).cpu().numpy()
            del lg, li, q
        print(f"{vname}: done, {time.time() - t0:.0f} s; sd at t >= 200 by sigma "
              f"{np.round(OUTS[vi, :, :, 200:, 1].mean((1, 2)), 1).tolist()}", flush=True)

OUT = os.path.join(args.raw, args.model)
os.makedirs(OUT, exist_ok=True)
np.save(os.path.join(OUT, "ablate_out.npy"), OUTS)
json.dump({"model": args.model, "variants": list(VARIANTS), "sigmas": SIGMAS, "layers": LAYERS, "single": args.single,
           "means": {f"{n}@{l}@{k}": v for (n, l, k), v in means.items()}, "T": T, "seqs": args.seqs, "first": FIRST,
           "args": vars(args), "seconds": round(time.time() - t0),
           "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev},
          open(os.path.join(OUT, "ablate_info.json"), "w"), indent=1)
print(f"Done in {time.time() - t0:.0f} s: {OUT}/ablate_out.npy", flush=True)
