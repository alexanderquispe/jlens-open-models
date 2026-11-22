"""Experiment 09, cue readouts (GPU): what the lenses say when the number stream is followed by words.

At the comma the model always predicts a number, so a lens read there is dominated by numbers (run_model.py). Here
each stream is cut at a chosen point and followed by a short text cue that invites words ("Wait,", "The last number
is", "Q: Did the numbers change? A:", ...). The lenses are read at every token of the cue. The model's own next-token
distribution there is the logit lens at the last block.

Contexts (matched pairs; only the numbers after the cut point differ):
  switch    10 x 8: sequence k of N(300, 100) (the paper's, 1000 numbers), then e + 1 numbers (e in CUTS) drawn
            from N(700, 100) ("switch") or N(300, 100) ("control") with the same standard normal draws (seed
            1000 + k). e = -1 is the first segment alone (both arms identical: a sanity check).
  outlier   10 x 3: sequence k of N(500, 100) up to number t (t near 200, 500, 900, moved to the first number within
            50 of 500), with number t kept ("typical", |x - 500| <= 50), set to 750 or 250 ("moderate", 2.5 sd), or
            to 950 or 50 ("extreme", 4.5 sd; the side alternates with k).

Recorded for every prompt, at every cue token, per lens and layer:
  words     log-softmax at the TARGETS words (single tokens with a leading space)
  lex       z-scores of the common.LEXICON groups (as run_model.py)
  top       the 20 top tokens
  cells     at the first and the last cue token, the mean log-softmax over the vocabulary per cell
            (switch: e x arm; outlier: level)

    sbatch --qos=debug --time=00:30:00 hpc/gpu.sbatch .venv-tf5/bin/python experiments/09_beliefs/cue_lens.py --model llama3.1-8b
    sbatch --gres=gpu:nvidia_h200:1 --mem=160G hpc/gpu.sbatch .venv-tf5/bin/python experiments/09_beliefs/cue_lens.py --model olmo-3-32b
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
from common import HI, LEXICON, LO, RAW, cond_name  # noqa: E402
from jlens_open import MODELS, LensModel  # noqa: E402
from jlens_open.models import lens_path  # noqa: E402
from stimuli import draw  # noqa: E402

CUES = {
    "comma": ",",
    "wait": ",\n\nWait,",
    "last": ",\n\nThe last number is",
    "q_surprise": ",\n\nQ: Was the last number surprising? A:",
    "q_change": ",\n\nQ: Did the numbers change? A:",
    "these": ",\n\nThese numbers are",
}
CUTS = (-1, 0, 1, 2, 4, 8, 32, 128)
LEVELS = ("typical", "moderate", "extreme")
TARGETS = {
    "surprise": ["surprising", "unexpected", "unusual", "strange", "odd", "weird", "outlier", "anomaly", "surprise",
                 "sudden", "suddenly", "wrong", "error", "interesting", "wait"],
    "change": ["change", "changed", "changing", "different", "new", "shift", "jump", "switch", "increased",
               "increasing", "higher", "rising", "bigger", "now", "moved"],
    "normal": ["normal", "typical", "expected", "usual", "consistent", "similar", "same", "fine", "ok", "random",
               "stable", "average", "around", "about", "between"],
    "answer": ["Yes", "No", "yes", "no"],
    "magnitude": ["large", "small", "big", "huge", "high", "low", "larger", "smaller", "greater", "less", "much"],
    "numberword": ["three", "four", "five", "six", "seven", "eight", "nine", "hundred", "thousand", "million"],
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--lenses", default=None, help="comma list of logit, jlens, jpp (default: all the model has)")
parser.add_argument("--device", default=None)
parser.add_argument("--raw", default=RAW)
parser.add_argument("--seqs", type=int, default=10)
parser.add_argument("--prefix", type=int, default=1000, help="numbers before the cut (smoke: fewer)")
parser.add_argument("--top", type=int, default=20)
args = parser.parse_args()

t_start = time.time()
lm = LensModel(args.model, args.device)
L, d, dev = lm.n_blocks, lm.d, lm.device
lenses = {"logit": None, "jlens": lm.J}
jpp_key = f"{args.model}-jpp"
want = args.lenses.split(",") if args.lenses else ["logit", "jlens"] + (["jpp"] if jpp_key in MODELS else [])
if "jpp" in want:
    f = torch.load(lens_path(MODELS[jpp_key]), weights_only=True)
    lenses["jpp"] = {layer: m.float().to(dev) for layer, m in f["J"].items()}
    del f
lenses = {k: v for k, v in lenses.items() if k in want}
OUT = os.path.join(args.raw, args.model)
os.makedirs(OUT, exist_ok=True)
V = lm.model.lm_head.weight.shape[0]
config = getattr(lm.model.config, "text_config", lm.model.config)
window = getattr(config, "sliding_window", None)

int_id = [lm.tok.encode(str(k), add_special_tokens=False)[0] for k in range(1000)]
comma = lm.tok.encode(",", add_special_tokens=False)[0]
BOS = [lm.tok.bos_token_id] if lm.tok.bos_token_id is not None else []
PAD = lm.tok.eos_token_id if lm.tok.eos_token_id is not None else 0


def single(words):
    keep, drop = [], []
    for w in words:
        t = lm.tok.encode(" " + w, add_special_tokens=False)
        (keep.append((w, t[0])) if len(t) == 1 else drop.append(w))
    return keep, drop


targets, dropped = [], {}
for g, words in TARGETS.items():
    keep, drop = single(words)
    targets += [(g, w, i) for w, i in keep]
    dropped[g] = drop
W_IDS = torch.tensor([i for _, _, i in targets], device=dev)
lex_ids = {g: torch.tensor([i for _, i in single(ws)[0]], device=dev) for g, ws in LEXICON.items()}
G = len(lex_ids)
print(f"{args.model}: {L} blocks, lenses {list(lenses)}, {len(targets)} target words (dropped {dropped})", flush=True)


# ------------------------------------------------------------------------------------------------- contexts
def clip(x):
    return np.clip(np.round(x).astype(int), LO, HI)


contexts = []  # dicts: block, seq, e / t, arm / level, cell, numbers
first = draw(300, 100)[: args.seqs, : args.prefix]
for k in range(args.seqs):
    z = np.random.default_rng(1000 + k).standard_normal(max(CUTS) + 1)
    for ci, e in enumerate(CUTS):
        for ai, (arm, mu) in enumerate((("switch", 700), ("control", 300))):
            nums = first[k].tolist() + clip(mu + 100 * z[: e + 1]).tolist()
            contexts.append({"block": "switch", "seq": k, "e": e, "arm": arm, "cell": 2 * ci + ai, "numbers": nums})
base = draw(500, 100)[: args.seqs, : args.prefix]
for k in range(args.seqs):
    for t0 in (200, 500, 900):
        t0 = min(t0, args.prefix - 1)
        t = next((t for t in range(t0, base.shape[1]) if abs(int(base[k, t]) - 500) <= 50), t0)
        for li, level in enumerate(LEVELS):
            last = int(base[k, t]) if level == "typical" else (500 + (1 if k % 2 == 0 else -1) * (250 if level == "moderate" else 450))
            nums = base[k, :t].tolist() + [last]
            contexts.append({"block": "outlier", "seq": k, "t": t, "level": level, "last": last,
                             "cell": 2 * len(CUTS) + li, "numbers": nums})
n_cells = 2 * len(CUTS) + len(LEVELS)


def prompt_ids(numbers, cue):
    """(common, tail): the stream up to and including the last number, and the cue tokens (from the joint
    tokenization of the text, checked to extend the stream's own tokens)."""
    common = BOS + [x for n in numbers for x in (int_id[n], comma)][:-1]
    ids = lm.encode(",".join(map(str, numbers)) + cue).tolist()
    assert ids[: len(common)] == common, "the cue changes the tokenization of the stream"
    return common, ids[len(common):]


# ------------------------------------------------------------------------------------------------- run
@torch.no_grad()
def tail_states(common, tails):
    """Block outputs at the tail tokens, {layer: (n_cues, max_tail, d)}: the stream is run once, the cues are run as a
    right-padded batch on its cache (causal attention: padding after a cue cannot change it)."""
    from transformers.cache_utils import DynamicCache, DynamicLayer

    if window:
        assert len(common) + max(map(len, tails)) <= window
    out = lm.model(torch.tensor([common], device=dev), use_cache=True, logits_to_keep=1)
    kv = [(layer.keys, layer.values) for layer in out.past_key_values.layers]
    del out
    b, k = len(tails), max(map(len, tails))
    batch = torch.tensor([t + [PAD] * (k - len(t)) for t in tails], device=dev)
    cache = DynamicCache()
    for K, Vv in kv:
        layer = DynamicLayer()
        layer.lazy_initialization(K, Vv)
        layer.keys, layer.values = K.expand(b, -1, -1, -1), Vv.expand(b, -1, -1, -1)
        cache.layers.append(layer)
    H = {}

    def hook(i):
        def f(module, inputs, output):
            H[i] = (output[0] if isinstance(output, tuple) else output).float()
        return f

    handles = [blk.register_forward_hook(hook(i)) for i, blk in enumerate(lm.blocks)]
    try:
        lm.model(batch, past_key_values=cache, use_cache=True, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return H


cue_names = list(CUES)
C = len(cue_names)
max_tail = max(len(prompt_ids(contexts[0]["numbers"], c)[1]) for c in CUES.values()) + 2
P = len(contexts)
REC = {k: {"words": np.full((P, C, max_tail, L, len(targets)), np.nan, np.float16),
           "lex": np.full((P, C, max_tail, L, G), np.nan, np.float16),
           "top": np.full((P, C, max_tail, L, args.top), -1, np.int32)} for k in lenses}
acc = {k: torch.zeros(C, 2, n_cells, L, V, device=dev) for k in lenses}
count = np.zeros((n_cells,))
n_tail = np.zeros((P, C), int)
tail_text = {}
for p, ctx in enumerate(contexts):
    commons, tails = zip(*(prompt_ids(ctx["numbers"], CUES[c]) for c in cue_names))
    assert all(c == commons[0] for c in commons)
    for j, t in enumerate(tails):
        n_tail[p, j] = len(t)
        tail_text.setdefault(cue_names[j], [lm.tok.decode([x]) for x in t])
    H = tail_states(commons[0], list(tails))
    count[ctx["cell"]] += 1
    rows = [(j, i) for j in range(C) for i in range(n_tail[p, j])]
    rj = torch.tensor([r[0] for r in rows], device=dev)
    ri = torch.tensor([r[1] for r in rows], device=dev)
    is_first = ri == 0
    is_last = ri == torch.tensor(n_tail[p], device=dev)[rj] - 1
    nj, ni = rj.cpu().numpy(), ri.cpu().numpy()
    for k, J in lenses.items():
        for layer in range(L):
            X = H[layer][rj, ri]
            lg = lm.unembed(X @ J[layer].T if J is not None and layer in J else X)
            lse = torch.logsumexp(lg, 1)
            var, mean = torch.var_mean(lg, 1)
            REC[k]["words"][p, nj, ni, layer] = (lg[:, W_IDS] - lse[:, None]).cpu().numpy()
            REC[k]["lex"][p, nj, ni, layer] = (torch.stack(
                [(lg[:, i] - mean[:, None]).mean(1) for i in lex_ids.values()], 1) / var.sqrt()[:, None]).cpu().numpy()
            REC[k]["top"][p, nj, ni, layer] = lg.topk(args.top, 1).indices.cpu().numpy()
            lsm = lg - lse[:, None]
            acc[k][rj[is_first], 0, ctx["cell"], layer] += lsm[is_first]
            acc[k][rj[is_last], 1, ctx["cell"], layer] += lsm[is_last]
    del H
    if (p + 1) % 20 == 0 or p + 1 == P:
        print(f"{p + 1}/{P} contexts, {time.time() - t_start:.0f} s", flush=True)

for k in lenses:
    for name, arr in REC[k].items():
        np.save(os.path.join(OUT, f"cues_{k}_{name}.npy"), arr)
    np.save(os.path.join(OUT, f"cues_{k}_cells.npy"),
            (acc[k] / torch.tensor(np.maximum(count, 1), device=dev, dtype=torch.float32)[None, None, :, None, None]).half().cpu().numpy())
meta = {"model": args.model, "hf": MODELS[args.model]["hf"], "n_blocks": L, "lenses": list(lenses), "cues": CUES,
        "cue_names": cue_names, "tail_tokens": tail_text, "cuts": CUTS, "levels": LEVELS, "n_cells": n_cells,
        "cell_count": count.tolist(), "targets": [[g, w, int(i)] for g, w, i in targets], "dropped": dropped,
        "groups": list(lex_ids), "n_tail": n_tail.tolist(),
        "contexts": [{k: v for k, v in c.items() if k != "numbers"} | {"n_numbers": len(c["numbers"])} for c in contexts],
        "args": vars(args), "seconds": round(time.time() - t_start),
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev}
json.dump(meta, open(os.path.join(OUT, "cues_meta.json"), "w"), indent=1)
print(f"Done in {time.time() - t_start:.0f} s: {OUT}/cues_*", flush=True)
