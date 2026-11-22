"""Experiment 09, word sets at every com2num position (GPU), for two analyses in checks_words.py:
  - echo for number words: is the lens's "seventy" about the last number (7xx) or about mu?
  - magnitude words and uncertainty: does their readout track the output's sd across positions, as steering
    (steer.py) suggests it drives it in Llama-3.1-8B?

Recorded, per sequence (stationary and switch), com2num position, lens and layer: log-sum-exp of the lens's
log-softmax over each word set (single tokens with a leading space).

    sbatch --qos=debug --time=00:30:00 hpc/gpu.sbatch .venv-tf5/bin/python experiments/09_beliefs/wordsets.py --model llama3.1-8b
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
from jlens_open.models import lens_path  # noqa: E402
from stimuli import sequences  # noqa: E402

DECADES = ["thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
SETS = {
    "magnitude": ["trillion", "billion", "billions", "million", "millions", "thousand", "thousands", "infinite",
                  "huge", "hundreds"],
    **{d: [d, d.capitalize()] for d in DECADES},
    "surprise": LEXICON["surprise"],
    "change": LEXICON["change"],
    "normal": ["normal", "typical", "expected", "usual", "consistent", "similar", "same", "fine", "ok", "random",
               "stable", "average"],
    "control": LEXICON["control"],
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--device", default=None)
parser.add_argument("--raw", default=RAW)
parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--chunk", type=int, default=2048)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--numbers", type=int, default=None)
args = parser.parse_args()

t0 = time.time()
lm = LensModel(args.model, args.device)
L, dev = lm.n_blocks, lm.device
lenses = {"logit": None, "jlens": lm.J}
if f"{args.model}-jpp" in MODELS:
    f = torch.load(lens_path(MODELS[f"{args.model}-jpp"]), weights_only=True)
    lenses["jpp"] = {layer: m.float().to(dev) for layer, m in f["J"].items()}
    del f
int_id = [lm.tok.encode(str(k), add_special_tokens=False)[0] for k in range(1000)]
comma = lm.tok.encode(",", add_special_tokens=False)[0]
BOS = [lm.tok.bos_token_id] if lm.tok.bos_token_id is not None else []
set_ids, kept = {}, {}
for name, words in SETS.items():
    ids = [t[0] for t in (lm.tok.encode(" " + w, add_special_tokens=False) for w in words) if len(t) == 1]
    set_ids[name] = torch.tensor(sorted(set(ids)), device=dev)
    kept[name] = len(set(ids))
S = len(SETS)
print(f"{args.model}: lenses {list(lenses)}, sets {kept}", flush=True)

OUT = os.path.join(args.raw, args.model)
seqs = sequences()
info = {"model": args.model, "sets": list(SETS), "kept": kept, "lenses": list(lenses), "n_blocks": L}
for name in ("stationary", "switch"):
    ss = [s for s in seqs if s["dataset"] == name]
    if args.limit:
        ss = [s for s in ss if s["seq"] < args.limit]
    if args.numbers:
        ss = [{**s, "numbers": s["numbers"][: args.numbers]} for s in ss]
    T = len(ss[0]["numbers"])
    com = torch.tensor([len(BOS) + 2 * t + 1 for t in range(T)], device=dev)
    X = {k: np.zeros((len(ss), T, L, S), np.float16) for k in lenses}
    batch = max(1, args.batch // (2 if name == "switch" else 1))
    for b0 in range(0, len(ss), batch):
        bs = ss[b0 : b0 + batch]
        ids = torch.tensor([BOS + [x for n in s["numbers"] for x in (int_id[n], comma)] for s in bs])
        hs, _ = lm.run(ids)
        for k, J in lenses.items():
            for layer in range(L):
                H = hs[layer][:, com].reshape(-1, lm.d)
                vals = []
                for c0 in range(0, len(H), args.chunk):
                    x = H[c0 : c0 + args.chunk]
                    lg = lm.unembed(x @ J[layer].T if J is not None and layer in J else x)
                    lsm = lg - torch.logsumexp(lg, 1, keepdim=True)
                    vals.append(torch.stack([torch.logsumexp(lsm[:, i], 1) for i in set_ids.values()], 1))
                X[k][b0 : b0 + len(bs), :, layer] = torch.cat(vals).reshape(len(bs), T, S).cpu().numpy()
        del hs
        print(f"{name}: {b0 + len(bs)}/{len(ss)} sequences, {time.time() - t0:.0f} s", flush=True)
    for k in lenses:
        np.save(os.path.join(OUT, f"{name}_{k}_wordsets.npy"), X[k])
    info[f"{name}_rows"] = [{"cond": s["cond"], "seq": s["seq"]} for s in ss]
info["seconds"] = round(time.time() - t0)
json.dump(info, open(os.path.join(OUT, "wordsets_info.json"), "w"), indent=1)
print(f"Done in {time.time() - t0:.0f} s", flush=True)
