"""Read number streams with the logit lens, the J-lens and the J++ lens (GPU). Used by experiments 10 and 12.

The model reads every stream once (teacher forcing). The input is a JSON-lines file of streams of equal length, each
with "numbers" (integers 0..999; every one is a single token, as is the comma) and any other keys (kept in the
meta file). Tokens: [BOS] n_0 , n_1 , ... n_{T-1} , -- number t is token BOS + 2t and the comma after it (where the
model predicts number t + 1) token BOS + 2t + 1.

Recorded (N streams, T numbers, L blocks):
  {name}_out_dist.npy      (N, T, 1000) float16  the output distribution over the integers at the comma after
                                                 number t, renormalized over them
  {name}_out.npy           (N, T, 5) float32     its mean, sd, entropy, its mass on the integers (before
                                                 renormalizing), and -log of the renormalized probability of the
                                                 number that comes next (NaN at the last comma)
  {name}_{lens}_scalars.npy (N, T, L, 3) float32 at the comma: the lens's distribution restricted to the integer
                                                 tokens -- its softmax mass on them, and its renormalized mean and sd
  {name}_{lens}_sets.npy   (N, T, 2, L, S) float16  log P(word set) under the lens (log-sum-exp of its log-softmax
                                                 over the set's single-token " word" forms), at [the comma after
                                                 number t, the token of number t]
  {name}_meta.json         the streams' keys (without numbers), the sets kept, the layers, timing

    sbatch --qos=debug --time=00:30:00 hpc/gpu.sbatch .venv-tf5/bin/python experiments/10_unlikely_vs_update/run_streams.py \
        --model llama3.1-8b --stimuli experiments/10_unlikely_vs_update/data/streams.jsonl --name streams
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
from common import RAW, SETS  # noqa: E402
from jlens_open import MODELS, LensModel  # noqa: E402
from jlens_open.models import lens_path  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--stimuli", required=True, help="JSON lines, one stream per line, all of the same length")
parser.add_argument("--name", required=True, help="prefix of the output files")
parser.add_argument("--raw", default=RAW, help="output folder (default $JLENS_SHARED/experiments/10_unlikely_vs_update)")
parser.add_argument("--lenses", default=None, help="comma list of logit, jlens, jpp (default: all the model has)")
parser.add_argument("--device", default=None)
parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--chunk", type=int, default=2048, help="rows per lens readout call")
parser.add_argument("--limit", type=int, default=None, help="first N streams only (smoke tests)")
parser.add_argument("--numbers", type=int, default=None, help="cut every stream to its first N numbers (smoke)")
args = parser.parse_args()

t_start = time.time()
streams = [json.loads(line) for line in open(args.stimuli)]
if args.limit:
    streams = streams[: args.limit]
if args.numbers:
    for s in streams:
        s["numbers"] = s["numbers"][: args.numbers]
T = len(streams[0]["numbers"])
assert all(len(s["numbers"]) == T for s in streams), "streams must have equal length"
assert all(0 <= x <= 999 for s in streams for x in s["numbers"])
N = len(streams)

lm = LensModel(args.model, args.device, lens=MODELS[args.model]["lens"] is not None)
L, d, dev = lm.n_blocks, lm.d, lm.device
lenses = {"logit": None}
if lm.J:
    lenses["jlens"] = lm.J
jpp_key = f"{args.model}-jpp"
if jpp_key in MODELS:
    f = torch.load(lens_path(MODELS[jpp_key]), weights_only=True)
    lenses["jpp"] = {layer: m.float().to(dev) for layer, m in f["J"].items()}
    del f
if args.lenses:
    want = args.lenses.split(",")
    assert set(want) <= set(lenses), f"{args.model} has lenses {list(lenses)}"
    lenses = {k: v for k, v in lenses.items() if k in want}
OUT = os.path.join(args.raw, args.model)
os.makedirs(OUT, exist_ok=True)

int_id = []
for k in range(1000):
    t = lm.tok.encode(str(k), add_special_tokens=False)
    assert len(t) == 1, f"{args.model}: {k} is {len(t)} tokens"
    int_id.append(t[0])
comma = lm.tok.encode(",", add_special_tokens=False)
assert len(comma) == 1
comma = comma[0]
INT = torch.tensor(int_id, device=dev)
KV = torch.arange(1000, device=dev, dtype=torch.float64)
BOS = [lm.tok.bos_token_id] if lm.tok.bos_token_id is not None else []
B0 = len(BOS)

set_ids, kept = [], {}
for g, words in SETS.items():
    keep = [w for w in words if len(lm.tok.encode(" " + w, add_special_tokens=False)) == 1]
    kept[g] = keep
    set_ids.append(torch.tensor([lm.tok.encode(" " + w, add_special_tokens=False)[0] for w in keep], device=dev))
assert all(len(i) for i in set_ids), f"empty word set: {kept}"
NS = len(set_ids)
print(f"{args.model}: {L} blocks, d = {d}, lenses {list(lenses)}; {N} streams of {T} numbers; "
      f"set sizes {[len(i) for i in set_ids]}", flush=True)


def token_ids(numbers):
    ids = list(BOS)
    for x in numbers:
        ids += [int_id[x], comma]
    return ids


# The streams are tokenized number by number; check once that this is how the tokenizer reads the text
_s = streams[0]["numbers"][:50]
assert lm.encode(",".join(map(str, _s)) + ",").tolist() == token_ids(_s), "tokenization differs from LensModel.encode"


@torch.no_grad()
def integer_stats(li):
    """li (n, 1000): log-softmax at the integer tokens -> (mass, mean, sd) (n, 3) and the renormalized q (n, 1000)."""
    li = li.double()
    lmass = torch.logsumexp(li, 1)
    q = torch.exp(li - lmass[:, None])
    mean = q @ KV
    sd = (q @ KV**2 - mean**2).clamp(min=0).sqrt()
    return torch.stack([lmass.exp(), mean, sd], 1), q


@torch.no_grad()
def lens_logits(X, layer, J):
    return lm.unembed(X @ J[layer].T if J is not None and layer in J else X)


com = torch.tensor([B0 + 2 * t + 1 for t in range(T)])
num = torch.tensor([B0 + 2 * t for t in range(T)])
pos = torch.cat([com, num])  # rows: T commas, then T number tokens
R = {"out_dist": np.zeros((N, T, 1000), np.float16), "out": np.full((N, T, 5), np.nan, np.float32)}
Sc = {k: np.zeros((N, T, L, 3), np.float32) for k in lenses}
St = {k: np.zeros((N, T, 2, L, NS), np.float16) for k in lenses}
with torch.no_grad():
    for b0 in range(0, N, args.batch):
        bs = streams[b0 : b0 + args.batch]
        B = len(bs)
        ids = torch.tensor([token_ids(s["numbers"]) for s in bs])
        hs, logits = lm.run(ids, all_logits=True)
        li = torch.log_softmax(logits[:, com.to(logits.device)], -1)[..., INT.to(logits.device)]  # (B, T, 1000)
        del logits
        for j, s in enumerate(bs):
            st, q = integer_stats(li[j].to(dev))
            ent = -(q * torch.log(q.clamp(min=1e-300))).sum(1)
            nxt = torch.tensor(s["numbers"][1:], device=dev)
            surp = torch.full((T,), float("nan"), dtype=torch.float64, device=dev)
            surp[:-1] = -torch.log(q[torch.arange(T - 1, device=dev), nxt].clamp(min=1e-300))
            R["out"][b0 + j] = torch.stack([st[:, 1], st[:, 2], ent, st[:, 0], surp], 1).float().cpu().numpy()
            R["out_dist"][b0 + j] = q.float().cpu().numpy().astype(np.float16)
        del li
        P = pos.to(dev)
        for k, J in lenses.items():
            for layer in range(L):
                X = hs[layer][:, P].reshape(-1, d)  # (B * 2T, d), row b * 2T + i
                sc, se = [], []
                for c0 in range(0, len(X), args.chunk):
                    lg = lens_logits(X[c0 : c0 + args.chunk], layer, J)
                    lse = torch.logsumexp(lg, 1, keepdim=True)
                    lsm = lg - lse
                    sc.append(integer_stats(lsm[:, INT])[0].float())
                    se.append(torch.stack([torch.logsumexp(lsm[:, i], 1) for i in set_ids], 1))
                sc = torch.cat(sc).reshape(B, 2 * T, 3)[:, :T]
                se = torch.cat(se).reshape(B, 2, T, NS).permute(0, 2, 1, 3)  # (B, T, [comma, number], NS)
                Sc[k][b0 : b0 + B, :, layer] = sc.cpu().numpy()
                St[k][b0 : b0 + B, :, :, layer] = se.cpu().numpy().astype(np.float16)
        del hs
        print(f"{args.name}: {b0 + B}/{N} streams, {time.time() - t_start:.0f} s", flush=True)

np.save(os.path.join(OUT, f"{args.name}_out_dist.npy"), R["out_dist"])
np.save(os.path.join(OUT, f"{args.name}_out.npy"), R["out"])
for k in lenses:
    np.save(os.path.join(OUT, f"{args.name}_{k}_scalars.npy"), Sc[k])
    np.save(os.path.join(OUT, f"{args.name}_{k}_sets.npy"), St[k])
meta = {"model": args.model, "hf": MODELS[args.model]["hf"], "n_blocks": L, "d": d, "lenses": list(lenses),
        "T": T, "bos": B0, "sets": list(SETS), "sets_kept": kept,
        "streams": [{k: v for k, v in s.items() if k != "numbers"} for s in streams],
        "args": vars(args), "seconds": round(time.time() - t_start),
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev}
json.dump(meta, open(os.path.join(OUT, f"{args.name}_meta.json"), "w"), indent=1)
print(f"Done in {meta['seconds']} s: {OUT}/{args.name}_*", flush=True)
