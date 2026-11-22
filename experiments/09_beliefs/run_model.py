"""Experiment 09 on one model (GPU): what the logit lens, the J-lens and the J++ lens say, layer by layer, while the
model reads the number streams of Sarfati et al. (2026), "The Shape of Beliefs".

The model reads every sequence once (teacher forcing; it never generates). Recorded:
  behaviour   at every token, the log-probability of the next token; at every com2num position (the comma after
              number t, where the model predicts number t + 1), its distribution over the 1000 integer tokens:
              mean, sd, entropy, mass on integers, and KL(model || N(mu, sigma)) as in the paper's Fig. 7 (the
              target is the Gaussian density on the integers 0..999, renormalized; for the switch, the segment of
              the number being predicted); the whole distribution at a few times.
  per lens and layer, at every token:
    scalars   the lens's distribution restricted to the integer tokens: its softmax mass on them, and its mean and
              sd renormalized over them ("the belief in number words")
    lex       the mean z-score (over the vocabulary) of the lens logits of each common.LEXICON group's words
    top       the 20 top tokens
    cells     at the com2num positions, the mean log-softmax over the whole vocabulary in cells: (condition, bin of
              t) for the stationary sequences, bin of event time for the switch (common.T_BINS, common.E_BINS)

Lenses: "logit" (unembed h), "jlens" (unembed J_l h, the model's lens in jlens_open.MODELS) and "jpp" (the J++ lens,
when MODELS has "<model>-jpp"; same file format and readout as the J-lens). The last block has no J: every lens is
the logit lens there.

    sbatch hpc/gpu.sbatch .venv-tf5/bin/python experiments/09_beliefs/run_model.py --model llama3.1-8b
    sbatch --gres=gpu:nvidia_h200:1 --mem=160G hpc/gpu.sbatch .venv-tf5/bin/python experiments/09_beliefs/run_model.py --model olmo-3-32b
    .venv-tf5/bin/python experiments/09_beliefs/run_model.py --model llama3.1-8b --device cpu --limit 1 --numbers 40 --raw /tmp/smoke
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import CONDITIONS, E_BINS, LEXICON, RAW, T_BINS, cond_name  # noqa: E402
from jlens_open import MODELS, LensModel  # noqa: E402
from jlens_open.models import lens_path  # noqa: E402
from stimuli import sequences  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--lenses", default=None, help="comma list of logit, jlens, jpp (default: all the model has)")
parser.add_argument("--device", default=None)
parser.add_argument("--raw", default=RAW, help="output folder (default $JLENS_SHARED/experiments/09_beliefs)")
parser.add_argument("--datasets", default="stationary,switch")
parser.add_argument("--batch", type=int, default=4, help="sequences per forward pass (halved for the switch)")
parser.add_argument("--chunk", type=int, default=1024, help="rows per lens readout call")
parser.add_argument("--top", type=int, default=20)
parser.add_argument("--limit", type=int, default=None, help="first N sequences of each condition (smoke tests)")
parser.add_argument("--numbers", type=int, default=None, help="cut every sequence to its first N numbers (smoke)")
args = parser.parse_args()

t_start = time.time()
lm = LensModel(args.model, args.device)
L, d, dev = lm.n_blocks, lm.d, lm.device
lenses = {"logit": None, "jlens": lm.J}
lens_prompts = {"logit": 0, "jlens": lm.n_lens_prompts}
jpp_key = f"{args.model}-jpp"
want = args.lenses.split(",") if args.lenses else ["logit", "jlens"] + (["jpp"] if jpp_key in MODELS else [])
if "jpp" in want:
    assert jpp_key in MODELS, f"no J++ lens registered for {args.model}"
    f = torch.load(lens_path(MODELS[jpp_key]), weights_only=True)
    lenses["jpp"] = {layer: m.float().to(dev) for layer, m in f["J"].items()}
    lens_prompts["jpp"] = f.get("n_prompts")
    del f
lenses = {k: v for k, v in lenses.items() if k in want}
OUT = os.path.join(args.raw, args.model)
os.makedirs(OUT, exist_ok=True)
info = {"model": args.model, "hf": MODELS[args.model]["hf"], "n_blocks": L, "d": d, "lenses": list(lenses),
        "lens_prompts": lens_prompts, "args": vars(args), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev}

# The integer tokens: every integer 0..999 must be one token (the paper's design; true for Llama 3 and OLMo 3)
int_id = []
for k in range(1000):
    t = lm.tok.encode(str(k), add_special_tokens=False)
    assert len(t) == 1, f"{args.model}: {k} is {len(t)} tokens; the stimuli need one token per integer"
    int_id.append(t[0])
comma = lm.tok.encode(",", add_special_tokens=False)
assert len(comma) == 1
comma = comma[0]
INT = torch.tensor(int_id, device=dev)
KV = torch.arange(1000, device=dev, dtype=torch.float64)
BOS = 1 if lm.tok.bos_token_id is not None else 0

lex_ids, dropped = {}, []
for g, words in LEXICON.items():
    keep = []
    for w in words:
        t = lm.tok.encode(" " + w, add_special_tokens=False)
        (keep.append(t[0]) if len(t) == 1 else dropped.append(w))
    lex_ids[g] = torch.tensor(keep, device=dev)
info["lexicon_kept"] = {g: len(v) for g, v in lex_ids.items()}
info["lexicon_dropped"] = dropped
G = len(lex_ids)
print(f"{args.model}: {L} blocks, d = {d}, lenses {list(lenses)}; lexicon dropped {dropped}", flush=True)


def token_ids(numbers):
    """[BOS] n_0 , n_1 , ... n_{T-1} , as LensModel.encode tokenizes the text (checked)."""
    ids = [lm.tok.bos_token_id] if BOS else []
    for x in numbers:
        ids += [int_id[x], comma]
    return ids


@torch.no_grad()
def padded(X, chunk):
    """X with copies of its first row appended to a multiple of `chunk` rows (same-shape GPU calls, see exp. 08)."""
    pad = (-len(X)) % chunk
    return (torch.cat([X, X[:1].expand(pad, -1)]) if pad else X), len(X)


@torch.no_grad()
def lens_logits(X, layer, J):
    return lm.unembed(X @ J[layer].T if J is not None and layer in J else X)


@torch.no_grad()
def integer_stats(li):
    """li (n, 1000): log-softmax at the integer tokens. Mass on them, mean and sd of the distribution renormalized
    over them, and that distribution."""
    li = li.double()
    lmass = torch.logsumexp(li, 1)
    q = torch.exp(li - lmass[:, None])
    mean = q @ KV
    sd = (q @ KV**2 - mean**2).clamp(min=0).sqrt()
    return torch.stack([lmass.exp(), mean, sd], 1).float(), q


def gauss_logp(mu, sigma):
    """log of the Gaussian density on the integers 0..999, renormalized (the target of the paper's KL)."""
    lp = -((KV - mu) ** 2) / (2.0 * sigma**2)
    return lp - torch.logsumexp(lp, 0)


@torch.no_grad()
def run_dataset(name, seqs):
    T = len(seqs[0]["numbers"])
    n_tok = BOS + 2 * T
    N = len(seqs)
    if name == "stationary":
        conds = [cond_name(m, s) for m, s in CONDITIONS]
        bins = np.searchsorted(T_BINS, np.arange(T), side="right") - 1
        n_bins = len(T_BINS) - 1
        cell_of = lambda s, t: conds.index(s["cond"]) * n_bins + bins[t]  # noqa: E731
        n_cells = len(conds) * n_bins
        snap = [t for t in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 999) if t < T]
    else:
        at = seqs[0]["switch_at"]
        e = np.arange(T) - at
        bins = np.searchsorted(E_BINS, e, side="right") - 1
        n_cells = len(E_BINS) - 1
        cell_of = lambda s, t: bins[t] if 0 <= bins[t] < n_cells else -1  # noqa: E731
        snap = [t for t in (at - 1, at, at + 1, at + 2, at + 4, at + 8, at + 16, at + 32, at + 64, at + 128,
                            at + 256, at + 512, 2 * at - 1) if 0 <= t < T]
    com = np.array([BOS + 2 * t + 1 for t in range(T)])  # com2num positions
    R = {"out_logp": np.full((N, n_tok - 1), np.nan, np.float32),
         "out_int": np.full((N, T, 5), np.nan, np.float32),  # mean, sd, entropy, mass, KL(model || target)
         "out_snap": np.zeros((N, len(snap), 1000), np.float16)}
    S = {k: np.zeros((N, n_tok, L, 3), np.float32) for k in lenses}
    X = {k: np.zeros((N, n_tok, L, G), np.float16) for k in lenses}
    TOP = {k: np.zeros((N, n_tok, L, args.top), np.int32) for k in lenses}
    acc = {k: torch.zeros(L, n_cells, lm.model.lm_head.weight.shape[0], device=dev) for k in lenses}
    count = torch.zeros(n_cells, device=dev)
    batch = max(1, args.batch // (2 if name == "switch" else 1))
    targets = {}
    for b0 in range(0, N, batch):
        bs = seqs[b0 : b0 + batch]
        B = len(bs)
        ids = torch.tensor([token_ids(s["numbers"]) for s in bs])
        for s, row in zip(bs, ids):
            assert lm.encode(s["text"]).tolist() == row.tolist(), "tokenization differs from LensModel.encode"
        hs, logits = lm.run(ids, all_logits=True)
        # behaviour
        lsm = torch.log_softmax(logits, -1)
        R["out_logp"][b0 : b0 + B] = lsm[:, :-1].gather(-1, ids[:, 1:, None].to(dev))[..., 0].cpu().numpy()
        for j, s in enumerate(bs):
            st, q = integer_stats(lsm[j, com][:, INT])
            lq = torch.log(q.clamp(min=1e-300))
            ent = -(q * lq).sum(1)
            # the number predicted at the comma after number t is number t + 1: its segment's Gaussian is the target
            keys = [(s["mu"], s["sigma"]) if name == "stationary" or t + 1 < s["switch_at"] else (s["mu2"], s["sigma2"])
                    for t in range(T)]
            for key in set(keys):
                if key not in targets:
                    targets[key] = gauss_logp(*key)
            kl = (q * (lq - torch.stack([targets[key] for key in keys]))).sum(1)
            R["out_int"][b0 + j] = torch.stack([st[:, 1], st[:, 2], ent.float(), st[:, 0], kl.float()], 1).cpu().numpy()
            R["out_snap"][b0 + j] = q[snap].cpu().numpy().astype(np.float16)
        del lsm, logits
        # cell index of every row of the batch (com2num rows only; -1 elsewhere)
        cell = np.full((B, n_tok), -1)
        for j, s in enumerate(bs):
            cell[j, com] = [cell_of(s, t) for t in range(T)]
        cell = torch.tensor(cell.reshape(-1), device=dev)
        valid = cell >= 0
        count.index_add_(0, cell[valid], torch.ones(int(valid.sum()), device=dev))
        for k, J in lenses.items():
            for layer in range(L):
                H, n = padded(hs[layer].reshape(-1, d), args.chunk)
                sc, lx, tp = [], [], []
                for c0 in range(0, len(H), args.chunk):
                    lg = lens_logits(H[c0 : c0 + args.chunk], layer, J)
                    lse = torch.logsumexp(lg, 1)
                    sc.append(integer_stats(lg[:, INT] - lse[:, None])[0])
                    var, mean = torch.var_mean(lg, 1)  # z-scores over the vocabulary (unbiased sd, as experiment 08)
                    lx.append(torch.stack([(lg[:, i] - mean[:, None]).mean(1) for i in lex_ids.values()], 1)
                              / var.sqrt()[:, None])
                    tp.append(lg.topk(args.top, 1).indices)
                    rows = torch.arange(c0, c0 + len(lg), device=dev)
                    rows = rows[(rows < n) & valid[rows.clamp(max=n - 1)]]
                    if len(rows):
                        # per-cell sums of the log-softmax by a one-hot product (rows of a chunk share few cells)
                        u, inv = torch.unique(cell[rows], return_inverse=True)
                        onehot = torch.zeros(len(u), len(rows), device=dev)
                        onehot[inv, torch.arange(len(rows), device=dev)] = 1
                        r = rows - c0
                        acc[k][layer][u] += onehot @ lg[r] - (onehot @ lse[r])[:, None]
                S[k][b0 : b0 + B, :, layer] = torch.cat(sc)[:n].reshape(B, n_tok, 3).cpu().numpy()
                X[k][b0 : b0 + B, :, layer] = torch.cat(lx)[:n].reshape(B, n_tok, G).cpu().numpy().astype(np.float16)
                TOP[k][b0 : b0 + B, :, layer] = torch.cat(tp)[:n].reshape(B, n_tok, args.top).cpu().numpy()
        del hs
        print(f"{name}: {min(b0 + batch, N)}/{N} sequences, {time.time() - t_start:.0f} s", flush=True)
    np.savez(os.path.join(OUT, f"behaviour_{name}.npz"), snap_t=np.array(snap), **R)
    np.save(os.path.join(OUT, f"cells_count_{name}.npy"), count.cpu().numpy())
    for k in lenses:
        np.save(os.path.join(OUT, f"{name}_{k}_scalars.npy"), S[k])
        np.save(os.path.join(OUT, f"{name}_{k}_lex.npy"), X[k])
        np.save(os.path.join(OUT, f"{name}_{k}_top.npy"), TOP[k])
        mean = acc[k] / count.clamp(min=1)[None, :, None]
        np.save(os.path.join(OUT, f"{name}_{k}_cells.npy"), mean.half().cpu().numpy())
    meta = [{key: s[key] for key in s if key not in ("numbers", "text")} for s in seqs]
    with open(os.path.join(OUT, f"meta_{name}.json"), "w") as f:
        json.dump({"T": T, "bos": BOS, "sequences": meta}, f)
    info[f"{name}_sequences"] = N


all_seqs = sequences()
for name in args.datasets.split(","):
    seqs = [s for s in all_seqs if s["dataset"] == name and (args.limit is None or s["seq"] < args.limit)]
    if args.numbers:
        for s in seqs:
            if name == "switch":  # keep the change in the middle
                h = args.numbers // 2
                at = s["switch_at"]
                s["numbers"] = s["numbers"][at - h : at + h]
                s["switch_at"] = h
            else:
                s["numbers"] = s["numbers"][: args.numbers]
            s["text"] = ",".join(map(str, s["numbers"])) + ","
    run_dataset(name, seqs)

info["integers_one_token"] = True
info["seconds"] = round(time.time() - t_start)
with open(os.path.join(OUT, "run_info.json"), "w") as f:
    json.dump(info, f, indent=1)
print(f"Done in {info['seconds']} s: {OUT}", flush=True)
