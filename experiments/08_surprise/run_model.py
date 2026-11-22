"""Experiment 08 on one model (GPU): behaviour and internal signals on the change-point and arithmetic stimuli.

The model reads every sequence once (teacher forcing: it never generates). Recorded:
  behaviour  Part A: the model's log-probability of each number, and its predictive distribution over the next
             integer (mean, sd, entropy, mass on integers). Part B: the log-probability of each token in a window
             around the edit, and at the token of r: the probabilities of r, of the near and the far wrong value.
  at every layer, at the positions read (Part A: the num / sep / pred slots of every number, see common.SLOTS;
  Part B: a window from 5 tokens before the edit to 15 after it, and around the later reuse of r):
    norm   |h|
    delta  |h - h_prev| (Part A: the same slot of the previous number; Part B: the previous token)
    div    |h - h0| / |h0|, with h0 the control arm (Part A) or the clean twin (Part B) at the same position
    lex    the J-lens readout of each common.LEXICON group: the mean z-score (over the vocabulary) of its words
    top    the 10 top J-lens tokens (Part A, sep slot)
  at 8 layers (common.decomp_layers), the J-space: nonnegative matching pursuit (jlens_open.jspace, k = 25) of
  the activation minus the mean activation over the part's stimuli: atom indices, coefficients and residual
  fractions (Part A: sep and pred slots; Part B: the window).
  With --save-hidden, Part A's hidden states at the sep slot at common.probe_layers (for belief probes).

Validity checks (an item that fails is dropped and counted): the twins have the same tokens except inside the
edit; Part A's numbers are one token each, at the same positions in every arm.

    sbatch hpc/gpu.sbatch python experiments/08_surprise/run_model.py --model llama3.1-8b --raw /resnick/scratch/$USER/surprise
    python experiments/08_surprise/run_model.py --model gpt2-small --device cpu --limit 4 --raw /tmp/smoke   # smoke test
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from common import ARMS, DATA, GROUPS, LEXICON, RAW, SKIP, T, TWINS, band, decomp_layers, probe_layers
from jlens_open import MODELS, LensModel
from jlens_open.jspace import decompose, lens_atoms
from prep import integer_table, prepare_arith, prepare_changepoint, tokenize

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--device", default=None)
parser.add_argument("--part", choices=["both", "changepoint", "arith"], default="both")
parser.add_argument("--data", default=DATA, help="folder with changepoint.csv and arith.jsonl (stimuli.py)")
parser.add_argument("--raw", default=RAW, help="output folder (large files; default $JLENS_SHARED/experiments/08_surprise)")
parser.add_argument("--limit", type=int, default=None, help="first N items of each part (smoke tests)")
parser.add_argument("--batch", type=int, default=24, help="sequences per forward pass")
parser.add_argument("--k", type=int, default=25, help="J-space atoms per activation")
parser.add_argument("--pre", type=int, default=5, help="Part B: tokens read before the edit")
parser.add_argument("--post", type=int, default=15, help="Part B: tokens read after the edit")
parser.add_argument("--save-hidden", action="store_true")
args = parser.parse_args()

t_start = time.time()
lm = LensModel(args.model, args.device)
L, d, dev = lm.n_blocks, lm.d, lm.device
DEC, PROBE = decomp_layers(L), probe_layers(L)
OUT = os.path.join(args.raw, args.model)
os.makedirs(OUT, exist_ok=True)
info = {"model": args.model, "hf": MODELS[args.model]["hf"], "n_blocks": L, "d": d, "lens_prompts": lm.n_lens_prompts,
        "decomp_layers": DEC, "probe_layers": PROBE, "band": band(L), "k": args.k, "args": vars(args),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev}


def check_tokenize(text):
    assert tokenize(lm.tok, text)[0] == lm.encode(text).tolist(), "tokenization differs from LensModel.encode"


# J-lens readout of the lexicon
lex_ids, dropped_words = {}, []
for g, words in LEXICON.items():
    keep = []
    for w in words:
        t = lm.tok.encode(" " + w, add_special_tokens=False)
        (keep.append(t[0]) if len(t) == 1 else dropped_words.append(w))
    lex_ids[g] = torch.tensor(keep, device=dev)
info["lexicon_kept"] = {g: len(v) for g, v in lex_ids.items()}
info["lexicon_dropped"] = dropped_words
print(f"{args.model}: {L} blocks, d = {d}; lexicon words kept {info['lexicon_kept']}, dropped {dropped_words}", flush=True)


@torch.no_grad()
def padded(X, chunk):
    """X (n, d) with copies of its first row appended up to a multiple of `chunk` rows, and n. On a GPU the
    matrix-multiply kernel depends on the number of rows, so a short last chunk can give a row slightly different
    numbers than a full chunk would; twins whose rows fell in different chunks then differed before the edit (by up
    to 0.008 z-units in the J-lens readout of the first Llama run). With every call the same shape, identical rows
    give identical outputs wherever they sit."""
    pad = (-len(X)) % chunk
    return (torch.cat([X, X[:1].expand(pad, -1)]) if pad else X), len(X)


@torch.no_grad()
def readout(H, layer, top=0, chunk=512):
    """J-lens readout of H (n, d) at `layer`: lexicon group scores (n, G) and the top-`top` tokens (n, top)."""
    H, n = padded(H, chunk)
    lex, tops = [], []
    for s in range(0, len(H), chunk):
        logits = lm.unembed(lm.transport(H[s : s + chunk], layer))
        z = (logits - logits.mean(1, keepdim=True)) / logits.std(1, keepdim=True)
        lex.append(torch.stack([z[:, ids].mean(1) for ids in lex_ids.values()], 1))
        if top:
            tops.append(logits.topk(top, 1).indices)
    return torch.cat(lex)[:n], (torch.cat(tops)[:n] if top else None)


@torch.no_grad()
def mean_activation(id_lists):
    """Mean block output at each layer over the positions >= SKIP, for centering the J-space."""
    tot, n = torch.zeros(L, d, device=dev), 0
    for ids in id_lists:
        hs, _ = lm.run(torch.tensor([ids]))
        tot += torch.stack([h[0, SKIP:].sum(0) for h in hs])
        n += max(len(ids) - SKIP, 0)
    return tot / n


@torch.no_grad()
def jspace(X, layer, mu):
    """Nonnegative matching pursuit of X - mu on the J-lens atoms: indices, coefficients, residual fractions."""
    Xc, n = padded(X - mu, 1024)  # decompose works in chunks of 1024 rows
    idx, coef, res = (v[:n] for v in decompose(Xc, ATOMS[layer], args.k))
    Xc = Xc[:n]
    return idx.int(), coef.half(), (res / (Xc**2).sum(1, keepdim=True).clamp(min=1e-12)).half()


def batches(groups, size):
    """Lists of items with sequences of the same length, at most `size` sequences per list."""
    for _, its in groups.items():
        for s in range(0, len(its), size):
            yield its[s : s + size]


# --------------------------------------------------------------------------------------------------------------
# Part A: change points
# --------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def run_changepoint():
    df = pd.read_csv(os.path.join(args.data, "changepoint.csv"))
    items = list(df.item.unique())[: args.limit] if args.limit else list(df.item.unique())
    check_tokenize(df.text.iloc[0])
    # Token id of each integer as it appears after SEP (for the model's predictive distribution over integers)
    table = integer_table(lm.tok)
    table_k = torch.tensor(list(table), device=dev, dtype=torch.float64)
    table_id = torch.tensor(list(table.values()), device=dev)
    col_of = np.full(1000, -1)
    col_of[list(table)] = np.arange(len(table))
    info["integers_one_token"] = len(table)
    prepared, n_drop = prepare_changepoint(df, lm.tok, items)
    info["changepoint_items"], info["changepoint_dropped"] = len(prepared), n_drop
    print(f"Part A: {len(prepared)} items valid, {n_drop} dropped; {len(table)} integers are one token", flush=True)
    if not prepared:
        return

    valid = list(prepared)
    row_of = {(it, j): 3 * i + j for i, it in enumerate(valid) for j in range(3)}
    N = 3 * len(valid)
    mu = mean_activation([prepared[it][0][0] for it in valid[:64]])
    G, nD, K = len(GROUPS), len(DEC), args.k
    R = {
        "logp": np.full((N, T), np.nan, np.float32),        # log p(x_t | prefix), full softmax
        "pred": np.full((N, T, 4), np.nan, np.float32),     # next integer: mean, sd, entropy, mass on integers
        "logp_next": np.full((N, T), np.nan, np.float32),   # log p(x_{t+1}), renormalized over integers
        "norm": np.full((N, T, 3, L), np.nan, np.float32),
        "delta": np.full((N, T, 3, L), np.nan, np.float32),
        "div": np.full((N, T, 3, L), np.nan, np.float32),
        "lex": np.full((N, T, 3, L, G), np.nan, np.float32),
        "top": np.full((N, T, L, 10), -1, np.int32),
        "dec_idx": np.full((N, T, 2, nD, K), -1, np.int32),
        "dec_coef": np.zeros((N, T, 2, nD, K), np.float16),
        "dec_res": np.full((N, T, 2, nD, K), np.nan, np.float16),
    }
    hidden = np.zeros((N, T, len(PROBE), d), np.float16) if args.save_hidden else None
    groups = {}
    for it in valid:
        groups.setdefault(len(prepared[it][0][0]), []).append(it)
    done = 0
    for its in batches(groups, max(1, args.batch // 3)):
        ids = torch.tensor([a[0] for it in its for a in prepared[it]])
        P = torch.tensor(np.stack([a[1] for it in its for a in prepared[it]]))
        nums = np.array([a[2] for it in its for a in prepared[it]])
        rows = [row_of[(it, j)] for it in its for j in range(3)]
        B = len(ids)
        b_idx = torch.arange(B)
        ctrl = torch.arange(B) // 3 * 3  # the control arm of each row's item
        hs, logits = lm.run(ids, all_logits=True)
        ids_d = ids.to(dev)
        # behaviour: log p of each number, from the logits at the token before it
        num_pos = P[:, :, 0].to(dev)
        lg = logits[b_idx[:, None].to(dev), num_pos - 1]  # (B, T, V)
        lp = torch.log_softmax(lg, -1).gather(-1, ids_d.gather(1, num_pos)[..., None]).squeeze(-1)
        # the model's distribution over the next integer, at the pred slot of x_t (t < T)
        pp = P[:, :-1, 2].to(dev)
        lpk = torch.log_softmax(logits[b_idx[:, None].to(dev), pp], -1)[..., table_id].double()  # (B, T-1, K)
        mass = lpk.exp().sum(-1)
        lq = lpk - torch.logsumexp(lpk, -1, keepdim=True)
        q = lq.exp()
        mean = (q * table_k).sum(-1)
        sd = ((q * table_k**2).sum(-1) - mean**2).clamp(min=0).sqrt()
        ent = -(q * lq).sum(-1)
        nxt_col = torch.tensor(col_of[nums[:, 1:]], device=dev)
        lp_next = torch.where(nxt_col >= 0, lq.gather(-1, nxt_col.clamp(min=0)[..., None]).squeeze(-1),
                              torch.full_like(mean, float("nan")))
        R["logp"][rows] = lp.cpu().numpy()
        R["pred"][rows, :-1] = torch.stack([mean, sd, ent, mass], -1).cpu().numpy()
        R["logp_next"][rows, :-1] = lp_next.cpu().numpy()
        del logits, lg
        Pc = P.clamp(min=0).to(dev)
        for layer in range(L):
            H = hs[layer][b_idx[:, None, None].to(dev), Pc]  # (B, T, 3, d)
            R["norm"][rows, :, :, layer] = H.norm(dim=-1).cpu().numpy()
            R["delta"][rows, 1:, :, layer] = (H[:, 1:] - H[:, :-1]).norm(dim=-1).cpu().numpy()
            H0 = H[ctrl.to(dev)]
            R["div"][rows, :, :, layer] = ((H - H0).norm(dim=-1) / H0.norm(dim=-1)).cpu().numpy()
            lex, top = readout(H.reshape(-1, d), layer, top=10)
            R["lex"][rows, :, :, layer] = lex.reshape(B, T, 3, G).cpu().numpy()
            R["top"][rows, :, layer] = top.reshape(B, T, 3, 10)[:, :, 1].cpu().numpy()
            if layer in DEC:
                j = DEC.index(layer)
                idx, coef, res = jspace(H[:, :, 1:3].reshape(-1, d), layer, mu[layer])
                R["dec_idx"][rows, :, :, j] = idx.reshape(B, T, 2, K).cpu().numpy()
                R["dec_coef"][rows, :, :, j] = coef.reshape(B, T, 2, K).cpu().numpy()
                R["dec_res"][rows, :, :, j] = res.reshape(B, T, 2, K).cpu().numpy()
            if hidden is not None and layer in PROBE:
                hidden[rows, :, PROBE.index(layer)] = H[:, :, 1].half().cpu().numpy()
        # slots that do not exist (sep and pred of the last number) are NaN / -1
        miss = (P < 0).numpy()
        for key in ("norm", "delta", "div"):
            R[key][rows] = np.where(miss[..., None], np.nan, R[key][rows])
        R["lex"][rows] = np.where(miss[..., None, None], np.nan, R["lex"][rows])
        R["top"][rows] = np.where(miss[:, :, 1, None, None], -1, R["top"][rows])
        done += B
        print(f"  Part A: {done}/{N} sequences ({time.time() - t_start:.0f} s)", flush=True)
    # sep / pred of the last number do not exist
    R["dec_idx"][:, -1], R["dec_coef"][:, -1], R["dec_res"][:, -1] = -1, 0, np.nan
    if hidden is not None:
        hidden[:, -1] = np.nan
    meta = df.set_index(["item", "arm"]).loc[[(it, a) for it in valid for a in ARMS]].reset_index()
    meta["row"] = np.arange(N)
    meta.drop(columns="text").to_csv(os.path.join(OUT, "changepoint_meta.csv"), index=False)
    np.savez(os.path.join(OUT, "changepoint.npz"), **R)
    if hidden is not None:
        np.save(os.path.join(OUT, "changepoint_hidden.npy"), hidden)


# --------------------------------------------------------------------------------------------------------------
# Part B: arithmetic twins
# --------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def run_arith():
    items = [json.loads(line) for line in open(os.path.join(args.data, "arith.jsonl"))]
    items = items[: args.limit] if args.limit else items
    check_tokenize(items[0]["texts"]["clean"])
    prepared, reasons, WMAX = prepare_arith(items, lm.tok, args.pre, args.post)
    n_drop = sum(reasons.values())
    info["arith_items"], info["arith_dropped"] = len(prepared), reasons
    print(f"Part B: {len(prepared)} items valid, {n_drop} dropped {reasons}", flush=True)
    if not prepared:
        return
    M, nT, G, nD, K = len(prepared), len(TWINS), len(GROUPS), len(DEC), args.k
    mu = mean_activation([p["ids"][0] for p in prepared[:64]])
    R = {
        "pos": np.full((M, WMAX), -1, np.int32),               # absolute token positions read (same for all twins)
        "logp": np.full((M, nT, WMAX), np.nan, np.float32),     # log p(token at pos | prefix)
        "res_logp": np.full((M, nT, 3), np.nan, np.float32),    # at the token of r: log p of r, near, far
        "res_top1": np.zeros((M, nT), bool),                    # the model's top-1 there is r
        "norm": np.full((M, nT, WMAX, L), np.nan, np.float32),
        "delta": np.full((M, nT, WMAX, L), np.nan, np.float32),
        "div": np.full((M, nT, WMAX, L), np.nan, np.float32),
        "lex": np.full((M, nT, WMAX, L, G), np.nan, np.float32),
        "dec_idx": np.full((M, nT, WMAX, nD, K), -1, np.int32),
        "dec_coef": np.zeros((M, nT, WMAX, nD, K), np.float16),
        "dec_res": np.full((M, nT, WMAX, nD, K), np.nan, np.float16),
    }
    for i, p in enumerate(prepared):
        R["pos"][i, : len(p["window"])] = p["window"]
    groups = {}
    for i, p in enumerate(prepared):
        groups.setdefault(p["n"], []).append(i)
    done = 0
    for its in batches(groups, max(1, args.batch // nT)):
        ids = torch.tensor([s for i in its for s in prepared[i]["ids"]])
        B = len(ids)
        b_idx = torch.arange(B, device=dev)
        pos = torch.tensor(R["pos"][its], device=dev).repeat_interleave(nT, 0)  # (B, WMAX)
        valid = pos >= 0
        pc = pos.clamp(min=1)
        hs, logits = lm.run(ids, all_logits=True)
        ids_d = ids.to(dev)
        lp = torch.log_softmax(logits[b_idx[:, None], pc - 1], -1).gather(-1, ids_d.gather(1, pc)[..., None]).squeeze(-1)
        R["logp"][its] = torch.where(valid, lp, torch.nan).reshape(len(its), nT, -1).cpu().numpy()
        res_pos = torch.tensor([prepared[i]["result"] for i in its], device=dev).repeat_interleave(nT)
        cand = torch.stack([ids_d[(b_idx // nT) * nT + TWINS.index(name), res_pos] for name in ("clean", "near", "far")], 1)
        lr = torch.log_softmax(logits[b_idx, res_pos - 1], -1)
        R["res_logp"][its] = lr.gather(1, cand).reshape(len(its), nT, 3).cpu().numpy()
        R["res_top1"][its] = (lr.argmax(1) == cand[:, 0]).reshape(len(its), nT).cpu().numpy()
        del logits, lr
        clean_row = b_idx // nT * nT
        for layer in range(L):
            H = hs[layer][b_idx[:, None], pc]  # (B, WMAX, d)
            Hp = hs[layer][b_idx[:, None], pc - 1]
            H0 = H[clean_row]
            for key, val in (("norm", H.norm(dim=-1)), ("delta", (H - Hp).norm(dim=-1)),
                             ("div", (H - H0).norm(dim=-1) / H0.norm(dim=-1))):
                R[key][its, :, :, layer] = torch.where(valid, val, torch.nan).reshape(len(its), nT, -1).cpu().numpy()
            lex, _ = readout(H.reshape(-1, d), layer)
            lex = torch.where(valid[..., None], lex.reshape(B, -1, G), torch.nan)
            R["lex"][its, :, :, layer] = lex.reshape(len(its), nT, -1, G).cpu().numpy()
            if layer in DEC:
                j = DEC.index(layer)
                idx, coef, res = jspace(H.reshape(-1, d), layer, mu[layer])
                shape = (len(its), nT, -1, K)
                idx = torch.where(valid[..., None], idx.reshape(B, -1, K), -1)
                R["dec_idx"][its, :, :, j] = idx.reshape(shape).cpu().numpy()
                R["dec_coef"][its, :, :, j] = coef.reshape(shape).cpu().numpy()
                R["dec_res"][its, :, :, j] = res.reshape(shape).cpu().numpy()
        done += len(its)
        print(f"  Part B: {done}/{M} items ({time.time() - t_start:.0f} s)", flush=True)
    meta = pd.DataFrame([{
        "item": p["it"]["id"], "n_tokens": p["n"], "result_pos": p["result"], "reuse_pos": p["reuse"],
        **{f"first_{name}": p["edit"][name][0] for name in TWINS}, **{f"last_{name}": p["edit"][name][1] for name in TWINS},
        "r": p["it"]["r"], "near_value": p["it"]["near_value"], "far_value": p["it"]["far_value"],
        "a": p["it"]["a"], "b": p["it"]["b"], "op": p["it"]["op"]} for p in prepared])
    meta.to_csv(os.path.join(OUT, "arith_meta.csv"), index=False)
    np.savez(os.path.join(OUT, "arith.npz"), **R)


ATOMS = {layer: lens_atoms(lm, layer) for layer in DEC}
if args.part in ("both", "changepoint"):
    run_changepoint()
if args.part in ("both", "arith"):
    run_arith()
info["seconds"] = round(time.time() - t_start)
with open(os.path.join(OUT, "run_info.json"), "w") as f:
    json.dump(info, f, indent=1)
print(f"Saved to {OUT} ({info['seconds']} s)")
