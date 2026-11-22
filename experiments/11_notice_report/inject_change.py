"""Experiment 11, step 3 (GPU): inject the *pure change* direction of the 2 x 2 design.

inject.py adds the jump-minus-control difference, which mixes "the stream changed" with "the recent numbers are
high". Here the four cells of the 2 x 2 design at cut e (default 4) are used (history low / high x recent numbers
low / high; contexts.jsonl and contexts_down.jsonl, same draws):
  LL  low throughout (the up control)      LH  up jump (300 -> 700)
  HL  down jump (700 -> 300)               HH  high throughout (the down control)
At every chosen layer and tail token (the tail is everything after the stream's last number, the same tokens in
every cell), three directions are fitted on the other streams (leave one stream out):
  change   C = (LH + HL - LL - HH) / 2    changed cells minus unchanged cells: no main effect of the numbers
  recent   R = (LH + HH - LL - HL) / 2    recent numbers high minus low
  history  S = (HL + HH - LL - LH) / 2    history high minus low
Pass 2 re-runs each tail of each cell with one change at one layer, at every tail token:
  none               nothing
  change  alpha      add alpha C   (+1 on LL or HH gives the unchanged cell the average "changed" state; -1 on LH
                     or HL removes it)
  recent  alpha      add alpha R   (control: the other part of the old jump-minus-control difference)
  history alpha      add alpha S
  randm{r} alpha     a random direction with the norm of alpha C at each tail token (norm-matched control)
  proj               set the component along the unit C direction to the other class's mean (changed <-> unchanged)
Readouts at the last tail token: the answers, the surprise / normal / change word sets, and (comma) the integer
mean and sd. If C carries a report of change, +C must move every question toward its "changed" answer, including
the reversed-polarity question (q_same) and both orders of the two-option question (q_ab, q_ba), and must not move
the number guess the way R does.

    sbatch --qos=debug --time=00:30:00 --gres=gpu:nvidia_h200:1 hpc/gpu.sbatch .venv-tf5/bin/python \
        experiments/11_notice_report/inject_change.py --model llama3.1-8b
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import prompts  # noqa: E402
from jlens_open import MODELS, LensModel  # noqa: E402

SET_WORDS = {  # as inject.py
    "surprise": ["surprising", "unexpected", "unusual", "strange", "odd", "weird", "outlier", "anomaly", "surprise",
                 "sudden", "suddenly", "wrong", "error", "interesting", "wait"],
    "normal": ["normal", "typical", "expected", "usual", "consistent", "similar", "same", "fine", "ok", "random",
               "stable", "average", "around", "about", "between"],
    "change": ["change", "changed", "changing", "different", "new", "shift", "jump", "switch", "increased",
               "increasing", "higher", "rising", "bigger", "now", "moved"],
}
CELLS = {"LL": ("switch", "control"), "LH": ("switch", "jump"), "HL": ("switch_down", "jump"), "HH": ("switch_down", "control")}
CHANGED = {"LL": False, "LH": True, "HL": True, "HH": False}
ALPHAS = (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0)
ALPHAS_MAIN = (-1.0, 1.0, 4.0)  # recent, history
ALPHAS_RAND = (1.0, 4.0)
N_RANDOM = 2
DEPTHS = (0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
RAW = (os.path.join(os.environ["JLENS_SHARED"], "experiments", "11_notice_report") if os.environ.get("JLENS_SHARED")
       else os.path.join(HERE, "raw"))

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--up", default=os.path.join(HERE, "data", "contexts.jsonl"))
parser.add_argument("--down", default=os.path.join(HERE, "data", "contexts_down.jsonl"))
parser.add_argument("--raw", default=RAW)
parser.add_argument("--device", default=None)
parser.add_argument("--e", type=int, default=4)
parser.add_argument("--layers", default=None, help="comma list (default: 6 layers over 1/4 .. 7/8 depth)")
parser.add_argument("--seqs", type=int, default=None, help="first N streams (smoke tests)")
parser.add_argument("--numbers", type=int, default=None, help="keep the last N numbers of each context (smoke)")
args = parser.parse_args()

t0 = time.time()
by = {}
for path in (args.up, args.down):
    for c in map(json.loads, open(path)):
        for cell, (block, arm) in CELLS.items():
            if c["block"] == block and c.get("e") == args.e and c["arm"] == arm:
                by[(c["seq"], cell)] = c
seqs = sorted({k for k, _ in by})
if args.seqs:
    seqs = [k for k in seqs if k < args.seqs]
assert len(seqs) >= 3 and all((k, cell) in by for k in seqs for cell in CELLS), "every stream needs the four cells"
for k in seqs:  # the design: tails swap between the blocks
    n = args.e + 1
    assert by[(k, "LH")]["numbers"][-n:] == by[(k, "HH")]["numbers"][-n:]
    assert by[(k, "LL")]["numbers"][-n:] == by[(k, "HL")]["numbers"][-n:]
    assert by[(k, "LL")]["numbers"][:-n] == by[(k, "LH")]["numbers"][:-n]
    assert by[(k, "HL")]["numbers"][:-n] == by[(k, "HH")]["numbers"][:-n]
if args.numbers:
    for c in by.values():
        c["numbers"] = c["numbers"][-args.numbers:]

lm = LensModel(args.model, args.device, lens=False)
L, d, dev, tok = lm.n_blocks, lm.d, lm.device, lm.tok
WINDOW = getattr(getattr(lm.model.config, "text_config", lm.model.config), "sliding_window", None)
layers = [int(x) for x in args.layers.split(",")] if args.layers else sorted({round(L * f) for f in DEPTHS})
chat = prompts.is_chat(args.model)
fmt_names = list(prompts.CHAT) if chat else list(prompts.BASE)
ans = prompts.answer_ids(tok)
set_ids = [torch.tensor([t[0] for t in (tok.encode(" " + w, add_special_tokens=False) for w in ws) if len(t) == 1],
                        device=dev) for ws in SET_WORDS.values()]
INT = torch.tensor([tok.encode(str(k), add_special_tokens=False)[0] for k in range(1000)], device=dev)
KV = torch.arange(1000, device=dev, dtype=torch.float64)


def split(numbers, name):
    """(prefix ids up to and including the last number, tail ids). As inject.py."""
    full = prompts.build(tok, args.model, numbers, name)
    stream = ",".join(map(str, numbers))
    if chat:
        msg = [{"role": "user", "content": prompts.CHAT_INTRO + stream + "\n\n" + prompts.CHAT[name][0]}]
        text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        cut = text.index(stream) + len(stream)
        pre = tok(text[:cut], add_special_tokens=False).input_ids
    else:
        pre = tok(stream).input_ids
        if tok.bos_token_id is not None and pre[0] != tok.bos_token_id:
            pre = [tok.bos_token_id] + pre
    assert full[: len(pre)] == pre, f"{name}: the text after the stream changes its tokenization"
    return pre, full[len(pre):]


def cache_of(prefix):
    if WINDOW:
        assert len(prefix) + 64 <= WINDOW, "prompt longer than the sliding window"
    out = lm.model(torch.tensor([prefix], device=dev), use_cache=True, logits_to_keep=1)
    kv = [(layer.keys, layer.values) for layer in out.past_key_values.layers]
    del out
    return kv


@torch.no_grad()
def run_tail(kv, tail, layer=None, A=None, V=None, target=None, read_layers=()):
    """Run the tail on the cached prefix with h <- h + A + (target - h.V) V at block `layer` (A, V: (B, n, d);
    target: (B, n); rows with V = 0 are not projected). Returns the last token's output log-softmax (B, vocab) and
    {layer: block output (B, n, d)} for read_layers."""
    from transformers.cache_utils import DynamicCache, DynamicLayer

    B = 1 if A is None else len(A)
    cache = DynamicCache()
    for K, Vv in kv:
        lay = DynamicLayer()
        lay.lazy_initialization(K, Vv)
        lay.keys, lay.values = K.expand(B, -1, -1, -1), Vv.expand(B, -1, -1, -1)
        cache.layers.append(lay)
    got = {}

    def change(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        hf = h.float()
        delta = A + (target[..., None] - (hf * V).sum(-1, keepdim=True)) * V
        h = (hf + delta).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def reader(i):
        def f(module, inputs, output):
            got[i] = (output[0] if isinstance(output, tuple) else output).float()
        return f

    handles = [lm.blocks[i].register_forward_hook(reader(i)) for i in read_layers]
    if layer is not None:
        handles.append(lm.blocks[layer].register_forward_hook(change))
    try:
        out = lm.model(torch.tensor([tail] * B, device=dev), past_key_values=cache, use_cache=True, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return torch.log_softmax(out.logits[:, -1].float(), -1), got


def readouts(lsm, name):
    r = {f"lp_{a}": torch.logsumexp(lsm[:, ids], 1) for a, ids in ans.items() if ids}
    for g, ids in zip(SET_WORDS, set_ids):
        r[f"set_{g}"] = torch.logsumexp(lsm[:, ids], 1)
    if name == "comma":
        li = lsm[:, INT].double()
        q = torch.exp(li - torch.logsumexp(li, 1, keepdim=True))
        mean = q @ KV
        r["int_mean"] = mean
        r["int_sd"] = (q @ KV**2 - mean**2).clamp(min=0).sqrt()
    return [{k: float(v[b]) for k, v in r.items()} for b in range(len(lsm))]


conds = ([("none", 0.0)] + [("change", a) for a in ALPHAS] + [("recent", a) for a in ALPHAS_MAIN]
         + [("history", a) for a in ALPHAS_MAIN] + [(f"randm{r}", a) for r in range(N_RANDOM) for a in ALPHAS_RAND]
         + [("proj", 1.0)])
B = len(conds)

with torch.no_grad():
    # ---------------------------------------------------------------- pass 1: tail states at the chosen layers
    tails, prefix, H = {}, {}, {}
    for k in seqs:
        for cell in CELLS:
            pre_tail = {n: split(by[(k, cell)]["numbers"], n) for n in fmt_names}
            pres = {tuple(p) for p, _ in pre_tail.values()}
            assert len(pres) == 1, "formats must share the prefix"
            kv = cache_of(list(pres.pop()))
            for n, (_, tail) in pre_tail.items():
                tails.setdefault(n, tail)
                assert tails[n] == tail, "the tail must be the same tokens in every context"
                _, got = run_tail(kv, tail, read_layers=layers)
                H[(k, cell, n)] = torch.stack([got[l][0] for l in layers]).cpu()  # (n_layers, n_tail, d)
            prefix[(k, cell)] = pre_tail[fmt_names[0]][0]
            del kv
    assert len({len(p) for p in prefix.values()}) == 1, "prefixes must have the same length in every cell"
    print(f"pass 1 done, {time.time() - t0:.0f} s; tails {[(n, len(t)) for n, t in tails.items()]}", flush=True)

    # ---------------------------------------------------------------- pass 2: one change at one layer
    rows = []
    for k in seqs:
        others = [j for j in seqs if j != k]
        kv_cell = {cell: cache_of(prefix[(k, cell)]) for cell in CELLS}
        for n in fmt_names:
            nt = len(tails[n])
            M = {cell: torch.stack([H[(j, cell, n)] for j in others]).mean(0).to(dev) for cell in CELLS}  # (nl, nt, d)
            C = (M["LH"] + M["HL"] - M["LL"] - M["HH"]) / 2
            R = (M["LH"] + M["HH"] - M["LL"] - M["HL"]) / 2
            S = (M["HL"] + M["HH"] - M["LL"] - M["LH"]) / 2
            m_changed, m_unchanged = (M["LH"] + M["HL"]) / 2, (M["LL"] + M["HH"]) / 2
            for li, layer in enumerate(layers):
                vhat = C[li] / C[li].norm(dim=-1, keepdim=True)  # (nt, d)
                g = torch.Generator(device="cpu").manual_seed(1000 * layer + 11)
                rnd = [torch.randn(d, generator=g).to(dev) for _ in range(N_RANDOM)]
                info = {"norm_change": float(C[li].norm(dim=-1).mean()), "norm_recent": float(R[li].norm(dim=-1).mean()),
                        "norm_history": float(S[li].norm(dim=-1).mean()),
                        "cos_change_recent": float(torch.cosine_similarity(C[li], R[li], dim=-1).mean()),
                        "cos_change_history": float(torch.cosine_similarity(C[li], S[li], dim=-1).mean())}
                for cell in CELLS:
                    A = torch.zeros(B, nt, d, device=dev)
                    V = torch.zeros(B, nt, d, device=dev)
                    T = torch.zeros(B, nt, device=dev)
                    for b, (name, a) in enumerate(conds):
                        if name == "change":
                            A[b] = a * C[li]
                        elif name == "recent":
                            A[b] = a * R[li]
                        elif name == "history":
                            A[b] = a * S[li]
                        elif name.startswith("randm"):
                            u = rnd[int(name[5:])]
                            A[b] = a * C[li].norm(dim=-1, keepdim=True) * (u / u.norm())[None]
                        elif name == "proj":
                            V[b] = vhat
                            T[b] = ((m_unchanged if CHANGED[cell] else m_changed)[li] * vhat).sum(-1)
                    lsm, _ = run_tail(kv_cell[cell], tails[n], layer, A, V, T)
                    for (name, a), r in zip(conds, readouts(lsm, n)):
                        rows.append({"seq": k, "cell": cell, "format": n, "layer": layer, "cond": name, "coef": a,
                                     **info, **r})
        del kv_cell
        print(f"stream {k}: done, {time.time() - t0:.0f} s", flush=True)

OUT = os.path.join(args.raw, args.model)
os.makedirs(OUT, exist_ok=True)
import pandas as pd  # noqa: E402

pd.DataFrame(rows).to_csv(os.path.join(OUT, f"inject_change_e{args.e}.csv.gz"), index=False)
json.dump({"model": args.model, "layers": layers, "n_blocks": L, "formats": fmt_names, "seqs": seqs,
           "tails": {n: [tok.decode([x]) for x in t] for n, t in tails.items()},
           "conds": conds, "args": vars(args), "seconds": round(time.time() - t0),
           "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev},
          open(os.path.join(OUT, f"inject_change_e{args.e}_info.json"), "w"), indent=1)
print(f"Done in {time.time() - t0:.0f} s: {OUT}/inject_change_e{args.e}.csv.gz", flush=True)
