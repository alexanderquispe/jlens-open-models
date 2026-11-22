"""Experiment 11, step 1 (GPU): does the inside signal of the jump cause the report?

Contexts: the switch contexts at cut e (default 4): stream k of N(300, 100), then e + 1 numbers from N(700, 100)
("jump") or N(300, 100) ("control"), same draws. Formats: base models "comma" (the number guess), "wait" (the
"Wait," cue), "q_surprise", "q_change"; instruct models the chat questions q_surprise, q_change (prompts.py).

The tail of a prompt is everything after the stream's last number (the same tokens in both arms). Pass 1 reads the
block outputs at the tail tokens at the chosen layers. Pass 2 re-runs each tail with one change at one layer l,
at every tail token, and reads the model's output at the last tail token. Changes (directions are fitted without the
test stream: leave one stream out):
  none                 nothing (the baseline)
  diff_pos  alpha      add alpha times the mean jump-minus-control difference at that tail token and layer
                       (alpha in ALPHAS; +1 on a control context transplants the average jump state at layer l)
  diff_comma alpha     (base models) add alpha times the mean difference at the comma after the last number,
                       at every tail token
  proj                 set the component along the unit diff_pos direction to the other arm's mean (control ->
                       jump, jump -> control): necessity along the one direction
  surprise c, ctrlw c  (models with a J-lens) add c ||h|| u, with u the lens direction of the surprise words or of
                       control words (experiment 09's steering), c in COEFS
  random{r} c          a random unit direction, c ||h||, c in (-1, 1)
  randm alpha          random0's direction with the norm of alpha times diff_pos at each tail token (alpha 1, 4): a
                       norm-matched control for diff_pos
Readouts at the last tail token: log P of the answers (yes / no / a / b, log-sum-exp over variants), log P of the
surprise, normal and change sets, and (comma) the integer mean and sd.

    sbatch --qos=debug --time=00:30:00 hpc/gpu.sbatch .venv-tf5/bin/python experiments/11_notice_report/inject.py --model llama3.1-8b
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
sys.path.insert(0, HERE)
import prompts  # noqa: E402
from jlens_open import MODELS, LensModel  # noqa: E402

SET_WORDS = {
    "surprise": ["surprising", "unexpected", "unusual", "strange", "odd", "weird", "outlier", "anomaly", "surprise",
                 "sudden", "suddenly", "wrong", "error", "interesting", "wait"],
    "normal": ["normal", "typical", "expected", "usual", "consistent", "similar", "same", "fine", "ok", "random",
               "stable", "average", "around", "about", "between"],
    "change": ["change", "changed", "changing", "different", "new", "shift", "jump", "switch", "increased",
               "increasing", "higher", "rising", "bigger", "now", "moved"],
}
CTRL_WORDS = ["table", "river", "green", "letter", "music", "paper", "window", "garden", "chair", "yellow", "bottle",
              "mountain", "pencil", "coffee", "island", "horse", "cloud", "kitchen", "forest", "shirt"]
ALPHAS = (-2.0, -1.0, 1.0, 2.0, 4.0)
COEFS = (-1.0, -0.5, 0.5, 1.0)
N_RANDOM = 2
RAW = (os.path.join(os.environ["JLENS_SHARED"], "experiments", "11_notice_report") if os.environ.get("JLENS_SHARED")
       else os.path.join(HERE, "raw"))

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--contexts", default=os.path.join(HERE, "data", "contexts.jsonl"))
parser.add_argument("--raw", default=RAW)
parser.add_argument("--device", default=None)
parser.add_argument("--e", type=int, default=4)
parser.add_argument("--layers", default=None, help="comma list (default: 8 layers over 1/4 .. 7/8 depth)")
parser.add_argument("--seqs", type=int, default=None, help="first N streams (smoke tests)")
parser.add_argument("--numbers", type=int, default=None, help="keep the last N numbers of each context (smoke)")
args = parser.parse_args()

t0 = time.time()
ctxs = [c for c in map(json.loads, open(args.contexts)) if c["block"] == "switch" and c["e"] == args.e]
if args.seqs:
    ctxs = [c for c in ctxs if c["seq"] < args.seqs]
if args.numbers:
    for c in ctxs:
        c["numbers"] = c["numbers"][-args.numbers :]
seqs = sorted({c["seq"] for c in ctxs})
by = {(c["seq"], c["arm"]): c for c in ctxs}
assert all((k, a) in by for k in seqs for a in ("jump", "control"))

has_lens = MODELS[args.model]["lens"] is not None
lm = LensModel(args.model, args.device, lens=has_lens)
L, d, dev, tok = lm.n_blocks, lm.d, lm.device, lm.tok
WINDOW = getattr(getattr(lm.model.config, "text_config", lm.model.config), "sliding_window", None)
layers = ([int(x) for x in args.layers.split(",")] if args.layers
          else sorted({round(L * f) for f in (0.25, 0.375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.875)}))
chat = prompts.is_chat(args.model)
fmt_names = ["q_surprise", "q_change"] if chat else ["comma", "wait", "q_surprise", "q_change"]
ans = prompts.answer_ids(tok)
set_ids = [torch.tensor([t[0] for t in (tok.encode(" " + w, add_special_tokens=False) for w in ws) if len(t) == 1],
                        device=dev) for ws in SET_WORDS.values()]
INT = torch.tensor([tok.encode(str(k), add_special_tokens=False)[0] for k in range(1000)], device=dev)
KV = torch.arange(1000, device=dev, dtype=torch.float64)


def split(numbers, name):
    """(prefix ids up to and including the last number, tail ids)."""
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


def lens_dir(words, layer):
    ids = [t[0] for t in (tok.encode(" " + w, add_special_tokens=False) for w in words) if len(t) == 1]
    w = lm.model.lm_head.weight[ids].float().to(dev) * lm.gain
    if layer in lm.J:
        w = w @ lm.J[layer]
    w = w / w.norm(dim=1, keepdim=True)
    u = w.mean(0)
    return u / u.norm()


def cache_of(prefix):
    if WINDOW:
        assert len(prefix) + 64 <= WINDOW, "prompt longer than the sliding window"
    out = lm.model(torch.tensor([prefix], device=dev), use_cache=True, logits_to_keep=1)
    kv = [(layer.keys, layer.values) for layer in out.past_key_values.layers]
    del out
    return kv


@torch.no_grad()
def run_tail(kv, tail, layer=None, A=None, U=None, c=None, V=None, target=None, read_layers=()):
    """Run the tail (same tokens in every row) on the cached prefix, with h <- h + A + c ||h|| U + (target - h.V) V
    at block `layer` (all (B, n, d) / (B, 1, 1) / (B, n)). Returns the last token's output log-softmax (B, vocab) and
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
        delta = A + c * hf.norm(dim=-1, keepdim=True) * U + (target[..., None] - (hf * V).sum(-1, keepdim=True)) * V
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
    """Rows of readouts from the output log-softmax (B, vocab)."""
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


with torch.no_grad():
    # ---------------------------------------------------------------- pass 1: tail states at the chosen layers
    tails, prefix, H = {}, {}, {}
    for (k, arm), c in by.items():
        pre_tail = {n: split(c["numbers"], n) for n in fmt_names}
        pres = {tuple(p) for p, _ in pre_tail.values()}
        assert len(pres) == 1, "formats must share the prefix"
        kv = cache_of(list(pres.pop()))
        for n, (_, tail) in pre_tail.items():
            tails.setdefault(n, tail)
            assert tails[n] == tail, "the tail must be the same tokens in every context"
            _, got = run_tail(kv, tail, read_layers=layers)
            H[(k, arm, n)] = torch.stack([got[l][0] for l in layers])  # (n_layers, n_tail, d)
        prefix[(k, arm)] = pre_tail[fmt_names[0]][0]
        del kv
    print(f"pass 1 done, {time.time() - t0:.0f} s; tails {[(n, len(t)) for n, t in tails.items()]}", flush=True)

    # ---------------------------------------------------------------- pass 2: changes at one layer
    rows = []
    for k in seqs:
        others = [j for j in seqs if j != k]
        kv_arm = {arm: cache_of(prefix[(k, arm)]) for arm in ("jump", "control")}
        for n in fmt_names:
            nt = len(tails[n])
            dj = torch.stack([H[(j, "jump", n)] - H[(j, "control", n)] for j in others]).mean(0)  # (nl, nt, d)
            mj = torch.stack([H[(j, "jump", n)] for j in others]).mean(0)
            mc = torch.stack([H[(j, "control", n)] for j in others]).mean(0)
            if not chat:
                dcomma = torch.stack([H[(j, "jump", "comma")][:, -1] - H[(j, "control", "comma")][:, -1]
                                      for j in others]).mean(0)  # (nl, d)
            for li, layer in enumerate(layers):
                conds = [("none", 0.0)] + [("diff_pos", a) for a in ALPHAS] + ([] if chat else [("diff_comma", a) for a in ALPHAS])
                conds += [("proj", 1.0)]
                if has_lens:
                    conds += [("surprise", c) for c in COEFS] + [("ctrlw", c) for c in COEFS]
                conds += [(f"random{r}", c) for r in range(N_RANDOM) for c in (-1.0, 1.0)]
                conds += [("randm", a) for a in (1.0, 4.0)]  # random direction with the norm of a times diff_pos
                B = len(conds)
                vhat = dj[li] / dj[li].norm(dim=-1, keepdim=True)  # (nt, d)
                g = torch.Generator(device="cpu").manual_seed(1000 * layer + 7)
                rnd = [torch.randn(d, generator=g).to(dev) for _ in range(N_RANDOM)]
                lens_u = {"surprise": lens_dir(SET_WORDS["surprise"], layer), "ctrlw": lens_dir(CTRL_WORDS, layer)} if has_lens else {}
                for arm in ("jump", "control"):
                    A = torch.zeros(B, nt, d, device=dev)
                    U = torch.zeros(B, nt, d, device=dev)
                    Cc = torch.zeros(B, 1, 1, device=dev)
                    V = torch.zeros(B, nt, d, device=dev)
                    T = torch.zeros(B, nt, device=dev)
                    for b, (name, a) in enumerate(conds):
                        if name == "diff_pos":
                            A[b] = a * dj[li]
                        elif name == "diff_comma":
                            A[b] = a * dcomma[li][None]
                        elif name == "proj":
                            V[b] = vhat
                            T[b] = ((mc if arm == "jump" else mj)[li] * vhat).sum(-1)
                        elif name in lens_u:
                            U[b], Cc[b] = lens_u[name][None], a
                        elif name == "randm":
                            u = rnd[0] / rnd[0].norm()
                            A[b] = a * dj[li].norm(dim=-1, keepdim=True) * u[None]
                        elif name.startswith("random"):
                            u = rnd[int(name[6:])]
                            U[b], Cc[b] = (u / u.norm())[None], a
                    lsm, _ = run_tail(kv_arm[arm], tails[n], layer, A, U, Cc, V, T)
                    for (name, a), r in zip(conds, readouts(lsm, n)):
                        rows.append({"seq": k, "arm": arm, "format": n, "layer": layer, "cond": name, "coef": a,
                                     "diff_norm": float(dj[li].norm(dim=-1).mean()), **r})
        del kv_arm
        print(f"stream {k}: done, {time.time() - t0:.0f} s", flush=True)

OUT = os.path.join(args.raw, args.model)
os.makedirs(OUT, exist_ok=True)
import pandas as pd  # noqa: E402

pd.DataFrame(rows).to_csv(os.path.join(OUT, f"inject_e{args.e}.csv.gz"), index=False)
json.dump({"model": args.model, "layers": layers, "formats": fmt_names,
           "tails": {n: [tok.decode([x]) for x in t] for n, t in tails.items()}, "alphas": ALPHAS, "coefs": COEFS,
           "args": vars(args), "seconds": round(time.time() - t0),
           "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev},
          open(os.path.join(OUT, f"inject_e{args.e}_info.json"), "w"), indent=1)
print(f"Done in {time.time() - t0:.0f} s: {OUT}/inject_e{args.e}.csv.gz", flush=True)
