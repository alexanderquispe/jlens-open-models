"""Experiment 11, step 0 (GPU): ask the model about the stream, and read the answer inside it.

For every context (data/contexts.jsonl) and prompt format (prompts.formats: cues and questions for a base model,
chat questions for an instruct model), at the last prompt token:
  ans_{lens}.npy    (P, F, L, A) float32  log-softmax at the answer tokens (prompts.ANSWERS variants, in the order
                                          of meta["answer_tokens"]), per layer; at the last block this is the
                                          model's own answer
  sets_{lens}.npy   (P, F, L, 3) float32  log P(surprise set), log P(normal set), log P(change set) (experiment
                                          10's sets)
  intstats.npy      (P, 2) float32        the output's integer mean and sd at the "comma" format (base models)
  top.npy           (P, F, 10) int32      the output's 10 top tokens
  resid.npy         (P, F, L', d) float16 the residual stream (block outputs) at the layers meta["probe_layers"],
                                          for the probes (is the jump decodable where the model answers?)
Lenses: the logit lens, and the model's J-lens when it has one (instruct OLMo models have none).

    sbatch --qos=debug --time=00:30:00 hpc/gpu.sbatch .venv-tf5/bin/python experiments/11_notice_report/ask.py --model llama3.1-8b
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

SET_WORDS = {  # experiment 10's sets
    "surprise": ["surprising", "unexpected", "unusual", "strange", "odd", "weird", "outlier", "anomaly", "surprise",
                 "sudden", "suddenly", "wrong", "error", "interesting", "wait"],
    "normal": ["normal", "typical", "expected", "usual", "consistent", "similar", "same", "fine", "ok", "random",
               "stable", "average", "around", "about", "between"],
    "change": ["change", "changed", "changing", "different", "new", "shift", "jump", "switch", "increased",
               "increasing", "higher", "rising", "bigger", "now", "moved"],
}
RAW = (os.path.join(os.environ["JLENS_SHARED"], "experiments", "11_notice_report") if os.environ.get("JLENS_SHARED")
       else os.path.join(HERE, "raw"))

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--contexts", default=os.path.join(HERE, "data", "contexts.jsonl"))
parser.add_argument("--raw", default=RAW)
parser.add_argument("--device", default=None)
parser.add_argument("--limit", type=int, default=None, help="first N contexts (smoke tests)")
parser.add_argument("--numbers", type=int, default=None, help="keep the last N numbers of each context (smoke)")
parser.add_argument("--probe-every", type=int, default=2, help="store residuals at every k-th layer")
args = parser.parse_args()

t0 = time.time()
contexts = [json.loads(line) for line in open(args.contexts)]
if args.limit:
    contexts = contexts[: args.limit]
if args.numbers:
    for c in contexts:
        c["numbers"] = c["numbers"][-args.numbers :]
has_lens = MODELS[args.model]["lens"] is not None
lm = LensModel(args.model, args.device, lens=has_lens)
L, d, dev, tok = lm.n_blocks, lm.d, lm.device, lm.tok
WINDOW = getattr(getattr(lm.model.config, "text_config", lm.model.config), "sliding_window", None)
lenses = {"logit": None} | ({"jlens": lm.J} if has_lens else {})
FMTS = prompts.formats(args.model)
F = len(FMTS)
PAD = tok.eos_token_id if tok.eos_token_id is not None else 0
if isinstance(PAD, list):
    PAD = PAD[0]
probe_layers = list(range(0, L, args.probe_every))
if L - 1 not in probe_layers:
    probe_layers.append(L - 1)

ans = prompts.answer_ids(tok)
ans_tokens = [(a, i) for a, ids in ans.items() for i in ids]
A_IDS = torch.tensor([i for _, i in ans_tokens], device=dev)
set_ids = []
for g, ws in SET_WORDS.items():
    ids = [tok.encode(" " + w, add_special_tokens=False) for w in ws]
    set_ids.append(torch.tensor([t[0] for t in ids if len(t) == 1], device=dev))
int_id = [tok.encode(str(k), add_special_tokens=False) for k in range(1000)]
assert all(len(t) == 1 for t in int_id)
INT = torch.tensor([t[0] for t in int_id], device=dev)
KV = torch.arange(1000, device=dev, dtype=torch.float64)
print(f"{args.model}: {L} blocks, lenses {list(lenses)}, {len(contexts)} contexts, formats {[f[0] for f in FMTS]}, "
      f"answer tokens {[(a, tok.decode([i])) for a, i in ans_tokens]}", flush=True)


@torch.no_grad()
def run_family(common, tails):
    """Block outputs at each tail's last token, (n_tails, L, d) float32: the shared prefix is run once and the tails
    as a right-padded batch on its KV cache."""
    from transformers.cache_utils import DynamicCache, DynamicLayer

    if WINDOW:
        assert len(common) + max(map(len, tails)) <= WINDOW, "prompt longer than the sliding window"
    out = lm.model(torch.tensor([common], device=dev), use_cache=True, logits_to_keep=1)
    kv = [(layer.keys, layer.values) for layer in out.past_key_values.layers]
    del out
    b, k = len(tails), max(map(len, tails))
    batch = torch.tensor([t + [PAD] * (k - len(t)) for t in tails], device=dev)
    last = torch.tensor([len(t) - 1 for t in tails], device=dev)
    cache = DynamicCache()
    for K, Vv in kv:
        layer = DynamicLayer()
        layer.lazy_initialization(K, Vv)
        layer.keys, layer.values = K.expand(b, -1, -1, -1), Vv.expand(b, -1, -1, -1)
        cache.layers.append(layer)
    H = torch.zeros(b, L, d, device=dev)

    def hook(i):
        def f(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            H[:, i] = h[torch.arange(b, device=h.device), last.to(h.device)].float().to(dev)
        return f

    handles = [blk.register_forward_hook(hook(i)) for i, blk in enumerate(lm.blocks)]
    try:
        lm.model(batch, past_key_values=cache, use_cache=True, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return H


P = len(contexts)
ANS = {k: np.full((P, F, L, len(ans_tokens)), np.nan, np.float32) for k in lenses}
SETS = {k: np.full((P, F, L, len(set_ids)), np.nan, np.float32) for k in lenses}
INTS = np.full((P, 2), np.nan, np.float32)
TOP = np.full((P, F, 10), -1, np.int32)
RES = np.zeros((P, F, len(probe_layers), d), np.float16)
families = {}
for j, (name, fam, kind, changed) in enumerate(FMTS):
    families.setdefault(fam, []).append(j)
example = {}
for p, ctx in enumerate(contexts):
    for fam, js in families.items():
        ids = [prompts.build(tok, args.model, ctx["numbers"], FMTS[j][0]) for j in js]
        n = min(map(len, ids)) - 1
        lcp = next((i for i in range(n) if any(x[i] != ids[0][i] for x in ids)), n)
        if p == 0:
            for j, x in zip(js, ids):
                example[FMTS[j][0]] = tok.decode(x[-60:])
        H = run_family(ids[0][:lcp], [x[lcp:] for x in ids])  # (len(js), L, d)
        RES[p, js] = H[:, probe_layers].cpu().numpy().astype(np.float16)
        for k, J in lenses.items():
            for layer in range(L):
                X = H[:, layer]
                lg = lm.unembed(X @ J[layer].T if J is not None and layer in J else X)
                lsm = lg - torch.logsumexp(lg, 1, keepdim=True)
                ANS[k][p, js, layer] = lsm[:, A_IDS].cpu().numpy()
                SETS[k][p, js, layer] = torch.stack([torch.logsumexp(lsm[:, i], 1) for i in set_ids], 1).cpu().numpy()
                if k == "logit" and layer == L - 1:
                    TOP[p, js] = lg.topk(10, 1).indices.cpu().numpy()
                    for r, j in enumerate(js):
                        if FMTS[j][2] == "int":
                            li = lsm[r, INT].double()
                            q = torch.exp(li - torch.logsumexp(li, 0))
                            mean = (q * KV).sum()
                            INTS[p] = [mean.item(), ((q * KV**2).sum() - mean**2).clamp(min=0).sqrt().item()]
    if (p + 1) % 20 == 0 or p + 1 == P:
        print(f"{p + 1}/{P} contexts, {time.time() - t0:.0f} s", flush=True)

OUT = os.path.join(args.raw, args.model)
os.makedirs(OUT, exist_ok=True)
for k in lenses:
    np.save(os.path.join(OUT, f"ask_ans_{k}.npy"), ANS[k])
    np.save(os.path.join(OUT, f"ask_sets_{k}.npy"), SETS[k])
np.save(os.path.join(OUT, "ask_intstats.npy"), INTS)
np.save(os.path.join(OUT, "ask_top.npy"), TOP)
np.save(os.path.join(OUT, "ask_resid.npy"), RES)
meta = {"model": args.model, "hf": MODELS[args.model]["hf"], "n_blocks": L, "d": d, "lenses": list(lenses),
        "formats": FMTS, "answer_tokens": [[a, int(i), tok.decode([i])] for a, i in ans_tokens],
        "sets": list(SET_WORDS), "probe_layers": probe_layers, "example_prompt_ends": example,
        "contexts": [{k: v for k, v in c.items() if k != "numbers"} | {"n_numbers": len(c["numbers"])} for c in contexts],
        "args": vars(args), "seconds": round(time.time() - t0),
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev}
json.dump(meta, open(os.path.join(OUT, "ask_meta.json"), "w"), indent=1)
print(f"Done in {meta['seconds']} s: {OUT}/ask_*", flush=True)
