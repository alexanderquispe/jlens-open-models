"""Cue readouts (GPU): what the lenses say when a number stream is cut and followed by words. Used by experiment 10.

Each context (a JSON line with "numbers" and any other keys) is cut after its last number and followed by each cue of
common.CUES ("Wait,", "The last number is"), tokenized jointly with the stream (the cue text starts with the comma
after the last number). The stream is run once and the cues as a right-padded batch on its KV cache, as in
experiment 09's cue_lens.py. The model's own next-token distribution is the logit lens at the last block.

Recorded, per lens and layer, at every cue token:
  cues_{lens}_words.npy  (P, C, max_tail, L, W) float16  log-softmax at the words of common.SETS (single tokens
                                                         with a leading space), in the order of meta["words"]
  cues_{lens}_top.npy    (P, C, L, 20) int32             the 20 top tokens at the last cue token
  cues_meta.json         contexts (without numbers), cues, tail tokens, words, layers

    sbatch --qos=debug --time=00:30:00 hpc/gpu.sbatch .venv-tf5/bin/python experiments/10_unlikely_vs_update/run_cues.py \
        --model llama3.1-8b --contexts experiments/10_unlikely_vs_update/data/cue_contexts.jsonl
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
from common import CUES, RAW, SETS  # noqa: E402
from jlens_open import MODELS, LensModel  # noqa: E402
from jlens_open.models import lens_path  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--contexts", required=True)
parser.add_argument("--raw", default=RAW)
parser.add_argument("--prefix", default="cues", help="prefix of the output files")
parser.add_argument("--lenses", default=None)
parser.add_argument("--device", default=None)
parser.add_argument("--limit", type=int, default=None, help="first N contexts (smoke tests)")
parser.add_argument("--numbers", type=int, default=None, help="keep only the last N numbers of each context (smoke)")
parser.add_argument("--top", type=int, default=20)
args = parser.parse_args()

t_start = time.time()
contexts = [json.loads(line) for line in open(args.contexts)]
if args.limit:
    contexts = contexts[: args.limit]
if args.numbers:
    for c in contexts:
        c["numbers"] = c["numbers"][-args.numbers :]
lm = LensModel(args.model, args.device, lens=MODELS[args.model]["lens"] is not None)
L, d, dev = lm.n_blocks, lm.d, lm.device
lenses = {"logit": None}
if lm.J:
    lenses["jlens"] = lm.J
if f"{args.model}-jpp" in MODELS:
    f = torch.load(lens_path(MODELS[f"{args.model}-jpp"]), weights_only=True)
    lenses["jpp"] = {layer: m.float().to(dev) for layer, m in f["J"].items()}
    del f
if args.lenses:
    lenses = {k: v for k, v in lenses.items() if k in args.lenses.split(",")}
OUT = os.path.join(args.raw, args.model)
os.makedirs(OUT, exist_ok=True)
config = getattr(lm.model.config, "text_config", lm.model.config)
window = getattr(config, "sliding_window", None)

int_id = [lm.tok.encode(str(k), add_special_tokens=False)[0] for k in range(1000)]
comma = lm.tok.encode(",", add_special_tokens=False)[0]
BOS = [lm.tok.bos_token_id] if lm.tok.bos_token_id is not None else []
PAD = lm.tok.eos_token_id if lm.tok.eos_token_id is not None else 0

words, dropped = [], []
for g, ws in SETS.items():
    for w in ws:
        t = lm.tok.encode(" " + w, add_special_tokens=False)
        (words.append((g, w, t[0])) if len(t) == 1 else dropped.append(w))
W_IDS = torch.tensor([i for _, _, i in words], device=dev)
print(f"{args.model}: {L} blocks, lenses {list(lenses)}, {len(contexts)} contexts, {len(words)} words "
      f"(dropped {dropped})", flush=True)


def prompt_ids(numbers, cue):
    """(common, tail): the stream up to and including the last number, and the cue tokens (joint tokenization,
    checked to extend the stream's own tokens)."""
    common = BOS + [x for n in numbers for x in (int_id[n], comma)][:-1]
    ids = lm.encode(",".join(map(str, numbers)) + cue).tolist()
    assert ids[: len(common)] == common, "the cue changes the tokenization of the stream"
    return common, ids[len(common):]


@torch.no_grad()
def tail_states(common, tails):
    """Block outputs at the tail tokens, {layer: (n_cues, max_tail, d)} (float32)."""
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
REC = {k: {"words": np.full((P, C, max_tail, L, len(words)), np.nan, np.float16),
           "top": np.full((P, C, L, args.top), -1, np.int32)} for k in lenses}
n_tail = np.zeros((P, C), int)
tail_text = {}
for p, ctx in enumerate(contexts):
    commons, tails = zip(*(prompt_ids(ctx["numbers"], CUES[c]) for c in cue_names))
    assert all(c == commons[0] for c in commons)
    for j, t in enumerate(tails):
        n_tail[p, j] = len(t)
        assert len(t) <= max_tail
        tail_text.setdefault(cue_names[j], [lm.tok.decode([x]) for x in t])
    H = tail_states(commons[0], list(tails))
    rows = [(j, i) for j in range(C) for i in range(n_tail[p, j])]
    rj = torch.tensor([r[0] for r in rows], device=dev)
    ri = torch.tensor([r[1] for r in rows], device=dev)
    nj, ni = rj.cpu().numpy(), ri.cpu().numpy()
    last = torch.tensor([sum(n_tail[p, :j + 1]) - 1 for j in range(C)], device=dev)  # row of each cue's last token
    for k, J in lenses.items():
        for layer in range(L):
            X = H[layer][rj, ri]
            lg = lm.unembed(X @ J[layer].T if J is not None and layer in J else X)
            lsm = lg - torch.logsumexp(lg, 1, keepdim=True)
            REC[k]["words"][p, nj, ni, layer] = lsm[:, W_IDS].cpu().numpy()
            REC[k]["top"][p, :, layer] = lg[last].topk(args.top, 1).indices.cpu().numpy()
    del H
    if (p + 1) % 50 == 0 or p + 1 == P:
        print(f"{p + 1}/{P} contexts, {time.time() - t_start:.0f} s", flush=True)

for k in lenses:
    for name, arr in REC[k].items():
        np.save(os.path.join(OUT, f"{args.prefix}_{k}_{name}.npy"), arr)
meta = {"model": args.model, "hf": MODELS[args.model]["hf"], "n_blocks": L, "lenses": list(lenses), "cues": CUES,
        "cue_names": cue_names, "tail_tokens": tail_text, "words": [[g, w, int(i)] for g, w, i in words],
        "dropped": dropped, "n_tail": n_tail.tolist(),
        "contexts": [{k: v for k, v in c.items() if k != "numbers"} | {"n_numbers": len(c["numbers"])} for c in contexts],
        "args": vars(args), "seconds": round(time.time() - t_start),
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev}
json.dump(meta, open(os.path.join(OUT, f"{args.prefix}_meta.json"), "w"), indent=1)
print(f"Done in {time.time() - t_start:.0f} s: {OUT}/{args.prefix}_*", flush=True)
