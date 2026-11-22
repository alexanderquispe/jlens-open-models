"""Experiment 09, Concept Lens readouts (GPU): multi-token concepts in the workspace while the model reads the streams.

At the com2num position of number t (the comma after it, where the model predicts number t + 1) the Concept Lens
(jlens_open/concept.py; Gong, Huang and Wang 2026) is read at each of its source layers (Llama-3.1-8B: 10-30). The
prompt it sees is the stream cut after that comma, so the readout position is the prompt's last token, as in the
authors' code. Readouts:
  stationary  every condition, the first --seqs sequences, at t in --times (default 127 and 999)
  switch      the first --seqs sequences, at event times e = t - 1000 in --events (default -1 0 1 2 4 8 16 64 256 999)

Before the readouts, the carrier reader is fitted on the authors' 300 calibration concepts (downloaded once from
their repository at a fixed commit, checked by SHA-256).

Writes to $JLENS_SHARED/experiments/09_beliefs/<model>/: concepts_<lens>.jsonl (one line per readout and layer: the
candidates with their joint rank among candidates and the whole vocabulary, and the 10 best vocabulary tokens) and
concepts_<lens>_info.json (the reader fit, counts and timings).

    sbatch hpc/gpu.sbatch .venv-tf5/bin/python experiments/09_beliefs/concept_lens.py --model llama3.1-8b
    sbatch --gres=gpu:nvidia_h200:1 --mem=160G hpc/gpu.sbatch .venv-tf5/bin/python experiments/09_beliefs/concept_lens.py --model olmo-3-32b --lens jpp
    .venv-tf5/bin/python experiments/09_beliefs/concept_lens.py --model llama3.1-8b --device cpu --seqs 1 --times 39 \\
        --events none --conds m500_s100 --numbers 40 --src-layers 16,27 --raw /tmp/smoke   # smoke test
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import RAW  # noqa: E402
from jlens_open import MODELS, LensModel  # noqa: E402
from jlens_open.concept import ConceptLens, calibration_concepts  # noqa: E402
from jlens_open.models import lens_path  # noqa: E402
from stimuli import sequences  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--lens", choices=["jlens", "jpp"], default="jlens")
parser.add_argument("--device", default=None)
parser.add_argument("--raw", default=RAW)
parser.add_argument("--times", default="127,999", help="stationary: numbers t read (comma list, or none)")
parser.add_argument("--events", default="-1,0,1,2,4,8,16,64,256,999", help="switch: event times (or none)")
parser.add_argument("--seqs", type=int, default=10, help="sequences per condition")
parser.add_argument("--conds", default=None, help="comma list of stationary conditions (default all 17)")
parser.add_argument("--src-layers", default=None, help="comma list of source layers (default the authors', scaled)")
parser.add_argument("--kv-gb", type=float, default=16.0, help="memory for the batched BIND decode")
parser.add_argument("--numbers", type=int, default=None, help="cut every sequence to its first N numbers (smoke)")
args = parser.parse_args()

t0 = time.time()
lm = LensModel(args.model, args.device)
J = lm.J
if args.lens == "jpp":
    f = torch.load(lens_path(MODELS[f"{args.model}-jpp"]), weights_only=True)
    J = {layer: m.float().to(lm.device) for layer, m in f["J"].items()}
    del f
src = [int(x) for x in args.src_layers.split(",")] if args.src_layers else None
cl = ConceptLens(lm, J, source_layers=src, kv_budget_gb=args.kv_gb)
OUT = os.path.join(args.raw, args.model)
os.makedirs(OUT, exist_ok=True)
concepts = calibration_concepts(os.path.join(args.raw, "clens_singleton_calibration_300.json"))
fit = cl.fit_reader(concepts)
print(f"{args.model} ({args.lens}): source layers {cl.src[0]}-{cl.src[-1]}, carrier {cl.car}; reader on "
      f"{fit['used']}/{fit['requested']} concepts, paired cosine {fit['paired_cosine']:.3f}; setup {time.time() - t0:.0f} s",
      flush=True)
BOS = 1 if lm.tok.bos_token_id is not None else 0


def parse(s):
    return [] if s in (None, "", "none") else [int(x) for x in s.split(",")]


times, events = parse(args.times), parse(args.events)
conds = set(args.conds.split(",")) if args.conds else None
plan = []
for s in sequences():
    if s["seq"] >= args.seqs:
        continue
    if args.numbers:
        if s["dataset"] == "switch":
            h, at = args.numbers // 2, s["switch_at"]
            s["numbers"], s["switch_at"] = s["numbers"][at - h : at + h], h
        else:
            s["numbers"] = s["numbers"][: args.numbers]
        s["text"] = ",".join(map(str, s["numbers"])) + ","
    T = len(s["numbers"])
    if s["dataset"] == "stationary" and (conds is None or s["cond"] in conds):
        reads = [t for t in times if 0 <= t < T]
    elif s["dataset"] == "switch":
        reads = [s["switch_at"] + e for e in events if 0 <= s["switch_at"] + e < T]
    else:
        reads = []
    if reads:
        plan.append((s, reads))
n_read = sum(len(r) for _, r in plan)
print(f"{n_read} readouts on {len(plan)} sequences", flush=True)

path = os.path.join(OUT, f"concepts_{args.lens}.jsonl")
done, t_read = 0, time.time()
with open(path, "w") as out:
    for s, reads in plan:
        ids = lm.encode(s["text"]).tolist()
        hs = cl.block_outputs([ids], cl.src)
        for t in reads:
            pos = BOS + 2 * t + 1  # the comma after number t
            cloze = ",".join(map(str, s["numbers"][: t + 1])) + ","
            assert cl.encode(cloze) == ids[: pos + 1], "the cut prompt does not tokenize as the stream's prefix"
            res = cl.readout(cloze, {layer: hs[layer][0, pos] for layer in cl.src})
            base = {"dataset": s["dataset"], "cond": s["cond"], "seq": s["seq"], "mu": s["mu"], "sigma": s["sigma"], "t": t,
                    "e": t - s["switch_at"] if s["dataset"] == "switch" else None}
            for layer, rec in res.items():
                out.write(json.dumps({**base, "layer": layer,
                                      "concepts": [list(c) for c in rec["concepts"]],
                                      "tokens": [[lm.tok.decode([i]), sc] for i, sc in rec["tokens"]]}) + "\n")
            done += 1
            if done % 10 == 0 or done == n_read:
                el = time.time() - t_read
                print(f"{done}/{n_read} readouts, {el / done:.1f} s each, {len(cl.vC)} concepts carried", flush=True)
        del hs

info = {"model": args.model, "lens": args.lens, "source_layers": cl.src, "carrier_layer": cl.car, "reader": fit,
        "readouts": n_read, "concepts_carried": len(cl.vC), "concepts_unaligned": sum(v is None for v in cl.vC.values()),
        "seconds": round(time.time() - t0), "seconds_per_readout": round((time.time() - t_read) / max(n_read, 1), 2),
        "args": vars(args), "gpu": torch.cuda.get_device_name(0) if lm.device == "cuda" else lm.device}
with open(os.path.join(OUT, f"concepts_{args.lens}_info.json"), "w") as f:
    json.dump(info, f, indent=1)
print(f"Done in {info['seconds']} s: {path}", flush=True)
