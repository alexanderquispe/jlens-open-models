"""What does each lens read? Top-10 tokens of the J-lens and the logit lens at four layers (45, 55, 65 and
75% of depth), at the experiment-01 readout position, for the first items of one prompt set.

    python experiments/01_lens_eval/qwen/top_tokens.py --model qwen3-8b --set typo
"""
import argparse
import json
import os

from jlens_open import MODELS, LensModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--set", default="typo", choices=["multihop", "multilingual", "order-ops", "poetry", "association", "typo"])
parser.add_argument("--items", type=int, default=8)
parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: the first available)")
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
TXT_OUT = os.path.join(HERE, f"results/top_tokens_{args.model}_{args.set}.txt")

lm = LensModel(args.model, args.device)
L = lm.n_blocks
layers = [round(f * (L - 1)) for f in (0.45, 0.55, 0.65, 0.75)]
with open(os.path.join(HERE, f"../data/lens-eval-{args.set}.json")) as f:
    items = json.load(f)["items"][: args.items]

lines = []
for item in items:
    ids = lm.encode(item["prompt"].rstrip(" "))
    position = len(ids) - 1
    if args.set == "poetry":
        position = max(i for i, t in enumerate(ids.tolist()) if "\n" in lm.tok.decode([t]))
    hs, _ = lm.run(ids[None])
    lines.append(f"\n### {item['name']} (intermediates {item['intermediates']})")
    for layer in layers:
        for use_jacobian, lens in ((True, "J"), (False, "logit")):
            logits = lm.unembed(lm.transport(hs[layer][0, position], layer, use_jacobian)[None])[0]
            top = [lm.tok.decode([t]) for t in logits.topk(10).indices.tolist()]
            lines.append(f"  L{layer:2d} {lens:5s} {top}")
with open(TXT_OUT, "w") as f:
    f.write(f"{args.model}, {args.set}: top-10 tokens of each lens\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nSaved: {TXT_OUT}")
