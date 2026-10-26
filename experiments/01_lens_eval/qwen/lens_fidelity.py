"""How well does each lens predict the model's own output distribution, layer by layer?

For every layer l: KL(p_final || p_lens), top-1 agreement with the model's next-token prediction, and whether
the model's top token is in the lens's top 10. Lenses: logit lens (h_l), J-lens (J_l h_l), and two
off-by-one variants (J_{l-1} h_l, J_{l+1} h_l) that would do better if our layer indexing were shifted.
Positions: 16 wikitext paragraphs from ../../03_jspace/data at positions 16..127 (the lens's fitting
distribution), and the readout positions of experiment 01.

    python experiments/01_lens_eval/qwen/lens_fidelity.py --model qwen3-8b
"""
import argparse
import json
import os

import pandas as pd
import torch

from jlens_open import MODELS, LensModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: the first available)")
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_OUT = os.path.join(HERE, f"results/lens_fidelity_{args.model}.csv")
SETS = ["multihop", "multilingual", "order-ops", "poetry", "association", "typo"]

lm = LensModel(args.model, args.device)
L = lm.n_blocks

with open(os.path.join(HERE, "../../03_jspace/data/wikitext_sample.json")) as f:
    paragraphs = json.load(f)["paragraphs"]
text_ids = torch.stack([lm.encode(p)[:128] for p in paragraphs if len(lm.encode(p)) >= 128][:16])
hs, _ = lm.run(text_ids)
wiki = [h[:, 16:].reshape(-1, lm.d) for h in hs]
del hs

evals = [[] for _ in range(L)]
for eval_name in SETS:
    with open(os.path.join(HERE, f"../data/lens-eval-{eval_name}.json")) as f:
        items = json.load(f)["items"]
    for item in items:
        ids = lm.encode(item["prompt"].rstrip(" "))
        position = len(ids) - 1
        if eval_name == "poetry":
            position = max(i for i, t in enumerate(ids.tolist()) if "\n" in lm.tok.decode([t]))
        hs, _ = lm.run(ids[None])
        for layer in range(L):
            evals[layer].append(hs[layer][0, position])
evals = [torch.stack(e) for e in evals]
print(f"{args.model}: {len(wiki[0])} wikitext positions, {len(evals[0])} readout positions", flush=True)


def lens(x, layer, variant):
    if variant == "logit lens":
        return x
    j = {"J-lens": layer, "J of layer l-1": layer - 1, "J of layer l+1": layer + 1}[variant]
    return x @ lm.J[j].T if j in lm.J else x


rows = []
for data_name, acts in (("wikitext", wiki), ("01 readout", evals)):
    final = acts[L - 1]
    for layer in range(L - 1):
        for variant in ("logit lens", "J-lens", "J of layer l-1", "J of layer l+1"):
            if variant == "J of layer l-1" and layer == 0:
                continue
            kl, top1, in10 = 0.0, 0, 0
            for s in range(0, len(final), 256):
                logp_final = torch.log_softmax(lm.unembed(final[s : s + 256]), -1)
                logp_lens = torch.log_softmax(lm.unembed(lens(acts[layer][s : s + 256], layer, variant)), -1)
                kl += (logp_final.exp() * (logp_final - logp_lens)).sum().item()
                target = logp_final.argmax(-1)
                top1 += (logp_lens.argmax(-1) == target).sum().item()
                in10 += (logp_lens.topk(10, -1).indices == target[:, None]).any(-1).sum().item()
            n = len(final)
            rows.append({"data": data_name, "layer": layer, "lens": variant, "KL": kl / n,
                         "top-1 agreement": top1 / n, "final top-1 in lens top-10": in10 / n})
    print(f"done: {data_name}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(CSV_OUT, index=False)
pd.set_option("display.width", 250)
for data_name in ("wikitext", "01 readout"):
    for metric in ("KL", "final top-1 in lens top-10"):
        print(f"\n{args.model}, {data_name}: {metric}")
        print(df[df.data == data_name].pivot(index="lens", columns="layer", values=metric).round(2).to_string())
print(f"\nSaved: {CSV_OUT}")
