"""Replication of the Jacobian lens quality evaluation (Gurnee et al. 2026, §methods-comparison).

Six prompt sets from the reference code (data/evaluations at commit 581d398, Apache-2.0; copied to data/): multihop, multilingual,
order-ops, poetry, association, typo. Each item names intermediate concepts that never appear in
the prompt (e.g. "Brazil" in "The ocean on the coast of the country where Carnival is most famously
celebrated is the"). At one readout position, for every layer, the rank of each intermediate in the
lens readout over the full vocabulary is computed; the item score is the fraction of intermediates
whose min-over-layers rank is <= k, and pass@k is the mean over items. J-lens vs. logit lens (J = I).

Readout position, as in data/evaluations/README.md: the last prompt token (the token before the
target), except poetry, which reads at the newline ending line 1. Prompts ending in a space are
right-stripped, since these tokenizers attach the space to the next word. A word's rank is the
minimum over the first tokens of "word" and " word"; for order-ops, over the single-token synonyms
(numbers as digits and words, operations as symbols and words).

    python experiments/01_lens_eval/lens_eval.py --model gpt2-small
    python experiments/01_lens_eval/lens_eval.py --model llama3.1-8b
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from jlens_open import MODELS, LensModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: the first available)")
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_OUT = os.path.join(HERE, f"results/{args.model}.csv")
CSV_LAYERS = os.path.join(HERE, f"results/{args.model}_by_layer.csv")
FIG_OUT = os.path.join(HERE, f"figures/{args.model}.pdf")
SETS = ["multihop", "multilingual", "order-ops", "poetry", "association", "typo"]
NUMBER_WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four".split()
OPERATIONS = {
    "addition": ["+", "plus", "add", "addition", "sum"],
    "subtraction": ["-", "minus", "subtract", "subtraction"],
    "multiplication": ["*", "×", "times", "multiply", "multiplication"],
    "division": ["/", "÷", "divide", "divided", "division"],
    "squared": ["squared", "square", "²", "^"],
    "mod": ["mod", "%", "modulo", "remainder"],
}

items = {}
for name in SETS:
    with open(os.path.join(HERE, f"data/lens-eval-{name}.json")) as f:
        items[name] = json.load(f)["items"]

lm = LensModel(args.model, args.device)
tok = lm.tok
print(f"{args.model}: {lm.n_blocks} blocks, lens averaged over {lm.n_lens_prompts} prompts")


def target_ids(word, eval_name):
    if eval_name != "order-ops":
        return lm.first_token(word)
    synonyms = OPERATIONS.get(word) or [word] + ([NUMBER_WORDS[int(word)]] if word.isdigit() and int(word) < len(NUMBER_WORDS) else [])
    ids = set()
    for s in synonyms:
        for w in (s, " " + s):
            t = tok.encode(w, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
    return ids


def encode_with_position(prompt, eval_name):
    """Token ids (with BOS) and the readout position."""
    ids = lm.encode(prompt.rstrip(" "))
    if eval_name != "poetry":
        return ids, len(ids) - 1
    # Line 1 ends at the prompt's last newline; Llama 3's offset mapping is unreliable, so decode
    return ids, max(i for i, t in enumerate(ids.tolist()) if "\n" in tok.decode([t]))


rows, layer_rows = [], []
for eval_name in SETS:
    for item in items[eval_name]:
        ids, position = encode_with_position(item["prompt"], eval_name)
        hs, _ = lm.run(ids[None])
        for use_jacobian, lens in ((True, "J-lens"), (False, "logit lens")):
            x = torch.stack([lm.transport(hs[layer][0, position], layer, use_jacobian) for layer in range(lm.n_blocks)])
            logits = lm.unembed(x)  # (layers, vocab)
            for word in item["intermediates"]:
                t = sorted(target_ids(word, eval_name))
                if not t:
                    continue
                ranks = (logits[:, None, :] > logits[:, t][:, :, None]).sum(-1).min(1).values.cpu().numpy() + 1
                rows.append({"eval": eval_name, "item": item["name"], "word": word, "lens": lens, "min_rank": ranks.min()})
                layer_rows += [{"eval": eval_name, "lens": lens, "layer": layer, "rank": r} for layer, r in enumerate(ranks)]
    print(f"done: {eval_name} ({len(items[eval_name])} items)", flush=True)

df = pd.DataFrame(rows)
by_item = df.assign(**{f"pass@{k}": df.min_rank <= k for k in (1, 10)}).groupby(["eval", "lens", "item"])[["pass@1", "pass@10"]].mean()
table = by_item.groupby(["eval", "lens"]).mean().unstack("lens")
table.loc["macro (6 evals)"] = table.mean()
table.to_csv(CSV_OUT)
by_layer = pd.DataFrame(layer_rows).assign(hit10=lambda d: d["rank"] <= 10).groupby(["lens", "layer"]).hit10.mean().unstack("lens")
by_layer.to_csv(CSV_LAYERS)

pd.set_option("display.width", 200)
print(f"\npass@k (fraction of intermediates whose minimum rank over layers is <= k), {args.model}")
print(table.round(3).to_string())
print("\nFraction of intermediates in the top 10, by layer")
print(by_layer.round(3).to_string())

fig, ax = plt.subplots(figsize=(5, 3))
for lens, style in (("J-lens", "-"), ("logit lens", "--")):
    ax.plot(by_layer.index, by_layer[lens], style, marker="o", markersize=3, label=lens)
ax.set_xlabel("Layer (block output)")
ax.set_ylabel("intermediates in top 10")
ax.set_title(f"{args.model}: lens eval, 6 prompt sets", fontsize=9)
ax.grid(linestyle="--", alpha=0.4)
ax.legend(fontsize=7)
plt.tight_layout()
# No CreationDate in the PDF, so that rerunning gives byte-identical files and git diff shows real changes only
fig.savefig(FIG_OUT, bbox_inches="tight", metadata={"CreationDate": None})
fig.savefig(FIG_OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: {CSV_OUT}\n       {CSV_LAYERS}\n       {FIG_OUT}")
