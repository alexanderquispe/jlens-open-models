"""Does the truth show up in the workspace when the context pushes a falsehood?

Halawi, Denain and Steinhardt (ICLR 2024, "Overthinking the Truth") give a model few-shot demonstrations with
wrong labels: the model copies the wrong pattern, but early-exit (logit lens) readouts at middle layers still
pick the correct label, until "critical layers" override it. Here the same setup is read with the J-lens:
does the true answer become verbalizable before the model outputs the false one, and does the J-lens see it
better or earlier than the logit lens? This is the situation of an agent fed adversarial context by another
agent.

Three tasks, each with true demonstrations and with false ones (labels permuted consistently):
  parity     "Number: 37\nParity: odd"         even <-> odd
  continent  "Country: France\nContinent: Europe"   Europe -> Asia -> Africa -> Europe
  color      "Object: banana\nColor: yellow"        red -> yellow -> green -> blue -> white -> black -> red
At the last token of each query, every layer is read with both lenses, restricted to the task's labels:
`true` = the lens picks the true label, `false` = it picks the permuted label the demonstrations teach.

    python experiments/06_false_context/false_context.py --model llama3.1-8b
"""
import argparse
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
parser.add_argument("--demos", type=int, default=12, help="demonstrations per prompt")
parser.add_argument("--seeds", type=int, default=3, help="demonstration samples per query")
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_OUT = os.path.join(HERE, f"results/{args.model}.csv")
FIG_OUT = os.path.join(HERE, f"figures/{args.model}.pdf")
os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)

CONTINENTS = {
    "Europe": ["France", "Germany", "Italy", "Spain", "Portugal", "Greece", "Poland", "Sweden", "Norway", "Finland",
               "Hungary", "Denmark", "Austria", "Belgium", "Ireland", "Romania"],
    "Asia": ["Japan", "China", "India", "Thailand", "Vietnam", "Indonesia", "Iran", "Pakistan", "Mongolia", "Nepal",
             "Malaysia", "Cambodia", "Bangladesh", "Laos", "Afghanistan", "Uzbekistan"],
    "Africa": ["Egypt", "Kenya", "Nigeria", "Ethiopia", "Ghana", "Morocco", "Algeria", "Senegal", "Uganda", "Tanzania",
               "Zambia", "Angola", "Mali", "Sudan", "Cameroon", "Madagascar"],
}
COLORS = {
    "red": ["strawberry", "tomato", "blood", "cherry", "ruby"],
    "yellow": ["banana", "lemon", "sunflower", "egg yolk", "butter"],
    "green": ["grass", "frog", "lettuce", "emerald", "cucumber"],
    "blue": ["sky", "ocean", "sapphire", "blueberry", "denim"],
    "white": ["snow", "milk", "cloud", "sugar", "chalk"],
    "black": ["coal", "crow", "ink", "tar", "soot"],
}


def cycle(labels):
    return {a: labels[(i + 1) % len(labels)] for i, a in enumerate(labels)}


TASKS = {
    "parity": {
        "items": [(str(n), "even" if n % 2 == 0 else "odd") for n in range(10, 100)],
        "labels": ["even", "odd"], "fmt": "Number: {}\nParity:",
    },
    "continent": {
        "items": [(c, k) for k, cs in CONTINENTS.items() for c in cs],
        "labels": list(CONTINENTS), "fmt": "Country: {}\nContinent:",
    },
    "color": {
        "items": [(o, k) for k, os_ in COLORS.items() for o in os_],
        "labels": list(COLORS), "fmt": "Object: {}\nColor:",
    },
}

lm = LensModel(args.model, args.device)
L = lm.n_blocks
rng = np.random.default_rng(0)

records = []  # one per (prompt, layer, lens)
final_rows = []
for task_name, task in TASKS.items():
    labels = task["labels"]
    label_ids = [lm.tok.encode(" " + w, add_special_tokens=False)[0] for w in labels]
    assert len(set(label_ids)) == len(labels), labels
    perm = cycle(labels)
    items = task["items"]
    prompts, meta = [], []
    for qi, (x, y) in enumerate(items):
        for seed in range(args.seeds):
            others = [j for j in range(len(items)) if j != qi]
            demo_idx = rng.choice(others, size=args.demos, replace=False)
            for condition in ("true", "false"):
                demos = "".join(
                    task["fmt"].format(items[j][0]) + " " + (items[j][1] if condition == "true" else perm[items[j][1]]) + "\n\n"
                    for j in demo_idx
                )
                prompts.append(lm.encode(demos + task["fmt"].format(x)))
                meta.append({"task": task_name, "item": x, "seed": seed, "condition": condition,
                             "true": labels.index(y), "false": labels.index(perm[y])})
    by_len = pd.Series(range(len(prompts))).groupby([len(p) for p in prompts])
    # Batches of at most 8: the model computes logits at every position (batch x ~500 x vocab), and larger
    # batches exceed the size an MPS kernel accepts
    batches = [g.to_list()[s : s + 8] for _, g in by_len for s in range(0, len(g), 8)]
    for idx in batches:
        hs, logits = lm.run(torch.stack([prompts[i] for i in idx]))
        final = logits[:, label_ids].argmax(1).cpu().numpy()
        for b, i in enumerate(idx):
            m = meta[i]
            final_rows.append({**m, "final_true": final[b] == m["true"], "final_false": final[b] == m["false"]})
        for layer in range(L):
            h = hs[layer][:, -1]
            for use_jacobian, lens in ((True, "J-lens"), (False, "logit lens")):
                full = lm.unembed(lm.transport(h, layer, use_jacobian))
                pick = full[:, label_ids].argmax(1).cpu().numpy()
                true_ids = torch.tensor([label_ids[meta[i]["true"]] for i in idx], device=full.device)
                true_logit = full.gather(1, true_ids[:, None])
                rank = ((full > true_logit).sum(1) + 1).cpu().numpy()
                for b, i in enumerate(idx):
                    m = meta[i]
                    records.append({"task": task_name, "condition": m["condition"], "layer": layer, "lens": lens,
                                    "true": pick[b] == m["true"], "false": pick[b] == m["false"], "true_rank": rank[b]})
    print(f"done: {task_name} ({len(prompts)} prompts)", flush=True)

df = pd.DataFrame(records)
final_df = pd.DataFrame(final_rows)
summary = df.groupby(["task", "condition", "lens", "layer"]).agg(
    true=("true", "mean"), false=("false", "mean"), median_true_rank=("true_rank", "median")).reset_index()
final_summary = final_df.groupby(["task", "condition"])[["final_true", "final_false"]].mean()
summary.to_csv(CSV_OUT, index=False)
final_summary.to_csv(CSV_OUT.replace(".csv", "_final.csv"))
pd.set_option("display.width", 250)
print("\nModel output among the labels (fraction true / fraction following the permuted demonstrations)")
print(final_summary.round(3).to_string())
for task_name in TASKS:
    t = summary[(summary.task == task_name) & (summary.condition == "false")]
    print(f"\n{task_name}, false demonstrations: lens picks the TRUE label (and the permuted one)")
    print(t.pivot_table(index="layer", columns="lens", values=["true", "false"]).round(3).to_string())

fig, axs = plt.subplots(1, len(TASKS), figsize=(4 * len(TASKS), 3.2), sharey=True)
for ax, task_name in zip(axs, TASKS):
    for (condition, lens), style in {("false", "J-lens"): ("C3", "-"), ("false", "logit lens"): ("C3", "--"),
                                     ("true", "J-lens"): ("C0", "-"), ("true", "logit lens"): ("C0", "--")}.items():
        t = summary[(summary.task == task_name) & (summary.condition == condition) & (summary.lens == lens)]
        ax.plot(t.layer, t["true"], linestyle=style[1], color=style[0], label=f"{lens}, {condition} demos")
    f = final_summary.loc[(task_name, "false"), "final_true"]
    ax.axhline(f, color="C3", linestyle=":", linewidth=1, label="model output, false demos")
    ax.axhline(1 / len(TASKS[task_name]["labels"]), color="grey", linestyle=":", linewidth=0.8, label="chance")
    ax.set_title(f"{task_name}: lens picks the true label", fontsize=9)
    ax.set_xlabel("Layer (block output)", fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)
axs[0].legend(fontsize=6)
fig.suptitle(args.model, fontsize=10)
plt.tight_layout()
# No CreationDate in the PDF, so that rerunning gives byte-identical files and git diff shows real changes only
fig.savefig(FIG_OUT, bbox_inches="tight", metadata={"CreationDate": None})
fig.savefig(FIG_OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: {CSV_OUT}\n       {FIG_OUT}")
