"""How much of the residual stream is J-space, compared with chance, and what does removing it break?

Gurnee et al. (2026) define the J-space at layer l as the sparse nonnegative combinations (k <= 25) of the
J-lens vectors, the rows of W_U diag(g) J_l. They report that the J-space component carries "never more than
10%" of the activation variance, and that ablating it (k = 10) in a band of layers destroys multi-hop
reasoning while leaving MMLU intact. Both claims are tested here on an open model, with the controls the
group asked about (is 10% more than any large dictionary would give?):

  A. Variance. Mean-centered activations on wikitext are decomposed by nonnegative matching pursuit on five
     dictionaries of the same size (one atom per vocabulary token): J-lens vectors, logit-lens vectors
     (J = I), random isotropic directions, random directions shaped like the activations' covariance, and
     random directions shaped like the J-lens vectors' covariance (same second moments as the J-lens
     dictionary, but no atom is tied to a token).
     Reported: the fraction of variance explained after k = 1..25 atoms, and by the top-k principal components.
     The lens vectors are centered over the vocabulary (a shift shared by all rows of W_U does not change the
     softmax); the uncentered J-lens and logit-lens dictionaries are reported too, with the norm of their mean
     unit atom (1 = all atoms parallel).
  B. Ablation. In a band of 4 layers, every position except BOS loses its k-atom component (k = 10, 25) for
     each dictionary. Reported: accuracy on two-hop prompts ("the official language of the country whose
     capital is Lima is"), on their one-hop controls ("the official language of Peru is"; same answers, no
     intermediate entity), and the next-token loss on wikitext.

    python experiments/03_jspace/jspace.py --model llama3.1-8b
    python experiments/03_jspace/jspace.py --model llama3.1-8b --dicts "random (J-lens covariance)"   # add one
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
from jlens_open.facts import COUNTRIES, ONE_HOP, TWO_HOP
from jlens_open.jspace import component, decompose, lens_atoms, random_atoms

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: the first available)")
parser.add_argument("--n-text", type=int, default=16, help="wikitext paragraphs (128 tokens each)")
parser.add_argument("--band", type=int, default=4, help="layers per ablation band")
parser.add_argument("--skip-ablation", action="store_true")
parser.add_argument("--dicts", default=None,
                    help="comma-separated dictionaries to (re)compute; rows of the others are kept from existing results")
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_VAR = os.path.join(HERE, f"results/{args.model}_variance.csv")
CSV_ABL = os.path.join(HERE, f"results/{args.model}_ablation.csv")
FIG_OUT = os.path.join(HERE, f"figures/{args.model}.pdf")
os.makedirs(os.path.dirname(CSV_VAR), exist_ok=True)
os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
SKIP = 16  # the lenses were fit without the first 16 positions
K_MAX, K_ABLATE = 25, (10, 25)
DICTS = ["J-lens", "logit lens", "random (isotropic)", "random (activation covariance)", "random (J-lens covariance)"]
EXTRA = ["J-lens (uncentered)", "logit lens (uncentered)"]
RUN = args.dicts.split(",") if args.dicts else DICTS + EXTRA
assert set(RUN) <= set(DICTS + EXTRA), RUN

torch.manual_seed(0)
lm = LensModel(args.model, args.device)
L, dev = lm.n_blocks, lm.device

# Wikitext: the first 128 tokens of each paragraph
with open(os.path.join(HERE, "data/wikitext_sample.json")) as f:
    paragraphs = json.load(f)["paragraphs"]
text_ids = torch.stack([lm.encode(p)[:128] for p in paragraphs if len(lm.encode(p)) >= 128][: args.n_text])
hs, _ = lm.run(text_ids)
acts = [h[:, SKIP:].reshape(-1, lm.d) for h in hs]  # per layer (n_text * 112, d) on the device
del hs
n_tokens = acts[0].shape[0]
print(f"{args.model}: {n_tokens} wikitext tokens per layer")
mean = [a.mean(0) for a in acts]


def sqrt_of(cov):
    evals, evecs = torch.linalg.eigh(cov.cpu().double())
    return (evecs * evals.clamp(min=0).sqrt()).T.float().to(dev)  # z @ sqrt_of(cov) ~ N(0, cov)


def activation_cov(layer):
    x = acts[layer] - mean[layer]
    return x.T @ x / len(x)


def lens_cov(layer):
    a = lens_atoms(lm, layer)
    cov = sum(a[s : s + 16384].float().T @ a[s : s + 16384].float() for s in range(0, len(a), 16384))
    return cov / len(a)


def atoms_for(name, layer):
    vocab = lm.model.lm_head.weight.shape[0]
    if name.startswith("J-lens") or name.startswith("logit lens"):
        return lens_atoms(lm, layer, name.startswith("J-lens"), center="uncentered" not in name)
    if name == "random (isotropic)":
        return random_atoms(vocab, lm.d, dev, seed=1)
    if name == "random (J-lens covariance)":
        return random_atoms(vocab, lm.d, dev, seed=3, cov_sqrt=sqrt_of(lens_cov(layer)))
    return random_atoms(vocab, lm.d, dev, seed=2, cov_sqrt=sqrt_of(activation_cov(layer)))


def merged(df, path, first, first_order=None):
    """With --dicts, add the rows of the other dictionaries from `path`. Rows are sorted by `first` (in
    `first_order` if given), dictionary and k, so a partial run gives the same file as a full one."""
    if args.dicts and os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old[~old.dictionary.isin(RUN + ["none"])], df], ignore_index=True)
    order = {d: i for i, d in enumerate(["none", "top-k PCA"] + DICTS + EXTRA)}
    keys = pd.DataFrame({"a": df[first].map(first_order) if first_order else df[first], "b": df.dictionary.map(order), "c": df.k})
    return df.loc[keys.sort_values(["a", "b", "c"], kind="stable").index].reset_index(drop=True)


# A. Variance explained
var_rows = []
for layer in range(L):
    x = acts[layer] - mean[layer]
    total = (x**2).sum().item()
    if not args.dicts:
        pcs = torch.linalg.svdvals(x.cpu()) ** 2
        for k in (1, 10, 16, 25):
            var_rows.append({"layer": layer, "dictionary": "top-k PCA", "k": k, "explained": (pcs[:k].sum() / total).item()})
    for name in RUN:
        atoms = atoms_for(name, layer)
        cone = atoms.float().mean(0).norm().item()
        _, coef, res = decompose(x, atoms, K_MAX)
        for k in range(1, K_MAX + 1):
            var_rows.append({"layer": layer, "dictionary": name, "k": k, "explained": 1 - res[:, k - 1].sum().item() / total,
                             "mean unit atom norm": cone})
        del atoms
    print(f"variance: layer {layer} done", flush=True)
var_df = merged(pd.DataFrame(var_rows), CSV_VAR, "layer")
var_df.to_csv(CSV_VAR, index=False, float_format="%.10g")
pd.set_option("display.width", 250)
for k in (10, 25):
    print(f"\nFraction of centered activation variance explained with k = {k}")
    print(var_df[var_df.k == k].pivot_table(index="layer", columns="dictionary", values="explained").round(3).to_string())

if args.skip_ablation:
    raise SystemExit

# B. Ablation
lang_names = sorted({c[2] for c in COUNTRIES})
lang_ids = [lm.tok.encode(" " + w, add_special_tokens=False)[0] for w in lang_names]
assert len(set(lang_ids)) == len(lang_ids), "languages must start with different tokens"
prompts = {"two-hop": [], "one-hop": []}
answers = {"two-hop": [], "one-hop": []}
for country, capital, language in COUNTRIES:
    for t2, t1 in zip(TWO_HOP, ONE_HOP):
        prompts["two-hop"].append(lm.encode(t2.format(capital=capital)))
        prompts["one-hop"].append(lm.encode(t1.format(country=country)))
        for kind in prompts:
            answers[kind].append(lang_names.index(language))


def evaluate(hooks=None):
    """Accuracy among the languages on both prompt sets, and wikitext loss, with optional hooks {layer: fn}."""
    handles = [lm.blocks[l].register_forward_hook(fn) for l, fn in (hooks or {}).items()]
    try:
        out = {}
        for kind, ps in prompts.items():
            pred = np.zeros(len(ps), dtype=int)
            for _, idx in pd.Series(range(len(ps))).groupby([len(p) for p in ps]):
                idx = idx.to_list()
                with torch.no_grad():
                    logits = lm.model(torch.stack([ps[i] for i in idx]).to(dev), use_cache=False).logits[:, -1]
                pred[idx] = logits[:, lang_ids].argmax(1).cpu().numpy()
            out[f"{kind} accuracy"] = (pred == np.array(answers[kind])).mean()
        with torch.no_grad():
            logits = lm.model(text_ids.to(dev), use_cache=False).logits[:, SKIP:-1].float()
            nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), text_ids[:, SKIP + 1 :].reshape(-1).to(dev))
        out["wikitext loss"] = nll.item()
        return out
    finally:
        for h in handles:
            h.remove()


def ablation_hook(atoms, mu, k, stats):
    def f(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        x = hidden[:, 1:].float()
        flat = x.reshape(-1, x.shape[-1]) - mu
        idx, coef, _ = decompose(flat, atoms, k)
        comp = component(idx, coef, atoms)
        stats.append(((comp**2).sum() / (flat**2).sum()).item())
        hidden = hidden.clone()
        hidden[:, 1:] = (x - comp.reshape(x.shape)).to(hidden.dtype)
        return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden

    return f


abl_rows = [{"band": "none", "dictionary": "none", "k": 0, "removed variance": 0.0, **evaluate()}]
print("\nunablated:", abl_rows[0], flush=True)
bands = [list(range(s, min(s + args.band, L))) for s in range(0, L, args.band)]
for band in bands:
    for name in [d for d in DICTS if d in RUN]:
        atoms = {l: atoms_for(name, l) for l in band}
        for k in K_ABLATE:
            stats = []
            res = evaluate({l: ablation_hook(atoms[l], mean[l], k, stats) for l in band})
            abl_rows.append({"band": f"{band[0]}-{band[-1]}", "dictionary": name, "k": k, "removed variance": np.mean(stats), **res})
            print(abl_rows[-1], flush=True)
        del atoms
abl_df = merged(pd.DataFrame(abl_rows), CSV_ABL, "band", {"none": -1} | {f"{b[0]}-{b[-1]}": i for i, b in enumerate(bands)})
abl_df.to_csv(CSV_ABL, index=False, float_format="%.10g")
print("\nAblation (accuracy among languages; wikitext loss)")
print(abl_df.round(3).to_string(index=False))

fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
ax = axs[0]
for name in DICTS + ["top-k PCA"]:
    t = var_df[(var_df.dictionary == name) & (var_df.k == 25)]
    ax.plot(t.layer, t.explained, marker="o", markersize=2, label=name)
ax.set_title("Centered variance explained, k = 25", fontsize=9)
ax.set_xlabel("Layer (block output)", fontsize=8)
ax.legend(fontsize=6)
for ax, metric in zip(axs[1:], ["two-hop accuracy", "one-hop accuracy"]):
    base = abl_rows[0][metric]
    for name in DICTS:
        t = abl_df[(abl_df.dictionary == name) & (abl_df.k == 10)]
        ax.plot(range(len(t)), t[metric], marker="o", markersize=3, label=name)
    ax.axhline(base, color="black", linestyle=":", linewidth=0.8, label="no ablation")
    ax.set_xticks(range(len(bands)), [f"{b[0]}-{b[-1]}" for b in bands], fontsize=6)
    ax.set_title(f"{metric} after ablating k = 10 in a band", fontsize=9)
    ax.set_xlabel("Ablated layers", fontsize=8)
axs[1].legend(fontsize=6)
for ax in axs:
    ax.grid(linestyle="--", alpha=0.4)
fig.suptitle(args.model, fontsize=10)
plt.tight_layout()
# No CreationDate in the PDF, so that rerunning gives byte-identical files and git diff shows real changes only
fig.savefig(FIG_OUT, bbox_inches="tight", metadata={"CreationDate": None})
fig.savefig(FIG_OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: {CSV_VAR}\n       {CSV_ABL}\n       {FIG_OUT}")
