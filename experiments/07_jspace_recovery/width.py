"""Occupancy against width: does the J-lens occupancy of real activations, and its excess over a Gaussian with
their covariance, grow with the width d? Within a family the vocabulary is fixed (Qwen3: N = 151,936 at d = 2560,
4096, 5120; OLMo-3: N = 100,278 at d = 4096, 5120), so only d changes.

Reads results/<model>_recovery.csv of recovery.py for each model (the noises alone, run with --native-only, and
the isotropic grid), d, N and the number of blocks from the model's config in the Hugging Face cache, and the
activations' covariances recovery.py saved in $JLENS_SHARED: where one dimension carries most of the variance
(massive activations), a Gaussian with that covariance puts it in every sample and is no baseline.
Outputs: results/width.csv (one row per model and layer), figures/width.{pdf,png}.

    python experiments/07_jspace_recovery/width.py
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from matplotlib.lines import Line2D

from jlens_open import MODELS

HERE = os.path.dirname(os.path.abspath(__file__))
FAMILIES = {"Qwen3": ["qwen3-4b", "qwen3-8b", "qwen3-32b"], "OLMo-3": ["olmo-3-7b", "olmo-3-32b"], "Llama-3.1": ["llama3.1-8b"]}
MARKERS = {"Qwen3": ("o", "-"), "OLMo-3": ("s", "--"), "Llama-3.1": ("^", ":")}
DEPTHS = (1 / 4, 1 / 2, 5 / 8, 3 / 4)  # recovery.py's default layers, as fractions of the blocks
BLUES = ["#6da7ec", "#2a78d6", "#184f95", "#0d366b"]  # shallow to deep, as in recovery.py
OCC = "occupancy (median over trials)"
S = 0.1  # planted share for the isotropic check
MASSIVE = 0.2  # share of the variance in one dimension above which the Gaussian is marked as no baseline


def config(name):
    with open(hf_hub_download(MODELS[name]["hf"], "config.json")) as f:
        c = json.load(f)
    c = c.get("text_config", c)
    return c["hidden_size"], c["vocab_size"], c["num_hidden_layers"]


rows = []
for family, names in FAMILIES.items():
    for name in names:
        path = os.path.join(HERE, f"results/{name}_recovery.csv")
        if not os.path.exists(path):
            print(f"{name}: no results yet")
            continue
        rec = pd.read_csv(path)
        d, N, blocks = config(name)
        cov_path = os.path.join(os.environ.get("JLENS_SHARED", ""), "experiments/07_jspace_recovery", f"{name}_activation_covariances.pt")
        covs = torch.load(cov_path, map_location="cpu")["covariance"] if os.path.exists(cov_path) else {}
        native = rec[(rec.k == 0) & (rec.dictionary == "J-lens")]
        iso = rec[(rec.noise == "isotropic") & (rec.dictionary == "J-lens") & (rec.weights == "equal") & (rec["share s"] == S) & (rec.k >= 50)]
        for layer in sorted(native["activation layer"].unique()):
            var = covs[int(layer)].diagonal().double() if int(layer) in covs else None
            row = {"family": family, "model": name, "d": d, "N": N, "blocks": blocks, "layer": int(layer),
                   "depth": min(DEPTHS, key=lambda f: abs(f - layer / blocks)),
                   "largest dimension's share of the variance": (var.max() / var.sum()).item() if var is not None else np.nan}
            for noise, tag in (("activations", "activations"), ("activation covariance", "Gaussian")):
                r = native[(native.noise == noise) & (native["activation layer"] == layer)].iloc[0]
                row |= {f"occupancy, {tag}": r[OCC], f"occupancy 25th percentile, {tag}": r["occupancy (25th percentile over trials)"],
                        f"occupancy 75th percentile, {tag}": r["occupancy (75th percentile over trials)"],
                        f"explained in 25 steps, {tag}": r["explained at k"], f"control explained in 25 steps, {tag}": r["explained by the control at k"]}
            # The share s for which the isotropic back-of-envelope s d / ((1 - s) 2 ln N) gives this occupancy
            r = row["occupancy, activations"] * 2 * math.log(N) / d
            row["share s the isotropic back-of-envelope needs"] = r / (1 + r)
            row["share of the excess explained that the Gaussian reproduces"] = (
                (row["explained in 25 steps, Gaussian"] - row["control explained in 25 steps, Gaussian"])
                / (row["explained in 25 steps, activations"] - row["control explained in 25 steps, activations"]))
            t = iso[iso.layer == layer]
            row |= {f"isotropic s = {S:g}, k >= 50: occupancy min": t[OCC].min() if len(t) else np.nan,
                    f"isotropic s = {S:g}, k >= 50: occupancy max": t[OCC].max() if len(t) else np.nan,
                    f"isotropic s = {S:g}: predicted": S * d / ((1 - S) * 2 * math.log(N))}
            rows.append(row)
width = pd.DataFrame(rows)
width.to_csv(os.path.join(HERE, "results/width.csv"), index=False, float_format="%.6g")
pd.set_option("display.width", 250)
print(width[["model", "d", "layer", "depth", "largest dimension's share of the variance", "occupancy, activations", "occupancy, Gaussian",
             "share s the isotropic back-of-envelope needs",
             "share of the excess explained that the Gaussian reproduces", f"isotropic s = {S:g}, k >= 50: occupancy min",
             f"isotropic s = {S:g}, k >= 50: occupancy max", f"isotropic s = {S:g}: predicted"]].round(2).to_string(index=False))

fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, tag in zip(axs[:2], ("activations", "Gaussian")):
    for family, names in FAMILIES.items():
        marker, style = MARKERS[family]
        for depth, color in zip(DEPTHS, BLUES):
            t = width[(width.family == family) & (width.depth == depth)].sort_values("d")
            ax.plot(t.d, t[f"occupancy, {tag}"], color=color, marker=marker, linestyle=style, markersize=5)
    ax.set_xlabel("Width d", fontsize=8)
    ax.set_ylabel("J-lens occupancy, median over positions", fontsize=8)
massive = width[width["largest dimension's share of the variance"] >= MASSIVE]
axs[1].scatter(massive.d, massive["occupancy, Gaussian"], s=110, facecolors="none", edgecolors="#d03b3b", linewidths=1.2, zorder=3,
               label=f"one dimension carries >= {MASSIVE:.0%} of the variance: no baseline")
axs[1].legend(fontsize=6, loc="upper left")
axs[0].set_title("Activations alone", fontsize=9)
axs[1].set_title("Gaussian with the activations' covariance (no words)", fontsize=9)
top = max(width["occupancy, activations"].max(), width["occupancy, Gaussian"].max()) * 1.1
for ax in axs[:2]:
    ax.set_ylim(0, top)
axs[0].legend(handles=[Line2D([], [], color=c, label=f"depth {f:g} of the blocks") for f, c in zip(DEPTHS, BLUES)]
              + [Line2D([], [], color="black", marker=m, linestyle=s, label=f) for f, (m, s) in MARKERS.items()], fontsize=6)

ax = axs[2]
lo, hi = f"isotropic s = {S:g}, k >= 50: occupancy min", f"isotropic s = {S:g}, k >= 50: occupancy max"
t = width.dropna(subset=[lo])
for family in FAMILIES:
    marker, _ = MARKERS[family]
    for depth, color in zip(DEPTHS, BLUES):
        u = t[(t.family == family) & (t.depth == depth)]
        pred = u[f"isotropic s = {S:g}: predicted"]
        ax.errorbar(pred, (u[lo] + u[hi]) / 2, yerr=(u[hi] - u[lo]) / 2, color=color, marker=marker, linestyle="none", markersize=5, capsize=2)
if len(t):
    m = max(t[hi].max(), t[f"isotropic s = {S:g}: predicted"].max()) * 1.1
    ax.plot([0, m], [0, m], color="black", linewidth=0.8, label="measured = predicted")
    ax.legend(fontsize=6)
ax.set_title(f"Planted words in isotropic noise, s = {S:g}, k from 50 to 400", fontsize=9)
ax.set_xlabel("Predicted s d / ((1 - s) 2 ln N)", fontsize=8)
ax.set_ylabel("J-lens occupancy, median over trials", fontsize=8)
for ax in axs:
    ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
out = os.path.join(HERE, "figures/width.pdf")
fig.savefig(out, bbox_inches="tight", metadata={"CreationDate": None})
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: {os.path.join(HERE, 'results/width.csv')}\n       {out}")
