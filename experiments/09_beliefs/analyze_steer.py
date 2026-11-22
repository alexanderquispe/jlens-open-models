"""Experiment 09, analysis of steer.py (CPU). Writes results/<model>_steer.md and figures/<model>_steer.png.

For each direction, layer and strength c: the change of the output's integer mean and sd (mean over the 10
sequences), and its z against the 8 random directions at the same layer and c (z = (named - mean of random) / sd of
random). Predictions if the lens words are used: "seventy" (and the integers 700-799) raise the mean, "thirty" (300-399)
lower it; "wide", "magnitude" or "surprise" widen the distribution (sd up).

    python experiments/09_beliefs/analyze_steer.py --model llama3.1-8b
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NAMED = ["int700", "seventy", "int300", "thirty", "magnitude", "surprise", "change", "wide", "control"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--raw", default=RAW)
    p.add_argument("--out", default=HERE)
    args = p.parse_args()
    d = pd.read_csv(os.path.join(args.raw, args.model, "steer.csv.gz"))
    g = d.groupby(["t", "basis", "layer", "dir", "c"])[["d_mean", "d_sd", "d_entropy", "d_mass"]].mean().reset_index()
    rnd = g[g.dir.str.startswith("random")].groupby(["t", "basis", "layer", "c"])[["d_mean", "d_sd"]].agg(["mean", "std"])
    rows = []
    for _, r in g[~g.dir.str.startswith("random")].iterrows():
        q = rnd.loc[(r.t, r.basis, r.layer, r.c)]
        rows.append({**r.to_dict(), "z_mean": (r.d_mean - q[("d_mean", "mean")]) / (q[("d_mean", "std")] + 1e-9),
                     "z_sd": (r.d_sd - q[("d_sd", "mean")]) / (q[("d_sd", "std")] + 1e-9)})
    z = pd.DataFrame(rows)
    base = d.groupby("t")[["base_mean", "base_sd"]].mean()
    out = [f"# Experiment 09 · steering with lens word directions: {args.model}", "",
           "Change of the output's integer mean and sd when c ||h|| u is added at the last comma (u: the unit "
           "direction raising the word set's lens logits), mean over 10 sequences of N(500, 100). z: against 8 random "
           "unit directions at the same layer and c. Unsteered output: "
           + "; ".join(f"t = {t}: mean {r.base_mean:.0f}, sd {r.base_sd:.0f}" for t, r in base.iterrows()) + ".", ""]
    for basis in ("jlens", "logit"):
        out += [f"## Directions from the {'J-lens' if basis == 'jlens' else 'logit lens'} (c = +0.5 and -0.5)", "",
                "| t | direction | " + " | ".join(f"L{l}" for l in sorted(z.layer.unique())) + " |",
                "|---|---|" + "---|" * z.layer.nunique()]
        for t in sorted(z.t.unique()):
            for name in NAMED:
                cells = []
                for l in sorted(z.layer.unique()):
                    a = z[(z.t == t) & (z.basis == basis) & (z.layer == l) & (z.dir == name) & (z.c == 0.5)]
                    b = z[(z.t == t) & (z.basis == basis) & (z.layer == l) & (z.dir == name) & (z.c == -0.5)]
                    if a.empty:
                        cells.append("")
                        continue
                    a, b = a.iloc[0], b.iloc[0]
                    cells.append(f"mean {a.d_mean:+.0f} (z {a.z_mean:+.1f}) / {b.d_mean:+.0f}; "
                                 f"sd {a.d_sd:+.0f} (z {a.z_sd:+.1f}) / {b.d_sd:+.0f}")
                out.append(f"| {t} | {name} | " + " | ".join(cells) + " |")
        out.append("")
    # figure: dose response at the layer of the largest |z| for the mean (int700, seventy) and sd (wide, magnitude)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.2))
    for ax, (name, col) in zip(axs, (("seventy", "d_mean"), ("thirty", "d_mean"), ("magnitude", "d_sd"), ("surprise", "d_sd"))):
        sub = g[(g.t == g.t.max()) & (g.basis == "jlens")]
        for l in sorted(sub.layer.unique()):
            s = sub[(sub.layer == l) & (sub.dir == name)].sort_values("c")
            ax.plot(s.c, s[col], marker="o", ms=3, label=f"L{l}")
            rr = sub[(sub.layer == l) & sub.dir.str.startswith("random")].groupby("c")[col].agg(["min", "max"])
            ax.fill_between(rr.index, rr["min"], rr["max"], color="grey", alpha=0.04)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"J-lens '{name}': {col} (t = {g.t.max()})", fontsize=9)
        ax.set_xlabel("c (fraction of ||h||)")
    axs[0].legend(fontsize=6, ncol=2)
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "figures", f"{args.model}_steer.png"), dpi=120)
    plt.close(fig)
    path = os.path.join(args.out, "results", f"{args.model}_steer.md")
    open(path, "w").write("\n".join(out))
    z.to_csv(os.path.join(args.out, "results", f"{args.model}_steer.csv"), index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
