"""Can the J-space reading be trusted? Geometry of the dictionary, and planted words recovered by the pursuit.

Gurnee et al. (2026) read an activation as a nonnegative combination of at most k ~ 25 J-lens vectors, one per
vocabulary token, found by a greedy pursuit. With far more vectors than dimensions (Llama-3.1-8B: 128,256 in
4096), is that reading stable, does the pursuit find it, and is the occupancy of ~25 a property of the model or
of the measurement? Part A and part B with isotropic noise use only the unembedding, the final norm's gain and
the lens (the model is loaded for its weights; it runs only for the activation noises of part B):

  A. Geometry, at every layer with a lens. For the decoder's unit atoms (centered over the vocabulary, as in
     experiment 03) of the J-lens, the logit lens and a random isotropic dictionary:
       - each atom's nearest neighbour (largest cosine with another atom), the coherence (largest |cosine|), the
         pairs above 0.9 / 0.95 / 0.99 and the clusters of near-duplicates they form;
       - the spectrum of (d / N) V^T V against the Marchenko-Pastur interval (1 +- sqrt(d / N))^2 of N random
         unit vectors in d dimensions;
       - the atoms' norms before normalization (an atom much shorter than the rest was mostly the shared mean
         that centering removed), and the singular values of J_l.
  B. Plant and recover. In each trial, k atoms are drawn at random, given positive weights (equal, or
     lognormal) and summed; isotropic Gaussian noise is added so that the planted part is a share s of |y|^2.
     The pursuit of experiment 03 (jlens_open.jspace.decompose) runs on y. Reported: the share of planted atoms
     among its first k picks, exactly and up to a near-duplicate (cosine >= 0.9); the share of trials with all
     k right; and the occupancy the paper's stopping rule reports, read as: the first step at which the
     dictionary's gain in explained |y|^2 falls below a random isotropic dictionary's gain on the same y. For
     equal weights and isotropic noise, a back-of-envelope puts that step near s d / ((1 - s) 2 ln N).
     --noise picks what the planted words are added to, at the same share s:
       - isotropic: Gaussian noise;
       - activation covariance: Gaussian noise with the covariance of real activations at the layer;
       - activations: real activations at the layer (centered block outputs on wikitext, positions >= 16), which
         carry their own J-space words. Here the rule is also applied to the activations alone (k = 0, s = 0).
     The activation noises are drawn at each J-lens layer, and the logit-lens and random dictionaries are run on
     the activations of each of those layers.

    python experiments/07_jspace_recovery/recovery.py --model llama3.1-8b
    python experiments/07_jspace_recovery/recovery.py --model llama3.1-8b --skip-geometry --noise "activation covariance,activations"
    python experiments/07_jspace_recovery/recovery.py --model llama3.1-8b --plot-only
"""
import argparse
import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

from jlens_open import MODELS, LensModel
from jlens_open.jspace import decompose, lens_atoms, random_atoms

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: the first available)")
parser.add_argument("--layers", default=None, help="comma-separated layers for part B (default: 1/4, 1/2, 5/8, 3/4 of the depth)")
parser.add_argument("--ks", default="5,10,25,50,100,200,400", help="planted atoms per trial")
parser.add_argument("--shares", default="1,0.5,0.2,0.1,0.05", help="planted share s of |y|^2")
parser.add_argument("--trials", type=int, default=128, help="trials per (dictionary, weights, s, k)")
parser.add_argument("--noise", default="isotropic",
                    help="comma-separated: isotropic, activation covariance, activations; rows of the others are kept from existing results")
parser.add_argument("--n-text", type=int, default=64, help="wikitext paragraphs (128 tokens each) for the activation noises")
parser.add_argument("--skip-geometry", action="store_true", help="part B only (keeps existing part A results)")
parser.add_argument("--skip-recovery", action="store_true", help="part A only (keeps existing part B results)")
parser.add_argument("--plot-only", action="store_true", help="redraw the figure from existing results")
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = {part: os.path.join(HERE, f"results/{args.model}_{part}.csv") for part in ("geometry", "distributions", "duplicates", "recovery")}
FIG_OUT = os.path.join(HERE, f"figures/{args.model}.pdf")
os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
os.makedirs(os.path.dirname(CSV["geometry"]), exist_ok=True)
DUP = 0.9  # cosine at which two atoms count as near-duplicates
THRESHOLDS = (0.9, 0.95, 0.99)
QS = np.concatenate([np.linspace(0, 0.98, 50), [0.99, 0.995, 0.999, 0.9995, 0.9999, 1.0]])
CONTROL_SEED = 11  # random dictionary of the stopping rule (seed 1 is the "random (isotropic)" dictionary, as in 03)
PROFILES = ("equal", "lognormal")
NOISES = ("isotropic", "activation covariance", "activations")
KS = [int(k) for k in args.ks.split(",")]
SHARES = [float(s) for s in args.shares.split(",")]
RUN_NOISES = args.noise.split(",")
assert set(RUN_NOISES) <= set(NOISES), RUN_NOISES
WIKITEXT = os.path.join(HERE, "../03_jspace/data/wikitext_sample.json")  # the paragraphs of experiment 03
SKIP = 16  # the lenses were fit without the first 16 positions
NATIVE_STEPS = 100  # steps of the pursuit on the activations alone
BLUES = ["#6da7ec", "#2a78d6", "#184f95", "#0d366b"]  # J-lens layers, shallow to deep
LOGIT, RANDOM = "#1baf7a", "#8c8c8c"


def rows_of(df, name, layer, noise="isotropic", act_layer=None):
    """Rows of one dictionary (layer None for the ones that do not depend on the layer); in the recovery results,
    with one noise and, for the activation noises, the layer the activations come from."""
    t = df[(df.dictionary == name) & (df.layer.isna() if layer is None else df.layer == layer)]
    if "noise" in t:
        t = t[t.noise == noise]
    return t if act_layer is None else t[t["activation layer"] == act_layer]


def instance_style(name, layer, layers):
    if name == "J-lens":
        return {"color": BLUES[layers.index(layer) % len(BLUES)], "label": f"J-lens, layer {layer}"}
    return {"color": LOGIT if name == "logit lens" else RANDOM, "label": name}


def plot(geo, dist, dup, rec):
    """Row 1: geometry (nearest neighbours, spectrum, near-duplicates across layers). Row 2: recovery and
    occupancy against the number of planted atoms, isotropic noise. Row 3 (if run): the activation noises."""
    layers = sorted(int(l) for l in rec[rec.dictionary == "J-lens"].layer.unique())
    mid = layers[len(layers) // 2 - 1] if len(layers) > 1 else layers[0]
    instances = [("J-lens", l) for l in layers] + [("logit lens", None), ("random (isotropic)", None)]
    oranges = dict(zip(sorted(SHARES), plt.cm.Oranges(np.linspace(0.35, 0.95, len(SHARES)))))
    s_real = min(SHARES, key=lambda s: abs(s - 0.1))
    extra = "noise" in rec and (rec.noise != "isotropic").any()
    fig, axs = plt.subplots(3 if extra else 2, 3, figsize=(16, 13 if extra else 8.6))

    ax = axs[0, 0]
    for name, layer in instances:
        t = rows_of(dist, name, layer)
        t = t[(t.quantity == "nearest-neighbour cosine") & (t.q < 1)]
        ax.plot(t.value, 1 - t.q, **instance_style(name, layer, layers))
    ax.axvline(DUP, color="black", linestyle=":", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_title("Nearest neighbour of each atom", fontsize=9)
    ax.set_xlabel("Largest cosine with another atom", fontsize=8)
    ax.set_ylabel("Share of atoms at least this close", fontsize=8)
    ax.legend(fontsize=6)

    ax = axs[0, 1]
    for name, layer in instances:
        t = rows_of(dist, name, layer)
        t = t[t.quantity == "eigenvalue"]
        ax.plot(t.q, t.value, **instance_style(name, layer, layers))
    g = geo.iloc[0]
    ax.axhspan(g["MP lower"], g["MP upper"], color=RANDOM, alpha=0.2, label="Marchenko-Pastur interval")
    ax.set_yscale("log")
    ax.set_title("Spectrum of (d / N) V$^T$V (mean 1)", fontsize=9)
    ax.set_xlabel("Eigenvalue rank (fraction of d)", fontsize=8)
    ax.legend(fontsize=6)

    ax = axs[0, 2]
    t = geo[geo.dictionary == "J-lens"].sort_values("layer")
    for thr, style in zip(THRESHOLDS, ["-", "--", ":"]):
        ax.plot(t.layer, t[f"atoms with a neighbour >= {thr}"] / t.N, color=BLUES[1], linestyle=style, marker="o",
                markersize=2, label=f"J-lens, cosine >= {thr}")
    for name, color in (("logit lens", LOGIT), ("random (isotropic)", RANDOM)):
        ax.axhline(rows_of(geo, name, None)[f"atoms with a neighbour >= {DUP}"].iloc[0] / g.N, color=color,
                   linewidth=1, label=f"{name}, cosine >= {DUP}")
    ax.set_title("Atoms with a near-duplicate", fontsize=9)
    ax.set_xlabel("Layer (block output)", fontsize=8)
    ax.set_ylabel("Share of atoms", fontsize=8)
    ax.legend(fontsize=6)

    eq = rec[rec.weights == "equal"]
    share_lines = [Line2D([], [], color=oranges[s], label=f"s = {s:g}") for s in sorted(SHARES)]
    dict_lines = [Line2D([], [], color="black", linestyle="-", label=f"J-lens, layer {mid}"),
                  Line2D([], [], color="black", linestyle="--", label="random (isotropic)")]
    for ax, metric in ((axs[1, 0], "recall"), (axs[1, 2], "occupancy")):
        for (name, layer), style in ((("J-lens", mid), "-"), (("random (isotropic)", None), "--")):
            for s in SHARES:
                t = rows_of(eq, name, layer)
                t = t[t["share s"] == s].sort_values("k")
                ax.plot(t.k, t[metric], color=oranges[s], linestyle=style, marker="o", markersize=2)
        ax.set_xscale("log")
        ax.set_xlabel("Planted atoms k", fontsize=8)
    ax = axs[1, 0]
    ax.set_ylim(0, 1.05)
    ax.set_title("Planted atoms among the first k picks (equal weights)", fontsize=9)
    ax.set_ylabel("Recall", fontsize=8)
    ax.legend(handles=share_lines + dict_lines, fontsize=6)
    ax = axs[1, 2]
    ax.plot(KS, KS, color="black", linewidth=0.8, label="occupancy = k")
    for s in SHARES:
        if s < 1:
            ax.axhline(s * g.d / ((1 - s) * 2 * math.log(g.N)), color=oranges[s], linestyle=":", linewidth=1)
    ax.set_yscale("log")
    ax.set_title("Occupancy from the stopping rule (dotted: s d / ((1 - s) 2 ln N))", fontsize=9)
    ax.set_ylabel("Reported occupancy", fontsize=8)
    ax.legend(handles=share_lines + dict_lines + [Line2D([], [], color="black", linewidth=0.8, label="occupancy = k")], fontsize=6)

    ax = axs[1, 1]
    for name, layer in instances:
        t = rows_of(eq, name, layer)
        t = t[t["share s"] == s_real].sort_values("k")
        style = instance_style(name, layer, layers)
        ax.plot(t.k, t["recall"], marker="o", markersize=2, **style)
        ax.plot(t.k, t["recall up to near-duplicates"], linestyle=":", color=style["color"])
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Recall at s = {s_real:g} (dotted: up to a near-duplicate)", fontsize=9)
    ax.set_xlabel("Planted atoms k", fontsize=8)
    ax.legend(fontsize=6)

    if extra:
        ax = axs[2, 0]
        for name, layer in instances:
            t = rows_of(eq, name, layer, "activations", layer if name == "J-lens" else mid)
            t = t[t["share s"] == s_real].sort_values("k")
            style = instance_style(name, layer, layers)
            if name != "J-lens":
                style["label"] += f" (activations of layer {mid})"
            ax.plot(t.k, t["recall"], marker="o", markersize=2, **style)
            ax.plot(t.k, t["recall within all steps"], linestyle=":", color=style["color"])
        ax.set_title(f"Planted into real activations, s = {s_real:g} (dotted: among all picks)", fontsize=9)
        ax.set_ylabel("Recall", fontsize=8)
        ax.legend(fontsize=6)

        ax = axs[2, 1]
        noise_styles = {"isotropic": "-", "activation covariance": "--", "activations": ":"}
        for name, layer, color in (("J-lens", mid, BLUES[1]), ("random (isotropic)", None, RANDOM)):
            for noise, style in noise_styles.items():
                t = rows_of(eq, name, layer, noise, None if noise == "isotropic" else mid)
                t = t[t["share s"] == s_real].sort_values("k")
                ax.plot(t.k, t["recall"], color=color, linestyle=style, marker="o", markersize=2)
        ax.set_title(f"Recall at s = {s_real:g}, by noise (activations of layer {mid})", fontsize=9)
        ax.legend(handles=[Line2D([], [], color=BLUES[1], label=f"J-lens, layer {mid}"), Line2D([], [], color=RANDOM, label="random (isotropic)")]
                  + [Line2D([], [], color="black", linestyle=st, label=f"noise: {n}") for n, st in noise_styles.items()], fontsize=6)

        ax = axs[2, 2]
        for noise, style in (("activations", "-"), ("activation covariance", "--")):
            for s in SHARES:
                if s < 1:
                    t = rows_of(eq, "J-lens", mid, noise, mid)
                    t = t[t["share s"] == s].sort_values("k")
                    ax.plot(t.k, t["occupancy"], color=oranges[s], linestyle=style, marker="o", markersize=2)
        ax.plot(KS, KS, color="black", linewidth=0.8)
        native = rows_of(rec, "J-lens", mid, "activations", mid)
        native = native[native.k == 0]
        handles = [Line2D([], [], color=oranges[s], label=f"s = {s:g}") for s in sorted(SHARES) if s < 1]
        handles += [Line2D([], [], color="black", linestyle="-", label="noise: activations"),
                    Line2D([], [], color="black", linestyle="--", label="noise: activation covariance"),
                    Line2D([], [], color="black", linewidth=0.8, label="occupancy = k")]
        if len(native):
            ax.axhline(native["occupancy"].iloc[0], color="black", linestyle=":", linewidth=1)
            handles.append(Line2D([], [], color="black", linestyle=":", label="activations alone"))
        ax.set_yscale("log")
        ax.set_title(f"Occupancy from the stopping rule, J-lens layer {mid}", fontsize=9)
        ax.set_ylabel("Reported occupancy", fontsize=8)
        ax.legend(handles=handles, fontsize=6)
        for ax in axs[2]:
            ax.set_xscale("log")
            ax.set_xlabel("Planted atoms k", fontsize=8)
        for ax in axs[2, :2]:
            ax.set_ylim(0, 1.05)
    for ax in axs.flat:
        ax.grid(linestyle="--", alpha=0.4)
    fig.suptitle(f"{args.model}: N = {int(g.N):,} atoms in d = {int(g.d)}", fontsize=10)
    plt.tight_layout()
    # No CreationDate in the PDF, so that rerunning gives byte-identical files and git diff shows real changes only
    fig.savefig(FIG_OUT, bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(FIG_OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")


if args.plot_only:
    plot(*(pd.read_csv(CSV[p]) for p in ("geometry", "distributions", "duplicates", "recovery")))
    raise SystemExit

torch.manual_seed(0)
lm = LensModel(args.model, args.device)
dev, d = lm.device, lm.d
N = lm.model.lm_head.weight.shape[0]
token = lambda i: repr(lm.tok.decode([int(i)]))
print(f"{args.model}: N = {N} atoms in d = {d}; lens at layers {min(lm.J)}-{max(lm.J)}")


def atoms_for(name, layer):
    if name == "random (isotropic)":
        return random_atoms(N, d, dev, seed=1)
    return lens_atoms(lm, layer if layer is not None else 0, use_jacobian=name == "J-lens")


@torch.no_grad()
def atom_norms(layer, use_jacobian, chunk=16384):
    """Norms of the centered lens vectors before lens_atoms normalizes them (same centering)."""
    W, g = lm.model.lm_head.weight, lm.gain
    mean_row = sum(W[s : s + chunk].to(dev).float().sum(0) for s in range(0, N, chunk)) * g / N
    out = []
    for s in range(0, N, chunk):
        w = W[s : s + chunk].to(dev).float() * g - mean_row
        if use_jacobian and layer in lm.J:
            w = w @ lm.J[layer]
        out.append(w.norm(dim=1).cpu())
    return torch.cat(out)


@torch.no_grad()
def neighbours(atoms, chunk=4096, top=8):
    """Each atom's largest cosine with another atom, signed and absolute; the number of pairs above each threshold;
    and edges to up to `top` neighbours at cosine >= DUP (enough to connect clusters of near-duplicates)."""
    best, best_abs = torch.empty(N), torch.empty(N)
    counts = dict.fromkeys(THRESHOLDS, 0)
    edges, edge_cos = [], []
    for s in range(0, N, chunk):
        c = (atoms[s : s + chunk] @ atoms.T).float()
        rows = torch.arange(len(c), device=c.device)
        c[rows, rows + s] = 0
        best_abs[s : s + len(c)] = c.abs().max(1).values.cpu()
        c[rows, rows + s] = -1
        best[s : s + len(c)] = c.max(1).values.cpu()
        for thr in THRESHOLDS:
            counts[thr] += (c >= thr).sum().item()
        vals, cols = c.topk(top, dim=1)
        mask = vals >= DUP
        edges.append(torch.stack([rows[:, None].expand_as(cols)[mask] + s, cols[mask]], 1).cpu())
        edge_cos.append(vals[mask].cpu())
    return best, best_abs, {t: n // 2 for t, n in counts.items()}, torch.cat(edges), torch.cat(edge_cos)


def components(edges):
    """Connected components of the graph on the N atoms with these edges: one label per atom (min-label
    propagation with pointer jumping)."""
    lab = torch.arange(N)
    if not len(edges):
        return lab
    a, b = edges[:, 0], edges[:, 1]
    while True:
        m = torch.minimum(lab[a], lab[b])
        new = lab.clone()
        new.scatter_reduce_(0, a, m, reduce="amin")
        new.scatter_reduce_(0, b, m, reduce="amin")
        new = new[new]
        if torch.equal(new, lab):
            return lab
        lab = new


@torch.no_grad()
def spectrum(atoms, chunk=16384):
    """Eigenvalues of (d / N) V^T V, ascending (mean 1 for unit atoms)."""
    G = sum(atoms[s : s + chunk].float().T @ atoms[s : s + chunk].float() for s in range(0, N, chunk))
    return (torch.linalg.eigvalsh(G.double()) * d / N).cpu().numpy()


# A. Geometry
if not args.skip_geometry:
    mp = ((1 - math.sqrt(d / N)) ** 2, (1 + math.sqrt(d / N)) ** 2)
    welch = math.sqrt((N - d) / (d * (N - 1)))
    geo_rows, dist_rows, dup_rows = [], [], []
    for name, layer in [("logit lens", None), ("random (isotropic)", None)] + [("J-lens", l) for l in sorted(lm.J)]:
        t0 = time.time()
        atoms = atoms_for(name, layer)
        best, best_abs, counts, edges, edge_cos = neighbours(atoms)
        lab = components(edges)
        sizes = torch.bincount(lab, minlength=N)
        clustered = sizes[lab] >= 2
        ev = spectrum(atoms)
        row = {"dictionary": name, "layer": layer, "N": N, "d": d, "coherence": best_abs.max().item(), "Welch bound": welch,
               "nearest-neighbour cosine, median": best.median().item(), "nearest-neighbour cosine, p99": best.quantile(0.99).item()}
        row |= {f"pairs >= {t}": counts[t] for t in THRESHOLDS}
        row |= {f"atoms with a neighbour >= {t}": (best >= t).sum().item() for t in THRESHOLDS}
        row |= {"clusters": (sizes >= 2).sum().item(), "atoms in clusters": clustered.sum().item(), "largest cluster": sizes.max().item(),
                "eigenvalue min": ev[0], "eigenvalue max": ev[-1], "MP lower": mp[0], "MP upper": mp[1],
                "eigenvalues above MP": int((ev > mp[1]).sum()), "eigenvalues below MP": int((ev < mp[0]).sum()),
                "effective rank / d": ev.sum() ** 2 / (ev**2).sum() / d}
        if name != "random (isotropic)":
            norms = atom_norms(layer, name == "J-lens")
            med = norms.median()
            row |= {"norm min / median": (norms.min() / med).item(), "norm p01 / median": (norms.quantile(0.01) / med).item(),
                    "atoms shorter than 0.1 median": (norms < 0.1 * med).sum().item(),
                    "shortest atoms": " | ".join(token(i) for i in norms.argsort()[:5])}
        if name == "J-lens":
            sv = torch.linalg.svdvals(lm.J[layer].double())
            row |= {"J condition number": (sv[0] / sv[-1]).item(), "J top singular share": (sv[0] ** 2 / (sv**2).sum()).item()}
        geo_rows.append(row)
        for quantity, values in (("nearest-neighbour cosine", best.numpy()), ("eigenvalue", ev)):
            for q, v in zip(QS, np.quantile(values, QS)):
                dist_rows.append({"dictionary": name, "layer": layer, "quantity": quantity, "q": q, "value": v})
        # The largest clusters (members in token order) and the closest pairs
        for root in sizes.argsort(descending=True)[:25]:
            if sizes[root] < 2:
                break
            members = (lab == root).nonzero().squeeze(1)
            dup_rows.append({"dictionary": name, "layer": layer, "kind": "cluster", "size": sizes[root].item(),
                             "cosine": best[members].min().item(), "tokens": " | ".join(token(i) for i in members[:12])})
        if len(edges):
            a, b = edges.min(1).values, edges.max(1).values
            key = torch.unique(a * N + b, return_inverse=True)[1]
            first = torch.full((key.max().item() + 1,), len(key)).scatter_reduce(0, key, torch.arange(len(key)), reduce="amin")
            for e in first[edge_cos[first].argsort(descending=True)[:25]]:
                dup_rows.append({"dictionary": name, "layer": layer, "kind": "pair", "size": 2, "cosine": edge_cos[e].item(),
                                 "tokens": f"{token(a[e])} | {token(b[e])}"})
        del atoms
        print(f"geometry: {name}{'' if layer is None else f', layer {layer}'}: coherence {row['coherence']:.3f}, "
              f"atoms with a neighbour >= {DUP}: {row[f'atoms with a neighbour >= {DUP}']}, "
              f"eigenvalues {ev[0]:.3f}-{ev[-1]:.2f} ({time.time() - t0:.0f} s)", flush=True)
    geo = pd.DataFrame(geo_rows)
    geo.to_csv(CSV["geometry"], index=False, float_format="%.6g")
    pd.DataFrame(dist_rows).to_csv(CSV["distributions"], index=False, float_format="%.6g")
    pd.DataFrame(dup_rows).to_csv(CSV["duplicates"], index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    cols = ["dictionary", "layer", "coherence", "nearest-neighbour cosine, median", f"atoms with a neighbour >= {DUP}",
            "largest cluster", "eigenvalue max", "effective rank / d", "atoms shorter than 0.1 median", "J condition number"]
    print("\n" + geo[cols].round(3).to_string(index=False))


# B. Plant and recover
@torch.no_grad()
def activations(layers):
    """Centered block outputs at positions >= SKIP of the wikitext paragraphs, {layer: (tokens, d)} on the device."""
    with open(WIKITEXT) as f:
        paragraphs = json.load(f)["paragraphs"]
    ids = torch.stack([t for t in (lm.encode(p)[:128] for p in paragraphs) if len(t) == 128][: args.n_text])
    acts = {l: [] for l in layers}
    for s in range(0, len(ids), 8):
        hs, _ = lm.run(ids[s : s + 8])
        for l in layers:
            acts[l].append(hs[l][:, SKIP:].reshape(-1, d))
        del hs
    acts = {l: torch.cat(a) for l, a in acts.items()}
    return {l: a - a.mean(0) for l, a in acts.items()}


def sqrt_of(cov):
    """S with z @ S ~ N(0, cov) for z ~ N(0, I), as in experiment 03."""
    evals, evecs = torch.linalg.eigh(cov.double())
    return (evecs * evals.clamp(min=0).sqrt()).T.float()


@torch.no_grad()
def plant(atoms, k, share, profile, seed, noise="isotropic", act_layer=None):
    """y (trials, d) and the planted indices (trials, k). The cell's seed fixes the indices, weights and noise, so
    every dictionary and every noise sees the same planted atoms (in index space)."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.rand(args.trials, N, generator=gen).topk(k, dim=1).indices
    w = torch.ones(args.trials, k) if profile == "equal" else torch.randn(args.trials, k, generator=gen).exp()
    if noise == "activations":
        noise = acts[act_layer][torch.randperm(len(acts[act_layer]), generator=gen)[: args.trials].to(dev)].clone()
    else:
        noise = torch.randn(args.trials, d, generator=gen).to(dev)
        if act_layer is not None:
            noise = noise @ cov_sqrt[act_layer]
    idx, w = idx.to(dev), w.to(dev)
    signal = (w[..., None] * atoms[idx].float()).sum(1)
    if share < 1:
        noise *= (signal.norm(dim=1) * math.sqrt((1 - share) / share) / noise.norm(dim=1))[:, None]
        return signal + noise, idx
    return signal, idx


def first_below(gain, gain_ctrl, steps):
    """First step (1-based) at which the dictionary's gain falls below the control's, steps + 1 if never."""
    below = gain < gain_ctrl
    return torch.where(below.any(-1), below.float().argmax(-1) + 1, torch.full(below.shape[:-1], steps + 1, device=below.device))


def gains(y, res):
    """Gain in explained |y|^2 at each step (trials, steps), from the squared residual norms after each step."""
    total = (y**2).sum(1, keepdim=True)
    return torch.cat([total, res[:, :-1]], 1) - res


if not args.skip_recovery:
    L = lm.n_blocks
    layers = [int(l) for l in args.layers.split(",")] if args.layers else sorted({L // 4, L // 2, 5 * L // 8, 3 * L // 4} & set(lm.J))
    if set(RUN_NOISES) - {"isotropic"}:
        acts = activations(layers)
        cov_sqrt = {l: sqrt_of(a.T @ a / len(a)) for l, a in acts.items()}
        print(f"activations: {len(acts[layers[0]])} wikitext tokens per layer", flush=True)
    control = random_atoms(N, d, dev, seed=CONTROL_SEED)
    rec_rows = []
    for noise in RUN_NOISES:
        if noise == "isotropic":
            instances = [("J-lens", l, None) for l in layers] + [("logit lens", None, None), ("random (isotropic)", None, None)]
        else:
            instances = [(name, l if name == "J-lens" else None, l) for l in layers for name in ("J-lens", "logit lens", "random (isotropic)")]
        for name, layer, act_layer in instances:
            t0 = time.time()
            atoms = atoms_for(name, layer)
            base = {"noise": noise, "dictionary": name, "layer": layer, "activation layer": act_layer}
            if noise == "activations":
                # The rule on the activations alone: their own J-space words, before anything is planted
                gen = torch.Generator().manual_seed(999)
                x = acts[act_layer][torch.randperm(len(acts[act_layer]), generator=gen)[: args.trials].to(dev)]
                _, _, res = decompose(x, atoms, NATIVE_STEPS)
                _, _, res_ctrl = decompose(x, control, NATIVE_STEPS)
                gain, gain_ctrl = gains(x, res), gains(x, res_ctrl)
                occ = first_below(gain.mean(0), gain_ctrl.mean(0), NATIVE_STEPS).item()
                rec_rows.append(base | {"weights": "none", "share s": 0.0, "k": 0, "steps": NATIVE_STEPS,
                                        "explained at k": (1 - res[:, 24] / (x**2).sum(1)).mean().item(),  # after 25 steps
                                        "occupancy": occ, "occupancy censored": occ > NATIVE_STEPS,
                                        "occupancy (median over trials)": first_below(gain, gain_ctrl, NATIVE_STEPS).float().median().item()})
            for pi, profile in enumerate(PROFILES):
                for si, share in enumerate(SHARES):
                    if share == 1 and noise != "isotropic":
                        continue  # nothing is added at s = 1: the isotropic rows
                    for ki, k in enumerate(KS):
                        y, idx = plant(atoms, k, share, profile, 1_000_000 * pi + 1000 * si + ki, noise, act_layer)
                        steps = k + max(10, k // 4)
                        picked, _, res = decompose(y, atoms, steps)
                        _, _, res_ctrl = decompose(y, control, steps)
                        gain, gain_ctrl = gains(y, res), gains(y, res_ctrl)
                        sel = picked[:, :k]
                        hit = (sel[:, :, None] == idx[:, None, :]).any(1)  # (trials, k): planted atom among the first k picks
                        near = (atoms[sel].float() @ atoms[idx].float().transpose(1, 2)).max(1).values >= DUP
                        occ = first_below(gain.mean(0), gain_ctrl.mean(0), steps).item()
                        rec_rows.append(base | {"weights": profile, "share s": share, "k": k, "steps": steps,
                                                "recall": hit.float().mean().item(), "recall up to near-duplicates": near.float().mean().item(),
                                                "all k right": hit.all(1).float().mean().item(),
                                                "explained at k": (1 - res[:, k - 1] / (y**2).sum(1)).mean().item(),
                                                "occupancy": occ, "occupancy censored": occ > steps,
                                                "occupancy (median over trials)": first_below(gain, gain_ctrl, steps).float().median().item(),
                                                "predicted occupancy": share * d / ((1 - share) * 2 * math.log(N)) if share < 1 else float("nan"),
                                                "recall within all steps": (picked[:, :, None] == idx[:, None, :]).any(1).float().mean().item()})
                print(f"recovery, {noise} noise: {name}{'' if layer is None else f', layer {layer}'}"
                      f"{'' if act_layer is None else f' (activations of layer {act_layer})'}, {profile} weights done ({time.time() - t0:.0f} s)", flush=True)
            del atoms
    rec = pd.DataFrame(rec_rows)
    if os.path.exists(CSV["recovery"]):
        # Keep the rows of the other noises (results written before the noise column existed are isotropic)
        old = pd.read_csv(CSV["recovery"])
        if "noise" not in old:
            old.insert(0, "noise", "isotropic")
            old.insert(3, "activation layer", float("nan"))
        rec = pd.concat([old[~old.noise.isin(RUN_NOISES)], rec], ignore_index=True)
    rec = rec.iloc[rec.noise.map(NOISES.index).argsort(kind="stable")].reset_index(drop=True)
    rec.to_csv(CSV["recovery"], index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    label = lambda t: t.dictionary + t.layer.map(lambda l: "" if pd.isna(l) else f" {int(l)}") \
        + t["activation layer"].map(lambda l: "" if pd.isna(l) else f" (acts {int(l)})")
    for noise in RUN_NOISES:
        eq = rec[(rec.noise == noise) & (rec.weights == "equal")].assign(dictionary=label)
        for metric in ("recall", "recall within all steps", "occupancy"):
            print(f"\n{metric} ({noise} noise, equal weights), by dictionary and s (rows: k)")
            print(eq.pivot_table(index="k", columns=["dictionary", "share s"], values=metric).round(2).to_string())
        native = rec[(rec.noise == noise) & (rec.k == 0)].assign(dictionary=label)
        if len(native):
            print(f"\noccupancy of the activations alone (up to {NATIVE_STEPS} steps; explained after 25 steps)")
            print(native[["dictionary", "occupancy", "occupancy censored", "occupancy (median over trials)", "explained at k"]].round(3).to_string(index=False))

if all(os.path.exists(p) for p in CSV.values()):
    plot(*(pd.read_csv(CSV[p]) for p in ("geometry", "distributions", "duplicates", "recovery")))
    print(f"\nSaved: {', '.join(CSV.values())}\n       {FIG_OUT}")
