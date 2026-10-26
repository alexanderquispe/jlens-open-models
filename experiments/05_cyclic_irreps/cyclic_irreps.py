"""Which irreducible representations of Z/n carry the answer? Days (Z/7) and months (Z/12).

Engels et al. (2025) patch the 2-dim circle, the k = 1 Fourier plane of the day/month centroids. The n
centroids span (at most) the regular representation of Z/n minus its trivial part, which splits into real
irreps: one plane per frequency k < n/2 and, for even n, a line at k = n/2. Here every irrep is patched on
its own, to see which frequencies the model actually uses.

Prompts (Engels et al.):
  days    "Let's do some days of the week math. Two days from Monday is"   7 days x durations 1-7
  months  "Let's do some calendar math. Two months from January is"       12 months x durations 1-12
Fit on odd durations, test on even ones. Positions: `start` (the named day/month, Engels' a_token) and `last`.

  A. Irrep spectrum: energy share of each frequency k in the centroids, per layer.
  B. Regularity of the k = 1 polygon: an equivariant (rotation-like) code puts the n centroids at equal angles
     2*pi/n in cyclic order and at equal radii. Reported: cyclic order, spread of the angle gaps and radii.
  C. Rotation by irrep: add P_K (c_{s+r} - c_s) after block l at that position (c_j: training centroid of j,
     P_K: projection on the frequencies K) and count how often the answer moves from a to a + r, averaged
     over r = 1..n-1 (chance 1/n). K = {k} for every k, K = {1, 2}, and all k (the full centroid difference);
     controls: the span of the n lens directions of the names, and a random direction with the k = 1 norm.

    python experiments/05_cyclic_irreps/cyclic_irreps.py --model llama3.1-8b --concept months --position start
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from jlens_open import MODELS, LensModel, irrep_basis, pair_energies

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: the first available)")
parser.add_argument("--layers", default="::1", help="layers to patch, as start:stop:step (default: all)")
parser.add_argument("--concept", choices=["days", "months"], default="months")
parser.add_argument("--position", choices=["start", "last"], default="start")
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = f"{args.model}_{args.concept}_{args.position}"
CSV_LAYERS = os.path.join(HERE, f"results/{TAG}_layers.csv")
CSV_ROTATION = os.path.join(HERE, f"results/{TAG}_rotation.csv")
FIG_OUT = os.path.join(HERE, f"figures/{TAG}.pdf")
os.makedirs(os.path.dirname(CSV_LAYERS), exist_ok=True)
os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)

torch.manual_seed(0)
rng = np.random.default_rng(0)
NUMBERS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve"]
if args.concept == "days":
    NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    # As in intervention/days_of_week_task.py of github.com/JoshEngels/MultiDimensionalFeatures
    INTERVALS = ["Zero days", "One day", "Two days", "Three days", "Four days", "Five days", "Six days", "Seven days"]
    template = "Let's do some days of the week math. {} from {} is"
else:
    NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    # Engels et al. write "One months" too (intervention/months_of_year_task.py)
    INTERVALS = [f"{w} months" for w in NUMBERS]
    template = "Let's do some calendar math. {} from {} is"
N = len(NAMES)
FREQS = list(range(1, N // 2 + 1))

lm = LensModel(args.model, args.device)
name_ids = []
for name in NAMES:
    t = lm.tok.encode(" " + name, add_special_tokens=False)
    assert len(t) == 1, f"' {name}' is {len(t)} tokens"
    name_ids.append(t[0])
L = lm.n_blocks
LAYERS = list(range(L))[slice(*[int(x) if x else None for x in args.layers.split(":")])]

rows, prompts = [], []
for start in range(N):
    for duration in range(1, N + 1):
        prompts.append(lm.encode(template.format(INTERVALS[duration], NAMES[start])))
        rows.append({"train": duration % 2 == 1, "start": start, "answer": (start + duration) % N})
meta = pd.DataFrame(rows)
answers = meta.answer.to_numpy()
train = meta.train.to_numpy()
test = ~train
for i, p in enumerate(prompts):
    assert p[-2].item() == name_ids[meta.start[i]], lm.tok.decode(p)
positions = np.array([len(p) - (2 if args.position == "start" else 1) for p in prompts])
labels = meta.start.to_numpy() if args.position == "start" else answers
groups = [g.to_list() for _, g in pd.Series(range(len(prompts))).groupby([len(p) for p in prompts])]

acts = np.zeros((len(prompts), L, lm.d), dtype=np.float32)
model_logits = np.zeros((len(prompts), N), dtype=np.float32)
for idx in groups:
    hs, logits = lm.run(torch.stack([prompts[i] for i in idx]))
    for layer, h in enumerate(hs):
        acts[idx, layer] = h[torch.arange(len(idx)), positions[idx]].cpu().numpy()
    model_logits[idx] = logits[:, name_ids].cpu().numpy()
correct = model_logits.argmax(1) == answers
print(f"{TAG}: {len(prompts)} prompts; the model picks the right {args.concept[:-1]} among the {N} in {correct.mean():.1%} "
      f"({correct[test].mean():.1%} of the held-out ones)")


def polygon_stats(centroids):
    """Cyclic order, angle-gap spread and radius spread of the centroids in their k = 1 plane."""
    z = (centroids - centroids.mean(0)) @ irrep_basis(centroids, [1])
    angle = np.arctan2(z[:, 1], z[:, 0])
    gaps = np.angle(np.exp(1j * (np.roll(angle, -1) - angle)))  # step from j to j + 1
    gaps *= np.sign(gaps.sum()) or 1.0  # orientation-free
    radius = np.linalg.norm(z, axis=1)
    return {
        "k1 cyclic order": float(np.all(gaps > 0) and np.isclose(gaps.sum(), 2 * np.pi, atol=1e-3)),
        "k1 angle gap std / (2pi/n)": gaps.std() / (2 * np.pi / N),
        "k1 radius CV": radius.std() / radius.mean(),
    }


layer_rows = []
for layer in range(L):
    c = np.stack([acts[labels == j, layer].mean(0) for j in range(N)])
    e = pair_energies(c)
    row = {"layer": layer, **{f"energy k={k}": e[k - 1] / e.sum() for k in FREQS}, **polygon_stats(c)}
    layer_rows.append(row)
layers_df = pd.DataFrame(layer_rows)
layers_df.to_csv(CSV_LAYERS, index=False)
pd.set_option("display.width", 250)
print(f"(one-hot: every k has the same share per real dimension; k=1 share {pair_energies(np.eye(N))[0] / pair_energies(np.eye(N)).sum():.2f})")
print(layers_df.round(3).to_string(index=False))


def patched_logits(layer, deltas):
    """Model logits over the n names on the test prompts with deltas added after block `layer`."""
    out = np.zeros((len(prompts), N), dtype=np.float32)
    for idx in groups:
        idx = [i for i in idx if test[i]]
        if idx:
            d = torch.from_numpy(deltas[idx]).to(lm.device)
            _, logits = lm.run(torch.stack([prompts[i] for i in idx]), deltas={layer: d}, positions=positions[idx], store=False)
            out[idx] = logits[:, name_ids].cpu().numpy()
    return out[test]


patch_sets = {f"k={k}": [k] for k in FREQS}
patch_sets["k=1,2"] = [1, 2]
original = model_logits.argmax(1)[test]
rot_rows = []
for layer in LAYERS:
    x = acts[:, layer]
    centroids = np.stack([x[train & (labels == j)].mean(0) for j in range(N)])
    bases = {name: irrep_basis(centroids, ks) for name, ks in patch_sets.items()}
    V = lm.lens_directions(name_ids, layer)
    lens_basis = np.linalg.qr((V - V.mean(0)).T)[0]
    for shift in range(1, N):
        target = (answers + shift) % N
        diff = centroids[(labels + shift) % N] - centroids[labels]
        patches = {name: diff @ B @ B.T for name, B in bases.items()}
        random_dir = rng.standard_normal(diff.shape).astype(np.float32)
        random_dir *= np.linalg.norm(patches["k=1"], axis=1, keepdims=True) / np.linalg.norm(random_dir, axis=1, keepdims=True)
        patches["all k (full centroid difference)"] = diff
        patches["name lens span"] = diff @ lens_basis @ lens_basis.T
        patches["random direction, k=1 norm"] = random_dir
        for name, delta in patches.items():
            pred = patched_logits(layer, delta.astype(np.float32)).argmax(1)
            moved = pred == target[test]
            rot_rows.append({
                "layer": layer, "shift": shift, "patch": name,
                "moved_to_target": moved.mean(), "moved_if_correct_unpatched": moved[correct[test]].mean(),
                "changed_answer": (pred != original).mean(),
                "patch_norm": np.linalg.norm(delta[test], axis=1).mean(),
            })
    print(f"rotation: layer {layer} done", flush=True)
rot_df = pd.DataFrame(rot_rows)
rot_df.to_csv(CSV_ROTATION, index=False)
rot_table = rot_df.pivot_table(index="layer", columns="patch", values="moved_if_correct_unpatched", aggfunc="mean")
print(f"\nRotation: fraction of correctly solved held-out prompts whose answer moves from a to a + r (chance 1/{N})")
print(rot_table.round(3).to_string())
print("\nFraction whose answer changes at all")
print(rot_df.pivot_table(index="layer", columns="patch", values="changed_answer", aggfunc="mean").round(3).to_string())

fig, axs = plt.subplots(2, 2, figsize=(9, 6.5))
ax = axs[0, 0]
ax.stackplot(layers_df.layer, *[layers_df[f"energy k={k}"] for k in FREQS], labels=[f"k={k}" for k in FREQS], alpha=0.85)
ax.set_title(f"Irrep spectrum of the {args.concept} centroids", fontsize=9)
ax.set_ylim(0, 1)
ax.legend(fontsize=6, loc="lower right")
ax = axs[0, 1]
for name in [f"k={k}" for k in FREQS] + ["k=1,2", "all k (full centroid difference)", "name lens span", "random direction, k=1 norm"]:
    style = "-." if name == "k=1,2" else "-" if name.startswith("k=") else "--"
    ax.plot(rot_table.index, rot_table[name], style, marker="o", markersize=2, label=name)
ax.axhline(1 / N, color="grey", linestyle=":", linewidth=0.8, label="chance")
ax.set_title("Rotation by irrep (correct unpatched prompts)", fontsize=9)
ax.legend(fontsize=5.5)
ax = axs[1, 0]
mid = int(rot_table["k=1"].idxmax())
c = np.stack([acts[labels == j, mid].mean(0) for j in range(N)])
Q = irrep_basis(c, [1])
z = (acts[:, mid] - c.mean(0)) @ Q
zc = (c - c.mean(0)) @ Q
colors = plt.cm.twilight(np.linspace(0, 1, N + 1))[:N]
for j in range(N):
    ax.scatter(z[labels == j, 0], z[labels == j, 1], s=8, color=colors[j])
    ax.annotate(NAMES[j][:3], zc[j], fontsize=7, ha="center", va="center", weight="bold")
ax.set_title(f"Layer {mid} (best k=1 rotation): k = 1 plane", fontsize=9)
ax = axs[1, 1]
heat = rot_table[[f"k={k}" for k in FREQS]].T.to_numpy()
im = ax.imshow(heat, aspect="auto", cmap="viridis", vmin=0, vmax=max(heat.max(), 1e-3))
ax.set_yticks(range(len(FREQS)), [f"k={k}" for k in FREQS])
ax.set_xticks(range(len(rot_table.index))[::4], rot_table.index[::4])
ax.set_title("Moved to target, by frequency", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.04)
for ax in (axs[0, 0], axs[0, 1], axs[1, 1]):
    ax.set_xlabel("Layer (block output)", fontsize=8)
for ax in (axs[0, 0], axs[0, 1]):
    ax.grid(linestyle="--", alpha=0.4)
fig.suptitle(TAG, fontsize=10)
plt.tight_layout()
# No CreationDate in the PDF, so that rerunning gives byte-identical files and git diff shows real changes only
fig.savefig(FIG_OUT, bbox_inches="tight", metadata={"CreationDate": None})
fig.savefig(FIG_OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: {CSV_LAYERS}\n       {CSV_ROTATION}\n       {FIG_OUT}")
