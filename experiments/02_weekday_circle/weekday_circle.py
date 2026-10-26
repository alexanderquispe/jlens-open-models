"""What does the Jacobian lens see of the weekday circle?

Two tasks:
  sequence  "<prefix> Tuesday, Wednesday," -> " Thursday" (8 prefixes of >= 16 tokens, 1-4 listed
            days, 7 starting days = 224 prompts). Fit on prefixes 0, 2, 4, 6; test on 1, 3, 5, 7.
  engels    the prompt of Engels et al. (ICLR 2025, "Not All Language Model Features Are One-Dimensionally
            Linear"): "Let's do some days of the week math. Two days from Monday is"
            (7 days x durations 1-7 = 49 prompts). Fit on odd durations, test on 2, 4, 6.
and two positions:
  last      the last token (the comma / " is"); the circle is grouped by the answer day.
  day       the last day named in the prompt (" Wednesday" / " Monday", Engels' a_token); the circle
            is grouped by that day.

  B. Readout at that position: does the answer day come out of the J-lens / logit lens, and from
     which layer? Fourier (irreps of Z/7) spectrum of the day centroids,
     and a circle decoder (angle in the k = 1 plane, nearest training centroid).
  C. Lens directions of the 7 day tokens, v_k = J_l^T (g * w_k): their own Fourier spectrum, and
     how much of the activations' k = 1 circle plane lies in their 6-dim span.
  D. Rotation: add P(c_{d+r} - c_d) to the residual at that position after block l, where c_j is
     the centroid of day j and P projects onto the k = 1 circle plane, and check whether the
     model's answer moves from a to a + r. Controls: the same displacement in a random direction,
     the projection on the span of the day lens directions, and the full centroid difference.

The lens was fit without the first 16 positions; Engels' day token is at position 14, so lens
readouts there are outside the fit's range (the rotation does not use the lens).

    python experiments/02_weekday_circle/weekday_circle.py --model llama3.1-8b --task engels --position day
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from jlens_open import MODELS, LensModel, circle_plane, pair_energies

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: the first available)")
parser.add_argument("--task", choices=["sequence", "engels"], default="sequence")
parser.add_argument("--position", choices=["last", "day"], default="last")
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = f"{args.model}_{args.task}_{args.position}"
CSV_LAYERS = os.path.join(HERE, f"results/{TAG}_layers.csv")
CSV_ROTATION = os.path.join(HERE, f"results/{TAG}_rotation.csv")
FIG_OUT = os.path.join(HERE, f"figures/{TAG}.pdf")

torch.manual_seed(0)
rng = np.random.default_rng(0)
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
N = 7
# Prefixes are >= 16 tokens: the lens was fit without the first 16 positions (attention sinks)
PREFIXES = [
    "The following is a list of the days of the week, in the order they appear on our office calendar:",
    "Here is the weekly rota for the volunteers at the community garden, with one name for each day:",
    "Our small bakery posts its opening schedule on the door every single week, and it reads as follows:",
    "My grandmother taught me to recite the days of the week when I was a little child, like this:",
    "The museum is open to visitors on the following days, as printed on the back of every ticket:",
    "In the spreadsheet that tracks our team's weekly meetings, the first column lists the days:",
    "When the teacher asked the class to name the days of the week in order, the students answered:",
    "The bus timetable for the northern route lists departures on these days of the week only:",
]
# As in intervention/days_of_week_task.py of github.com/JoshEngels/MultiDimensionalFeatures
DAY_INTERVALS = ["Zero days", "One day", "Two days", "Three days", "Four days", "Five days", "Six days", "Seven days"]

lm = LensModel(args.model, args.device)
day_ids = []
for d in DAYS:
    t = lm.tok.encode(" " + d, add_special_tokens=False)
    assert len(t) == 1, f"' {d}' is {len(t)} tokens"
    day_ids.append(t[0])
L = lm.n_blocks

# Each prompt ends "<day><one token>" ("," or " is"), so the named day is the second-to-last token
rows, prompts = [], []
if args.task == "sequence":
    for p, prefix in enumerate(PREFIXES):
        for length in (1, 2, 3, 4):
            for start in range(N):
                seq = ", ".join(DAYS[(start + i) % N] for i in range(length)) + ","
                prompts.append(lm.encode(prefix + " " + seq))
                last_day = (start + length - 1) % N
                rows.append({"train": p % 2 == 0, "day": last_day, "answer": (last_day + 1) % N})
else:
    for start in range(N):
        for duration in range(1, 8):
            prompts.append(lm.encode(f"Let's do some days of the week math. {DAY_INTERVALS[duration]} from {DAYS[start]} is"))
            rows.append({"train": duration % 2 == 1, "day": start, "answer": (start + duration) % N})
meta = pd.DataFrame(rows)
answers = meta.answer.to_numpy()
train = meta.train.to_numpy()
test = ~train
for i, p in enumerate(prompts):
    assert p[-2].item() == day_ids[meta.day[i]], lm.tok.decode(p)
positions = np.array([len(p) - (2 if args.position == "day" else 1) for p in prompts])
labels = meta.day.to_numpy() if args.position == "day" else answers
groups = [g.to_list() for _, g in pd.Series(range(len(prompts))).groupby([len(p) for p in prompts])]

acts = np.zeros((len(prompts), L, lm.d), dtype=np.float32)
model_day_logits = np.zeros((len(prompts), N), dtype=np.float32)
for idx in groups:
    hs, logits = lm.run(torch.stack([prompts[i] for i in idx]))
    for layer, h in enumerate(hs):
        acts[idx, layer] = h[torch.arange(len(idx)), positions[idx]].cpu().numpy()
    model_day_logits[idx] = logits[:, day_ids].cpu().numpy()
model_acc = (model_day_logits.argmax(1) == answers).mean()
print(f"{TAG}: {len(prompts)} prompts; the model picks the right day among the 7 in {model_acc:.1%}")


def rank_of_answer(full):
    correct = full[np.arange(len(answers)), np.array(day_ids)[answers]]
    return (full > correct[:, None]).sum(1) + 1


def day_basis(layer):
    """Orthonormal basis (d, 6) of the centered day lens directions at `layer`."""
    V = lm.lens_directions(day_ids, layer)
    return np.linalg.qr((V - V.mean(0)).T)[0]


onehot_k1 = pair_energies(np.eye(N))[0] / pair_energies(np.eye(N)).sum()
layer_rows = []
for layer in range(L):
    x = acts[:, layer]
    centroids_train = np.stack([x[train & (labels == j)].mean(0) for j in range(N)])
    e = pair_energies(np.stack([x[labels == j].mean(0) for j in range(N)]))
    row = {"layer": layer, "k1_share_acts": e[0] / e.sum()}
    h = torch.from_numpy(x).to(lm.device)
    for use_jacobian, lens in ((True, "J-lens"), (False, "logit lens")):
        full = lm.unembed(lm.transport(h, layer, use_jacobian)).cpu().numpy()
        row[f"{lens}: top1 among days"] = (full[:, day_ids].argmax(1) == answers).mean()
        row[f"{lens}: median rank"] = np.median(rank_of_answer(full))
    Q = circle_plane(centroids_train)
    z_test = (x[test] - centroids_train.mean(0)) @ Q
    z_cent = (centroids_train - centroids_train.mean(0)) @ Q
    angle_err = np.angle(np.exp(1j * (np.arctan2(z_test[:, 1:], z_test[:, :1]) - np.arctan2(z_cent[:, 1], z_cent[:, 0]))))
    row["circle decoder acc (held out)"] = (np.abs(angle_err).argmin(1) == labels[test]).mean()
    ev = pair_energies(lm.lens_directions(day_ids, layer))
    row["k1_share_lens_dirs"] = ev[0] / ev.sum()
    row["circle plane in lens span"] = (np.linalg.norm(day_basis(layer).T @ Q, axis=0) ** 2).mean()
    row["random plane in lens span"] = (N - 1) / lm.d
    layer_rows.append(row)
layers_df = pd.DataFrame(layer_rows)
layers_df.to_csv(CSV_LAYERS, index=False)
pd.set_option("display.width", 250)
print(f"(k=1 share: one-hot = {onehot_k1:.2f}, pure circle = 1.00; the lens columns score the answer day)")
print(layers_df.round(3).to_string(index=False))


def patched_day_logits(layer, deltas):
    """Model day logits on the test prompts when deltas[i] is added at positions[i] after block `layer`."""
    out = np.zeros((len(prompts), N), dtype=np.float32)
    for idx in groups:
        idx = [i for i in idx if test[i]]
        if idx:
            d = torch.from_numpy(deltas[idx]).to(lm.device)
            _, logits = lm.run(torch.stack([prompts[i] for i in idx]), deltas={layer: d}, positions=positions[idx])
            out[idx] = logits[:, day_ids].cpu().numpy()
    return out[test]


correct_test = (model_day_logits.argmax(1) == answers)[test]
rot_rows = []
for layer in range(L):
    x = acts[:, layer]
    centroids = np.stack([x[train & (labels == j)].mean(0) for j in range(N)])
    Q = circle_plane(centroids)
    B = day_basis(layer)
    for shift in range(1, N):
        target = (answers + shift) % N
        diff = centroids[(labels + shift) % N] - centroids[labels]
        circle = diff @ Q @ Q.T
        random_dir = rng.standard_normal(diff.shape).astype(np.float32)
        random_dir *= np.linalg.norm(circle, axis=1, keepdims=True) / np.linalg.norm(random_dir, axis=1, keepdims=True)
        for name, delta in (
            ("circle plane (2-dim)", circle),
            ("random direction, same norm", random_dir),
            ("day lens span (6-dim)", diff @ B @ B.T),
            ("full centroid difference", diff),
        ):
            moved = patched_day_logits(layer, delta.astype(np.float32)).argmax(1) == target[test]
            rot_rows.append(
                {"layer": layer, "shift": shift, "patch": name, "moved_to_target": moved.mean(), "moved_if_correct_unpatched": moved[correct_test].mean()}
            )
    print(f"rotation: layer {layer} done", flush=True)
rot_df = pd.DataFrame(rot_rows)
rot_df.to_csv(CSV_ROTATION, index=False)
rot_table = rot_df.pivot_table(index="layer", columns="patch", values="moved_to_target", aggfunc="mean")
print(f"\nUnpatched, the model answers {correct_test.mean():.1%} of the held-out prompts correctly")
print("Rotation: fraction of held-out prompts whose answer moves from a to a + r, mean over r = 1..6 (chance = 1/7 = 0.14)")
print(rot_table.round(3).to_string())
print("\nThe same, only on the prompts the model gets right unpatched")
print(rot_df.pivot_table(index="layer", columns="patch", values="moved_if_correct_unpatched", aggfunc="mean").round(3).to_string())

fig, axs = plt.subplots(2, 2, figsize=(8, 6))
ax = axs[0, 0]
for lens, style in (("J-lens", "-"), ("logit lens", "--")):
    ax.plot(layers_df.layer, layers_df[f"{lens}: top1 among days"], style, marker="o", markersize=3, label=f"{lens} (answer day)")
ax.plot(layers_df.layer, layers_df["circle decoder acc (held out)"], ":", marker="s", markersize=3, label="circle decoder (held out)")
ax.axhline(1 / N, color="grey", linestyle="--", linewidth=0.8, label="chance")
ax.set_title(f"Read out at the {args.position} token", fontsize=9)
ax.legend(fontsize=6)
ax = axs[0, 1]
ax.plot(layers_df.layer, layers_df.k1_share_acts, marker="o", markersize=3, label="day centroids (activations)")
ax.plot(layers_df.layer, layers_df.k1_share_lens_dirs, marker="s", markersize=3, label="day lens directions")
ax.plot(layers_df.layer, layers_df["circle plane in lens span"], marker="^", markersize=3, label="circle plane inside lens span")
ax.axhline(onehot_k1, color="grey", linestyle="--", linewidth=0.8, label="one-hot (regular rep.)")
ax.set_title("k = 1 (circle) share", fontsize=9)
ax.legend(fontsize=6)
ax = axs[1, 0]
mid = int(layers_df.k1_share_acts[L // 4 : 3 * L // 4].idxmax())
x = acts[:, mid]
centroids = np.stack([x[labels == j].mean(0) for j in range(N)])
Q = circle_plane(centroids)
z = (x - centroids.mean(0)) @ Q
colors = plt.cm.twilight(np.linspace(0, 1, N + 1))[:N]
for j in range(N):
    ax.scatter(z[labels == j, 0], z[labels == j, 1], s=8, color=colors[j], label=DAYS[j][:3])
v = (lm.lens_directions(day_ids, mid) - lm.lens_directions(day_ids, mid).mean(0)) @ Q
v *= np.abs(z).max() / np.abs(v).max()
for j in range(N):
    ax.annotate("", xy=v[j], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=colors[j], lw=1.2))
ax.set_title(f"Layer {mid}: activations in the k = 1 plane\n(arrows: day lens directions projected, rescaled to fit)", fontsize=9)
ax.legend(fontsize=6, ncol=2)
ax = axs[1, 1]
for name in rot_table.columns:
    ax.plot(rot_table.index, rot_table[name], marker="o", markersize=3, label=name)
ax.axhline(1 / N, color="grey", linestyle="--", linewidth=0.8, label="chance")
ax.axhline(correct_test.mean(), color="black", linestyle=":", linewidth=0.8, label="unpatched accuracy")
ax.set_title("Rotation: answer moves from a to a + r", fontsize=9)
ax.legend(fontsize=6)
for ax in (axs[0, 0], axs[0, 1], axs[1, 1]):
    ax.set_xlabel("Layer (block output)", fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)
fig.suptitle(TAG, fontsize=10)
plt.tight_layout()
# No CreationDate in the PDF, so that rerunning gives byte-identical files and git diff shows real changes only
fig.savefig(FIG_OUT, bbox_inches="tight", metadata={"CreationDate": None})
fig.savefig(FIG_OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: {CSV_LAYERS}\n       {CSV_ROTATION}\n       {FIG_OUT}")
