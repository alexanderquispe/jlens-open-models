"""Is the intermediate that the J-lens reads causally used? Swapping the country in two-hop prompts.

"The official language of the country whose capital is Lima is" needs the intermediate "Peru". Gurnee et al.
(2026) swap the J-lens coordinate of such an intermediate for another entity's and the answer follows in
54-70% of trials (Claude models). Here the same test runs on an open model, with controls: replace country A
by country B (with a different language) after block l, at the last token or at the capital's last token,
and count how often the answer becomes B's language (among the languages; on prompts answered correctly).

Displacements added to the residual stream:
  J-lens swap        s (v_B - v_A), unit J-lens vectors of " B" and " A" (centered over the vocabulary)
  logit-lens swap    the same with logit-lens vectors (J = I)
  random swap        the same with two random unit vectors
     with s = alpha * |h - mean| for alpha in 0.5, 1, 2 (|h - mean|: mean norm of the centered activations there)
  coordinate swap    c_A (v_B - v_A), c_A the coefficient of v_A in the k = 16 J-space decomposition of h - mean
                     (zero if v_A is not among the 16 atoms), and 4 c_A (the paper's scaled-up version)
  centroid swap      m_B - m_A, m_X the mean activation over the other phrasings of country X's prompt;
                     also its projection on span(v_A, v_B) and the remainder outside it

    python experiments/04_multihop_swap/multihop_swap.py --model llama3.1-8b
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
from jlens_open.facts import COUNTRIES, TWO_HOP
from jlens_open.jspace import decompose, lens_atoms

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: the first available)")
parser.add_argument("--targets", type=int, default=3, help="countries B per prompt")
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_SWAP = os.path.join(HERE, f"results/{args.model}_swap.csv")
CSV_READ = os.path.join(HERE, f"results/{args.model}_readout.csv")
FIG_OUT = os.path.join(HERE, f"figures/{args.model}.pdf")
os.makedirs(os.path.dirname(CSV_SWAP), exist_ok=True)
os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
ALPHAS = (0.5, 1.0, 2.0)
K = 16

torch.manual_seed(0)
rng = np.random.default_rng(0)
lm = LensModel(args.model, args.device)
L, dev = lm.n_blocks, lm.device

lang_names = sorted({c[2] for c in COUNTRIES})
lang_ids = [lm.tok.encode(" " + w, add_special_tokens=False)[0] for w in lang_names]
assert len(set(lang_ids)) == len(lang_ids), "languages must start with different tokens"
country_ids = {c: lm.tok.encode(" " + c, add_special_tokens=False) for c, _, _ in COUNTRIES}
single = [i for i, (c, _, _) in enumerate(COUNTRIES) if len(country_ids[c]) == 1]
print(f"{args.model}: {len(single)} of {len(COUNTRIES)} countries are one token")

# Prompts: every country x phrasing; positions of the last token and of the capital's last token
prompts, meta = [], []
for ci, (country, capital, language) in enumerate(COUNTRIES):
    for ti, template in enumerate(TWO_HOP):
        ids = lm.encode(template.format(capital=capital))
        cap_pos = len(lm.encode(template.split("{capital}")[0] + capital)) - 1
        assert capital[-2:] in lm.tok.decode(ids[: cap_pos + 1])[-4:], (capital, lm.tok.decode(ids[: cap_pos + 1]))
        prompts.append(ids)
        meta.append({"country": ci, "template": ti, "answer": lang_names.index(language), "last": len(ids) - 1, "capital": cap_pos})
meta = pd.DataFrame(meta)
groups = [g.to_list() for _, g in pd.Series(range(len(prompts))).groupby([len(p) for p in prompts])]
POSITIONS = ["last", "capital"]

acts = {p: np.zeros((len(prompts), L, lm.d), dtype=np.float32) for p in POSITIONS}
pred = np.zeros(len(prompts), dtype=int)
read_rows = []
for idx in groups:
    hs, logits = lm.run(torch.stack([prompts[i] for i in idx]))
    pred[idx] = logits[:, lang_ids].argmax(1).cpu().numpy()
    for p in POSITIONS:
        pos = torch.tensor(meta[p].to_numpy()[idx])
        for layer, h in enumerate(hs):
            hp = h[torch.arange(len(idx)), pos]
            acts[p][idx, layer] = hp.cpu().numpy()
            # Is the intermediate country readable? Its rank in each lens (one-token countries only)
            keep = [b for b, i in enumerate(idx) if meta.country[i] in single]
            t = torch.tensor([country_ids[COUNTRIES[meta.country[idx[b]]][0]][0] for b in keep], device=dev)
            for use_jacobian, lens in ((True, "J-lens"), (False, "logit lens")):
                full = lm.unembed(lm.transport(hp[keep], layer, use_jacobian))
                ranks = ((full > full.gather(1, t[:, None])).sum(1) + 1).cpu().numpy()
                read_rows += [{"position": p, "layer": layer, "lens": lens, "rank": r} for r in ranks]
correct = pred == meta.answer.to_numpy()
print(f"two-hop accuracy among {len(lang_names)} languages: {correct.mean():.1%}")
read_df = pd.DataFrame(read_rows).groupby(["position", "lens", "layer"])["rank"].agg(
    median="median", top10=lambda r: (r <= 10).mean()).reset_index()
read_df.to_csv(CSV_READ, index=False)

# Trials: prompt of a one-token country A, answered correctly, x targets B with another language
trials = []
for i in range(len(prompts)):
    a = meta.country[i]
    if a not in single or not correct[i]:
        continue
    options = [b for b in single if COUNTRIES[b][2] != COUNTRIES[a][2]]
    for b in rng.choice(options, size=min(args.targets, len(options)), replace=False):
        trials.append((i, a, int(b)))
trials = pd.DataFrame(trials, columns=["prompt", "A", "B"])
tA_idx, tB_idx, tP_idx = trials.A.to_numpy(), trials.B.to_numpy(), trials.prompt.to_numpy()
target = np.array([lang_names.index(COUNTRIES[b][2]) for b in trials.B])
print(f"{len(trials)} swap trials")
tgroups = [g.to_list() for _, g in pd.Series(range(len(trials))).groupby([len(prompts[i]) for i in trials.prompt])]


def run_patched(layer, position, deltas):
    out = np.zeros(len(trials), dtype=int)
    for tidx in tgroups:
        pidx = tP_idx[tidx]
        d = torch.from_numpy(deltas[tidx].astype(np.float32)).to(dev)
        _, logits = lm.run(torch.stack([prompts[i] for i in pidx]), deltas={layer: d}, positions=meta[position].to_numpy()[pidx])
        out[tidx] = logits[:, lang_ids].argmax(1).cpu().numpy()
    return out


swap_rows = []
gen = torch.Generator(device="cpu").manual_seed(1)
rand_vecs = torch.randn(len(COUNTRIES), lm.d, generator=gen)
rand_vecs = (rand_vecs / rand_vecs.norm(dim=1, keepdim=True)).numpy()
tok_rows = [country_ids[COUNTRIES[c][0]][0] if c in single else 0 for c in range(len(COUNTRIES))]
for layer in range(L):
    J_atoms = lens_atoms(lm, layer, True)
    vec = {"J-lens": J_atoms[tok_rows].float().cpu().numpy(),
           "logit lens": lens_atoms(lm, layer, False)[tok_rows].float().cpu().numpy(),
           "random": rand_vecs}
    for position in POSITIONS:
        X = acts[position][:, layer]
        mu = X.mean(0)
        scale = np.linalg.norm(X - mu, axis=1).mean()
        x_trial = X[tP_idx]
        deltas = {}
        for name, V in vec.items():
            for alpha in ALPHAS:
                deltas[f"{name} swap, alpha={alpha}"] = alpha * scale * (V[tB_idx] - V[tA_idx])
        # The paper's coordinate swap: v_A's coefficient in the k = 16 J-space decomposition
        idx, coef, _ = decompose(torch.from_numpy(x_trial - mu).to(dev), J_atoms, K)
        tA = torch.tensor([tok_rows[a] for a in tA_idx], device=dev)
        c_A = (coef * (idx == tA[:, None])).sum(1).cpu().numpy()
        for mult in (1, 4):
            deltas[f"coordinate swap x{mult}"] = mult * c_A[:, None] * (vec["J-lens"][tB_idx] - vec["J-lens"][tA_idx])
        # Centroids over the other phrasings of each country
        cent = np.zeros((len(trials), 2, lm.d), dtype=np.float32)
        for t, (i, a, b) in enumerate(trials[["prompt", "A", "B"]].itertuples(index=False)):
            others = meta.template != meta.template[i]
            cent[t, 0] = X[(meta.country == a) & others].mean(0)
            cent[t, 1] = X[(meta.country == b) & others].mean(0)
        diff = cent[:, 1] - cent[:, 0]
        span = np.stack([vec["J-lens"][tA_idx], vec["J-lens"][tB_idx]], 2)  # (n, d, 2)
        Q = np.linalg.qr(span)[0]
        in_span = np.einsum("ndk,nk->nd", Q, np.einsum("ndk,nd->nk", Q, diff))
        deltas["centroid swap"] = diff
        deltas["centroid swap, J-lens span part"] = in_span
        deltas["centroid swap, outside J-lens span"] = diff - in_span
        for name, delta in deltas.items():
            out = run_patched(layer, position, delta)
            swap_rows.append({"layer": layer, "position": position, "patch": name,
                              "success": (out == target).mean(), "changed": (out != meta.answer.to_numpy()[tP_idx]).mean(),
                              "patch_norm": np.linalg.norm(delta, axis=1).mean(),
                              "A in J-space top16": (c_A > 0).mean()})
    del J_atoms
    print(f"swap: layer {layer} done", flush=True)
swap_df = pd.DataFrame(swap_rows)
swap_df.to_csv(CSV_SWAP, index=False)
pd.set_option("display.width", 250)
for position in POSITIONS:
    print(f"\nSwap success at the {position} token (answer becomes B's language), by layer")
    print(swap_df[swap_df.position == position].pivot_table(index="layer", columns="patch", values="success").round(2).to_string())
print("\nIntermediate readout (median rank of the country, fraction in the top 10)")
print(read_df.pivot_table(index="layer", columns=["position", "lens"], values=["median", "top10"]).round(2).to_string())

fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
for ax, position in zip(axs[:2], POSITIONS):
    t = swap_df[swap_df.position == position]
    for name in ["J-lens swap, alpha=1.0", "logit lens swap, alpha=1.0", "random swap, alpha=1.0", "coordinate swap x1",
                 "coordinate swap x4", "centroid swap", "centroid swap, J-lens span part", "centroid swap, outside J-lens span"]:
        s = t[t.patch == name]
        ax.plot(s.layer, s.success, marker="o", markersize=2, label=name)
    ax.set_title(f"Swap at the {position} token: answer becomes B's language", fontsize=9)
    ax.set_xlabel("Layer (block output)", fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)
axs[0].legend(fontsize=6)
ax = axs[2]
for (position, lens), s in read_df.groupby(["position", "lens"]):
    ax.plot(s.layer, s.top10, linestyle="-" if lens == "J-lens" else "--", label=f"{lens}, {position} token")
ax.set_title("Intermediate country in the lens top 10", fontsize=9)
ax.set_xlabel("Layer (block output)", fontsize=8)
ax.grid(linestyle="--", alpha=0.4)
ax.legend(fontsize=6)
fig.suptitle(args.model, fontsize=10)
plt.tight_layout()
# No CreationDate in the PDF, so that rerunning gives byte-identical files and git diff shows real changes only
fig.savefig(FIG_OUT, bbox_inches="tight", metadata={"CreationDate": None})
fig.savefig(FIG_OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: {CSV_SWAP}\n       {CSV_READ}\n       {FIG_OUT}")
