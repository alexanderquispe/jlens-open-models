"""Experiment 09, analysis (CPU): the words the lenses surface while the model forms its beliefs.

Reads the raw outputs of run_model.py (and of concept_lens.py, if present) and writes:
  results/<model>_behaviour.csv     per condition, late in the stream (t >= 512): the model's mean, sd, KL to the
                                    target; the go / no-go of the belief (results/<model>_verdict.json)
  results/<model>_lens_belief.csv   per lens and layer: the lens's softmax mass on integer tokens, and how its
                                    integer distribution tracks mu (correlation, slope) and sigma (log-log)
  results/<model>_word_frequency.csv  per lens and layer: the tokens most often among the lens's top 20 at the
                                    com2num positions (t >= 8, all stationary sequences)
  results/<model>_top_words.csv     per lens, layer third, condition and bin of t: the top tokens by mean log-prob
  results/<model>_contrasts.csv     per lens and layer third: the tokens whose mean log-prob moves most with mu, with
                                    sigma, with time (late - early) and at the switch (after - before)
  results/<model>_lexicon.csv       the word groups of common.LEXICON (minus the control words), middle band
  results/<model>_concepts.csv      Concept Lens: concepts in the joint top 10, by condition / event and layer third
  results/<model>_words.md          a readable report of all of the above
  figures/<model>_behaviour, _lens_belief, _switch, _lexicon (PDF and PNG)

    python experiments/09_beliefs/analyze.py --model llama3.1-8b
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONDITIONS, E_BINS, HERE, MUS, RAW, SIGMA0, SIGMAS, MU0, T_BINS, band, bin_labels, cond_name, thirds  # noqa: E402

LENS_COLOR = {"logit": "C7", "jlens": "C0", "jpp": "C3"}
LENS_NAME = {"logit": "logit lens", "jlens": "J-lens", "jpp": "J++ lens"}
LATE, EARLY = len(T_BINS) - 2, 1  # t bins [512, 1000) and [8, 32)
SW_PRE = [E_BINS.index(-100)]                           # e in [-100, -10)
SW_POST = [E_BINS.index(1), E_BINS.index(2), E_BINS.index(3), E_BINS.index(4)]  # e in [1, 8)
SW_AT = [E_BINS.index(0)]                               # e = 0, the first new number


def save(fig, path):
    fig.savefig(path + ".pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(path + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def token_table(hf_id, V):
    """Display string and class (integer / word / other) of every token id."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id)
    disp, cls = [], []
    for i in range(V):
        try:
            s = tok.decode([i], clean_up_tokenization_spaces=False)
        except Exception:  # ids past the tokenizer (padded embedding rows)
            s = ""
        t = s.strip()
        disp.append(t if t else repr(s))
        if t.isdigit() and len(t) <= 3:
            cls.append("integer")
        elif t.isalpha() and len(t) >= 2:
            cls.append("word")
        else:
            cls.append("other")
    return np.array(disp, dtype=object), np.array(cls)


def family(name):
    """'mu' (sigma = 100), 'sigma' (mu = 500) or both for m500_s100."""
    out = []
    for m, s in CONDITIONS:
        if cond_name(m, s) == name:
            if s == SIGMA0:
                out.append("mu")
            if m == MU0:
                out.append("sigma")
    return out


# ------------------------------------------------------------------------------------------------------- behaviour
def behaviour(R, meta):
    T = meta["T"]
    late = np.arange(T) >= 512 if T > 512 else np.arange(T) >= T // 2
    window = "t >= 512" if T > 512 else f"t >= {T // 2} (short run)"
    rows = []
    for c, (m, s) in enumerate(CONDITIONS):
        idx = [i for i, q in enumerate(meta["sequences"]) if q["cond"] == cond_name(m, s)]
        if not idx:
            continue
        o = R["out_int"][idx][:, late]
        rows.append({"cond": cond_name(m, s), "mu": m, "sigma": s, "model_mean": np.nanmean(o[..., 0]),
                     "model_sd": np.nanmean(o[..., 1]), "entropy": np.nanmean(o[..., 2]),
                     "mass_on_integers": np.nanmean(o[..., 3]), "kl_to_target": np.nanmean(o[..., 4])})
    df = pd.DataFrame(rows)
    fm = df[df.sigma == SIGMA0]
    fs = df[df.mu == MU0]
    v = {"window": window}
    if len(fm) > 2:
        v["mean_vs_mu_r"] = float(np.corrcoef(fm.mu, fm.model_mean)[0, 1])
        v["mean_vs_mu_slope"] = float(np.polyfit(fm.mu, fm.model_mean, 1)[0])
    if len(fs) > 2:
        v["logsd_vs_logsigma_r"] = float(np.corrcoef(np.log(fs.sigma), np.log(fs.model_sd))[0, 1])
        v["logsd_vs_logsigma_slope"] = float(np.polyfit(np.log(fs.sigma), np.log(fs.model_sd), 1)[0])
    v["go"] = bool(v.get("mean_vs_mu_r", 0) >= 0.95 and v.get("logsd_vs_logsigma_r", 0) >= 0.9)
    return df, v


def fig_behaviour(R, meta, Rs, ms, path):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    T = meta["T"]
    t = np.arange(T) + 1
    cmap = plt.get_cmap("viridis")
    for c, (m, s) in enumerate(CONDITIONS):
        idx = [i for i, q in enumerate(meta["sequences"]) if q["cond"] == cond_name(m, s)]
        if idx:
            axs[0].plot(t, np.nanmean(R["out_int"][idx, :, 4], 0), color=cmap(c / len(CONDITIONS)), lw=0.8)
    axs[0].set(xscale="log", yscale="log", xlabel="numbers read (t + 1)", ylabel="KL(model || N(mu, sigma))",
               title="Convergence (paper Fig. 7A), each condition")
    df, _ = behaviour(R, meta)
    fm, fs = df[df.sigma == SIGMA0], df[df.mu == MU0]
    axs[1].plot(fm.mu, fm.model_mean, "o-", color="C0", label="mean (sigma = 100)")
    axs[1].plot([min(MUS), max(MUS)], [min(MUS), max(MUS)], color="grey", lw=0.8)
    axs[1].set(xlabel="true mu", ylabel="model's predicted mean", title="Belief late in the stream (t >= 512)")
    inset = axs[1].inset_axes([0.62, 0.12, 0.34, 0.34])
    inset.loglog(fs.sigma, fs.model_sd, "o-", color="C1", ms=3)
    inset.loglog([min(SIGMAS), max(SIGMAS)], [min(SIGMAS), max(SIGMAS)], color="grey", lw=0.6)
    inset.set_title("model sd vs true sigma (mu = 500)", fontsize=7)
    inset.tick_params(labelsize=6)
    if Rs is not None:
        at = ms["sequences"][0]["switch_at"]
        Ts = ms["T"]
        e = np.arange(Ts) - at
        keep = (e >= -50) & (e <= min(400, Ts - at - 1))
        mean = np.nanmean(Rs["out_int"][:, keep, 0], 0)
        sd = np.nanmean(Rs["out_int"][:, keep, 1], 0)
        axs[2].plot(e[keep], mean, color="C0", label="mean")
        axs[2].fill_between(e[keep], mean - sd, mean + sd, color="C0", alpha=0.2, label="mean +- sd")
        axs[2].axvline(0, color="grey", lw=0.6, ls=":")
        axs[2].set(xlabel="numbers after the switch", ylabel="model's prediction", title="Switch N(300) -> N(700) (Fig. 2B)")
        axs[2].legend(fontsize=7)
    save(fig, path)


# ------------------------------------------------------------------------------------------------------- lens belief
def com_positions(meta):
    return np.array([meta["bos"] + 2 * t + 1 for t in range(meta["T"])])


def lens_belief(raw, lenses, meta, L):
    T, com = meta["T"], com_positions(meta)
    late = np.arange(T) >= (512 if T > 512 else T // 2)
    seqs = meta["sequences"]
    mu = np.array([q["mu"] for q in seqs])
    sg = np.array([q["sigma"] for q in seqs])
    fam_m, fam_s = sg == SIGMA0, mu == MU0
    rows = []
    for k in lenses:
        S = np.load(os.path.join(raw, f"stationary_{k}_scalars.npy"), mmap_mode="r")
        x = np.asarray(S[:, com[late]]).mean(1)  # (N, L, 3): mass, mean, sd averaged late in each sequence
        for layer in range(L):
            r = {"lens": k, "layer": layer, "int_mass": float(x[:, layer, 0].mean())}
            if fam_m.sum() > 2:
                r["r_mu"] = float(np.corrcoef(mu[fam_m], x[fam_m, layer, 1])[0, 1])
                r["slope_mu"] = float(np.polyfit(mu[fam_m], x[fam_m, layer, 1], 1)[0])
            if fam_s.sum() > 2:
                ls = np.log(np.maximum(x[fam_s, layer, 2], 1e-6))
                r["r_logsigma"] = float(np.corrcoef(np.log(sg[fam_s]), ls)[0, 1])
                r["slope_logsigma"] = float(np.polyfit(np.log(sg[fam_s]), ls, 1)[0])
            rows.append(r)
    return pd.DataFrame(rows)


def fig_lens_belief(df, L, path):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
    for k, s in df.groupby("lens"):
        c = LENS_COLOR.get(k, "k")
        axs[0].plot(s.layer, s.int_mass, color=c, label=LENS_NAME.get(k, k))
        if "r_mu" in s:
            axs[1].plot(s.layer, s.r_mu, color=c, label=f"{LENS_NAME.get(k, k)}: r")
            axs[1].plot(s.layer, s.slope_mu, color=c, ls="--", lw=0.8, label=f"{LENS_NAME.get(k, k)}: slope")
        if "r_logsigma" in s:
            axs[2].plot(s.layer, s.r_logsigma, color=c, label=f"{LENS_NAME.get(k, k)}: r")
            axs[2].plot(s.layer, s.slope_logsigma, color=c, ls="--", lw=0.8)
    b = band(L)
    for ax in axs:
        ax.axvspan(b[0], b[-1], color="grey", alpha=0.08)
        ax.set_xlabel("layer")
    axs[0].set(ylabel="softmax mass on integer tokens")
    axs[0].set_title("Mass on integer tokens", fontsize=9)
    axs[1].set(ylabel="lens integer mean vs mu")
    axs[1].set_title("Integer mean tracks mu (sigma = 100)", fontsize=9)
    axs[2].set(ylabel="log lens sd vs log sigma")
    axs[2].set_title("Integer sd tracks sigma (mu = 500)", fontsize=9)
    axs[0].legend(fontsize=7)
    axs[1].legend(fontsize=6)
    save(fig, path)


def fig_switch(raw, lenses, ms, Rs, L, path):
    at, Ts = ms["sequences"][0]["switch_at"], ms["T"]
    com = com_positions(ms)
    e = np.arange(Ts) - at
    keep = (e >= -20) & (e <= min(200, Ts - at - 1))
    fig, axs = plt.subplots(1, len(lenses) + 1, figsize=(3.6 * (len(lenses) + 1), 3.3))
    out = np.nanmean(Rs["out_int"][:, keep, 0], 0)
    for ax, k in zip(axs, lenses):
        S = np.load(os.path.join(raw, f"switch_{k}_scalars.npy"), mmap_mode="r")
        m = np.asarray(S[:, com[keep]])[..., 1].mean(0).T  # (L, e)
        im = ax.imshow(m, aspect="auto", origin="lower", cmap="viridis", vmin=200, vmax=800,
                       extent=[e[keep][0] - 0.5, e[keep][-1] + 0.5, -0.5, L - 0.5])
        ax.axvline(0, color="w", lw=0.6, ls=":")
        ax.set(title=f"{LENS_NAME.get(k, k)}: integer mean", xlabel="numbers after the switch", ylabel="layer")
    fig.colorbar(im, ax=axs[: len(lenses)], shrink=0.8)
    axs[-1].plot(e[keep], out, color="k", label="model output")
    for k in lenses:
        S = np.load(os.path.join(raw, f"switch_{k}_scalars.npy"), mmap_mode="r")
        b = band(L)
        axs[-1].plot(e[keep], np.asarray(S[:, com[keep]][:, :, b, 1]).mean((0, 2)), color=LENS_COLOR.get(k, "k"),
                     label=f"{LENS_NAME.get(k, k)}, layers {b[0]}-{b[-1]}")
    axs[-1].axvline(0, color="grey", lw=0.6, ls=":")
    axs[-1].set(xlabel="numbers after the switch", ylabel="integer mean", title="Middle band vs output")
    axs[-1].legend(fontsize=7)
    save(fig, path)


# ------------------------------------------------------------------------------------------------------- words
def word_frequency(raw, lenses, meta, L, disp, cls, top_n=30):
    com = com_positions(meta)[np.arange(meta["T"]) >= 8]
    rows = []
    for k in lenses:
        TOP = np.load(os.path.join(raw, f"stationary_{k}_top.npy"), mmap_mode="r")
        for layer in range(L):
            ids = np.asarray(TOP[:, com, layer]).reshape(-1)
            cnt = np.bincount(ids, minlength=len(cls))
            n = len(com) * TOP.shape[0]
            for kind in ("word", "integer", "other"):
                c = np.where(cls == kind, cnt, 0)
                for rank, i in enumerate(np.argsort(-c)[:top_n]):
                    if c[i] == 0:
                        break
                    rows.append({"lens": k, "layer": layer, "class": kind, "rank": rank + 1, "token": disp[i],
                                 "token_id": int(i), "frac_in_top20": c[i] / n})
    return pd.DataFrame(rows)


def layer_mean(C, layers):
    return np.mean([np.asarray(C[layer], dtype=np.float32) for layer in layers], 0)  # (n_cells, V)


def filled(raw, ds):
    """Boolean per cell: True if any position fell in it. Empty cells are saved as zeros and must be skipped
    (they only occur in short test runs, e.g. no t >= 512 with --numbers 40). All True if counts are absent."""
    p = os.path.join(raw, f"cells_count_{ds}.npy")
    return np.load(p) > 0 if os.path.exists(p) else None


def top_words(raw, lenses, L, disp, cls, top_n=15):
    conds = [cond_name(m, s) for m, s in CONDITIONS]
    nb = len(T_BINS) - 1
    labels = bin_labels(T_BINS)
    ok = filled(raw, "stationary")
    ok = np.ones((len(conds), nb), bool) if ok is None else ok.reshape(len(conds), nb)
    rows = []
    for k in lenses:
        C = np.load(os.path.join(raw, f"stationary_{k}_cells.npy"), mmap_mode="r")
        for third, layers in thirds(L).items():
            M = layer_mean(C, layers).reshape(len(conds), nb, -1)
            cells = {(c, labels[b]): M[i, b] for i, c in enumerate(conds) for b in (EARLY, LATE) if ok[i, b]}
            for b in (LATE, EARLY):
                if ok[:, b].any():
                    cells[("all", labels[b])] = M[ok[:, b], b].mean(0)
            for (c, b), v in cells.items():
                for kind in ("word", "integer"):
                    vv = np.where(cls == kind, v, -np.inf)
                    for rank, i in enumerate(np.argsort(-vv)[:top_n]):
                        rows.append({"lens": k, "layers": third, "cond": c, "t_bin": b, "class": kind, "rank": rank + 1,
                                     "token": disp[i], "token_id": int(i), "mean_logp": float(v[i])})
    return pd.DataFrame(rows)


def ranked_delta(delta, base, cand, cls, disp, top_n, meta):
    rows = []
    for kind in ("word", "integer"):
        m = cand & (cls == kind)
        idx = np.where(m)[0]
        order = idx[np.argsort(-delta[idx])]
        up, down = order[delta[order] > 0][:top_n], order[::-1][delta[order[::-1]] < 0][:top_n]
        for direction, sel in (("up", up), ("down", down)):
            for rank, i in enumerate(sel):
                rows.append({**meta, "class": kind, "direction": direction, "rank": rank + 1, "token": disp[i],
                             "token_id": int(i), "delta": float(delta[i]), "base_logp": float(base[i])})
    return rows


def contrasts(raw, lenses, L, disp, cls, has_switch, top_n=15, pool=200):
    """Tokens whose mean log-prob moves most, among those in the top `pool` of at least one cell compared."""
    conds = [cond_name(m, s) for m, s in CONDITIONS]
    nb = len(T_BINS) - 1
    im = [conds.index(cond_name(m, SIGMA0)) for m in MUS]
    isg = [conds.index(cond_name(MU0, s)) for s in SIGMAS]
    ok = filled(raw, "stationary")
    ok = np.ones((len(conds), nb), bool) if ok is None else ok.reshape(len(conds), nb)
    ok_sw = filled(raw, "switch") if has_switch else None
    sw_ok = lambda sel: ok_sw is None or bool(ok_sw[sel].all())  # noqa: E731
    late_ok, early_ok = ok[:, LATE].all(), ok[:, EARLY].all()
    rows = []

    def pool_of(cells):
        cand = np.zeros(len(cls), bool)
        for v in cells:
            cand[np.argsort(-v)[:pool]] = True
        return cand

    for k in lenses:
        C = np.load(os.path.join(raw, f"stationary_{k}_cells.npy"), mmap_mode="r")
        Cs = np.load(os.path.join(raw, f"switch_{k}_cells.npy"), mmap_mode="r") if has_switch else None
        for third, layers in thirds(L).items():
            M = layer_mean(C, layers).reshape(len(conds), nb, -1)
            late = M[:, LATE]
            if late_ok:
                x = np.array(MUS, float)
                Y = late[im]
                slope = ((x - x.mean())[:, None] * (Y - Y.mean(0))).sum(0) / ((x - x.mean()) ** 2).sum()
                rows += ranked_delta(slope * (x.max() - x.min()), Y.mean(0), pool_of(Y), cls, disp, top_n,
                                     {"lens": k, "layers": third, "contrast": "mu 700 vs 300 (late)"})
                x = np.log(np.array(SIGMAS, float))
                Y = late[isg]
                slope = ((x - x.mean())[:, None] * (Y - Y.mean(0))).sum(0) / ((x - x.mean()) ** 2).sum()
                rows += ranked_delta(slope * (x.max() - x.min()), Y.mean(0), pool_of(Y), cls, disp, top_n,
                                     {"lens": k, "layers": third, "contrast": "sigma 200 vs 10 (late)"})
            if late_ok and early_ok:
                a, b = M[:, LATE].mean(0), M[:, EARLY].mean(0)
                rows += ranked_delta(a - b, b, pool_of([a, b]), cls, disp, top_n,
                                     {"lens": k, "layers": third, "contrast": "late (t >= 512) vs early (8 <= t < 32)"})
            if Cs is not None and sw_ok(SW_PRE):
                S = layer_mean(Cs, layers)
                pre = S[SW_PRE].mean(0)
                for name, sel in (("switch: e in [1, 8) vs [-100, -10)", SW_POST), ("switch: e = 0 vs [-100, -10)", SW_AT)):
                    if not sw_ok(sel):
                        continue
                    post = S[sel].mean(0)
                    rows += ranked_delta(post - pre, pre, pool_of([pre, post]), cls, disp, top_n,
                                         {"lens": k, "layers": third, "contrast": name})
    return pd.DataFrame(rows)


def lexicon(raw, lenses, meta, ms, L):
    from common import GROUPS

    b = band(L)
    gi = {g: i for i, g in enumerate(GROUPS)}
    rows = []
    for k in lenses:
        X = np.load(os.path.join(raw, f"stationary_{k}_lex.npy"), mmap_mode="r")
        com = com_positions(meta)
        x = np.asarray(X[:, com][:, :, b], dtype=np.float32).mean(2)  # (N, T, G)
        x = x - x[..., gi["control"]][..., None]
        bins = np.searchsorted(T_BINS, np.arange(meta["T"]), side="right") - 1
        for bi, lab in enumerate(bin_labels(T_BINS)):
            sel = bins == bi
            if sel.any():
                for g in GROUPS[:-1]:
                    rows.append({"lens": k, "dataset": "stationary", "bin": lab, "group": g,
                                 "score": float(x[:, sel, gi[g]].mean())})
        if ms is not None:
            Xs = np.load(os.path.join(raw, f"switch_{k}_lex.npy"), mmap_mode="r")
            coms = com_positions(ms)
            xs = np.asarray(Xs[:, coms][:, :, b], dtype=np.float32).mean(2)
            xs = xs - xs[..., gi["control"]][..., None]
            e = np.arange(ms["T"]) - ms["sequences"][0]["switch_at"]
            for ev in range(-10, 31):
                if (e == ev).any():
                    for g in GROUPS[:-1]:
                        rows.append({"lens": k, "dataset": "switch", "bin": ev, "group": g,
                                     "score": float(xs[:, e == ev, gi[g]].mean())})
    return pd.DataFrame(rows)


def fig_lexicon(df, path):
    sw = df[df.dataset == "switch"]
    if sw.empty:
        return
    lenses = list(dict.fromkeys(sw.lens))
    fig, axs = plt.subplots(1, len(lenses), figsize=(3.8 * len(lenses), 3.2), sharey=True, squeeze=False)
    for ax, k in zip(axs[0], lenses):
        for j, (g, s) in enumerate(sw[sw.lens == k].groupby("group", sort=False)):
            ax.plot(s.bin, s.score, color=f"C{j}", label=g, lw=1)
        ax.axvline(0, color="grey", lw=0.6, ls=":")
        ax.axhline(0, color="grey", lw=0.6)
        ax.set(title=f"{LENS_NAME.get(k, k)}, middle band", xlabel="numbers after the switch")
    axs[0][0].set_ylabel("word-group z-score minus control")
    axs[0][0].legend(fontsize=6)
    save(fig, path)


# ------------------------------------------------------------------------------------------------------- concepts
def concepts(raw, L, top_rank=10):
    """Per (lens, dataset, condition, time, layer third): how often each concept is in the joint top `top_rank`, as a
    fraction of the (readout, layer) pairs of that cell."""
    rows, n_read = [], {}
    th = thirds(L)
    third_of = {layer: name for name, ls in th.items() for layer in ls}
    for f in sorted(os.listdir(raw)):
        if not (f.startswith("concepts_") and f.endswith(".jsonl")):
            continue
        lens = f[len("concepts_") : -len(".jsonl")]
        for line in open(os.path.join(raw, f)):
            r = json.loads(line)
            key = (lens, r["dataset"], r["cond"] if r["dataset"] == "stationary" else "switch",
                   r["t"] if r["dataset"] == "stationary" else r["e"], third_of[r["layer"]])
            n_read[key] = n_read.get(key, 0) + 1
            for surface, rank, score, _ in r["concepts"]:
                if rank <= top_rank:
                    rows.append(dict(zip(("lens", "dataset", "cond", "time", "layers"), key), concept=surface.lower(),
                                     rank=rank))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    g = df.groupby(["lens", "dataset", "cond", "time", "layers", "concept"]).agg(n=("rank", "size"),
                                                                                 median_rank=("rank", "median"))
    g = g.reset_index()
    g["frac_of_readouts"] = [row.n / n_read[(row.lens, row.dataset, row.cond, row.time, row.layers)]
                             for row in g.itertuples()]
    return g.sort_values(["lens", "dataset", "cond", "time", "layers", "n"], ascending=[True, True, True, True, True, False])


# ------------------------------------------------------------------------------------------------------- report
def md_table(df, cols, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in df[cols].itertuples(index=False):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def report(model, info, verdict, beh, lb, wf, tw, ct, lx, cc, L, path):
    th = thirds(L)
    out = [f"# Experiment 09 · words of beliefs: {model}", "",
           f"Generated by analyze.py from the raw outputs ({info.get('gpu', '?')}, {info.get('seconds', '?')} s). "
           f"Lenses: {', '.join(LENS_NAME.get(k, k) for k in info['lenses'])}. Layer thirds: "
           + ", ".join(f"{n} {v[0]}-{v[-1]}" for n, v in th.items()) + ".", "",
           "## Behaviour: does the model hold the belief?", "",
           f"Late in the stream ({verdict.get('window', 't >= 512')}): mean vs mu r = {verdict.get('mean_vs_mu_r', float('nan')):.3f} "
           f"(slope {verdict.get('mean_vs_mu_slope', float('nan')):.2f}); log sd vs log sigma r = "
           f"{verdict.get('logsd_vs_logsigma_r', float('nan')):.3f} (slope {verdict.get('logsd_vs_logsigma_slope', float('nan')):.2f}). "
           f"Go: **{verdict['go']}**.", "",
           md_table(beh.round(3), ["cond", "model_mean", "model_sd", "kl_to_target", "mass_on_integers"],
                    ["condition", "model mean", "model sd", "KL to target", "mass on integers"]), "",
           "## The belief in number words, layer by layer", ""]
    for k, s in lb.groupby("lens", sort=False):
        best = s.loc[s.r_mu.idxmax()] if "r_mu" in s and s.r_mu.notna().any() else None
        out.append(f"- **{LENS_NAME.get(k, k)}**: integer mass {s.int_mass.min():.2f}-{s.int_mass.max():.2f}; "
                   + (f"best r(mean, mu) {best.r_mu:.3f} at layer {int(best.layer)}; " if best is not None else "")
                   + ("first layer with r(mean, mu) >= 0.9: "
                      + str(int(s[s.r_mu >= 0.9].layer.min()) if (s.r_mu >= 0.9).any() else "none") if "r_mu" in s else ""))
    out += ["", "## Words most often in the lens's top 20 (com2num positions, t >= 8, all conditions)", ""]
    for k in wf.lens.unique():
        for third, layers in th.items():
            s = wf[(wf.lens == k) & wf.layer.isin(layers) & (wf["class"] == "word")]
            agg = s.groupby("token").frac_in_top20.mean().sort_values(ascending=False).head(15)
            out.append(f"- {LENS_NAME.get(k, k)}, {third} ({layers[0]}-{layers[-1]}): "
                       + ", ".join(f"{t} ({v:.2f})" for t, v in agg.items()))
    out += ["", "## Top words by mean log-prob, late (t >= 512), all conditions pooled", ""]
    for k in (tw.lens.unique() if len(tw) else []):
        for third in th:
            s = tw[(tw.lens == k) & (tw.layers == third) & (tw.cond == "all") & (tw.t_bin == bin_labels(T_BINS)[LATE])]
            if s.empty:
                continue
            w = ", ".join(s[s["class"] == "word"].sort_values("rank").token.head(12))
            n = ", ".join(s[s["class"] == "integer"].sort_values("rank").token.head(6))
            out.append(f"- {LENS_NAME.get(k, k)}, {third}: words: {w}; integers: {n}")
    out += ["", "## What moves with the belief (word tokens; numbers are in the CSV)", ""]
    for contrast in (ct.contrast.unique() if len(ct) else []):
        out += [f"### {contrast}", ""]
        for k in ct.lens.unique():
            for third in th:
                s = ct[(ct.lens == k) & (ct.layers == third) & (ct.contrast == contrast) & (ct["class"] == "word")]
                up = ", ".join(f"{t} ({d:+.2f})" for t, d in s[s.direction == "up"].sort_values("rank")[["token", "delta"]].head(8).values)
                dn = ", ".join(f"{t} ({d:+.2f})" for t, d in s[s.direction == "down"].sort_values("rank")[["token", "delta"]].head(8).values)
                out.append(f"- {LENS_NAME.get(k, k)}, {third}: up: {up}; down: {dn}")
        out.append("")
    if not lx.empty:
        out += ["## Word groups (middle band, z-score minus control)", ""]
        s = lx[lx.dataset == "stationary"].pivot_table(index=["lens", "group"], columns="bin", values="score", sort=False)
        s = s.round(3).reset_index()
        out += [md_table(s, list(s.columns), ["lens", "group"] + [f"t in {c}" for c in s.columns[2:]]), ""]
    if not cc.empty:
        out += ["## Concept Lens: concepts in the joint top 10", ""]
        for (lens, ds, cond, time), s in cc.groupby(["lens", "dataset", "cond", "time"], sort=False):
            parts = []
            for third in th:
                x = s[s.layers == third].head(6)
                if len(x):
                    parts.append(f"{third}: " + ", ".join(f"{c} ({f:.2f})" for c, f in zip(x.concept, x.frac_of_readouts)))
            out.append(f"- {LENS_NAME.get(lens, lens)}, {ds} {cond}, {'t' if ds == 'stationary' else 'e'} = {time}: " + "; ".join(parts))
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--raw", default=RAW)
    parser.add_argument("--out", default=HERE, help="folder for results/ and figures/")
    args = parser.parse_args()
    raw = os.path.join(args.raw, args.model)
    info = json.load(open(os.path.join(raw, "run_info.json")))
    L, lenses = info["n_blocks"], info["lenses"]
    res, figs = os.path.join(args.out, "results"), os.path.join(args.out, "figures")
    os.makedirs(res, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    pre = f"{args.model}_"
    meta = json.load(open(os.path.join(raw, "meta_stationary.json")))
    R = dict(np.load(os.path.join(raw, "behaviour_stationary.npz")))
    has_switch = os.path.exists(os.path.join(raw, "meta_switch.json"))
    ms = json.load(open(os.path.join(raw, "meta_switch.json"))) if has_switch else None
    Rs = dict(np.load(os.path.join(raw, "behaviour_switch.npz"))) if has_switch else None
    V = np.load(os.path.join(raw, f"stationary_{lenses[0]}_cells.npy"), mmap_mode="r").shape[-1]
    disp, cls = token_table(info["hf"], V)

    beh, verdict = behaviour(R, meta)
    beh.to_csv(os.path.join(res, pre + "behaviour.csv"), index=False)
    fig_behaviour(R, meta, Rs, ms, os.path.join(figs, pre + "behaviour"))
    lb = lens_belief(raw, lenses, meta, L)
    lb.to_csv(os.path.join(res, pre + "lens_belief.csv"), index=False)
    fig_lens_belief(lb, L, os.path.join(figs, pre + "lens_belief"))
    if has_switch:
        fig_switch(raw, lenses, ms, Rs, L, os.path.join(figs, pre + "switch"))
    wf = word_frequency(raw, lenses, meta, L, disp, cls)
    wf.to_csv(os.path.join(res, pre + "word_frequency.csv"), index=False)
    tw = top_words(raw, lenses, L, disp, cls)
    tw.to_csv(os.path.join(res, pre + "top_words.csv"), index=False)
    ct = contrasts(raw, lenses, L, disp, cls, has_switch)
    ct.to_csv(os.path.join(res, pre + "contrasts.csv"), index=False)
    lx = lexicon(raw, lenses, meta, ms, L)
    lx.to_csv(os.path.join(res, pre + "lexicon.csv"), index=False)
    fig_lexicon(lx, os.path.join(figs, pre + "lexicon"))
    cc = concepts(raw, L)
    if not cc.empty:
        cc.to_csv(os.path.join(res, pre + "concepts.csv"), index=False)
    verdict.update({"model": args.model, "lenses": lenses})
    with open(os.path.join(res, pre + "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1)
    report(args.model, info, verdict, beh, lb, wf, tw, ct, lx, cc, L, os.path.join(res, pre + "words.md"))
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
