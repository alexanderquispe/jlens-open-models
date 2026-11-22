"""Experiment 11 analysis (CPU): the model notices but does not say?

For every model with outputs in the raw folder:
A. Report (ask.py). Change score at the last prompt token: log P(the answer that means "changed / surprising") -
   log P(the other answer) (yes / no or A / B, log-sum-exp over variants); for "wait", log P(surprise set) -
   log P(normal set); for "comma", the predicted mean. Jump - control by cut e (mean over streams, sign-flip p),
   pooled over e in {1, 2, 4, 8, 32}; outliers against the typical last number; AUC of jump vs control.
   The same score read inside, per lens and layer ("does the middle of the model say Yes?").
   Probes: a difference-of-means direction fitted on 19 streams (jump vs control, e in {1, 2, 4, 8, 32}) and
   applied to the held-out stream's residual at the last prompt token; AUC over all held-out scores, per layer and
   format. Also trained on e >= 1 and tested at e = 0 (one new number).
B. Injection (inject.py). Natural gap G = jump - control without any change. Each change's effect = changed -
   unchanged, same stream and arm. Transplant share = effect of diff_pos (+1) on control / G; removal share =
   -effect of diff_pos (-1) on jump / G; proj shares likewise. Dose-response against norm-matched random
   directions (randm), and lens-direction steering (surprise vs control words).
D. The 2 x 2 change test (ask.py on data/contexts_down.jsonl, raw/down/<model>/): history low / high x recent
   numbers low / high. Cells LL (up control), LH (up jump), HL (down jump), HH (level high). Change =
   (LH + HL - LL - HH) / 2 (the interaction: what "changed?" should track), recent = (LH + HH - LL - HL) / 2,
   history = (HL + HH - LL - LH) / 2. Probes: a change direction (mean of the change cells - mean of the others, 19
   streams) scored on the held-out stream; and the up-pair direction (LH - LL) tested on the down pair (HL vs HH):
   an AUC near 1 means it reads change, near 0 that it reads the recent numbers.
C. Ablation (ablate.py). The predicted sd over numbers 200..399 by sigma, per variant; the slope of log sd on
   log sigma, its change from no ablation (bootstrap over streams).

    python experiments/11_notice_report/analyze.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(os.path.dirname(HERE), "08_surprise"))
from stats import sign_flip_p  # noqa: E402  (experiment 08's)

RAW = (os.path.join(os.environ["JLENS_SHARED"], "experiments", "11_notice_report") if os.environ.get("JLENS_SHARED")
       else os.path.join(HERE, "raw"))
MODELS = ("llama3.1-8b", "llama3.1-8b-it", "olmo-3-7b", "olmo-3-7b-it", "olmo-3-32b", "olmo-3.1-32b-it")
POOL = (1, 2, 4, 8, 32)
OPP = {"yes": "no", "no": "yes", "a": "b", "b": "a"}


def lse(x, axis=-1):
    x = np.asarray(x, np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def auc(pos, neg):
    """P(score of a positive > score of a negative), ties counted half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def thirds(L):
    b = list(range(L // 3, (2 * L) // 3 + 1))
    return {"early": list(range(0, b[0])), "middle": b, "late": list(range(b[-1] + 1, L))}


def scores(meta, A, S, ints):
    """Change score (P, F, L) from answer log-probs A (P, F, L, n_ans), sets S (P, F, L, 3), int stats (P, 2)."""
    toks = meta["answer_tokens"]
    idx = {a: [i for i, t in enumerate(toks) if t[0] == a] for a in OPP}
    P, F, L = A.shape[:3]
    out = np.full((P, F, L), np.nan)
    for j, (name, fam, kind, changed) in enumerate(meta["formats"]):
        if kind in ("yesno", "ab"):
            out[:, j] = lse(A[:, j][..., idx[changed]]) - lse(A[:, j][..., idx[OPP[changed]]])
        elif kind == "words":
            out[:, j] = S[:, j, :, 0] - S[:, j, :, 1]  # surprise - normal
        elif kind == "int":
            out[:, j, L - 1] = ints[:, 0]
    return out


def part_a(model, R, md, rows_out):
    meta = json.load(open(os.path.join(R, "ask_meta.json")))
    ctx = meta["contexts"]
    L, fmts = meta["n_blocks"], meta["formats"]
    sw = {(c["seq"], c["e"], c["arm"]): i for i, c in enumerate(ctx) if c["block"] == "switch"}
    ol = {(c["seq"], c["level"]): i for i, c in enumerate(ctx) if c["block"] == "outlier"}
    seqs = sorted({c["seq"] for c in ctx})
    ints = np.load(os.path.join(R, "ask_intstats.npy"))
    md.append(f"\n## {model}\n\n### A. What the model says (output, last prompt token)\n")
    md.append("Change score = log P(answer meaning 'changed') - log P(other answer); wait: surprise - normal; comma: "
              "predicted mean. Jump - control (sign-flip p). Pooled = e in {1, 2, 4, 8, 32}. AUC: jump vs control, "
              "pooled cuts. Outlier: extreme - typical last number.\n")
    md.append("| format | control level | e=0 | e=1 | e=4 | e=32 | pooled | AUC | outlier extreme | outlier moderate |\n|---|---|---|---|---|---|---|---|---|---|")
    sc = {}
    for lens in meta["lenses"]:
        A = np.load(os.path.join(R, f"ask_ans_{lens}.npy"))
        S = np.load(os.path.join(R, f"ask_sets_{lens}.npy"))
        sc[lens] = scores(meta, A, S, ints)
    out = sc["logit"][:, :, L - 1]
    for j, (name, fam, kind, changed) in enumerate(fmts):
        d = lambda e: np.array([out[sw[(k, e, "jump")], j] - out[sw[(k, e, "control")], j] for k in seqs])  # noqa: E731
        pooled = np.array([np.mean([out[sw[(k, e, "jump")], j] - out[sw[(k, e, "control")], j] for e in POOL]) for k in seqs])
        cells = []
        for e in (0, 1, 4, 32):
            m_, p_ = sign_flip_p(d(e))
            cells.append(f"{m_:+.2f}{'*' if p_ < 0.05 else ''}")
        mp, pp = sign_flip_p(pooled)
        a = auc([out[sw[(k, e, "jump")], j] for k in seqs for e in POOL], [out[sw[(k, e, "control")], j] for k in seqs for e in POOL])
        ctrl = np.mean([out[sw[(k, e, "control")], j] for k in seqs for e in POOL])
        oe = sign_flip_p([out[ol[(k, "extreme")], j] - out[ol[(k, "typical")], j] for k in seqs])
        om = sign_flip_p([out[ol[(k, "moderate")], j] - out[ol[(k, "typical")], j] for k in seqs])
        md.append(f"| {name} | {ctrl:+.2f} | " + " | ".join(cells) + f" | {mp:+.2f} (p={pp:.3f}) | {a:.2f} | "
                  f"{oe[0]:+.2f} (p={oe[1]:.3f}) | {om[0]:+.2f} (p={om[1]:.3f}) |")
        rows_out.append({"model": model, "format": name, "kind": kind, "control": ctrl, "pooled": mp, "pooled_p": pp,
                         "auc": a, "outlier_extreme": oe[0], "outlier_extreme_p": oe[1]})
    # inside
    md.append("\n### A. The same score read inside (jump - control, pooled cuts)\n")
    md.append("Mean over the layers of each third, and the layer with the largest |difference| (its value). "
              "The last block is the output.\n")
    md.append("| lens | format | early | middle | late | max layer (value, p) |\n|---|---|---|---|---|---|")
    th = thirds(L)
    for lens, s in sc.items():
        for j, (name, fam, kind, changed) in enumerate(fmts):
            if kind == "int":
                continue
            D = np.array([np.mean([s[sw[(k, e, "jump")], j] - s[sw[(k, e, "control")], j] for e in POOL], 0) for k in seqs])  # (K, L)
            band = [np.nanmean(D[:, ls]) if ls else np.nan for ls in th.values()]
            mean_l = np.nanmean(D, 0)
            lmax = int(np.nanargmax(np.abs(mean_l)))
            mp, pp = sign_flip_p(D[:, lmax])
            md.append(f"| {lens} | {name} | " + " | ".join(f"{b:+.2f}" for b in band) + f" | L{lmax} ({mp:+.2f}, p={pp:.3f}) |")
    # probes
    res = np.load(os.path.join(R, "ask_resid.npy"), mmap_mode="r")
    pl = meta["probe_layers"]
    md.append("\n### A. Probes: is the jump decodable at the last prompt token?\n")
    md.append("Difference-of-means direction fitted on the other 19 streams; AUC of held-out scores (jump vs control, "
              "e in {1, 2, 4, 8, 32}), and trained on those cuts and tested at e = 0. Middle probe layer and best.\n")
    md.append("| format | AUC middle layer | best layer (AUC) | e=0 AUC, best layer | output AUC |\n|---|---|---|---|---|")
    for j, (name, fam, kind, changed) in enumerate(fmts):
        aucs, aucs0 = [], []
        for li, layer in enumerate(pl):
            Xj = {(k, e): np.asarray(res[sw[(k, e, "jump")], j, li], np.float32) for k in seqs for e in (0,) + POOL}
            Xc = {(k, e): np.asarray(res[sw[(k, e, "control")], j, li], np.float32) for k in seqs for e in (0,) + POOL}
            pos, neg, pos0, neg0 = [], [], [], []
            for k in seqs:
                tr = [q for q in seqs if q != k]
                mj = np.mean([Xj[(q, e)] for q in tr for e in POOL], 0)
                mc = np.mean([Xc[(q, e)] for q in tr for e in POOL], 0)
                w, mid = mj - mc, (mj + mc) / 2
                pos += [float((Xj[(k, e)] - mid) @ w) for e in POOL]
                neg += [float((Xc[(k, e)] - mid) @ w) for e in POOL]
                pos0.append(float((Xj[(k, 0)] - mid) @ w))
                neg0.append(float((Xc[(k, 0)] - mid) @ w))
            aucs.append(auc(pos, neg))
            aucs0.append(auc(pos0, neg0))
        mid_i = len(pl) // 2
        b = int(np.nanargmax(aucs))
        b0 = int(np.nanargmax(aucs0))
        oa = next(r["auc"] for r in rows_out if r["model"] == model and r["format"] == name)
        md.append(f"| {name} | {aucs[mid_i]:.2f} (L{pl[mid_i]}) | L{pl[b]} ({aucs[b]:.2f}) | L{pl[b0]} ({aucs0[b0]:.2f}) | {oa:.2f} |")


def part_b(model, R, md):
    path = os.path.join(R, "inject_e4.csv.gz")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    md.append("\n### B. Injection at one layer, every tail token (e = 4)\n")
    md.append("G = natural gap (jump - control, no change). Shares of G: transplant = diff_pos +1 on control; removal "
              "= diff_pos -1 on jump; proj = setting the diff direction's component to the other arm's mean (on "
              "control, on jump). x4 = effect of diff_pos +4 on control, against a random direction of the same norm "
              "(randm x4). steer = surprise-word lens direction at c = +1 (control words in brackets). Effects in the "
              "score's units, mean over streams; * p < 0.05 (sign-flip). Where |G| is small (the question formats) the "
              "shares are noisy ratios; read the x4 column against randm x4 there.\n")
    df["score"] = np.where(df.format.isin(["q_surprise", "q_change"]), df.lp_yes - df.lp_no,
                           np.where(df.format == "wait", df.set_surprise - df.set_normal,
                                    df.int_mean if "int_mean" in df else np.nan))
    key = ["seq", "arm", "format", "layer"]
    base = df[df.cond == "none"].set_index(key).score
    df["eff"] = df.score.values - base.reindex(pd.MultiIndex.from_frame(df[key])).values
    for fmt, g in df.groupby("format", sort=False):
        md.append(f"\n**{fmt}**" + (" (score = predicted mean; the sd is in the csv)" if fmt == "comma" else "") + "\n")
        md.append("| layer | G | transplant | removal | proj control | proj jump | x4 | randm x4 | steer surprise (ctrlw) |\n|---|---|---|---|---|---|---|---|---|")
        for layer, h in g.groupby("layer"):
            nb = h[h.cond == "none"].pivot(index="seq", columns="arm", values="score")
            G = (nb.jump - nb.control).mean()

            def eff(cond, coef, arm):
                v = h[(h.cond == cond) & (h.coef == coef) & (h.arm == arm)].set_index("seq").eff
                return v

            def cell(v, share=None):
                m_, p_ = sign_flip_p(v.values)
                s = f"{m_ / share:+.2f}" if share else f"{m_:+.2f}"
                return s + ("*" if p_ < 0.05 else "")
            tr = cell(eff("diff_pos", 1.0, "control"), G)
            rm = cell(-eff("diff_pos", -1.0, "jump"), G)
            pc = cell(eff("proj", 1.0, "control"), G)
            pj = cell(-eff("proj", 1.0, "jump"), G)
            x4 = cell(eff("diff_pos", 4.0, "control"))
            r4 = cell(eff("randm", 4.0, "control"))
            st = cell(eff("surprise", 1.0, "control")) if (h.cond == "surprise").any() else "-"
            sc_ = cell(eff("ctrlw", 1.0, "control")) if (h.cond == "ctrlw").any() else "-"
            md.append(f"| {layer} | {G:+.2f} | {tr} | {rm} | {pc} | {pj} | {x4} | {r4} | {st} ({sc_}) |")


def part_d(model, R, RD, md, rows_d):
    """The 2 x 2 change test: the up block (R) and the mirror block (RD)."""
    meta, mdn = json.load(open(os.path.join(R, "ask_meta.json"))), json.load(open(os.path.join(RD, "ask_meta.json")))
    assert [f[0] for f in meta["formats"]] == [f[0] for f in mdn["formats"]]
    assert meta["answer_tokens"] == mdn["answer_tokens"] and meta["probe_layers"] == mdn["probe_layers"]
    L, fmts, pl = meta["n_blocks"], meta["formats"], meta["probe_layers"]
    up = {(c["seq"], c["e"], c["arm"]): i for i, c in enumerate(meta["contexts"]) if c["block"] == "switch"}
    dn = {(c["seq"], c["e"], c["arm"]): i for i, c in enumerate(mdn["contexts"]) if c["block"] == "switch_down"}
    seqs = sorted({k for k, _, _ in dn})
    cells = {"LL": (0, "control"), "LH": (0, "jump"), "HL": (1, "jump"), "HH": (1, "control")}

    def idx(cell, k, e):
        b, arm = cells[cell]
        return (up if b == 0 else dn)[(k, e, arm)]

    sc = {}
    for lens in meta["lenses"]:
        pair = []
        for M, Rx in ((meta, R), (mdn, RD)):
            A = np.load(os.path.join(Rx, f"ask_ans_{lens}.npy"))
            S = np.load(os.path.join(Rx, f"ask_sets_{lens}.npy"))
            pair.append(scores(M, A, S, np.load(os.path.join(Rx, "ask_intstats.npy"))))
        sc[lens] = pair

    def get(s, cell, k, e, j, layer):
        b = cells[cell][0]
        return s[b][idx(cell, k, e), j, layer]

    def effects(s, j, layer, es):
        """Per stream (K, 3): change, recent, history, averaged over the cuts es."""
        out = []
        for k in seqs:
            v = {c: np.mean([get(s, c, k, e, j, layer) for e in es]) for c in cells}
            out.append([(v["LH"] + v["HL"] - v["LL"] - v["HH"]) / 2, (v["LH"] + v["HH"] - v["LL"] - v["HL"]) / 2,
                        (v["HL"] + v["HH"] - v["LL"] - v["LH"]) / 2])
        return np.array(out)

    md.append("\n### D. The 2 x 2 change test (history low / high x recent numbers low / high)\n")
    md.append("Cells: LL = low throughout (the up control), LH = up jump, HL = down jump, HH = high throughout. "
              "Change = (LH + HL - LL - HH) / 2, the part of the score a reader of the recent numbers alone cannot "
              "produce; recent = (LH + HH - LL - HL) / 2; history = (HL + HH - LL - LH) / 2. Output score at the last "
              "prompt token, pooled cuts (e = 0 separately); sign-flip p over streams. AUC: change cells vs the "
              "others; same-tail AUC: up jump vs high throughout (identical recent numbers, different history).\n")
    md.append("| format | change (pooled) | change e=0 | recent | history | AUC change | same-tail AUC (LH vs HH) |\n|---|---|---|---|---|---|---|")
    s = sc["logit"]
    for j, (name, fam, kind, changed) in enumerate(fmts):
        E = effects(s, j, L - 1, POOL)
        E0 = effects(s, j, L - 1, (0,))
        r = [sign_flip_p(E[:, i]) for i in range(3)]
        r0 = sign_flip_p(E0[:, 0])
        pos = [get(s, c, k, e, j, L - 1) for c in ("LH", "HL") for k in seqs for e in POOL]
        neg = [get(s, c, k, e, j, L - 1) for c in ("LL", "HH") for k in seqs for e in POOL]
        a = auc(pos, neg)
        a_st = auc([get(s, "LH", k, e, j, L - 1) for k in seqs for e in POOL],
                   [get(s, "HH", k, e, j, L - 1) for k in seqs for e in POOL])
        f = lambda m_, p_: f"{m_:+.2f} (p={p_:.3f})"  # noqa: E731
        md.append(f"| {name} | {f(*r[0])} | {f(*r0)} | {f(*r[1])} | {f(*r[2])} | {a:.2f} | {a_st:.2f} |")
        rows_d.append({"model": model, "format": name, "kind": kind, "change": r[0][0], "change_p": r[0][1],
                       "change_e0": r0[0], "change_e0_p": r0[1], "recent": r[1][0], "recent_p": r[1][1],
                       "history": r[2][0], "history_p": r[2][1], "auc_change": a, "auc_same_tail": a_st})
    md.append("\n**Inside** (change effect by layer third, pooled cuts; max layer and its p)\n")
    md.append("| lens | format | early | middle | late | max layer (value, p) |\n|---|---|---|---|---|---|")
    th = thirds(L)
    for lens, s in sc.items():
        for j, (name, fam, kind, changed) in enumerate(fmts):
            if kind == "int":
                continue
            D = np.stack([effects(s, j, layer, POOL)[:, 0] for layer in range(L)], 1)  # (K, L)
            band = [np.nanmean(D[:, ls]) if ls else np.nan for ls in th.values()]
            lmax = int(np.nanargmax(np.abs(np.nanmean(D, 0))))
            mp, pp = sign_flip_p(D[:, lmax])
            md.append(f"| {lens} | {name} | " + " | ".join(f"{b:+.2f}" for b in band) + f" | L{lmax} ({mp:+.2f}, p={pp:.3f}) |")
    # probes
    res = [np.load(os.path.join(Rx, "ask_resid.npy"), mmap_mode="r") for Rx in (R, RD)]
    md.append("\n**Probes at the last prompt token** (difference of means on 19 streams, held-out stream scored)\n")
    md.append("Change probe: change cells vs the others (pooled cuts; e = 0 tested with the same direction). "
              "Up-to-down transfer: the up direction (up jump - up control) scored on the down pair (down jump vs "
              "high throughout): near 1 = it reads change, near 0 = it reads the recent numbers. Same-tail: the up "
              "direction on up jump vs high throughout (above 0.5 = it uses the history).\n")
    md.append("| format | change AUC, middle layer | change AUC, best layer | change e=0, best | up-to-down transfer, middle (min, max over layers) | same-tail, middle |\n|---|---|---|---|---|---|")
    for j, (name, fam, kind, changed) in enumerate(fmts):
        ach, ach0, atr, ast = [], [], [], []
        for li, layer in enumerate(pl):
            X = {(c, k, e): np.asarray(res[cells[c][0]][idx(c, k, e), j, li], np.float32)
                 for c in cells for k in seqs for e in (0,) + POOL}
            pos, neg, pos0, neg0, tp, tn, sp, sn = [], [], [], [], [], [], [], []
            for k in seqs:
                tr = [q for q in seqs if q != k]
                m = {c: np.mean([X[(c, q, e)] for q in tr for e in POOL], 0) for c in cells}
                w = (m["LH"] + m["HL"] - m["LL"] - m["HH"]) / 2
                mid = sum(m.values()) / 4
                pos += [float((X[(c, k, e)] - mid) @ w) for c in ("LH", "HL") for e in POOL]
                neg += [float((X[(c, k, e)] - mid) @ w) for c in ("LL", "HH") for e in POOL]
                pos0 += [float((X[(c, k, 0)] - mid) @ w) for c in ("LH", "HL")]
                neg0 += [float((X[(c, k, 0)] - mid) @ w) for c in ("LL", "HH")]
                wu, mu_ = m["LH"] - m["LL"], (m["LH"] + m["LL"]) / 2
                tp += [float((X[("HL", k, e)] - mu_) @ wu) for e in POOL]
                tn += [float((X[("HH", k, e)] - mu_) @ wu) for e in POOL]
                sp += [float((X[("LH", k, e)] - mu_) @ wu) for e in POOL]
                sn += [float((X[("HH", k, e)] - mu_) @ wu) for e in POOL]
            ach.append(auc(pos, neg))
            ach0.append(auc(pos0, neg0))
            atr.append(auc(tp, tn))
            ast.append(auc(sp, sn))
        mi = len(pl) // 2
        b, b0 = int(np.nanargmax(ach)), int(np.nanargmax(ach0))
        md.append(f"| {name} | {ach[mi]:.2f} (L{pl[mi]}) | L{pl[b]} ({ach[b]:.2f}) | L{pl[b0]} ({ach0[b0]:.2f}) | "
                  f"{atr[mi]:.2f} ({np.nanmin(atr):.2f}, {np.nanmax(atr):.2f}) | {ast[mi]:.2f} |")
        row = next(r for r in rows_d if r["model"] == model and r["format"] == name)
        row.update({"probe_change_mid": ach[mi], "probe_change_best": ach[b], "probe_change_best_layer": pl[b],
                    "probe_change_e0_best": ach0[b0], "probe_transfer_mid": atr[mi], "probe_same_tail_mid": ast[mi],
                    "probe_change_by_layer": json.dumps([round(x, 3) for x in ach]),
                    "probe_transfer_by_layer": json.dumps([round(x, 3) for x in atr])})


def part_c(model, R, md):
    path = os.path.join(R, "ablate_out.npy")
    if not os.path.exists(path):
        return
    info = json.load(open(os.path.join(R, "ablate_info.json")))
    O = np.load(path)  # (V, S, K, T, 2)
    sig = np.array(info["sigmas"], float)
    t0 = min(200, O.shape[3] // 2)
    sd = O[:, :, :, t0:, 1].mean(3)  # (V, S, K)
    md.append(f"\n### C. Ablation of the magnitude-word direction (layers {info['layers']})\n")
    md.append(f"Predicted sd over numbers {t0}..{O.shape[3] - 1}, mean over {O.shape[2]} streams, by sigma; slope of "
              "log sd on log sigma (95% bootstrap interval over streams) and its change from no ablation.\n")
    md.append("| variant | " + " | ".join(f"s={int(s)}" for s in sig) + " | slope | change |\n|---|" + "---|" * (len(sig) + 2))
    rng = np.random.default_rng(0)
    K = sd.shape[2]
    boots = rng.integers(0, K, (2000, K))
    slope = lambda v: np.polyfit(np.log(sig), np.log(v), 1)[0]  # noqa: E731
    base_b = np.array([slope(sd[0][:, b].mean(1)) for b in boots])
    for vi, v in enumerate(info["variants"]):
        sl = slope(sd[vi].mean(1))
        bb = np.array([slope(sd[vi][:, b].mean(1)) for b in boots])
        ch = bb - base_b
        md.append(f"| {v} | " + " | ".join(f"{x:.1f}" for x in sd[vi].mean(1)) + f" | {sl:.3f} [{np.quantile(bb, 0.025):.3f}, "
                  f"{np.quantile(bb, 0.975):.3f}] | {sl - slope(sd[0].mean(1)):+.3f} [{np.quantile(ch, 0.025):+.3f}, {np.quantile(ch, 0.975):+.3f}] |")


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    md, rows_out, rows_d = ["# Experiment 11: the model notices, but does it say?\n"], [], []
    for model in MODELS:
        R = os.path.join(RAW, model)
        if not os.path.exists(os.path.join(R, "ask_meta.json")):
            continue
        part_a(model, R, md, rows_out)
        RD = os.path.join(RAW, "down", model)
        if os.path.exists(os.path.join(RD, "ask_meta.json")):
            part_d(model, R, RD, md, rows_d)
        part_b(model, R, md)
        part_c(model, R, md)
    md.append("\n(* p < 0.05, sign-flip over the 20 streams, not corrected for the number of cells.)")
    open(os.path.join(HERE, "results", "summary.md"), "w").write("\n".join(md) + "\n")
    pd.DataFrame(rows_out).to_csv(os.path.join(HERE, "results", "report_output.csv"), index=False)
    if rows_d:
        pd.DataFrame(rows_d).to_csv(os.path.join(HERE, "results", "change_2x2.csv"), index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
