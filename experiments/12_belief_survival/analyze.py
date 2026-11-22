"""Experiment 12 analysis (CPU): how the old belief dies.

For one model (arrays from experiment 10's run_streams.py on data/jumps.jsonl, ramps_fast.jsonl, ramps_slow.jsonl):
1. Switch time. For each jump arm, progress p(e) = (m(e) - before) / (after - before) of the predicted mean (before
   = e in [-100, -1], after = e in [200, 299]); e50 = first e >= 0 where its 3-number running mean reaches 0.5;
   95% bootstrap intervals over the 20 streams. The same for the ideal observer (experiment 10's, default settings)
   and for an exponential moving average of the numbers (EMA) whose rate is fitted to the model's jump curves.
2. Mixture or slide? Right after a jump an ideal change-point reader predicts a two-bump distribution (old level or
   new level), a moving average one bump that slides. Mass of the model's predicted distribution near the old level,
   the new level and halfway (each within DELTA / 4), for e = 0..30, against the observer's predictive distribution
   and against a single Gaussian with the model's own mean and sd. Bimodal = the halfway mass falls below the
   single Gaussian's (ratio < 1) when the mean is halfway.
3. Hysteresis. On the ramps (level slides 300 -> 700 -> 300, or the reverse), the true level at which the model's
   mean crosses 500 on the way up (mu_up) and on the way down (mu_down); loop width W = mu_up - mu_down. Any reader
   with memory lags, so the test is the model's W against the EMA's and the observer's W (bootstrap over streams).
   Also first ramp vs second ramp, and up vs down.
4. Layers. e50 of the lens's integer mean ("the belief in number words") per lens and layer for the 400 and 200
   jumps, with intervals; which layers switch first. A discrete-time hazard model of the switch (per stream, layer
   and arm; censored at e = 299): logit P(switch at e | not yet) on event-time bins, the layer third, log2(DELTA)
   and direction; 95% intervals from a bootstrap over streams.
5. Magnitude words (uncertainty) after the jump: peak excess over the pre-jump level and half-life, J-lens by layer
   third, against the output sd.
6. History length (when the history_t{tau} runs exist): a jump after tau = 25 .. 800 numbers at the old level. Switch
   time = first e >= 0 where the 3-number running mean of s (mean - 500) reaches 0 (the belief crosses the midpoint;
   up and down pooled with the sign s; censored at the last number). Model, observer, EMA (rate fitted on these
   runs), running mean (cumulative mean of the numbers: its switch time is about tau). Slope of log switch time on
   log tau (bootstrap over streams): 1 = running mean, 0 = moving average or change-point reader. The same for the
   lens's integer mean by layer, and the pre-jump predicted sd by tau.

    python experiments/12_belief_survival/analyze.py --model llama3.1-8b
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (DATA, DELTAS, HIGH, HIST_DELTAS, HIST_POST, HIST_TAUS, LOW, N_SEQ, PLATEAU, PRE,  # noqa: E402
                    RAMPS, RAW, SIGMA, TAU)

E = np.arange(-100, 300)
RNG = np.random.default_rng(0)
N_BOOT = 1000


def run3(v):
    v = np.asarray(v, float)
    return np.convolve(np.pad(v, 1, mode="edge"), np.ones(3) / 3, mode="valid")


def e50_of(curve):
    """curve over E -> e50 (NaN if never)."""
    before = curve[E < 0].mean()
    after = curve[E >= 200].mean()
    if not np.isfinite(after - before) or abs(after - before) < 1e-9:
        return np.nan
    p = run3((curve - before) / (after - before))
    ok = np.where((E >= 0) & (p >= 0.5))[0]
    return float(E[ok[0]]) if len(ok) else np.nan


def boot(M, f):
    """M (n_seq, ...) -> f(mean), 2.5% and 97.5% quantiles of f over bootstrap resamples of the streams."""
    pt = f(M.mean(0))
    bs = np.array([f(M[RNG.integers(0, len(M), len(M))].mean(0)) for _ in range(N_BOOT)])
    return pt, np.nanquantile(bs, 0.025), np.nanquantile(bs, 0.975)


def ema(X, alpha, start):
    """Exponential moving average of the numbers X (n, T), starting at `start` (n,): the prediction after number t."""
    out = np.empty(X.shape, float)
    m = start.astype(float)
    for t in range(X.shape[1]):
        m = alpha * X[:, t] + (1 - alpha) * m
        out[:, t] = m
    return out


def thirds(L):
    b = list(range(L // 3, (2 * L) // 3 + 1))
    return {"early": list(range(0, b[0])), "middle": b, "late": list(range(b[-1] + 1, L))}


def logit_fit(X, y, iters=50):
    """Logistic regression by Newton's method (X includes an intercept column)."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw", default=RAW)
    args = ap.parse_args()
    R = os.path.join(args.raw, args.model)
    RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    md = [f"# Experiment 12, {args.model}\n"]
    ob = np.load(os.path.join(DATA, "observer.npz"))
    streams = {n: [json.loads(x) for x in open(os.path.join(DATA, f"{n}.jsonl"))] for n in ("jumps", "ramps_fast", "ramps_slow")}

    # ------------------------------------------------------------------ jumps
    meta = json.load(open(os.path.join(R, "jumps_meta.json")))
    L, lenses = meta["n_blocks"], meta["lenses"]
    assert [(s["arm"], s["seq"]) for s in meta["streams"]] == [(s["arm"], s["seq"]) for s in streams["jumps"]]
    out = np.load(os.path.join(R, "jumps_out.npy"))
    dist = np.load(os.path.join(R, "jumps_out_dist.npy"), mmap_mode="r")
    arms = list(dict.fromkeys(s["arm"] for s in streams["jumps"]))
    idx = {a: np.array([i for i, s in enumerate(streams["jumps"]) if s["arm"] == a]) for a in arms}
    X = np.array([s["numbers"] for s in streams["jumps"]])
    t = TAU + E
    m = out[:, :, 0]
    on = ob["jumps/next_mean"]
    # EMA fitted to the model's jump curves (e in [-20, 100], all jump arms)
    jarms = [a for a in arms if a != "flat"]
    rows = np.concatenate([idx[a] for a in jarms])
    win = np.arange(TAU - 20, TAU + 101)
    start = X[:, :50].mean(1)
    best = min(((np.nanmean((ema(X[rows], al, start[rows])[:, win] - m[rows][:, win]) ** 2), al)
                for al in np.exp(np.linspace(np.log(0.005), np.log(0.9), 60))))
    alpha = best[1]
    E_ema = ema(X, alpha, start)
    md.append(f"## 1. Switch time\n\nEMA rate fitted to the model's jump curves: alpha = {alpha:.3f} (half-life "
              f"{np.log(0.5) / np.log(1 - alpha):.1f} numbers; RMSE {np.sqrt(best[0]):.1f}).\n")
    md.append("e50 of the predicted mean (95% bootstrap interval over streams):\n\n| arm | model | observer | EMA |\n|---|---|---|---|")
    e50_rows = []
    for a in jarms:
        r = [boot(v[idx[a]][:, t], e50_of) for v in (m, on, E_ema)]
        md.append(f"| {a} | " + " | ".join(f"{p:.0f} [{lo:.0f}, {hi:.0f}]" for p, lo, hi in r) + " |")
        e50_rows.append({"arm": a, **{f"{n}_{k}": v for n, x in zip(("model", "observer", "ema"), r) for k, v in zip(("e50", "lo", "hi"), x)}})
    pd.DataFrame(e50_rows).to_csv(os.path.join(RES, f"{args.model}_e50.csv"), index=False)

    # ------------------------------------------------------------------ 2. mixture or slide
    md.append("\n## 2. Mixture or slide\n\nMass near the old level / halfway / new level (each within DELTA/4) of "
              "the predicted distribution, mean over streams; 'single' = halfway mass of a Gaussian with the model's "
              "own mean and sd; observer = its predictive distribution.\n")
    kk = np.arange(1000)
    dt = list(ob["jumps/dist_t"])
    od = ob["jumps/dist"]  # (n, 32, 1000), after numbers TAU-1 .. TAU+30
    for dl in (400, 200):
        md.append(f"\n**DELTA = {dl}** (up and down pooled)\n\n| e | model old | model halfway | model new | single halfway | observer old | observer halfway | observer new |\n|---|---|---|---|---|---|---|---|")
        for e in (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 30):
            vals = []
            for a in (f"up{dl}", f"down{dl}"):
                s = 1 if a.startswith("up") else -1
                old, new = 500 - s * dl / 2, 500 + s * dl / 2
                reg = {"old": np.abs(kk - old) <= dl / 4, "mid": np.abs(kk - 500) <= dl / 4, "new": np.abs(kk - new) <= dl / 4}
                q = np.asarray(dist[idx[a], TAU + e], np.float64)
                mu, sd = out[idx[a], TAU + e, 0], out[idx[a], TAU + e, 1]
                g = np.exp(-0.5 * ((kk[None] - mu[:, None]) / sd[:, None]) ** 2)
                g /= g.sum(1, keepdims=True)
                o = od[idx[a], dt.index(TAU + e)]
                vals.append([q[:, reg["old"]].sum(1).mean(), q[:, reg["mid"]].sum(1).mean(), q[:, reg["new"]].sum(1).mean(),
                             g[:, reg["mid"]].sum(1).mean(), o[:, reg["old"]].sum(1).mean(), o[:, reg["mid"]].sum(1).mean(),
                             o[:, reg["new"]].sum(1).mean()])
            v = np.mean(vals, 0)
            md.append(f"| {e} | " + " | ".join(f"{x:.2f}" for x in v) + " |")

    # ------------------------------------------------------------------ 3. hysteresis
    md.append("\n## 3. Hysteresis on the ramps\n\nTrue level at which the mean crosses 500 on the way up (mu_up) and "
              "down (mu_down); loop width W = mu_up - mu_down; 95% bootstrap intervals over streams. The observer and "
              "the EMA (rate fitted on the jumps) read the same numbers.\n")
    md.append("| ramp | reader | mu_up | mu_down | W | first-ramp lag - second-ramp lag (numbers) |\n|---|---|---|---|---|---|")
    hyst = []
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for speed in ("fast", "slow"):
        name = f"ramps_{speed}"
        mt = json.load(open(os.path.join(R, f"{name}_meta.json")))
        assert [(s["arm"], s["seq"]) for s in mt["streams"]] == [(s["arm"], s["seq"]) for s in streams[name]]
        o2 = np.load(os.path.join(R, f"{name}_out.npy"))
        Xr = np.array([s["numbers"] for s in streams[name]])
        lev = np.array([s["level"] for s in streams[name]])
        readers = {"model": o2[:, :, 0], "observer": ob[f"{name}/next_mean"],
                   "EMA": ema(Xr, alpha, Xr[:, :50].mean(1))}
        r = RAMPS[speed]
        up0, top0, down0 = PRE, PRE + r, PRE + r + PLATEAU  # first ramp starts, plateau starts, second ramp starts
        for shape in ("updown", "downup"):
            rows_s = np.array([i for i, s in enumerate(streams[name]) if s["shape"] == shape])
            for rd, M in readers.items():
                def crossings(Mi, levi):
                    """per stream: (true level at the crossing on ramp 1, on ramp 2, lag 1, lag 2)."""
                    res = []
                    for mrow, lrow in zip(Mi, levi):
                        sm = run3(mrow)
                        out_ = []
                        for a, b, upward in ((up0, down0, shape == "updown"), (down0, len(sm), shape != "updown")):
                            seg = np.arange(a, b)
                            hit = seg[(sm[seg] >= 500) if upward else (sm[seg] <= 500)]
                            tc = hit[0] if len(hit) else np.nan
                            true_c = a + np.argmin(np.abs(lrow[a:b] - 500)) if r > 0 else a
                            out_ += [lrow[int(tc)] if np.isfinite(tc) else np.nan, tc - true_c if np.isfinite(tc) else np.nan]
                        res.append(out_)
                    return np.array(res)  # (n, 4): level1, lag1, level2, lag2
                C = crossings(M[rows_s], lev[rows_s])
                up_level = C[:, 0] if shape == "updown" else C[:, 2]
                dn_level = C[:, 2] if shape == "updown" else C[:, 0]
                Wv = up_level - dn_level
                lagdiff = C[:, 1] - C[:, 3]
                b = lambda v: (np.nanmean(v), *np.nanquantile([np.nanmean(v[RNG.integers(0, len(v), len(v))]) for _ in range(N_BOOT)], [0.025, 0.975]))  # noqa: E731
                cu, cd, cw, cl = b(up_level), b(dn_level), b(Wv), b(lagdiff)
                md.append(f"| {shape} {speed} | {rd} | {cu[0]:.0f} [{cu[1]:.0f}, {cu[2]:.0f}] | {cd[0]:.0f} [{cd[1]:.0f}, {cd[2]:.0f}] | "
                          f"{cw[0]:.0f} [{cw[1]:.0f}, {cw[2]:.0f}] | {cl[0]:+.1f} [{cl[1]:+.1f}, {cl[2]:+.1f}] |")
                hyst.append({"speed": speed, "shape": shape, "reader": rd, "mu_up": cu[0], "mu_down": cd[0], "W": cw[0],
                             "W_lo": cw[1], "W_hi": cw[2], "lagdiff": cl[0]})
                if shape == "updown":
                    ax = axs[0 if speed == "fast" else 1]
                    span = np.arange(PRE - 20, len(lev[0]) - 50)
                    ax.plot(lev[rows_s][:, span].mean(0), M[rows_s][:, span].mean(0), label=rd)
        ax = axs[0 if speed == "fast" else 1]
        ax.plot([LOW, HIGH], [LOW, HIGH], color="grey", lw=0.5)
        ax.set(title=f"{args.model}: up-down ramp, {speed}", xlabel="true level", ylabel="predicted mean")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{args.model}_hysteresis.png"), dpi=110)
    plt.close(fig)
    H = pd.DataFrame(hyst)
    H.to_csv(os.path.join(RES, f"{args.model}_hysteresis.csv"), index=False)
    # model W minus EMA W and minus observer W, paired by stream
    md.append("\nThe test: the model's W against the readers' W (positive = the model's loop is wider).\n")

    # ------------------------------------------------------------------ 4. layers
    md.append("\n## 4. Which layers switch first\n\ne50 of the lens's integer mean (95% interval). Only layers whose "
              "mean moves by at least DELTA / 4 are timed.\n")
    lay_rows, haz = [], []
    th = thirds(L)
    third_of = {l: n for n, ls in th.items() for l in ls}
    for lens in lenses:
        S = np.load(os.path.join(R, f"jumps_{lens}_scalars.npy"), mmap_mode="r")
        for a in [x for x in jarms if x.endswith(("400", "200"))]:
            dl = int(a[-3:])
            for layer in range(L):
                M = np.asarray(S[idx[a]][:, t, layer, 1], float)
                move = abs(M[:, E >= 200].mean() - M[:, E < 0].mean())
                if move < dl / 4:
                    continue
                p, lo, hi = boot(M, e50_of)
                lay_rows.append({"lens": lens, "arm": a, "layer": layer, "e50": p, "lo": lo, "hi": hi, "move": move})
                # per-stream switch times for the hazard model
                for si, row in enumerate(M):
                    ev = e50_of(row)
                    haz.append({"lens": lens, "arm": a, "seq": si, "layer": layer, "third": third_of[layer],
                                "e": ev if np.isfinite(ev) else 299, "event": int(np.isfinite(ev)), "delta": dl,
                                "down": int(a.startswith("down"))})
    LR = pd.DataFrame(lay_rows)
    LR.to_csv(os.path.join(RES, f"{args.model}_layers.csv"), index=False)
    oe = {a: boot(m[idx[a]][:, t], e50_of) for a in jarms}
    for lens in lenses:
        md.append(f"\n**{lens}** (output e50 in the header)\n\n| layer | " + " | ".join(f"{a} (out {oe[a][0]:.0f})" for a in jarms if a.endswith(("400", "200"))) + " |\n|---|" + "---|" * 4)
        sub = LR[LR.lens == lens]
        for layer in sorted(sub.layer.unique()):
            if layer % max(1, L // 16) and layer != L - 1:
                continue
            cells = []
            for a in [x for x in jarms if x.endswith(("400", "200"))]:
                r = sub[(sub.layer == layer) & (sub.arm == a)]
                cells.append(f"{r.e50.iloc[0]:.0f} [{r.lo.iloc[0]:.0f}, {r.hi.iloc[0]:.0f}]" if len(r) else "-")
            md.append(f"| {layer} | " + " | ".join(cells) + " |")
    HZ = pd.DataFrame(haz)
    HZ.to_csv(os.path.join(RES, f"{args.model}_switch_times.csv"), index=False)
    bins = [0, 1, 2, 3, 4, 8, 16, 32, 300]
    md.append("\nDiscrete-time hazard of the switch (J-lens if present, else logit lens; all timed layers; bins of e "
              f"{bins[:-1]}): coefficients on the log-odds of switching at e given not yet, 95% bootstrap interval over "
              "streams. Baseline: early third, DELTA = 200, up.\n")
    lens_h = "jlens" if "jlens" in lenses else "logit"
    hz = HZ[HZ.lens == lens_h]
    if len(hz) and hz.third.nunique() > 1:
        def design(df):
            Xs, ys, gs = [], [], []
            for _, r in df.iterrows():
                last = int(min(r.e, 299))
                for e in range(0, last + 1):
                    b = np.searchsorted(bins, e, side="right") - 1
                    x = np.zeros(len(bins) - 1)
                    x[b] = 1
                    Xs.append(np.concatenate([x, [r.third == "middle", r.third == "late", np.log2(r.delta / 200), r.down]]))
                    ys.append(int(e == last and r.event))
                    gs.append(r.seq)
            return np.array(Xs, float), np.array(ys, float), np.array(gs)
        # one row per (stream, layer, arm) in long format is large; subsample event times at the bin level
        Xh, yh, gh = design(hz)
        names = [f"e[{bins[i]},{bins[i + 1]})" for i in range(len(bins) - 1)] + ["middle third", "late third", "log2(DELTA/200)", "down"]
        bh = logit_fit(Xh, yh)
        seqs = np.unique(gh)
        bb = []
        for _ in range(200):
            pick = RNG.choice(seqs, len(seqs))
            rows_b = np.concatenate([np.where(gh == q)[0] for q in pick])
            bb.append(logit_fit(Xh[rows_b], yh[rows_b]))
        bb = np.array(bb)
        md.append("| term | coefficient | 95% interval |\n|---|---|---|")
        for i, n in enumerate(names):
            if n.startswith("e[") and i not in (0, 4, 7):
                continue
            md.append(f"| {n} | {bh[i]:+.2f} | [{np.quantile(bb[:, i], 0.025):+.2f}, {np.quantile(bb[:, i], 0.975):+.2f}] |")

    # ------------------------------------------------------------------ 5. magnitude words
    md.append("\n## 5. Magnitude words after the jump\n\nExcess of log P(magnitude) - log P(control) over its pre-jump "
              "level: peak (e 0..20) and half-life (first e after the peak where the 3-number running mean is below "
              "half the peak), J-lens by layer third; the output sd the same way.\n")
    sets = meta["sets"]
    mi, ci = sets.index("magnitude"), sets.index("control")

    def peak_half(curve):
        exc = curve - curve[E < 0].mean()
        ee = (E >= 0) & (E <= 20)
        pk = int(np.argmax(exc[ee]))
        pval = exc[ee][pk]
        e_pk = E[ee][pk]
        sm = run3(exc)
        after = np.where((E > e_pk) & (sm < pval / 2))[0]
        return pval, (E[after[0]] - e_pk) if len(after) else np.nan

    md.append("| arm | output sd peak, half-life | " + " | ".join(f"{b} peak, half-life" for b in th) + " |\n|---|---|---|---|---|")
    lens_m = "jlens" if "jlens" in lenses else "logit"
    Sm = np.load(os.path.join(R, f"jumps_{lens_m}_sets.npy"), mmap_mode="r")
    for a in jarms:
        sdc = out[idx[a]][:, t, 1].mean(0)
        cells = [("%.1f, %s" % peak_half(sdc))]
        for b, ls in th.items():
            if not ls:
                cells.append("-")
                continue
            v = (np.asarray(Sm[idx[a]][:, t, 0][:, :, ls, mi], np.float32) - np.asarray(Sm[idx[a]][:, t, 0][:, :, ls, ci], np.float32)).mean((0, 2))
            pk, hl = peak_half(v)
            cells.append(f"{pk:+.2f}, {hl}")
        md.append(f"| {a} | " + " | ".join(cells) + " |")
    history_section(R, RES, FIG, args.model, md)
    open(os.path.join(RES, f"{args.model}_summary.md"), "w").write("\n".join(md) + "\n")


def cross_e(curve):
    """curve (E_h,) of s (mean - 500) over e = -10 .. HIST_POST - 1 -> first e >= 0 with run3 >= 0 (NaN if never)."""
    p = run3(curve)
    ok = np.where((EH >= 0) & (p >= 0))[0]
    return float(EH[ok[0]]) if len(ok) else np.nan


EH = np.arange(-10, HIST_POST)


def history_section(R, RES, FIG, model, md):
    if not all(os.path.exists(os.path.join(R, f"history_t{tau}_out.npy")) for tau in HIST_TAUS):
        return
    ob = np.load(os.path.join(DATA, "observer_history.npz"))
    md.append("\n## 6. History length: does the old belief die more slowly the longer it held?\n")
    md.append("Switch time = first e >= 0 where the belief (predicted mean) crosses the midpoint 500 (3-number running "
              "mean; up and down pooled; 95% bootstrap interval over streams; '>199' = never within the "
              f"{HIST_POST} numbers). EMA rate fitted to the model on these runs; running mean = cumulative mean.\n")
    data, rows = {}, []
    for tau in HIST_TAUS:
        S = [json.loads(x) for x in open(os.path.join(DATA, f"history_t{tau}.jsonl"))]
        meta = json.load(open(os.path.join(R, f"history_t{tau}_meta.json")))
        assert [(s["arm"], s["seq"]) for s in meta["streams"]] == [(s["arm"], s["seq"]) for s in S]
        out = np.load(os.path.join(R, f"history_t{tau}_out.npy"))
        X = np.array([s["numbers"] for s in S], float)
        sg = np.array([s["sign"] for s in S], float)[:, None]
        dl = np.array([s["delta"] for s in S])
        seq = np.array([s["seq"] for s in S])
        lens = {}
        for ln in meta["lenses"]:
            lens[ln] = np.load(os.path.join(R, f"history_t{tau}_{ln}_scalars.npy"), mmap_mode="r")
        data[tau] = dict(S=S, out=out, X=X, sg=sg, dl=dl, seq=seq, lens=lens, obs=ob[f"t{tau}/next_mean"],
                         obs_sd=ob[f"t{tau}/next_sd"])
    # EMA rate fitted to the model on all history runs (e in [-10, 100])
    def ema_all(al):
        err = []
        for tau, d in data.items():
            st = d["X"][:, :10].mean(1)
            w = np.arange(tau - 10, tau + 101)
            err.append(((ema(d["X"], al, st)[:, w] - d["out"][:, w, 0]) ** 2).ravel())
        return np.nanmean(np.concatenate(err))
    grid = np.exp(np.linspace(np.log(0.005), np.log(0.9), 60))
    alpha = grid[int(np.argmin([ema_all(a) for a in grid]))]
    md.append(f"EMA alpha = {alpha:.3f} (half-life {np.log(0.5) / np.log(1 - alpha):.1f} numbers).\n")

    def curves(d, reader):
        if reader == "model":
            v = d["out"][:, :, 0]
        elif reader == "observer":
            v = d["obs"]
        elif reader == "EMA":
            v = ema(d["X"], alpha, d["X"][:, :10].mean(1))
        else:  # running mean
            v = np.cumsum(d["X"], 1) / np.arange(1, d["X"].shape[1] + 1)
        return d["sg"] * (v - 500)

    readers = ("model", "observer", "EMA", "running mean")
    fmt = lambda x: f"{x[0]:.0f} [{x[1]:.0f}, {x[2]:.0f}]" if np.isfinite(x[0]) else ">199"  # noqa: E731
    slope_rows = []
    for D in HIST_DELTAS:
        md.append(f"\n**DELTA = {D}**\n\n| tau | " + " | ".join(readers) + " | model pre-jump sd | observer pre-jump sd |\n|---|" + "---|" * (len(readers) + 2))
        per_seq = {}
        for tau, d in data.items():
            rws = np.where(d["dl"] == D)[0]
            t = tau + EH
            cells = []
            for rd in readers:
                c = curves(d, rd)[rws][:, t]  # (2 N_SEQ, E_h)
                # per stream: mean of its up and down curves
                M = np.stack([c[d["seq"][rws] == k].mean(0) for k in range(N_SEQ)])
                per_seq[(rd, tau)] = M
                r = boot(M, cross_e)
                cells.append(fmt(r))
                rows.append({"delta": D, "tau": tau, "reader": rd, "switch": r[0], "lo": r[1], "hi": r[2]})
            psd = d["out"][rws][:, tau - 10:tau, 1].mean()
            osd = d["obs_sd"][rws][:, tau - 10:tau].mean()
            md.append(f"| {tau} | " + " | ".join(cells) + f" | {psd:.1f} | {osd:.1f} |")
        # slope of log switch time on log tau
        md.append("\nSlope of log switch time on log tau (taus where it is finite; 95% bootstrap interval):\n")
        md.append("| reader | slope | interval | taus used |\n|---|---|---|---|")
        lt = np.log(np.array(HIST_TAUS, float))
        for rd in readers:
            def slope_of(idx):
                y = np.array([cross_e(per_seq[(rd, tau)][idx].mean(0)) for tau in HIST_TAUS])
                ok = np.isfinite(y) & (y > 0)
                return (np.polyfit(lt[ok], np.log(y[ok]), 1)[0] if ok.sum() >= 3 else np.nan), int(ok.sum())
            pt, n_used = slope_of(np.arange(N_SEQ))
            bs = np.array([slope_of(RNG.integers(0, N_SEQ, N_SEQ))[0] for _ in range(N_BOOT)])
            lo, hi = (np.nanquantile(bs, 0.025), np.nanquantile(bs, 0.975)) if np.isfinite(bs).any() else (np.nan, np.nan)
            md.append(f"| {rd} | {pt:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {n_used} |")
            slope_rows.append({"delta": D, "reader": rd, "slope": pt, "lo": lo, "hi": hi, "n_taus": n_used})
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"{model}_history.csv"), index=False)
    pd.DataFrame(slope_rows).to_csv(os.path.join(RES, f"{model}_history_slopes.csv"), index=False)
    # layers: the lens's integer mean, DELTA = 400 and 200
    for ln in data[HIST_TAUS[0]]["lens"]:
        Lb = data[HIST_TAUS[0]]["lens"][ln].shape[2]
        layers = list(range(Lb // 2, Lb, 2)) + ([Lb - 1] if (Lb - 1 - Lb // 2) % 2 else [])
        for D in HIST_DELTAS:
            md.append(f"\n**{ln}, DELTA = {D}**: switch time of the lens's integer mean by layer (crossing 500; only "
                      "layers whose mean moves by at least DELTA / 4 at tau = 400)\n")
            md.append("| layer | " + " | ".join(f"tau {tau}" for tau in HIST_TAUS) + " |\n|---|" + "---|" * len(HIST_TAUS))
            for layer in layers:
                cells = []
                d4 = data[400]
                rws4 = np.where(d4["dl"] == D)[0]
                v4 = d4["sg"][rws4] * (np.asarray(d4["lens"][ln][rws4, :, layer, 1]) - 500)
                move = v4[:, 400 + 150:400 + 200].mean() - v4[:, 400 - 10:400].mean()
                if not move >= D / 4:
                    continue
                for tau, d in data.items():
                    rws = np.where(d["dl"] == D)[0]
                    v = d["sg"][rws] * (np.asarray(d["lens"][ln][rws, :, layer, 1]) - 500)
                    M = np.stack([v[d["seq"][rws] == k].mean(0) for k in range(N_SEQ)])[:, tau + EH]
                    cells.append(fmt(boot(M, cross_e)))
                md.append(f"| {layer} | " + " | ".join(cells) + " |")
    # figure
    fig, axes = plt.subplots(1, len(HIST_DELTAS), figsize=(5 * len(HIST_DELTAS), 3.6), sharey=True)
    df = pd.DataFrame(rows)
    for ax, D in zip(np.atleast_1d(axes), HIST_DELTAS):
        for rd, mk in zip(readers, ("o", "s", "^", "v")):
            g = df[(df.delta == D) & (df.reader == rd)]
            y = g.switch.fillna(HIST_POST).values
            ax.plot(g.tau, y, marker=mk, lw=2, label=rd)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("history length tau (numbers at the old level)")
        ax.set_title(f"jump of {D}")
        ax.axhline(HIST_POST, color="0.6", lw=1, ls=":")
    np.atleast_1d(axes)[0].set_ylabel("switch time (numbers after the jump)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{model}_history.png"), dpi=150)
    plt.close(fig)
    print("\n".join(md))


if __name__ == "__main__":
    main()
