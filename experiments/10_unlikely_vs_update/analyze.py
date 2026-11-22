"""Experiment 10 analysis (CPU): which kind of surprise does the inside react to?

Reads run_streams.py's and run_cues.py's arrays for one model and data/observer.npz. Writes results/<model>_*.md
and .csv, and figures/<model>_*.png.

1. Behaviour (go / no-go, fixed before the run). The model's guess must ignore the outlier (one number after it,
   the mean has moved less than 20% of the way to the outlier's level) and follow the drift (slope of its mean on
   the true level during the drift >= 0.5). Also: which observer setting predicts the model's mean best (chosen on
   behaviour only; the regressions use it, and the default too).
2. Signals, read at the comma after every number and at the number's own token (lens log-probabilities of word
   sets, as log-ratios): surprise - normal, change - normal, magnitude - control, wide - narrow. Event-time curves
   of each arm minus control, per lens and layer, averaged over the early, middle and late thirds of the layers.
3. Design contrasts (no observer needed):
   lasting   jump - control and drift - control over e = 100..199 (the belief has moved 5 sd; nothing is
             surprising any more). Non-zero: the signal tracks how far the belief is from the old one.
   noise     noise - control over e = 20..199 (more uncertainty, same level).
   profile   over e = 0..10 and the four arms, the correlation of the signal's arm - control curve with the
             observer's surprisal and Bayesian surprise curves (same lag, and one number later).
4. Regression (secondary: the two surprises correlate at 0.57 within twins under the default observer). Signal on
   z(surprisal), z(Bayesian surprise), z(deviation), z(uncertainty), with (stream, number) fixed effects, numbers
   TAU - 5 .. TAU + 40, errors clustered by stream; and with the surprisal and Bayesian surprise of the number
   before added.
5. Cue readouts ("Wait,": surprise - normal; "The last number is": change - normal) at the last cue token: arm -
   control by cut, and the regression across arms and cuts with (stream, cut) fixed effects.

    python experiments/10_unlikely_vs_update/analyze.py --model llama3.1-8b
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
sys.path.append(os.path.join(os.path.dirname(HERE), "08_surprise"))
from common import ARMS, CUTS, DATA, DELTA, DRIFT, MU0, N_SEQ, RAW, SIGMA0, TAU  # noqa: E402
from stats import boot_mean_ci, ols_fe_cluster, sign_flip_p  # noqa: E402  (experiment 08's)

SIGNALS = {"surprise": ("surprise", "normal"), "change": ("change", "normal"), "magnitude": ("magnitude", "control"),
           "wide": ("wide", "narrow")}
E_LO, E_HI = -5, 199
REG_WIN = (TAU - 5, TAU + 40)
POS = ("comma", "number")


def thirds(L):
    b = list(range(L // 3, (2 * L) // 3 + 1))
    return {"early": list(range(0, b[0])), "middle": b, "late": list(range(b[-1] + 1, L))}


def zs(v):
    v = np.asarray(v, float)
    return (v - np.nanmean(v)) / np.nanstd(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw", default=RAW)
    args = ap.parse_args()
    R = os.path.join(args.raw, args.model)
    RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    md = [f"# Experiment 10, {args.model}\n"]

    meta = json.load(open(os.path.join(R, "streams_meta.json")))
    L, lenses, T = meta["n_blocks"], meta["lenses"], meta["T"]
    row = {(s["arm"], s["seq"]): i for i, s in enumerate(meta["streams"])}
    order = np.array([[row[(a, k)] for k in range(N_SEQ)] for a in ARMS])  # (A, K)
    sign = np.array([1 if k % 2 == 0 else -1 for k in range(N_SEQ)])
    out = np.load(os.path.join(R, "streams_out.npy"))[order]  # (A, K, T, 5)
    ob = np.load(os.path.join(DATA, "observer.npz"))
    settings = sorted({k.split("/")[0] for k in ob.files if "/" in k})
    O = {s: {k.split("/")[1]: ob[k] for k in ob.files if k.startswith(s + "/")} for s in settings}
    a_of = {a: i for i, a in enumerate(ARMS)}
    e_all = np.arange(T) - TAU
    # the numbers and the true level of every arm, (A, K, T)
    st = [json.loads(x) for x in open(os.path.join(DATA, "streams.jsonl"))]
    Xn = np.zeros((len(ARMS), N_SEQ, T))
    for s_ in st:
        Xn[a_of[s_["arm"]], s_["seq"]] = s_["numbers"][:T]
    tt_ = np.arange(T)[None]
    sg = sign[:, None]
    LV = np.stack([np.full((N_SEQ, T), MU0), MU0 + sg * DELTA * (tt_ >= TAU), np.full((N_SEQ, T), MU0),
                   MU0 + sg * DELTA * np.clip((tt_ - TAU + 1) / DRIFT, 0, 1), np.full((N_SEQ, T), MU0)]).astype(float)
    # the level-matched arm (always at the new level), when it has been run
    lev_ok = os.path.exists(os.path.join(R, "streams_level_meta.json"))
    if lev_ok:
        lmeta = json.load(open(os.path.join(R, "streams_level_meta.json")))
        lorder = np.array([{s_["seq"]: i for i, s_ in enumerate(lmeta["streams"])}[k] for k in range(N_SEQ)])

    # ------------------------------------------------------------------ 1. behaviour
    m, sd = out[..., 0], out[..., 1]
    rel = (m - m[a_of["control"]][None]) * sign[None, :, None] / DELTA  # fraction of the way to the new level
    md.append("## 1. Behaviour\n")
    md.append("Mean of the predicted next number, arm minus control, as a fraction of the way to the jump's level "
              "(signed so that +1 is the new level), at the comma after number TAU + e:\n")
    md.append("| arm | " + " | ".join(f"e={e}" for e in (0, 1, 2, 4, 8, 32, 100, 199)) + " |\n|---|" + "---|" * 8)
    for a in ARMS[1:]:
        md.append(f"| {a} | " + " | ".join(f"{rel[a_of[a], :, TAU + e].mean():.2f}" for e in (0, 1, 2, 4, 8, 32, 100, 199)) + " |")
    out_e1 = rel[a_of["outlier"], :, TAU + 1].mean()
    lvl = MU0 + sign[:, None] * DELTA * np.clip((np.arange(T)[None] - TAU + 1) / DRIFT, 0, 1)
    dw = np.arange(TAU, TAU + DRIFT + 50)
    x = (lvl[:, dw] - MU0).ravel()
    y = (m[a_of["drift"]][:, dw] - m[a_of["control"]][:, dw]).ravel()
    slope = np.polyfit(x, y, 1)[0]
    yo = (O["default"]["next_mean"][a_of["drift"]][:, dw] - O["default"]["next_mean"][a_of["control"]][:, dw]).ravel()
    slope_obs = np.polyfit(x, yo, 1)[0]
    go = bool(out_e1 < 0.2 and slope >= 0.5)
    md.append(f"\nGo / no-go: one number after the outlier the mean is {out_e1:.2f} of the way (< 0.2 needed); "
              f"during the drift its slope on the true level is {slope:.2f} (>= 0.5 needed; default observer "
              f"{slope_obs:.2f}). **{'GO' if go else 'NO-GO'}**\n")
    win = np.arange(TAU - 5, TAU + 150)
    fit = []
    for s in settings:
        r_m = np.sqrt(np.nanmean((m[:, :, win] - O[s]["next_mean"][:, :, win]) ** 2))
        r_s = np.sqrt(np.nanmean((sd[:, :, win] - O[s]["next_sd"][:, :, win]) ** 2))
        fit.append((s, r_m, r_s))
    best = min(fit, key=lambda f: f[1])[0]
    md.append("Which observer predicts the model's guess best (numbers TAU - 5 .. TAU + 149, all arms; RMSE of the "
              "predicted mean and sd):\n\n| observer | RMSE mean | RMSE sd |\n|---|---|---|")
    md += [f"| {s} | {a:.1f} | {b:.1f} |" for s, a, b in fit]
    md.append(f"\nBest: **{best}** (used below with the default).\n")
    noise_ratio = sd[a_of["noise"], :, TAU + 50 : TAU + 200].mean() / sd[a_of["control"], :, TAU + 50 : TAU + 200].mean()
    md.append(f"Noise arm: the model's predicted sd over e = 50..199 is {noise_ratio:.2f} times control's (the true "
              f"sd is 3 times; default observer "
              f"{O['default']['next_sd'][a_of['noise'], :, TAU + 50:TAU + 200].mean() / O['default']['next_sd'][a_of['control'], :, TAU + 50:TAU + 200].mean():.2f}).\n")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for a in ARMS:
        axs[0].plot(e_all[TAU - 20 :], rel[a_of[a], :, TAU - 20 :].mean(0), label=a)
        axs[1].plot(e_all[TAU - 20 :], sd[a_of[a], :, TAU - 20 :].mean(0), label=a)
    axs[0].set(xlabel="e (numbers after the event)", ylabel="mean, fraction of the way", title=f"{args.model}: guess")
    axs[1].set(xlabel="e", ylabel="predicted sd", title="spread")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{args.model}_behaviour.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------ observer regressors
    def regs(s):
        o = O[s]
        dev = np.abs(o["level_after"] - o["level_after"][:, :, TAU - 1 : TAU]) / SIGMA0
        lag = lambda v: np.concatenate([np.full(v.shape[:2] + (1,), np.nan), v[:, :, :-1]], 2)  # noqa: E731
        return {"surprisal": o["surprisal"], "bayes_kl": o["bayes_kl"], "deviation": dev, "uncertainty": o["next_sd"],
                "surprisal_lag": lag(o["surprisal"]), "bayes_kl_lag": lag(o["bayes_kl"]),
                "x": Xn, "x_sq": (Xn - 500.0) ** 2, "level": LV}

    REGS = {s: regs(s) for s in settings}

    # ------------------------------------------------------------------ 2-4. signals at every number
    sets_names = meta["sets"]
    rows_c, rows_r, curves = [], [], {}
    th = thirds(L)
    E = np.arange(E_LO, E_HI + 1)
    for lens in lenses:
        S = np.load(os.path.join(R, f"streams_{lens}_sets.npy"), mmap_mode="r")
        SL = np.load(os.path.join(R, f"streams_level_{lens}_sets.npy"), mmap_mode="r") if lev_ok else None
        for sig, (pos_set, neg_set) in SIGNALS.items():
            i1, i0 = sets_names.index(pos_set), sets_names.index(neg_set)
            for pi, pos in enumerate(POS):
                Y = (np.asarray(S[:, :, pi, :, i1], np.float32) - np.asarray(S[:, :, pi, :, i0], np.float32))[order]  # (A, K, T, L)
                D = Y - Y[a_of["control"]][None]  # arm - control
                YL = ((np.asarray(SL[:, :, pi, :, i1], np.float32) - np.asarray(SL[:, :, pi, :, i0], np.float32))[lorder]
                      if lev_ok else None)  # (K, T, L)
                for band, ls in list(th.items()) + [(f"L{l}", [l]) for l in range(L)]:
                    if not ls:
                        continue
                    Yb, Db = Y[..., ls].mean(-1), D[..., ls].mean(-1)  # (A, K, T)
                    if not band.startswith("L"):
                        curves[(lens, sig, pos, band)] = Db[:, :, TAU + E].mean(1)
                    rec = {"lens": lens, "signal": sig, "pos": pos, "band": band}
                    for name, lo, hi in (("after_outlier_e1_3", 1, 3), ("after_outlier_e4_10", 4, 10)):
                        est, p = sign_flip_p(Db[a_of["outlier"], :, TAU + lo : TAU + hi + 1].mean(1))
                        rec[name], rec[name + "_p"] = est, p
                    for name, arm in (("jump_vs_level_late", "jump"), ("drift_vs_level_late", "drift")):
                        if lev_ok:
                            YLb = YL[..., ls].mean(-1)
                            est, p = sign_flip_p((Yb[a_of[arm], :, TAU + 100 : TAU + 200] - YLb[:, TAU + 100 : TAU + 200]).mean(1))
                        else:
                            est, p = np.nan, np.nan
                        rec[name], rec[name + "_p"] = est, p
                    for name, arm, lo, hi in (("lasting_jump", "jump", 100, 199), ("lasting_drift", "drift", 100, 199),
                                              ("noise", "noise", 20, 199), ("event_jump_e0", "jump", 0, 0),
                                              ("jump_minus_outlier_e1_3", None, 1, 3)):
                        if arm is None:
                            d = (Yb[a_of["jump"], :, TAU + lo : TAU + hi + 1] - Yb[a_of["outlier"], :, TAU + lo : TAU + hi + 1]).mean(1)
                        else:
                            d = Db[a_of[arm], :, TAU + lo : TAU + hi + 1].mean(1)
                        est, p = sign_flip_p(d)
                        rec[name], rec[name + "_p"] = est, p
                    # profile: correlation over e = 0..10 and arms of arm - control with the observer's curves
                    o = REGS["default"]
                    ee = np.arange(0, 11)
                    yv = Db[1:, :, TAU + ee].mean(1).ravel()
                    for rn in ("surprisal", "bayes_kl"):
                        ov = o[rn] - o[rn][a_of["control"]][None]
                        rec[f"profile_r_{rn}"] = np.corrcoef(yv, ov[1:, :, TAU + ee].mean(1).ravel())[0, 1]
                        rec[f"profile_r_{rn}_lag"] = np.corrcoef(yv, ov[1:, :, TAU + ee - 1].mean(1).ravel())[0, 1]
                    rows_c.append(rec)
                    # regression (bands only, both observer settings)
                    if band.startswith("L") and int(band[1:]) % max(1, L // 16):
                        continue
                    t = np.arange(*REG_WIN)
                    kk, tt = np.meshgrid(np.arange(N_SEQ), t, indexing="ij")
                    y = Yb[:, :, t].ravel()
                    fe = np.stack([np.broadcast_to(kk, (len(ARMS),) + kk.shape).ravel(),
                                   np.broadcast_to(tt, (len(ARMS),) + tt.shape).ravel()], 1)
                    cl = fe[:, 0]
                    for s in sorted({"default", best}):
                        rg = REGS[s]
                        for spec, names in (("main", ["surprisal", "bayes_kl", "deviation", "uncertainty"]),
                                            ("lags", ["surprisal", "bayes_kl", "deviation", "uncertainty",
                                                      "surprisal_lag", "bayes_kl_lag"]),
                                            ("controls", ["surprisal", "bayes_kl", "deviation", "uncertainty",
                                                          "x", "x_sq", "level"])):
                            X = np.column_stack([zs(rg[n][:, :, t].ravel()) for n in names])
                            b, se, n, G = ols_fe_cluster(y, X, fe=fe, cluster=cl)
                            r = {"lens": lens, "signal": sig, "pos": pos, "band": band, "observer": s, "spec": spec, "n": n}
                            for nm, bi, si in zip(names, b, se):
                                r[f"b_{nm}"], r[f"t_{nm}"] = bi, bi / si
                            rows_r.append(r)
    C = pd.DataFrame(rows_c)
    Rg = pd.DataFrame(rows_r)
    C.to_csv(os.path.join(RES, f"{args.model}_contrasts.csv"), index=False)
    Rg.to_csv(os.path.join(RES, f"{args.model}_regression.csv"), index=False)
    # design table
    o = REGS["default"]
    t = np.arange(*REG_WIN)
    kk, tt = np.meshgrid(np.arange(N_SEQ), t, indexing="ij")
    fe = np.stack([np.broadcast_to(kk, (len(ARMS),) + kk.shape).ravel(), np.broadcast_to(tt, (len(ARMS),) + tt.shape).ravel()], 1)
    from stats import demean  # noqa: E402
    W = demean(np.column_stack([o[n][:, :, t].ravel() for n in ("surprisal", "bayes_kl", "deviation", "uncertainty")]), fe)
    ident = np.corrcoef(W.T)
    md.append("## 2-3. Signals: design contrasts\n")
    md.append("Within-(stream, number) correlation of the regressors (default observer): surprisal-KL "
              f"{ident[0, 1]:.2f}, surprisal-deviation {ident[0, 2]:.2f}, KL-deviation {ident[1, 2]:.2f}, "
              f"surprisal-uncertainty {ident[0, 3]:.2f}.\n")
    md.append("Nats of log-ratio, mean over 20 streams, sign-flip p in brackets; bands are thirds of the layers. "
              "e0 = jump - control at the event number (the outlier arm is the same input there). **after outlier** = "
              "outlier - control at e 1..3 and 4..10: from e = 1 on the two arms read identical numbers, so this is the "
              "clean contrast (only one past number differs). j-o = jump - outlier at e 1..3 and noise = noise - "
              "control at e 20..199 compare different current numbers (a readout of the numbers themselves enters). "
              "jump-level / drift-level = against the arm that was always at the new level, e 100..199 (same current "
              "level, different history). r = correlation over e 0..10 and the four arms of arm - control with the "
              "observer's surprisal / KL curves.\n")
    for sig in SIGNALS:
        md.append(f"\n### {sig} ({' - '.join(SIGNALS[sig])})\n")
        md.append("| lens | pos | band | e0 | after outlier e1..3 | after outlier e4..10 | j-o e1..3 | jump-level | drift-level | noise | r(surprisal) | r(KL) | r(KL, lag 1) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        sub = C[(C.signal == sig) & C.band.isin(["early", "middle", "late"])]
        for _, r in sub.iterrows():
            f = lambda k: f"{r[k]:+.3f} ({r[k + '_p']:.3f})"  # noqa: E731
            md.append(f"| {r.lens} | {r.pos} | {r.band} | {f('event_jump_e0')} | {f('after_outlier_e1_3')} | "
                      f"{f('after_outlier_e4_10')} | {f('jump_minus_outlier_e1_3')} | {f('jump_vs_level_late')} | "
                      f"{f('drift_vs_level_late')} | {f('noise')} | {r.profile_r_surprisal:.2f} | "
                      f"{r.profile_r_bayes_kl:.2f} | {r.profile_r_bayes_kl_lag:.2f} |")
    md.append("\n## 4. Regression (secondary)\n")
    md.append("Coefficients in nats per sd of each regressor, t-statistics in brackets; (stream, number) fixed "
              "effects, numbers TAU-5..TAU+40, clustered by stream. Middle and late thirds.\n")
    for s in sorted({"default", best}):
        for spec in ("main", "lags", "controls"):
            sub = Rg[(Rg.observer == s) & (Rg.spec == spec) & Rg.band.isin(["middle", "late"])]
            names = ["surprisal", "bayes_kl", "deviation", "uncertainty"] + {"main": [], "lags": ["surprisal_lag", "bayes_kl_lag"], "controls": ["x", "x_sq", "level"]}[spec]
            md.append(f"\n**Observer {s}, {spec}**\n\n| lens | signal | pos | band | " + " | ".join(names) + " |\n|---|---|---|---|" + "---|" * len(names))
            for _, r in sub.iterrows():
                md.append(f"| {r.lens} | {r.signal} | {r.pos} | {r.band} | " + " | ".join(f"{r['b_' + n]:+.3f} ({r['t_' + n]:.1f})" for n in names) + " |")
    # figure: middle and late band curves
    fig, axs = plt.subplots(len(SIGNALS), 2, figsize=(12, 3.2 * len(SIGNALS)), squeeze=False)
    lens_fig = "jpp" if "jpp" in lenses else "jlens" if "jlens" in lenses else "logit"
    for i, sig in enumerate(SIGNALS):
        for j, band in enumerate(("middle", "late")):
            cv = curves[(lens_fig, sig, "comma", band)]
            for a in ARMS[1:]:
                axs[i, j].plot(E, cv[a_of[a]], label=a)
            axs[i, j].axhline(0, color="grey", lw=0.5)
            axs[i, j].set(title=f"{sig}, {band} layers ({lens_fig}, comma)", xlabel="e")
    axs[0, 0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{args.model}_signals.png"), dpi=110)
    plt.close(fig)

    # ------------------------------------------------------------------ 5. cue readouts
    if os.path.exists(os.path.join(R, "cues_meta.json")):
        cm = json.load(open(os.path.join(R, "cues_meta.json")))
        words = cm["words"]
        gidx = {g: [i for i, w in enumerate(words) if w[0] == g] for g in ("surprise", "normal", "change")}
        crow = {(c["arm"], c["seq"], c["e"]): i for i, c in enumerate(cm["contexts"])}
        n_tail = np.array(cm["n_tail"])
        md.append("\n## 5. Cue readouts\n")
        md.append("At the last cue token. 'Wait,': log P(surprise set) - log P(normal set); 'The last number is': "
                  "change - normal. Arm minus control, mean over streams (sign-flip p); 'jump-level' and 'drift-level' "
                  "against the arm always at the new level. The output is the logit lens at the last block. From e = 1 "
                  "on, outlier and control read identical numbers.\n")
        rows_q = []
        for lens in cm["lenses"]:
            Wd = np.load(os.path.join(R, f"cues_{lens}_words.npy"), mmap_mode="r")
            for ci, cue in enumerate(cm["cue_names"]):
                pos_g = "surprise" if cue == "wait" else "change"
                last = n_tail[:, ci] - 1
                X = np.asarray(Wd[np.arange(len(last)), ci, last], np.float32)  # (P, L, W)
                lse = lambda g: np.log(np.exp(X[..., gidx[g]].astype(np.float64)).sum(-1))  # noqa: E731
                score = lse(pos_g) - lse("normal")  # (P, L)
                Y = np.array([[[score[crow[(a, k, e)]] for e in CUTS] for k in range(N_SEQ)] for a in ARMS])  # (A, K, E, L)
                YLc = None
                if os.path.exists(os.path.join(R, "cues_level_meta.json")):
                    lm_ = json.load(open(os.path.join(R, "cues_level_meta.json")))
                    assert [w[2] for w in lm_["words"]] == [w[2] for w in words]
                    lrow_ = {(c["seq"], c["e"]): i for i, c in enumerate(lm_["contexts"])}
                    nt_ = np.array(lm_["n_tail"])[:, ci] - 1
                    WL = np.load(os.path.join(R, f"cues_level_{lens}_words.npy"), mmap_mode="r")
                    XL = np.asarray(WL[np.arange(len(nt_)), ci, nt_], np.float32)
                    lseL = lambda g: np.log(np.exp(XL[..., gidx[g]].astype(np.float64)).sum(-1))  # noqa: E731
                    scL = lseL(pos_g) - lseL("normal")
                    YLc = np.array([[scL[lrow_[(k, e)]] for e in CUTS] for k in range(N_SEQ)])  # (K, E, L)
                bands = list(thirds(L).items()) + [("output", [L - 1])]
                for band, ls in bands:
                    if not ls:
                        continue
                    Yb = Y[..., ls].mean(-1)
                    for a in ARMS[1:]:
                        for ei, e in enumerate(CUTS):
                            d = Yb[a_of[a], :, ei] - Yb[a_of["control"], :, ei]
                            est, p = sign_flip_p(d)
                            rows_q.append({"lens": lens, "cue": cue, "band": band, "arm": a, "e": e, "diff": est, "p": p})
                    if YLc is not None:
                        YLb = YLc[..., ls].mean(-1)
                        for a in ("jump", "drift"):
                            for ei, e in enumerate(CUTS):
                                est, p = sign_flip_p(Yb[a_of[a], :, ei] - YLb[:, ei])
                                rows_q.append({"lens": lens, "cue": cue, "band": band, "arm": f"{a}-level", "e": e, "diff": est, "p": p})
                    # regression across arms x cuts
                    tcut = np.array([TAU + e for e in CUTS])
                    rg = REGS[best]
                    yv = Yb.ravel()
                    names = ["surprisal", "bayes_kl", "deviation", "uncertainty"]
                    Xr = np.column_stack([zs(rg[n][:, :, tcut].ravel()) for n in names])
                    kk2, ee2 = np.meshgrid(np.arange(N_SEQ), np.arange(len(CUTS)), indexing="ij")
                    fe2 = np.stack([np.broadcast_to(kk2, (len(ARMS),) + kk2.shape).ravel(),
                                    np.broadcast_to(ee2, (len(ARMS),) + ee2.shape).ravel()], 1)
                    b, se, n, G = ols_fe_cluster(yv, Xr, fe=fe2, cluster=fe2[:, 0])
                    rows_q.append({"lens": lens, "cue": cue, "band": band, "arm": "regression", "e": np.nan,
                                   **{f"b_{nm}": bi for nm, bi in zip(names, b)},
                                   **{f"t_{nm}": bi / si for nm, bi, si in zip(names, b, se)}})
        Q = pd.DataFrame(rows_q)
        Q.to_csv(os.path.join(RES, f"{args.model}_cues.csv"), index=False)
        for cue in cm["cue_names"]:
            md.append(f"\n### {cue}\n\n| lens | band | arm | " + " | ".join(f"e={e}" for e in CUTS) + " |\n|---|---|---|" + "---|" * len(CUTS))
            sub = Q[(Q.cue == cue) & (Q.arm != "regression")]
            for (lens, band, a), g in sub.groupby(["lens", "band", "arm"], sort=False):
                g = g.set_index("e")
                md.append(f"| {lens} | {band} | {a} | " + " | ".join(f"{g.loc[e, 'diff']:+.2f}{'*' if g.loc[e, 'p'] < 0.05 else ''}" for e in CUTS) + " |")
            md.append(f"\nRegression across arms and cuts (observer {best}; coefficient, t):\n\n| lens | band | surprisal | KL | deviation | uncertainty |\n|---|---|---|---|---|---|")
            for _, r in Q[(Q.cue == cue) & (Q.arm == "regression")].iterrows():
                md.append(f"| {r.lens} | {r.band} | " + " | ".join(f"{r['b_' + n]:+.3f} ({r['t_' + n]:.1f})" for n in ("surprisal", "bayes_kl", "deviation", "uncertainty")) + " |")
        md.append("\n(* p < 0.05, sign-flip over streams, not corrected for the number of cells.)")
    open(os.path.join(RES, f"{args.model}_summary.md"), "w").write("\n".join(md) + "\n")
    json.dump({"model": args.model, "go": go, "outlier_e1": float(out_e1), "drift_slope": float(slope),
               "best_observer": best, "identification": ident.tolist()},
              open(os.path.join(RES, f"{args.model}_verdict.json"), "w"), indent=1)
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
