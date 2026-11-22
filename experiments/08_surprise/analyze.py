"""Experiment 08 analysis (CPU): behaviour checks, regressions on the ideal observer's surprise, event studies.

Part A (change points)
  Behaviour (go / no-go): does the model's prediction of the next number move like the ideal observer's?
    update slope    OLS of the model's change in predicted mean on the observer's change (1 = Bayesian)
    surprisal corr  correlation of the model's -log p(x_t) with the observer's surprisal
    adaptation      after a jump, how far the model's predicted mean has moved to the new level (fraction)
  Signals at each layer and slot: jump (|h_t - h_{t-1}| over the layer's median norm), lex_surprise (J-lens
  readout of the surprise and change words minus the control words), lex_dir (up minus down words, times the
  sign of the jump), turnover10 (Jaccard distance between consecutive J-space sets of 10 atoms), div (distance
  to the control arm).
  Regressions: signal on z(Bayesian surprise) and z(surprisal) together, with fixed effects for (item, t): only
  the arms of the same item at the same number are compared (the twin design), errors clustered by item.
  Specs: twin (the above), pooled (fixed effects for t only), buffer (twin plus the model's own |change in
  predicted mean|, a mediator: what is left is surprise beyond "the next output changed").
  Event study: jump minus control and outlier minus control at k = t - tau, bootstrap CIs over items.
Part B (arithmetic twins)
  Behaviour (go / no-go): the share of items where the model's top-1 at r is r ("knows"); surprisal of the edit.
  Event study on items the model knows (a pre-edit property, so no selection on the outcome): twin minus clean at
  k tokens after each twin's last edited token; triple difference (error minus swap) averaged over k = 1..5 with a
  sign-flip randomization test; the same at the later reuse of r (identical tokens, different context).
Pipeline checks (raise an error when violated): every signal is identical across twins before the edit.

    python experiments/08_surprise/analyze.py --model llama3.1-8b --raw /resnick/scratch/$USER/surprise
"""
import argparse
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from bayes import observe
from common import ARMS, GROUPS, HERE, RAW, SLOTS, T, TWINS, band
from stats import boot_mean_ci, demean, ols_fe_cluster, sign_flip_p, turnover

K_A = range(-5, 9)  # event time in Part A (numbers after the change)
K_B = range(-5, 16)  # event time in Part B (tokens after the edit)
GI = {g: i for i, g in enumerate(GROUPS)}


def nanmean(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def zscore(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def save_fig(fig, path):
    fig.savefig(path + ".pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(path + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def pre_check(name, diff, tol):
    """diff: twin differences at positions before the edit, which must be zero (identical inputs)."""
    diff = np.asarray(diff, float)
    diff = diff[np.isfinite(diff)]
    worst = float(np.abs(diff).max()) if len(diff) else 0.0
    return {"check": name, "n": len(diff), "max_abs": worst, "ok": worst <= tol}


# ---------------------------------------------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------------------------------------------
def changepoint_signals(R, meta, dec_layers):
    """{name: (array (N, T, n_slots, n_layers), slot names, layer ids)}."""
    L = R["norm"].shape[-1]
    med = np.nanmedian(R["norm"], axis=(0, 1))  # (3, L)
    lex = R["lex"]
    sign = np.sign(meta["shift"].to_numpy())[:, None, None, None]
    sig = {
        "jump": (R["delta"] / med, SLOTS, list(range(L))),
        "lex_surprise": ((lex[..., GI["surprise"]] + lex[..., GI["change"]]) / 2 - lex[..., GI["control"]], SLOTS, list(range(L))),
        "lex_dir": ((lex[..., GI["up"]] - lex[..., GI["down"]]) * sign, SLOTS, list(range(L))),
        "div": (R["div"], SLOTS, list(range(L))),
    }
    tv = turnover(R["dec_idx"][:, 1:], R["dec_coef"][:, 1:], R["dec_idx"][:, :-1], R["dec_coef"][:, :-1], 10)
    sig["turnover10"] = (np.concatenate([np.full_like(tv[:, :1], np.nan), tv], 1), ("sep", "pred"), dec_layers)
    return sig


def band_mean(arr, layers, band_layers):
    cols = [i for i, layer in enumerate(layers) if layer in band_layers]
    if not cols:  # e.g. decomposition layers: take those inside the band, else the middle ones
        cols = list(range(len(layers) // 3, 2 * len(layers) // 3 + 1))
    return nanmean(arr[..., cols], -1)


def analyze_changepoint(raw, model, args, out_csv, out_fig):
    meta = pd.read_csv(os.path.join(raw, "changepoint_meta.csv"))
    R = dict(np.load(os.path.join(raw, "changepoint.npz")))
    info = json.load(open(os.path.join(raw, "run_info.json")))
    L = R["norm"].shape[-1]
    BAND = band(L)
    X = np.array([list(map(int, s.split())) for s in meta.numbers])
    obs = observe(X, hazard=args.hazard)
    N = len(meta)
    item = meta.item.to_numpy()
    tau = meta.tau.to_numpy()
    arm = meta.arm.to_numpy()
    items = np.unique(item)
    rows_of = {(it, a): r for r, (it, a) in enumerate(zip(item, arm))}
    trip = np.array([[rows_of[(it, a)] for a in ARMS] for it in items])  # (items, 3): control, jump, outlier rows
    tau_i = tau[trip[:, 0]]
    checks, summary = [], []

    # --- behaviour ----------------------------------------------------------------------------------------------
    pm = R["pred"][..., 0].astype(float)  # model's predicted mean of x_{t+1} after x_t
    d_model = pm[:, 1:-1] - pm[:, :-2]
    d_bayes = obs["next_mean"][:, 1:-1] - obs["next_mean"][:, :-2]
    cl = np.repeat(item[:, None], d_model.shape[1], 1)
    b, se, n, _ = ols_fe_cluster(d_model.ravel(), d_bayes.ravel()[:, None], cluster=cl.ravel())
    ok = np.isfinite(d_model) & np.isfinite(d_bayes)
    r2 = np.corrcoef(d_model[ok], d_bayes[ok])[0, 1] ** 2
    surp_m, surp_b = -R["logp"][:, 1:].astype(float), obs["surprisal"][:, 1:]
    ok2 = np.isfinite(surp_m) & np.isfinite(surp_b)
    corr = np.corrcoef(surp_m[ok2], surp_b[ok2])[0, 1]
    bs, ses, _, _ = ols_fe_cluster(surp_m.ravel(), surp_b.ravel()[:, None], cluster=np.repeat(item[:, None], T - 1, 1).ravel())
    mass = np.nanmean(R["pred"][..., 3])
    adapt = []
    for k in range(-2, 9):
        for a_i, a in enumerate(ARMS[1:], 1):
            r = trip[:, a_i]
            t = tau_i - 1 + k
            okk = (t >= 0) & (t <= T - 2)
            shift = meta["shift"].to_numpy()[r][okk]
            mu0 = meta["mu0"].to_numpy()[r][okk]
            fm = (pm[r[okk], t[okk]] - mu0) / shift
            fb = (obs["next_mean"][r[okk], t[okk]] - mu0) / shift
            adapt.append({"arm": a, "k": k, "model": np.nanmean(fm), "observer": np.nanmean(fb), "n": int(okk.sum())})
    adapt = pd.DataFrame(adapt)
    frac2 = adapt[(adapt.arm == "jump") & (adapt.k == 2)].model.iloc[0]
    go = bool(b[1] >= 0.5 and corr >= 0.5 and frac2 >= 0.5)
    beh = pd.DataFrame([
        {"quantity": "update slope (model on observer)", "value": b[1], "se": se[1], "n": n},
        {"quantity": "update R^2", "value": r2, "se": np.nan, "n": n},
        {"quantity": "surprisal corr (model, observer)", "value": corr, "se": np.nan, "n": int(ok2.sum())},
        {"quantity": "surprisal slope (model on observer)", "value": bs[1], "se": ses[1], "n": int(ok2.sum())},
        {"quantity": "mass on one-token integers", "value": mass, "se": np.nan, "n": np.nan},
        {"quantity": "jump: fraction of the way to the new level after 3 numbers", "value": frac2, "se": np.nan, "n": np.nan},
        {"quantity": "GO (slope >= 0.5, corr >= 0.5, fraction >= 0.5)", "value": float(go), "se": np.nan, "n": np.nan},
    ])
    beh.to_csv(out_csv + "changepoint_behaviour.csv", index=False)
    adapt.to_csv(out_csv + "changepoint_adaptation.csv", index=False)
    print("\nPart A, behaviour\n" + beh.round(3).to_string(index=False))

    # --- signals and the pre-period check ----------------------------------------------------------------------
    sig = changepoint_signals(R, meta, info["decomp_layers"])
    tt = np.arange(T)[None, :] + 1  # 1-based number index
    before = tt < tau_i[:, None]  # (items, T): every slot of x_t with t < tau has seen only pre-change numbers
    through = tt <= tau_i[:, None]  # jump and outlier are identical through tau
    for name, (arr, slots, layers) in sig.items():
        a = arr.reshape(N, T, -1)
        c, j, o = a[trip[:, 0]], a[trip[:, 1]], a[trip[:, 2]]
        scale = np.nanstd(a) if np.isfinite(a).any() else 1.0
        tol = args.tol * (scale if scale > 0 else 1.0)
        checks.append(pre_check(f"A {name}: jump - control before tau", (j - c)[before], tol))
        checks.append(pre_check(f"A {name}: outlier - control before tau", (o - c)[before], tol))
        checks.append(pre_check(f"A {name}: jump - outlier through tau", (j - o)[through], tol))

    # --- regressions -----------------------------------------------------------------------------------------
    # 0-based numbers 3..T-2: past the lens's first 16 positions (Llama-3: x_t at position 9 + 3t), x_{t-1} exists
    # (jump, turnover), and so do the sep and pred slots (not for the last number)
    t_lo, t_hi = 3, T - 2
    tsel = slice(t_lo, t_hi + 1)
    nt = t_hi - t_lo + 1
    zkl = zscore(obs["bayes_kl"][:, tsel])
    zsu = zscore(obs["surprisal"][:, tsel])
    upd = np.full_like(pm, np.nan)
    upd[:, 1:] = np.abs(pm[:, 1:] - pm[:, :-1])
    zbuf = zscore(upd[:, tsel])
    fe = (item[:, None] * 1000 + np.arange(t_lo, t_hi + 1)[None, :]).ravel()
    fe_t = np.repeat(np.arange(t_lo, t_hi + 1)[None, :], N, 0).ravel()
    clu = np.repeat(item[:, None], nt, 1).ravel()
    # identification: how much the two surprises move separately within (item, t)
    W = demean(np.column_stack([zkl.ravel(), zsu.ravel()]), fe)
    keep = np.abs(W).sum(1) > 0
    ident = np.corrcoef(W[keep].T)[0, 1]
    print(f"\nWithin (item, t), corr(Bayesian surprise, surprisal) = {ident:.3f} over {keep.sum()} rows that vary")
    specs = {"twin": (np.column_stack([zkl.ravel(), zsu.ravel()]), fe),
             "pooled": (np.column_stack([zkl.ravel(), zsu.ravel()]), fe_t),
             "buffer": (np.column_stack([zkl.ravel(), zsu.ravel(), zbuf.ravel()]), fe)}
    reg = []
    for name, (arr, slots, layers) in sig.items():
        if name == "div":
            continue  # a distance to the control arm, not a per-sequence signal
        for si, slot in enumerate(slots):
            if slot == "num" and not args.all_slots:
                continue
            cols = [(layer, arr[:, tsel, si, li]) for li, layer in enumerate(layers)]
            cols.append(("band", band_mean(arr[:, tsel, si], layers, BAND)))
            for layer, y in cols:
                y = y.ravel()
                sd = np.nanstd(y)
                if not np.isfinite(sd) or sd == 0:
                    continue
                for spec, (Xs, f) in specs.items():
                    beta, se, n, G = ols_fe_cluster(y / sd, Xs, fe=f, cluster=clu)
                    reg.append({"signal": name, "slot": slot, "layer": layer, "spec": spec,
                                "b_bayes_kl": beta[0], "se_bayes_kl": se[0], "b_surprisal": beta[1],
                                "se_surprisal": se[1], "n": n, "clusters": G})
    reg = pd.DataFrame(reg)
    reg.to_csv(out_csv + "changepoint_regression.csv", index=False)
    head = reg[(reg.layer == "band") & (reg.slot == "sep")].copy()
    for v in ("bayes_kl", "surprisal"):
        head[f"t_{v}"] = head[f"b_{v}"] / head[f"se_{v}"]
    print("\nPart A, headline (band-averaged signal at the sep slot, in sd of the signal per sd of the regressor)")
    print(head[["signal", "spec", "b_bayes_kl", "se_bayes_kl", "t_bayes_kl", "b_surprisal", "se_surprisal",
                "t_surprisal", "n"]].round(3).to_string(index=False))
    summary.append(head.assign(part="A"))

    # --- event study -------------------------------------------------------------------------------------------
    ev = []
    ref = {"bayes_kl": obs["bayes_kl"], "surprisal": obs["surprisal"], "model_surprisal": -R["logp"].astype(float)}
    series = {f"observer {k}": (v, "num") for k, v in ref.items()}
    for name, (arr, slots, layers) in sig.items():
        for si, slot in enumerate(slots):
            series[f"{name} [{slot}]"] = (band_mean(arr[:, :, si], layers, BAND), slot)
    for label, (y, slot) in series.items():
        for contrast, (a1, a0) in {"jump - control": (1, 0), "outlier - control": (2, 0), "jump - outlier": (1, 2)}.items():
            D = np.full((len(items), len(K_A)), np.nan)
            for ki, k in enumerate(K_A):
                t = tau_i - 1 + k
                okk = (t >= 0) & (t <= T - 1)
                D[okk, ki] = y[trip[okk, a1], t[okk]] - y[trip[okk, a0], t[okk]]
            est, lo, hi = boot_mean_ci(D, args.boot)
            for ki, k in enumerate(K_A):
                ev.append({"series": label, "contrast": contrast, "k": k, "est": est[ki], "lo": lo[ki], "hi": hi[ki]})
    ev = pd.DataFrame(ev)
    ev.to_csv(out_csv + "changepoint_event.csv", index=False)

    # --- figures -------------------------------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, color in (("jump", "C3"), ("outlier", "C0")):
        s = adapt[adapt.arm == a]
        axs[0].plot(s.k, s.model, color=color, label=f"model, {a}")
        axs[0].plot(s.k, s.observer, color=color, linestyle="--", label=f"ideal observer, {a}")
    axs[0].axhline(0, color="grey", lw=0.6)
    axs[0].axhline(1, color="grey", lw=0.6)
    axs[0].set(xlabel="numbers after the change (k)", ylabel="predicted mean: fraction of the way to the new level")
    axs[0].legend(fontsize=7)
    rng = np.random.default_rng(0)
    sel = np.flatnonzero(ok.ravel())
    sel = rng.choice(sel, min(4000, len(sel)), replace=False)
    axs[1].scatter(d_bayes.ravel()[sel], d_model.ravel()[sel], s=3, alpha=0.3)
    lim = np.nanquantile(np.abs(d_bayes.ravel()[sel]), 0.99)
    axs[1].plot([-lim, lim], [-lim, lim], color="grey", lw=0.8)
    axs[1].set(xlabel="observer's change in predicted mean", ylabel="model's change", title=f"slope {b[1]:.2f}, R² {r2:.2f}")
    fig.suptitle(f"{model}: does the model update like the ideal observer?", fontsize=10)
    plt.tight_layout()
    save_fig(fig, out_fig + "changepoint_behaviour")

    names = [n for n in sig if n != "div"]
    fig, axs = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.2), sharey=True)
    tw = reg[(reg.spec == "twin") & (reg.slot == "sep") & (reg.layer != "band")]
    for ax, name in zip(axs, names):
        s = tw[tw.signal == name]
        x = s.layer.astype(int)
        for v, color, lab in (("bayes_kl", "C3", "Bayesian surprise"), ("surprisal", "C0", "surprisal")):
            ax.errorbar(x, s[f"b_{v}"], yerr=1.96 * s[f"se_{v}"], color=color, marker=".", ms=3, lw=1, label=lab)
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvspan(BAND[0], BAND[-1], color="grey", alpha=0.08)
        ax.set(title=name, xlabel="layer")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_ylabel("coefficient (sd of signal per sd)")
    axs[0].legend(fontsize=7)
    fig.suptitle(f"{model}: which surprise does each signal track? (sep slot, twin fixed effects, 95% CI)", fontsize=10)
    plt.tight_layout()
    save_fig(fig, out_fig + "changepoint_regression")

    show = ["observer bayes_kl", "observer surprisal"] + [f"{n} [sep]" for n in names]
    fig, axs = plt.subplots(1, len(show), figsize=(2.9 * len(show), 3.1))
    for ax, label in zip(axs, show):
        for contrast, color in (("jump - control", "C3"), ("outlier - control", "C0")):
            s = ev[(ev.series == label) & (ev.contrast == contrast)]
            ax.plot(s.k, s.est, color=color, label=contrast)
            ax.fill_between(s.k, s.lo, s.hi, color=color, alpha=0.2)
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvline(-0.5, color="grey", lw=0.6, linestyle=":")
        ax.set(title=label, xlabel="k = t - tau")
    axs[0].legend(fontsize=7)
    fig.suptitle(f"{model}: event study around the change (band-averaged signals)", fontsize=10)
    plt.tight_layout()
    save_fig(fig, out_fig + "changepoint_event")
    return checks, summary, {"A_go": go, "A_update_slope": float(b[1]), "A_surprisal_corr": float(corr),
                             "A_fraction_after_3": float(frac2), "A_identification_corr": float(ident)}


# ---------------------------------------------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------------------------------------------
def analyze_arith(raw, model, args, out_csv, out_fig):
    meta = pd.read_csv(os.path.join(raw, "arith_meta.csv"))
    R = dict(np.load(os.path.join(raw, "arith.npz")))
    info = json.load(open(os.path.join(raw, "run_info.json")))
    L = R["norm"].shape[-1]
    BAND = band(L)
    M = len(meta)
    pos = R["pos"]
    TI = {w: i for i, w in enumerate(TWINS)}
    checks = []

    def col(i, p):
        j = np.flatnonzero(pos[i] == p)
        return j[0] if len(j) else -1

    # --- behaviour ----------------------------------------------------------------------------------------------
    knows = R["res_top1"][:, TI["clean"]]
    # the event study uses the items the model knows; if it knows none (a small model), all items, and says so
    sample_name = "items the model knows" if knows.any() else "all items (the model knows r in none)"
    surp = {}
    for w in TWINS:
        js = [col(i, meta.result_pos[i]) for i in range(M)]
        surp[w] = np.array([-R["logp"][i, TI[w], j] if j >= 0 else np.nan for i, j in enumerate(js)])
    swap_s = np.array([-np.nansum([R["logp"][i, TI["swap"], col(i, p)] for p in range(meta.first_swap[i], meta.last_swap[i] + 1)
                                   if col(i, p) >= 0]) for i in range(M)])
    clean_s = np.array([-np.nansum([R["logp"][i, TI["clean"], col(i, p)] for p in range(meta.first_swap[i], meta.last_swap[i] + 1)
                                    if col(i, p) >= 0]) for i in range(M)])
    reuse_d = {}
    for w in ("near", "far", "swap"):
        reuse_d[w] = np.array([R["logp"][i, TI[w], col(i, meta.reuse_pos[i])] - R["logp"][i, TI["clean"], col(i, meta.reuse_pos[i])]
                               if meta.reuse_pos[i] > 0 and col(i, meta.reuse_pos[i]) >= 0 else np.nan for i in range(M)])
    gap = nanmean((surp["near"] - surp["clean"])[knows], 0) if knows.any() else np.nan
    go = bool(knows.mean() >= 0.5 and gap >= 2)
    beh = pd.DataFrame([
        {"quantity": "items", "value": M},
        {"quantity": "model knows r (top-1 at r is r)", "value": knows.mean()},
        {"quantity": "surprisal of r, clean (knows)", "value": nanmean(surp["clean"][knows], 0)},
        {"quantity": "surprisal of the near error (knows)", "value": nanmean(surp["near"][knows], 0)},
        {"quantity": "surprisal of the far error (knows)", "value": nanmean(surp["far"][knows], 0)},
        {"quantity": "surprisal of the swapped operands minus clean (knows)", "value": nanmean((swap_s - clean_s)[knows], 0)},
        *[{"quantity": f"reuse of r: log p in {w} minus clean (knows)", "value": nanmean(reuse_d[w][knows], 0)}
          for w in ("near", "far", "swap")],
        {"quantity": "items with a reuse of r (knows)", "value": int(np.isfinite(reuse_d["near"][knows]).sum())},
        {"quantity": "GO (knows >= 0.5, near-error surprisal gap >= 2 nats)", "value": float(go)},
    ])
    beh.to_csv(out_csv + "arith_behaviour.csv", index=False)
    print("\nPart B, behaviour\n" + beh.round(3).to_string(index=False))

    # --- signals ---------------------------------------------------------------------------------------------
    med = np.nanmedian(R["norm"], axis=(0, 1, 2))
    lex = R["lex"]
    sig = {
        "jump": (R["delta"] / med, list(range(L))),
        "lex_error": ((lex[..., GI["error"]] + lex[..., GI["surprise"]]) / 2 - lex[..., GI["control"]], list(range(L))),
        "div": (R["div"], list(range(L))),
    }
    tv = turnover(R["dec_idx"][:, :, 1:], R["dec_coef"][:, :, 1:], R["dec_idx"][:, :, :-1], R["dec_coef"][:, :, :-1], 10)
    adjacent = (pos[:, 1:] - 1 == pos[:, :-1]) & (pos[:, 1:] >= 0)
    tv = np.where(adjacent[:, None, :, None], tv, np.nan)
    sig["turnover10"] = (np.concatenate([np.full_like(tv[:, :, :1], np.nan), tv], 2), info["decomp_layers"])

    # --- event study: twin minus clean at k tokens after each twin's last edited token -------------------------------
    ev, tests = [], []
    sample = np.flatnonzero(knows) if knows.any() else np.arange(M)
    for name, (arr, layers) in sig.items():
        series = {"band": band_mean(arr, layers, BAND)}
        if args.by_layer:
            series.update({layer: arr[..., li] for li, layer in enumerate(layers)})
        for layer, y in series.items():
            D = {w: np.full((M, len(K_B)), np.nan) for w in ("near", "far", "swap")}
            for w in ("near", "far", "swap"):
                pre = []
                for i in range(M):
                    first, last = meta[f"first_{w}"][i], meta[f"last_{w}"][i]
                    for ki, k in enumerate(K_B):
                        j = col(i, last + k)
                        if j >= 0:
                            D[w][i, ki] = y[i, TI[w], j] - y[i, TI["clean"], j]
                    # every position before the twin's first edited token, inside the window
                    pre += [y[i, TI[w], j] - y[i, TI["clean"], j] for j in np.flatnonzero((pos[i] >= 0) & (pos[i] < first))]
                if layer == "band":
                    scale = np.nanstd(y) if np.isfinite(y).any() else 1.0
                    checks.append(pre_check(f"B {name}: {w} - clean before the edit", pre, args.tol * (scale or 1.0)))
            contrasts = {"near - clean": D["near"], "far - clean": D["far"], "swap - clean": D["swap"],
                         "near - swap": D["near"] - D["swap"], "far - swap": D["far"] - D["swap"],
                         "far - near": D["far"] - D["near"]}
            for cname, Dc in contrasts.items():
                est, lo, hi = boot_mean_ci(Dc[sample], args.boot)
                for ki, k in enumerate(K_B):
                    ev.append({"signal": name, "layer": layer, "contrast": cname, "k": k, "est": est[ki], "lo": lo[ki], "hi": hi[ki]})
                post = nanmean(Dc[sample][:, [K_B.index(k) for k in range(1, 6)]], 1)
                m, p = sign_flip_p(post, args.perm)
                _, lo1, hi1 = boot_mean_ci(post[:, None], args.boot)
                tests.append({"signal": name, "layer": layer, "contrast": cname, "window": "k = 1..5", "mean": m,
                              "lo": lo1[0], "hi": hi1[0], "p_sign_flip": p, "n": int(np.isfinite(post).sum())})
            # the later reuse of r: identical tokens in every twin, so any difference is the context
            if layer == "band":
                for w in ("near", "far", "swap"):
                    d = np.array([y[i, TI[w], col(i, meta.reuse_pos[i])] - y[i, TI["clean"], col(i, meta.reuse_pos[i])]
                                  if meta.reuse_pos[i] > 0 and col(i, meta.reuse_pos[i]) >= 0 else np.nan for i in range(M)])
                    m, p = sign_flip_p(d[sample], args.perm)
                    tests.append({"signal": name, "layer": layer, "contrast": f"{w} - clean", "window": "reuse of r",
                                  "mean": m, "lo": np.nan, "hi": np.nan, "p_sign_flip": p, "n": int(np.isfinite(d[sample]).sum())})
    ev, tests = pd.DataFrame(ev), pd.DataFrame(tests)
    ev.to_csv(out_csv + "arith_event.csv", index=False)
    tests.to_csv(out_csv + "arith_tests.csv", index=False)
    head = tests[(tests.layer == "band")]
    print(f"\nPart B, headline (band-averaged signals; {sample_name})")
    print(head.round(4).to_string(index=False))

    fig, axs = plt.subplots(1, len(sig), figsize=(3.2 * len(sig), 3.1))
    for ax, name in zip(axs, sig):
        for cname, color in (("near - clean", "C1"), ("far - clean", "C3"), ("swap - clean", "C0")):
            s = ev[(ev.signal == name) & (ev.layer == "band") & (ev.contrast == cname)]
            ax.plot(s.k, s.est, color=color, label=cname)
            ax.fill_between(s.k, s.lo.astype(float), s.hi.astype(float), color=color, alpha=0.2)
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvline(0.5, color="grey", lw=0.6, linestyle=":")
        ax.set(title=name, xlabel="tokens after each twin's last edited token (k)")
    axs[0].legend(fontsize=7)
    fig.suptitle(f"{model}: planted errors vs a harmless edit (band-averaged, {sample_name})", fontsize=10)
    plt.tight_layout()
    save_fig(fig, out_fig + "arith_event")
    return checks, [head.assign(part="B")], {"B_go": go, "B_knows": float(knows.mean()), "B_error_surprisal_gap": float(gap),
                                            "B_sample": sample_name, "B_items": int(M)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--raw", default=RAW, help="run_model.py's output folder")
    parser.add_argument("--part", choices=["both", "changepoint", "arith"], default="both")
    parser.add_argument("--hazard", type=float, default=1 / 30, help="the ideal observer's change probability")
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--perm", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-3, help="pre-edit differences allowed, in sd of the signal")
    parser.add_argument("--all-slots", action="store_true", help="also regress the num slot (mixes token identity)")
    parser.add_argument("--by-layer", action="store_true", help="Part B: event study at every layer too")
    parser.add_argument("--tag", default="", help="suffix of the output names (e.g. _smoke)")
    parser.add_argument("--out-dir", default=HERE, help="results/ and figures/ go here")
    parser.add_argument("--no-strict", action="store_true", help="report failed pipeline checks without stopping")
    args = parser.parse_args()
    raw = os.path.join(args.raw, args.model)
    out_csv = os.path.join(args.out_dir, "results", f"{args.model}{args.tag}_")
    out_fig = os.path.join(args.out_dir, "figures", f"{args.model}{args.tag}_")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    pd.set_option("display.width", 250)
    checks, summary, verdict = [], [], {"model": args.model, "hazard": args.hazard}
    if args.part in ("both", "changepoint") and os.path.exists(os.path.join(raw, "changepoint.npz")):
        c, s, v = analyze_changepoint(raw, args.model, args, out_csv, out_fig)
        checks += c
        summary += s
        verdict.update(v)
    if args.part in ("both", "arith") and os.path.exists(os.path.join(raw, "arith.npz")):
        c, s, v = analyze_arith(raw, args.model, args, out_csv, out_fig)
        checks += c
        summary += s
        verdict.update(v)
    if summary:
        pd.concat(summary).to_csv(out_csv + "headline.csv", index=False)
    checks = pd.DataFrame(checks)
    checks.to_csv(out_csv + "checks.csv", index=False)
    with open(out_csv + "verdict.json", "w") as f:
        json.dump(verdict, f, indent=1)
    print("\nPipeline checks (twins identical before the edit)")
    print(checks.to_string(index=False))
    print("\n" + json.dumps(verdict, indent=1))
    if not checks.empty and not checks.ok.all() and not args.no_strict:
        raise SystemExit("A pipeline check failed: twins differ before the edit (see checks.csv)")


if __name__ == "__main__":
    main()
