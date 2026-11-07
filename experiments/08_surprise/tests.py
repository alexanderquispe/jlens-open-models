"""Tests of experiment 08 (CPU, no model, about a minute). From the repo root:

    python experiments/08_surprise/tests.py                          # math, statistics, stimuli, planted analysis
    python experiments/08_surprise/tests.py --tokenizer llama3.1-8b  # also the tokenization checks (HF cache)

  observer    exact BOCPD against brute-force enumeration of every change pattern (and numeric integration of the
              Gaussian part of the KL); the integer probability against an independent erfc formula, far in the tails
  stats       fixed-effects OLS and its clustered errors against a regression with dummies; sign-flip test; turnover
  stimuli     the invariants of data/changepoint.csv and data/arith.jsonl (arms and twins identical outside the change)
  analysis    analyze.py on synthetic model outputs with planted effects of known size: it must recover them exactly,
              pass its pre-period checks, and fail them when a difference is planted before the edit
  tokenizer   (with --tokenizer) prep.py on the real stimuli: how many items survive, and that r and the slots are the
              tokens they should be
"""
import argparse
import itertools
import json
import math
import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bayes import gauss_kl, log_int_prob, observe  # noqa: E402
from common import ARMS, DATA, GROUPS, PREFIX, SEP, T, TWINS, decomp_layers  # noqa: E402
from stats import boot_mean_ci, ols_fe_cluster, sign_flip_p, turnover  # noqa: E402

GI = {g: i for i, g in enumerate(GROUPS)}
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------------------------------------------
# observer
# ---------------------------------------------------------------------------------------------------------------
def int_prob(x, mean, sd):
    """P(x - 1/2 < X < x + 1/2), X ~ N(mean, sd^2), from erfc (independent of bayes.log_int_prob)."""
    a, b = (x - 0.5 - mean) / sd, (x + 0.5 - mean) / sd
    Q = lambda z: 0.5 * math.erfc(z / math.sqrt(2))  # noqa: E731  upper tail
    if a > 0:
        return Q(a) - Q(b)
    if b < 0:
        return Q(-b) - Q(-a)
    return 1 - Q(b) - Q(-a)


def test_log_int_prob():
    cases = [(x, m, s) for x in (0, 3, 10, 50, 80) for m in (0.0, 2.7, 10.0, 49.5) for s in (1.0, 2.5, 7.0)]
    cases = [c for c in cases if abs(c[0] - c[1]) / c[2] < 36]  # math.erfc underflows beyond
    got = log_int_prob(*(torch.tensor(np.array(c, dtype=np.float64)) for c in zip(*cases))).numpy()
    want = np.array([math.log(int_prob(*c)) for c in cases])
    assert np.allclose(got, want, rtol=1e-9, atol=1e-12), np.abs(got - want).max()
    assert want.min() < -500, "the cases should reach far into the tails"
    k = torch.arange(-60, 90, dtype=torch.float64)
    total = log_int_prob(k, torch.tensor(13.3, dtype=torch.float64), torch.tensor(4.0, dtype=torch.float64)).exp().sum()
    assert abs(total.item() - 1) < 1e-12


def test_gauss_kl():
    for m1, v1, m0, v0 in [(0.3, 0.5, -1.0, 2.0), (10.0, 4.0, 10.5, 9.0), (0.0, 1.0, 0.0, 1.0)]:
        x = np.linspace(m1 - 15 * math.sqrt(v1), m1 + 15 * math.sqrt(v1), 200001)
        lq = -0.5 * (x - m1) ** 2 / v1 - 0.5 * math.log(2 * math.pi * v1)
        lw = -0.5 * (x - m0) ** 2 / v0 - 0.5 * math.log(2 * math.pi * v0)
        num = TRAPZ(np.exp(lq) * (lq - lw), x)
        got = gauss_kl(*(torch.tensor(v, dtype=torch.float64) for v in (m1, v1, m0, v0))).item()
        assert abs(got - num) < 1e-8, (got, num)


def segment_posterior(data, sigma, m0, s0):
    prec = 1 / s0**2 + len(data) / sigma**2
    return (m0 / s0**2 + sum(data) / sigma**2) / prec, 1 / prec


def brute_force_observer(x, sigmas, h, m0, s0):
    """The observer's outputs by enumerating every (sigma, change pattern): no recursion, no run lengths."""
    out = {k: np.full(len(x), np.nan) for k in ("surprisal", "bayes_kl", "cpp", "level_after", "pred_mean", "pred_sd",
                                                 "next_mean", "next_sd", "sigma_mean")}
    for t in range(len(x)):
        rows = []  # per (sigma, pattern): log prior weight before x_t, sigma, prior and posterior of mu_t, change at t
        for sigma in sigmas:
            for c in itertools.product([0, 1], repeat=t):
                starts = (1,) + c  # starts[s] = 1 if a segment starts at number s (number 0 always does)
                logw = -math.log(len(sigmas)) + sum(math.log(h if cs else 1 - h) for cs in c)
                begin = 0
                for s in range(t + 1):
                    if starts[s]:
                        begin = s
                    m, v = segment_posterior(x[begin:s], sigma, m0, s0)
                    if s < t:
                        logw += math.log(int_prob(x[s], m, math.sqrt(v + sigma**2)))
                qm, qv = segment_posterior(x[begin : t + 1], sigma, m0, s0)
                lik = int_prob(x[t], m, math.sqrt(v + sigma**2))
                rows.append((logw, sigma, m, v, qm, qv, lik, starts[t]))
        logw = np.array([r[0] for r in rows])
        W = np.exp(logw - logw.max())
        W /= W.sum()
        sig, pm, pv, qm, qv, lik, new = (np.array([r[i] for r in rows]) for i in range(1, 8))
        ev = (W * lik).sum()
        Q = W * lik / ev
        kl_gauss = np.array([TRAPZ(np.exp(lq) * (lq - lw), xs) for xs, lq, lw in (
            (xs, -0.5 * (xs - a) ** 2 / va - 0.5 * np.log(2 * np.pi * va), -0.5 * (xs - b) ** 2 / vb - 0.5 * np.log(2 * np.pi * vb))
            for a, va, b, vb in zip(qm, qv, pm, pv)
            for xs in [np.linspace(a - 14 * np.sqrt(va), a + 14 * np.sqrt(va), 40001)])])
        out["surprisal"][t] = -math.log(ev)
        out["bayes_kl"][t] = (Q * np.log(Q / W)).sum() + (Q * kl_gauss).sum()
        out["cpp"][t] = Q[new == 1].sum() if t > 0 else np.nan
        out["level_after"][t] = (Q * qm).sum()
        out["pred_mean"][t] = (W * pm).sum()
        out["pred_sd"][t] = math.sqrt((W * (pv + sig**2 + pm**2)).sum() - out["pred_mean"][t] ** 2)
        nm = h * m0 + (1 - h) * out["level_after"][t]
        out["next_mean"][t] = nm
        out["next_sd"][t] = math.sqrt(h * (s0**2 + m0**2) + (1 - h) * (Q * (qv + qm**2)).sum() + (Q * sig**2).sum() - nm**2)
        out["sigma_mean"][t] = (Q * sig).sum()
    return out


def test_observer_brute_force():
    sigmas, h, m0, s0 = np.array([2.0, 5.0, 11.0]), 0.15, 50.0, 30.0
    X = [[50, 53, 48, 51, 72, 70, 71], [50, 53, 48, 80, 49, 51, 50], [50, 53, 48, 51, 49, 52, 50]]
    obs = observe(np.array(X), hazard=h, m0=m0, s0=s0, sigmas=sigmas)
    for i, x in enumerate(X):
        bf = brute_force_observer(x, sigmas, h, m0, s0)
        for key, want in bf.items():
            got = obs[key][i]
            ok = np.allclose(got, want, rtol=1e-7, atol=1e-9, equal_nan=True)
            assert ok, f"sequence {i}, {key}: observe {got} vs brute force {want}"
    # rows with the same prefix give bitwise the same outputs on it (the twins' pre-periods are exactly zero)
    for key in obs:
        assert np.array_equal(obs[key][0, :4], obs[key][2, :4], equal_nan=True), key
        assert np.array_equal(obs[key][0, :3], obs[key][1, :3], equal_nan=True), key
    # a jump in a calm sequence is a change; nothing happens in the third sequence
    assert obs["cpp"][0, 4] > 0.5 and obs["cpp"][2, 4] < 0.5
    assert obs["bayes_kl"][0, 4] > obs["bayes_kl"][2, 4] and obs["surprisal"][1, 3] > obs["surprisal"][2, 3]


def test_observer_default_grid():
    """On the stimuli's scale: sigma is learned, and a jump is detected faster in a calm sequence."""
    rng = np.random.default_rng(0)
    X = []
    for sigma in (4, 16):
        x = np.rint(500 + sigma * rng.standard_normal(T)).astype(int)
        x[15:] += 4 * sigma
        X.append(x)
    obs = observe(np.array(X))
    assert abs(obs["sigma_mean"][0, 13] - 4) < 2 and abs(obs["sigma_mean"][1, 13] - 16) < 8
    assert (obs["next_mean"][:, 20] - 500 > 3 * np.array([4, 16])).all(), "the observer should have adapted"
    assert np.isfinite(obs["bayes_kl"]).all() and (obs["bayes_kl"] >= -1e-12).all()


# ---------------------------------------------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------------------------------------------
def test_ols_fe_cluster():
    rng = np.random.default_rng(1)
    G, P, A = 40, 5, 3  # clusters, fixed-effect groups per cluster, rows per group
    n = G * P * A
    cl = np.repeat(np.arange(G), P * A)
    fe = np.repeat(np.arange(G * P), A)
    X = rng.standard_normal((n, 2)) + rng.standard_normal(G * P)[fe, None]
    y = 1.5 * X[:, 0] - 0.7 * X[:, 1] + 3 * rng.standard_normal(G * P)[fe] + rng.standard_normal(G)[cl] + rng.standard_normal(n)
    b, se, nobs, g = ols_fe_cluster(y, X, fe=fe, cluster=cl)
    D = np.zeros((n, G * P))
    D[np.arange(n), fe] = 1
    Z = np.column_stack([X, D])
    beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    u = y - Z @ beta
    bread = np.linalg.pinv(Z.T @ Z)
    s = np.zeros((G, Z.shape[1]))
    np.add.at(s, cl, Z * u[:, None])
    V = bread @ s.T @ s @ bread * G / (G - 1) * (n - 1) / (n - 2)  # nested fixed effects not counted (our convention)
    assert np.allclose(b, beta[:2], atol=1e-10) and np.allclose(se, np.sqrt(np.diag(V)[:2]), rtol=1e-8)
    assert nobs == n and g == G
    # without fixed effects: intercept first, HC1 when no cluster is given
    b0, se0, _, _ = ols_fe_cluster(y, X)
    Z = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    u = y - Z @ beta
    bread = np.linalg.inv(Z.T @ Z)
    V = bread @ (Z * u[:, None] ** 2).T @ Z @ bread * n / (n - 1) * (n - 1) / (n - 3)
    assert np.allclose(b0, beta) and np.allclose(se0, np.sqrt(np.diag(V)))
    # rows with a NaN are dropped
    y2 = y.copy()
    y2[[3, 50]] = np.nan
    keep = np.isfinite(y2)
    assert np.allclose(ols_fe_cluster(y2, X, fe=fe, cluster=cl)[0], ols_fe_cluster(y[keep], X[keep], fe=fe[keep], cluster=cl[keep])[0])


def test_sign_flip_boot_turnover():
    m, p = sign_flip_p(np.ones(10), n_perm=20000)
    assert m == 1 and 0.0005 < p < 0.005, p  # exact: 2 / 2^10
    assert sign_flip_p(np.array([-1.0, 1.0] * 10))[1] == 1.0
    est, lo, hi = boot_mean_ci(np.full((30, 2), 2.0))
    assert np.allclose([est, lo, hi], 2.0)
    est, lo, hi = boot_mean_ci(np.random.default_rng(0).standard_normal((400, 1)) + 1)
    assert lo[0] < 1 < hi[0] and hi[0] - lo[0] < 0.3
    ia, ca = np.array([1, 2, 3, -1]), np.array([0.5, 0.2, 0.1, 0.0])
    ib, cb = np.array([2, 3, 4, 5]), np.array([0.3, 0.3, 0.3, 0.3])
    assert abs(turnover(ia, ca, ib, cb, 4) - 0.6) < 1e-12  # {1,2,3} vs {2,3,4,5}
    cb0 = cb.copy()
    cb0[3] = 0
    assert abs(turnover(ia, ca, ib, cb0, 4) - 0.5) < 1e-12  # a zero coefficient is not in the set
    assert abs(turnover(ia, ca, ib, cb, 2) - (1 - 1 / 3)) < 1e-12  # first k atoms only: {1,2} vs {2,3}
    assert np.isnan(turnover(np.full(4, -1), ca, ib, cb, 4)) and turnover(ia, ca, ia, ca, 4) == 0


# ---------------------------------------------------------------------------------------------------------------
# stimuli
# ---------------------------------------------------------------------------------------------------------------
def test_stimuli(data=DATA):
    path = os.path.join(data, "changepoint.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run stimuli.py first")
    df = pd.read_csv(path)
    for item, sub in df.groupby("item"):
        sub = sub.set_index("arm").loc[list(ARMS)]
        assert sub[["sigma", "delta", "direction", "tau", "mu0", "shift"]].nunique().max() == 1
        tau, shift = int(sub.tau.iloc[0]), int(sub["shift"].iloc[0])
        x = {arm: np.array(list(map(int, sub.numbers[arm].split()))) for arm in ARMS}
        assert all(len(v) == T for v in x.values())
        c, j, o = x["control"], x["jump"], x["outlier"]
        assert (j[: tau - 1] == c[: tau - 1]).all() and (j[tau - 1 :] - c[tau - 1 :] == shift).all()
        assert (np.delete(o, tau - 1) == np.delete(c, tau - 1)).all() and o[tau - 1] == j[tau - 1]
        assert abs(shift) == sub.delta.iloc[0] * sub.sigma.iloc[0] and shift != 0
        for arm in ARMS:
            assert sub.text[arm] == PREFIX + SEP.join(map(str, x[arm]))
            assert 0 <= x[arm].min() and x[arm].max() < 1000
    items = [json.loads(line) for line in open(os.path.join(data, "arith.jsonl"))]
    assert len(items) > 100 or data != DATA, "too few arithmetic items"
    for it in items:
        tx = it["texts"]
        assert len({len(v) for v in tx.values()}) == 1
        rs, re_ = it["edit"]["clean"]
        a_s, b_e = it["edit"]["swap"]
        r, a, b = it["r"], it["a"], it["b"]
        assert tx["clean"][rs:re_] == str(r) and (a + b if it["op"] == "+" else a * b) == r
        assert tx["near"][rs:re_] == str(it["near_value"]) and tx["far"][rs:re_] == str(it["far_value"])
        for name, (s, e) in it["edit"].items():
            assert tx[name][:s] == tx["clean"][:s] and tx[name][e:] == tx["clean"][e:], name
        region = tx["clean"][a_s:b_e]
        assert region.startswith(str(a)) and region.endswith(str(b))
        assert tx["swap"][a_s:b_e] == str(b) + region[len(str(a)) : len(region) - len(str(b))] + str(a)
        assert tx["swap"] != tx["clean"] and b_e <= rs
        near, far = it["near_value"], it["far_value"]
        assert 0 < abs(near - r) <= 2 and len(str(near)) == len(str(r)) == len(str(far))
        lead = abs(int(str(far)[0]) - int(str(r)[0]))
        assert (lead >= 3 and str(far)[1:] == str(r)[1:]) if r >= 10 else abs(far - r) >= 4
        if it["reuse"]:
            s, e = it["reuse"]
            assert s >= re_ and all(v[s:e] == str(r) for v in tx.values())
            assert not tx["clean"][s - 1].isdigit() and (e == len(tx["clean"]) or not tx["clean"][e].isdigit())
    print(f"    {df.item.nunique()} change-point items, {len(items)} arithmetic items")


# ---------------------------------------------------------------------------------------------------------------
# analysis on planted data
# ---------------------------------------------------------------------------------------------------------------
def analysis_args():
    return argparse.Namespace(hazard=1 / 30, boot=200, perm=2000, tol=1e-3, all_slots=False, by_layer=False)


def planted_changepoint(raw, a_kl, b_su, violate=False):
    """Synthetic Part A outputs: the readout of the surprise words is a_kl * KL + b_su * surprisal (the observer's),
    the hidden-state jump and the J-space vary only with (item, t) (no effect), and the model predicts like the
    observer."""
    rng = np.random.default_rng(0)
    L, K = 8, 12
    DEC = decomp_layers(L)
    rows = []
    for item in range(48):
        sigma, delta = (4, 8)[item % 2], (2, 4)[(item // 2) % 2]
        direction = 1 if item % 4 < 2 else -1
        tau = int(rng.integers(12, 23))
        mu0 = int(rng.integers(200, 800))
        c = np.rint(mu0 + sigma * rng.standard_normal(T)).astype(int)
        j = c.copy()
        j[tau - 1 :] += direction * delta * sigma
        o = c.copy()
        o[tau - 1] = j[tau - 1]
        for arm, x in zip(ARMS, (c, j, o)):
            rows.append({"item": item, "arm": arm, "sigma": sigma, "delta": delta, "direction": direction, "tau": tau,
                         "mu0": mu0, "shift": direction * delta * sigma, "numbers": " ".join(map(str, x))})
    meta = pd.DataFrame(rows)
    meta["row"] = np.arange(len(meta))
    N = len(meta)
    X = np.array([list(map(int, s.split())) for s in meta.numbers])
    obs = observe(X)
    item = meta.item.to_numpy()
    shared = rng.standard_normal((48, T))[item]  # one value per (item, t), the same in the three arms
    R = {"logp": (-obs["surprisal"]).astype(np.float32), "pred": np.full((N, T, 4), np.nan, np.float32)}
    R["pred"][:, :-1, 0] = obs["next_mean"][:, :-1]
    R["pred"][:, :-1, 3] = 1
    R["norm"] = np.ones((N, T, 3, L), np.float32)
    R["delta"] = np.broadcast_to((1 + 0.1 * shared)[:, :, None, None], (N, T, 3, L)).astype(np.float32).copy()
    R["delta"][:, 0] = np.nan
    R["div"] = np.zeros((N, T, 3, L), np.float32)
    R["lex"] = np.zeros((N, T, 3, L, len(GROUPS)), np.float32)
    R["lex"][..., GI["surprise"]] = (2 * (a_kl * obs["bayes_kl"] + b_su * obs["surprisal"]))[:, :, None, None]
    if violate:  # a difference before the change in the jump arm: the pipeline check must catch it
        r = np.flatnonzero((meta.arm == "jump").to_numpy())[0]
        R["lex"][r, 3, 1, :, GI["surprise"]] += 1
    for key in ("norm", "delta", "div"):
        R[key][:, -1, 1:] = np.nan
    R["lex"][:, -1, 1:] = np.nan
    R["top"] = np.full((N, T, L, 10), -1, np.int32)
    idx = rng.integers(0, 1000, (48, T, 2, len(DEC), K))[item].astype(np.int32)
    idx[:, -1] = -1
    R["dec_idx"], R["dec_coef"] = idx, np.ones(idx.shape, np.float16)
    R["dec_res"] = np.full(idx.shape, 0.5, np.float16)
    os.makedirs(raw, exist_ok=True)
    meta.to_csv(os.path.join(raw, "changepoint_meta.csv"), index=False)
    np.savez(os.path.join(raw, "changepoint.npz"), **R)
    json.dump({"decomp_layers": DEC}, open(os.path.join(raw, "run_info.json"), "w"))
    return obs


def planted_arith(raw, violate=False):
    """Synthetic Part B outputs: the readout of the error words moves by e_near, e_far after the wrong value and by
    e_swap after the harmless swap; everything else is shared across twins (no effect)."""
    rng = np.random.default_rng(1)
    M, L, K, W = 60, 8, 12, 30
    DEC = decomp_layers(L)
    pos = np.tile(np.arange(10, 10 + W), (M, 1)).astype(np.int32)
    first_swap, last_swap, res = 18, 20, 23
    e = {"near": 1 + 0.2 * rng.standard_normal(M), "far": 1.5 + 0.2 * rng.standard_normal(M),
         "swap": 0.3 + 0.2 * rng.standard_normal(M)}
    start = {"near": res, "far": res, "swap": first_swap}
    bg = rng.standard_normal((M, W))
    sig = np.repeat(bg[:, None], 4, 1)  # (M, twin, W)
    for w in ("near", "far", "swap"):
        sig[:, TWINS.index(w), pos[0] >= start[w]] += e[w][:, None]
    if violate:
        sig[0, TWINS.index("swap"), 15 - 10] += 1
    R = {"pos": pos, "logp": np.full((M, 4, W), -1.0, np.float32), "res_logp": np.zeros((M, 4, 3), np.float32),
         "res_top1": np.ones((M, 4), bool)}
    R["logp"][:, 1:3, res - 10] = -8
    R["norm"] = np.ones((M, 4, W, L), np.float32)
    R["delta"] = np.broadcast_to((1 + 0.1 * bg)[:, None, :, None], (M, 4, W, L)).astype(np.float32).copy()
    R["div"] = np.zeros((M, 4, W, L), np.float32)
    R["lex"] = np.zeros((M, 4, W, L, len(GROUPS)), np.float32)
    R["lex"][..., GI["error"]] = 2 * sig[..., None]
    idx = np.broadcast_to(rng.integers(0, 1000, (M, 1, W, len(DEC), K)), (M, 4, W, len(DEC), K)).astype(np.int32).copy()
    R["dec_idx"], R["dec_coef"] = idx, np.ones(idx.shape, np.float16)
    R["dec_res"] = np.full(idx.shape, 0.5, np.float16)
    meta = pd.DataFrame({"item": np.arange(M), "n_tokens": 60, "result_pos": res,
                         "reuse_pos": np.where(np.arange(M) % 2 == 0, 35, -1),
                         **{f"first_{w}": (first_swap if w == "swap" else res) for w in TWINS},
                         **{f"last_{w}": (last_swap if w == "swap" else res) for w in TWINS},
                         "r": 12, "near_value": 13, "far_value": 42, "a": 5, "b": 7, "op": "+"})
    os.makedirs(raw, exist_ok=True)
    meta.to_csv(os.path.join(raw, "arith_meta.csv"), index=False)
    np.savez(os.path.join(raw, "arith.npz"), **R)
    json.dump({"decomp_layers": DEC}, open(os.path.join(raw, "run_info.json"), "w"))
    return e


def test_analysis_planted():
    import analyze

    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw")
        out_csv, out_fig = os.path.join(tmp, "res_"), os.path.join(tmp, "fig_")
        a_kl, b_su = 1.0, 0.4
        obs = planted_changepoint(raw, a_kl, b_su)
        checks, _, verdict = analyze.analyze_changepoint(raw, "synthetic", analysis_args(), out_csv, out_fig)
        assert all(c["ok"] for c in checks) and len(checks) == 15, checks
        assert abs(verdict["A_update_slope"] - 1) < 1e-3 and verdict["A_surprisal_corr"] > 0.999, verdict
        reg = pd.read_csv(out_csv + "changepoint_regression.csv")
        sel = slice(3, T - 1)
        y = a_kl * obs["bayes_kl"][:, sel] + b_su * obs["surprisal"][:, sel]
        row = reg[(reg.signal == "lex_surprise") & (reg.slot == "sep") & (reg.layer == "band") & (reg.spec == "twin")].iloc[0]
        want_kl = a_kl * np.std(obs["bayes_kl"][:, sel]) / np.std(y)
        want_su = b_su * np.std(obs["surprisal"][:, sel]) / np.std(y)
        assert abs(row.b_bayes_kl - want_kl) < 1e-4 and abs(row.b_surprisal - want_su) < 1e-4, (row, want_kl, want_su)
        null = reg[(reg.signal.isin(["jump", "turnover10"])) & (reg.spec == "twin")]
        assert len(null) and null[["b_bayes_kl", "b_surprisal"]].abs().max().max() < 1e-6, null
        ev = pd.read_csv(out_csv + "changepoint_event.csv")
        pre = ev[(ev.k < 0) & ev.contrast.isin(["jump - control", "outlier - control"])]
        assert pre.est.abs().max() == 0, pre[pre.est != 0]
        planted_changepoint(raw, a_kl, b_su, violate=True)
        checks, _, _ = analyze.analyze_changepoint(raw, "synthetic", analysis_args(), out_csv, out_fig)
        bad = [c["check"] for c in checks if not c["ok"]]
        assert bad == ["A lex_surprise: jump - control before tau", "A lex_surprise: jump - outlier through tau"], bad

        e = planted_arith(raw)
        checks, _, verdict = analyze.analyze_arith(raw, "synthetic", analysis_args(), out_csv, out_fig)
        assert all(c["ok"] for c in checks) and len(checks) == 12, checks
        assert verdict["B_go"] and verdict["B_knows"] == 1 and abs(verdict["B_error_surprisal_gap"] - 7) < 1e-6
        tests = pd.read_csv(out_csv + "arith_tests.csv")

        def get(signal, contrast, window="k = 1..5"):
            return tests[(tests.signal == signal) & (tests.contrast == contrast) & (tests.window == window)].iloc[0]

        for contrast, want in (("near - clean", e["near"]), ("far - clean", e["far"]), ("swap - clean", e["swap"]),
                               ("near - swap", e["near"] - e["swap"]), ("far - swap", e["far"] - e["swap"]),
                               ("far - near", e["far"] - e["near"])):
            row = get("lex_error", contrast)
            assert abs(row["mean"] - want.mean()) < 1e-5 and row.p_sign_flip < 0.01 and row.lo < want.mean() < row.hi, (contrast, row)
        row = get("lex_error", "near - clean", "reuse of r")
        assert abs(row["mean"] - e["near"][::2].mean()) < 1e-5 and row.n == 30
        for sig_name in ("jump", "turnover10", "div"):
            assert get(sig_name, "near - swap")["mean"] == 0, sig_name
        ev = pd.read_csv(out_csv + "arith_event.csv")
        s = ev[(ev.signal == "lex_error") & (ev.contrast == "near - clean")].set_index("k").est
        assert (s.loc[-5:-1] == 0).all() and abs(s.loc[0] - e["near"].mean()) < 1e-5
        planted_arith(raw, violate=True)
        checks, _, _ = analyze.analyze_arith(raw, "synthetic", analysis_args(), out_csv, out_fig)
        bad = [c["check"] for c in checks if not c["ok"]]
        assert bad == ["B lex_error: swap - clean before the edit"], bad


# ---------------------------------------------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------------------------------------------
def check_tokenizer(model, data=DATA):
    from transformers import AutoTokenizer

    from jlens_open import MODELS
    from prep import integer_table, prepare_arith, prepare_changepoint, tokenize

    tok = AutoTokenizer.from_pretrained(MODELS[model]["hf"])
    df = pd.read_csv(os.path.join(data, "changepoint.csv"))
    items = [json.loads(line) for line in open(os.path.join(data, "arith.jsonl"))]
    for text in (df.text.iloc[0], items[0]["texts"]["clean"]):  # the same ids as LensModel.encode
        ids = tok(text).input_ids
        if tok.bos_token_id is not None and ids[0] != tok.bos_token_id:
            ids = [tok.bos_token_id] + ids
        assert tokenize(tok, text)[0] == ids
    # every token's character span holds the token's text (tokens that are part of a multi-byte character excepted)
    texts = list(df.text) + [t for it in items for t in it["texts"].values()]
    bad, empty, n_tok = 0, 0, 0
    for text in texts:
        ids, offs = tokenize(tok, text)
        skip = 1 if tok.bos_token_id is not None else 0
        for i, (a, b) in zip(ids[skip:], offs[skip:]):
            piece = tok.decode([i])
            n_tok += 1
            empty += b <= a and piece.strip() != ""
            bad += "�" not in piece and piece != text[a:b] and piece.strip() != text[a:b].strip()
    print(f"    token spans: {bad} of {n_tok} do not match the decoded token, {empty} are empty")
    assert bad == 0 and empty == 0
    table = integer_table(tok)
    lo, hi = min(df.numbers.str.split().explode().astype(int)), max(df.numbers.str.split().explode().astype(int))
    print(f"    {len(table)} of the integers 0..999 are one token; the stimuli use {lo}..{hi}")
    prepared, n_drop = prepare_changepoint(df, tok, list(df.item.unique()))
    print(f"    Part A: {len(prepared)} items valid, {n_drop} dropped")
    ids, P, nums = prepared[next(iter(prepared))][0]
    slots = [repr(tok.decode([ids[P[0, s]]])) for s in range(3)]
    print(f"    Part A slots of the first number {nums[0]}: num / sep / pred = {' / '.join(slots)}")
    for it, arms in prepared.items():
        for ids, P, nums in arms:
            assert [tok.decode([ids[p]]).strip() for p in P[:, 0]] == [str(x) for x in nums]
            assert len({ids[p] for p in P[:-1, 1]}) == 1 and len({ids[p] for p in P[:-1, 2]}) == 1
    prepared, reasons, wmax = prepare_arith(items, tok, 5, 15)
    print(f"    Part B: {len(prepared)} items valid of {len(items)}, dropped {reasons}; longest window {wmax}")
    for p in prepared:
        it, r = p["it"], p["result"]
        for name, value in (("clean", it["r"]), ("near", it["near_value"]), ("far", it["far_value"])):
            assert tok.decode([p["ids"][TWINS.index(name)][r]]).strip() == str(value)
        if p["reuse"] > 0:
            assert tok.decode([p["ids"][0][p["reuse"]]]).strip() == str(it["r"])
            assert len({ids[p["reuse"]] for ids in p["ids"]}) == 1
        for name in TWINS:
            ids = np.array(p["ids"][TWINS.index(name)])
            first = p["edit"]["swap"][0] if name == "swap" else r
            assert (ids[:first] == np.array(p["ids"][0])[:first]).all()
        last = max(p["edit"]["near"][1], p["edit"]["swap"][1])
        assert all((np.array(ids)[last + 1 :] == np.array(p["ids"][0])[last + 1 :]).all() for ids in p["ids"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=None, help="also check prep.py with this model's tokenizer")
    parser.add_argument("--data", default=DATA)
    args = parser.parse_args()
    tests = [test_log_int_prob, test_gauss_kl, test_observer_brute_force, test_observer_default_grid, test_ols_fe_cluster,
             test_sign_flip_boot_turnover, lambda: test_stimuli(args.data), test_analysis_planted]
    names = ["log_int_prob", "gauss_kl", "observer vs brute force", "observer on the stimuli's scale", "ols_fe_cluster",
             "sign flip, bootstrap, turnover", "stimuli invariants", "analysis on planted data"]
    if args.tokenizer:
        tests.append(lambda: check_tokenizer(args.tokenizer, args.data))
        names.append(f"tokenizer of {args.tokenizer}")
    failed = 0
    for name, test in zip(names, tests):
        try:
            test()
            print(f"PASS  {name}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL  {name}\n{traceback.format_exc()}", flush=True)
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
