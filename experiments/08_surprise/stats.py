"""Statistics for experiment 08: fixed-effects OLS with clustered errors, bootstrap, sign-flip test, J-space turnover."""
import warnings

import numpy as np
import pandas as pd


def demean(cols, groups):
    """Subtract the group mean from each column of cols (n, k): the within transformation of fixed effects."""
    codes = pd.factorize(pd.Series(list(map(tuple, groups)) if np.ndim(groups) > 1 else groups))[0]
    counts = np.bincount(codes)
    out = np.empty_like(cols, dtype=float)
    for j in range(cols.shape[1]):
        out[:, j] = cols[:, j] - (np.bincount(codes, weights=cols[:, j]) / counts)[codes]
    return out


def ols_fe_cluster(y, X, fe=None, cluster=None):
    """OLS of y (n,) on X (n, k), with fixed effects for the groups in `fe` (absorbed by demeaning; an intercept
    is added when fe is None) and CR1 cluster-robust standard errors by `cluster` (heteroskedasticity-robust HC1
    when None). Fixed effects nested in clusters are not counted in the degrees of freedom (Stata's convention).
    Rows with a NaN are dropped. Returns (coef, se, n_obs, n_clusters); with fe=None the intercept comes first."""
    y, X = np.asarray(y, float), np.asarray(X, float).reshape(len(y), -1)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    fe = None if fe is None else np.asarray(fe)[ok]
    cluster = np.arange(len(y)) if cluster is None else np.asarray(cluster)[ok]
    if fe is None:
        X = np.column_stack([np.ones(len(y)), X])
        Z = np.column_stack([y, X])
    else:
        Z = demean(np.column_stack([y, X]), fe)
    yd, Xd = Z[:, 0], Z[:, 1:]
    n, k = Xd.shape
    bread = np.linalg.pinv(Xd.T @ Xd)
    beta = bread @ Xd.T @ yd
    u = yd - Xd @ beta
    codes = pd.factorize(cluster)[0]
    G = codes.max() + 1
    scores = np.zeros((G, k))
    np.add.at(scores, codes, Xd * u[:, None])
    V = bread @ (scores.T @ scores) @ bread * (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    return beta, np.sqrt(np.clip(np.diag(V), 0, None)), n, G


def boot_mean_ci(D, n_boot=1000, seed=0, level=0.95):
    """Mean over units (rows of D, NaN allowed) of each column, with a percentile bootstrap CI over units."""
    D = np.asarray(D, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(D), (n_boot, len(D)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns give NaN
        means = np.nanmean(D[idx], axis=1)
        est = np.nanmean(D, axis=0)
        a = (1 - level) / 2
        return est, np.nanquantile(means, a, axis=0), np.nanquantile(means, 1 - a, axis=0)


def sign_flip_p(d, n_perm=10000, seed=0):
    """Two-sided randomization p-value for mean(d) = 0, flipping the sign of each unit's d (a paired design:
    under the null, which twin is which is exchangeable). Returns (mean, p)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    stat = d.mean()
    null = (rng.choice([-1.0, 1.0], (n_perm, len(d))) * d).mean(1)
    return stat, (1 + np.sum(np.abs(null) >= abs(stat) - 1e-15)) / (1 + n_perm)


def turnover(idx_a, coef_a, idx_b, coef_b, k):
    """Jaccard distance between the J-space sets {first k atoms with a positive coefficient} of two activations.
    idx_*, coef_* (..., K); a missing activation has idx -1. Returns (...) in [0, 1], NaN when a set is empty."""
    def members(idx, coef):
        m = idx[..., :k].astype(np.int64)
        valid = (m >= 0) & (coef[..., :k].astype(float) > 0)
        return np.where(valid, m, -1), valid

    a, va = members(idx_a, coef_a)
    b, vb = members(idx_b, coef_b)
    inter = ((a[..., :, None] == b[..., None, :]) & va[..., :, None] & vb[..., None, :]).sum((-1, -2))
    union = va.sum(-1) + vb.sum(-1) - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        out = 1 - inter / union
    return np.where((va.sum(-1) > 0) & (vb.sum(-1) > 0), out, np.nan)
