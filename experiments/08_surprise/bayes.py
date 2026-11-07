"""The ideal observer of the change-point task: exact Bayesian online change-point detection.

How the observer believes the numbers are made (Adams and MacKay 2007, "Bayesian Online Changepoint
Detection", with the noise level learned):
  - the noise sd sigma is the same for the whole sequence, with a uniform prior on the grid SIGMAS;
  - the first number starts a segment, and before each later number a new segment starts with probability h;
  - each segment has a level mu ~ N(M0, S0^2), drawn independently of the previous level;
  - within a segment, x_t = mu + sigma * eps_t, with eps_t ~ N(0, 1) independent.
Given sigma and the number r of observations in the current segment, the level has a Gaussian posterior
(conjugate update), so the belief about (sigma, r, mu) is a finite mixture of Gaussians, updated exactly.

The numbers are integers. The observer gives the integer k the probability of [k - 1/2, k + 1/2] under its
predictive distribution (what a language model's token probability estimates), and updates the level as if k
were a continuous observation. With noise sd >= 4 in the stimuli, rounding changes a likelihood by under 1%.

Returned for each number x_t (nats unless stated):
  surprisal     -log p(x_t | x_1..x_{t-1}): how unlikely the number was ("Shannon surprise").
  bayes_kl      KL(belief after x_t || belief before x_t) over (sigma, r, mu_t): how far the belief moved
                ("Bayesian surprise", Itti and Baldi 2009). Exact: the two mixtures share their components.
  kl_discrete   the part of bayes_kl from the weights of (sigma, r) alone.
  cpp           posterior probability that x_t started a new segment ("change-point probability",
                Nassar et al. 2010); NaN for the first number, which always starts one.
  level_before, level_after   E[mu_t] before and after seeing x_t.
  shift         |level_after - level_before| / sigma_mean: how far the best guess of the level moved, in noise sds.
  pred_mean, pred_sd          predictive mean and sd of x_t before seeing it; z = (x_t - pred_mean) / pred_sd.
  next_mean, next_sd          predictive mean and sd of x_{t+1} after seeing x_t.
  sigma_mean                  E[sigma | x_1..x_t].
"""
import math

import numpy as np
import torch

SIGMAS = np.exp(np.linspace(np.log(1.0), np.log(40.0), 25))
HAZARD = 1 / 30
M0, S0 = 500.0, 250.0
KEYS = ("surprisal", "bayes_kl", "kl_discrete", "cpp", "level_before", "level_after", "shift", "pred_mean",
        "pred_sd", "z", "next_mean", "next_sd", "sigma_mean")


def log_int_prob(x, mean, sd):
    """log P(x - 1/2 < X < x + 1/2) for X ~ N(mean, sd^2), accurate far in both tails (torch float64)."""
    a = (x - 0.5 - mean) / sd
    b = (x + 0.5 - mean) / sd
    flip = a > 0  # in the upper tail, P(a < Z < b) = P(-b < Z < -a), which log_ndtr resolves accurately
    lo, hi = torch.where(flip, -b, a), torch.where(flip, -a, b)
    l_hi, l_lo = torch.special.log_ndtr(hi), torch.special.log_ndtr(lo)
    return l_hi + torch.log1p(-torch.exp(l_lo - l_hi))


def gauss_kl(m1, v1, m0, v0):
    """KL(N(m1, v1) || N(m0, v0)) for variances v1, v0."""
    return 0.5 * (torch.log(v0 / v1) + (v1 + (m1 - m0) ** 2) / v0 - 1)


@torch.no_grad()
def observe(X, hazard=HAZARD, m0=M0, s0=S0, sigmas=SIGMAS):
    """Run the observer on integer sequences X (n, T). Returns {key: (n, T) float64 array} for KEYS."""
    X = torch.as_tensor(np.asarray(X, dtype=np.float64))
    n, T = X.shape
    sig = torch.as_tensor(np.asarray(sigmas, dtype=np.float64))
    S = len(sig)
    sig2 = (sig**2)[None, :, None]  # (1, S, 1)
    log_h, log_1mh = math.log(hazard), math.log1p(-hazard)
    out = {k: np.full((n, T), np.nan) for k in KEYS}
    logw = m = v = None  # posterior after t numbers: (n, S, t), column j = run length j + 1
    for t in range(T):
        x = X[:, t][:, None, None]
        # Belief about (sigma, component, mu_t) before x_t. Component 0: a new segment starts at x_t (level from
        # the prior); component j >= 1: the segment that has j numbers so far goes on (with prob 1 - h)
        new_m = torch.full((n, S, 1), m0, dtype=torch.float64)
        new_v = torch.full((n, S, 1), s0**2, dtype=torch.float64)
        if t == 0:
            pw = torch.full((n, S, 1), -math.log(S), dtype=torch.float64)
            pm, pv = new_m, new_v
        else:
            pw = torch.cat([torch.logsumexp(logw, 2, keepdim=True) + log_h, logw + log_1mh], 2)
            pm, pv = torch.cat([new_m, m], 2), torch.cat([new_v, v], 2)
        lp = log_int_prob(x, pm, torch.sqrt(pv + sig2))
        joint = pw + lp
        log_ev = torch.logsumexp(joint.reshape(n, -1), 1)
        qw = joint - log_ev[:, None, None]
        qv = 1 / (1 / pv + 1 / sig2)  # conjugate update of the level
        qm = qv * (pm / pv + x / sig2)
        W, Q = pw.exp(), qw.exp()
        kl_d = (Q * (qw - pw)).sum((1, 2))
        kl = kl_d + (Q * gauss_kl(qm, qv, pm, pv)).sum((1, 2))
        before = (W * pm).sum((1, 2))
        after = (Q * qm).sum((1, 2))
        pred_var = (W * (pv + sig2 + pm**2)).sum((1, 2)) - before**2
        sigma_mean = (Q.sum(2) * sig[None]).sum(1)
        # Predictive for x_{t+1}: new segment with prob h (prior level), else the current one goes on
        nxt_mean = hazard * m0 + (1 - hazard) * after
        nxt_second = hazard * (s0**2 + m0**2) + (1 - hazard) * (Q * (qv + qm**2)).sum((1, 2)) + (Q.sum(2) * sig[None] ** 2).sum(1)
        vals = {
            "surprisal": -log_ev, "bayes_kl": kl, "kl_discrete": kl_d,
            "cpp": Q[:, :, 0].sum(1) if t > 0 else torch.full((n,), float("nan"), dtype=torch.float64),
            "level_before": before, "level_after": after, "shift": (after - before).abs() / sigma_mean,
            "pred_mean": before, "pred_sd": pred_var.clamp(min=0).sqrt(), "next_mean": nxt_mean,
            "next_sd": (nxt_second - nxt_mean**2).clamp(min=0).sqrt(), "sigma_mean": sigma_mean,
        }
        vals["z"] = (X[:, t] - before) / vals["pred_sd"]
        for k, val in vals.items():
            out[k][:, t] = val.numpy()
        logw, m, v = qw, qm, qv
    return out
