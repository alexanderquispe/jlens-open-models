"""The ideal observer of experiment 10: an exact Bayesian filter that knows streams can jump, change their noise, and
contain one-off outliers.

How the observer believes the numbers are made:
  - the stream has a level mu and a noise sd sigma (the state);
  - before each number after the first, mu is redrawn from its prior with probability h_mu (a change point), and,
    independently, sigma is redrawn from its prior with probability h_sig;
  - each number is, with probability rho, an outlier drawn uniformly from the integers 10..990; otherwise it is
    N(mu, sigma^2), rounded and clipped to [10, 990] (exactly how the stimuli are made).
Priors: mu uniform on MU_GRID (0..1000, step 2), sigma uniform on the log-spaced SIG_GRID (10..400). The state is
Markov (a change point redraws it independently of the past), so the posterior over the (sigma, mu) grid is updated
exactly at every number; the only approximation is the grid.

This is the observer O'Reilly et al. (2013, PNAS) and Nassar et al. (2010, J. Neurosci.) use to tell surprise from
belief updating; experiment 08's observer (08_surprise/bayes.py) had no outlier process, which is why its two
surprises moved together (r = 0.917).

Returned for each number x_t (nats unless stated):
  surprisal     -log p(x_t | x_1..x_{t-1}): how unlikely the number was.
  bayes_kl      KL(posterior after x_t || belief before x_t) over the (sigma, mu) grid: how far the belief moved.
  kl_mu         the same for the marginal belief about mu alone.
  cpp           posterior probability that mu was redrawn just before x_t (change-point probability).
  opp           posterior probability that x_t was an outlier.
  level_before, level_after   E[mu] before and after x_t;  level_sd  sd of mu after x_t.
  sigma_mean    E[sigma] after x_t.
  shift         |level_after - level_before| / sigma_mean.
  pred_mean, pred_sd          predictive mean and sd of x_t before seeing it.
  next_mean, next_sd          predictive mean and sd of x_{t+1} after seeing x_t (what the model predicts at the
                              comma after x_t).
`dist_at` also returns the whole predictive distribution of x_{t+1} over the integers 0..999 at chosen t.
"""
import numpy as np
import torch

LO, HI = 10, 990
MU_GRID = np.arange(0.0, 1001.0, 2.0)
SIG_GRID = np.exp(np.linspace(np.log(10.0), np.log(400.0), 24))
H_MU, H_SIG, RHO = 1 / 200, 1 / 500, 1 / 100
KEYS = ("surprisal", "bayes_kl", "kl_mu", "cpp", "opp", "level_before", "level_after", "level_sd", "sigma_mean",
        "shift", "pred_mean", "pred_sd", "next_mean", "next_sd")
N_INT = HI - LO + 1
_U = np.arange(LO, HI + 1, dtype=np.float64)
U_MEAN, U_SECOND = _U.mean(), (_U**2).mean()  # moments of the outlier distribution


def bin_probs(mu_grid=MU_GRID, sig_grid=SIG_GRID, lo=0, hi=999):
    """P(round-and-clip(N(mu, sigma^2)) = k) for k = lo..hi: (S, M, hi - lo + 1), float64. The end bins 10 and 990
    take the clipped tails; integers outside [10, 990] have probability 0."""
    k = torch.arange(lo, hi + 1, dtype=torch.float64)
    a = (k - 0.5)[None, None, :]
    b = (k + 0.5)[None, None, :]
    a = torch.where(k[None, None, :] == LO, torch.tensor(-float("inf"), dtype=torch.float64), a)
    b = torch.where(k[None, None, :] == HI, torch.tensor(float("inf"), dtype=torch.float64), b)
    mu = torch.as_tensor(mu_grid, dtype=torch.float64)[None, :, None]
    sig = torch.as_tensor(sig_grid, dtype=torch.float64)[:, None, None]
    za, zb = (a - mu) / sig, (b - mu) / sig
    # P(za < Z < zb), accurate in both tails: in the upper tail use P(-zb < Z < -za)
    flip = za > 0
    lo_, hi_ = torch.where(flip, -zb, za), torch.where(flip, -za, zb)
    l_hi, l_lo = torch.special.log_ndtr(hi_), torch.special.log_ndtr(lo_)
    p = torch.exp(l_hi + torch.log1p(-torch.exp(l_lo - l_hi)))
    inside = (k >= LO) & (k <= HI)
    return torch.where(inside[None, None, :], p, torch.zeros((), dtype=torch.float64))


def _transition(P, h_mu, h_sig):
    """Belief before the next number from the posterior P (n, S, M): mu redrawn with prob h_mu, sigma with h_sig.
    Returns (prior, part of the prior in which mu was redrawn)."""
    S, M = P.shape[1], P.shape[2]
    stay = (1 - h_mu) * P
    reset = h_mu * P.sum(2, keepdim=True).expand(-1, -1, M) / M
    sig = lambda Q: (1 - h_sig) * Q + h_sig * Q.sum(1, keepdim=True).expand(-1, S, -1) / S  # noqa: E731
    reset = sig(reset)
    return sig(stay) + reset, reset


@torch.no_grad()
def observe(X, h_mu=H_MU, h_sig=H_SIG, rho=RHO, mu_grid=MU_GRID, sig_grid=SIG_GRID, dist_at=()):
    """Run the observer on integer sequences X (n, T) (values in [10, 990]). Returns ({key: (n, T) array},
    {t: (n, 1000) predictive distribution of x_{t+1} after x_t} for t in dist_at)."""
    X = np.asarray(X)
    assert X.min() >= LO and X.max() <= HI
    n, T = X.shape
    mu = torch.as_tensor(mu_grid, dtype=torch.float64)
    sig = torch.as_tensor(sig_grid, dtype=torch.float64)
    S, M = len(sig), len(mu)
    table = bin_probs(mu_grid, sig_grid, LO, HI)              # (S, M, 981): likelihood of each integer
    full = bin_probs(mu_grid, sig_grid) if dist_at else None  # (S, M, 1000) for the predictive distributions
    out = {k: np.full((n, T), np.nan) for k in KEYS}
    dists = {}
    Xt = torch.as_tensor(X, dtype=torch.long)
    prior = torch.full((n, S, M), 1.0 / (S * M), dtype=torch.float64)
    reset = None
    mu2 = (mu**2)[None, None, :]
    sig2 = (sig**2)[None, :, None]

    def moments(P):
        m1 = (P * mu[None, None, :]).sum((1, 2))
        second = (P * (mu2 + sig2)).sum((1, 2))
        mean = (1 - rho) * m1 + rho * U_MEAN
        var = (1 - rho) * second + rho * U_SECOND - mean**2
        return mean, var.clamp(min=0).sqrt()

    for t in range(T):
        lik_n = table[:, :, Xt[:, t] - LO].permute(2, 0, 1)  # (n, S, M)
        lik = (1 - rho) * lik_n + rho / N_INT
        joint = prior * lik
        ev = joint.sum((1, 2))
        post = joint / ev[:, None, None]
        pos = post > 0
        logratio = torch.where(pos, torch.log(torch.where(pos, post, 1.0)) - torch.log(prior), 0.0)
        kl = (post * logratio).sum((1, 2))
        pm_post, pm_prior = post.sum(1), prior.sum(1)
        posm = pm_post > 0
        kl_mu = (pm_post * torch.where(posm, torch.log(torch.where(posm, pm_post, 1.0)) - torch.log(pm_prior), 0.0)).sum(1)
        before = (pm_prior * mu).sum(1)
        after = (pm_post * mu).sum(1)
        level_sd = ((pm_post * mu**2).sum(1) - after**2).clamp(min=0).sqrt()
        sigma_mean = (post.sum(2) * sig).sum(1)
        pred_mean, pred_sd = moments(prior)
        nxt, nxt_reset = _transition(post, h_mu, h_sig)
        next_mean, next_sd = moments(nxt)
        vals = {
            "surprisal": -torch.log(ev), "bayes_kl": kl, "kl_mu": kl_mu,
            "cpp": (reset * lik).sum((1, 2)) / ev if reset is not None else torch.full((n,), float("nan"), dtype=torch.float64),
            "opp": rho / N_INT / ev, "level_before": before, "level_after": after, "level_sd": level_sd,
            "sigma_mean": sigma_mean, "shift": (after - before).abs() / sigma_mean,
            "pred_mean": pred_mean, "pred_sd": pred_sd, "next_mean": next_mean, "next_sd": next_sd,
        }
        for k, v in vals.items():
            out[k][:, t] = v.numpy()
        if t in dist_at:
            d = torch.einsum("nsm,smk->nk", nxt, full) * (1 - rho)
            d[:, LO : HI + 1] += rho / N_INT
            dists[t] = (d / d.sum(1, keepdim=True)).numpy()
        prior, reset = nxt, nxt_reset
    return out, dists
