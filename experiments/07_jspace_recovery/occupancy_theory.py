"""The occupancy rule on a random dictionary in isotropic noise, from first principles (problem 1 of step 4).

Setting: N random unit atoms in R^d; y = a (v_1 + ... + v_k) + z with k planted atoms, z isotropic, and the share
s = |w|^2 / |y|^2 fixed by rescaling z (as plant() in recovery.py). The pursuit is nonnegative OMP (decompose in
jlens_open/jspace.py); the control is the same pursuit with N other random atoms; the occupancy of one y is the first
step K at which the dictionary's gain falls below the control's.

Reduced model. Write zeta_c = sqrt(d) <c, r> / |r| for the normalized score of an unpicked atom c against the
residual r. When the pursuit picks an atom e that explains a fraction rho of |r|^2, every unpicked atom's score
becomes, to leading order in 1/d,
    zeta_c  <-  sqrt(1 - rho) zeta_c - sqrt(rho) eta_c,    eta_c ~ N(0, 1) fresh and independent across atoms,
because <c, e> splits into a part along r (which shrinks zeta_c) and a part along e's new direction, which is
independent of everything c has seen. A planted atom scores alpha (1 - K/d) plus such a noise part, where the
planted amplitude alpha starts at a and is eaten by noise picks: a noise atom is picked for its correlation with
the whole residual, planted words included, so it removes the same fraction rho of their amplitude (alpha <- (1 -
rho) alpha); a planted pick leaves the other planted atoms' alpha alone (it removes only their cross terms). So the
pursuit in d dimensions becomes a process on N scalars: pick the largest, remove it, refresh the rest. This script
simulates that process, the full pursuit on random atoms (to check the reduction at sizes never run before), and the
closed forms derived from it.

    python experiments/07_jspace_recovery/occupancy_theory.py --part reduced   # Llama-3.1-8B and GPT-2 sizes
    python experiments/07_jspace_recovery/occupancy_theory.py --part check     # new sizes: reduced model and full OMP
    python experiments/07_jspace_recovery/occupancy_theory.py --part curves    # Llama size, finer k, no step cap
    python experiments/07_jspace_recovery/occupancy_theory.py --part compare   # tables and figures/occupancy_theory.*

Outputs: results/occupancy_theory_{reduced,check,curves}.csv (one row per model, size, s and k),
figures/occupancy_theory.{pdf,png}.
"""
import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = {"llama3.1-8b": (4096, 128256), "gpt2-small": (768, 50257),
         "check d=2048 N=16384": (2048, 16384), "check d=1024 N=131072": (1024, 131072), "check d=4096 N=16384": (4096, 16384)}
KS_DATA = (5, 10, 25, 50, 100, 200, 400)  # the grid of recovery.py
KS_CHECK = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200)
SHARES_DATA = (1.0, 0.5, 0.2, 0.1, 0.05)
SHARES_CHECK = (0.2, 0.1, 0.05)
OCC = "occupancy (median over trials)"


# Closed forms
def floor_b(N):
    """b_N with P(Z > b_N) = 1/N: the typical largest of N standard normal scores. B_N = b_N^2 is the noise floor
    in units of |y|^2 / d (2 ln N overstates it: 23.5 against 18.7 for N = 128,256)."""
    return -torch.special.ndtri(torch.tensor(1 / N, dtype=torch.float64)).item()


def kstar(s, d, N):
    """Planted atoms at which one carries as much as the best of N pure-noise atoms: s / k = B_N / d."""
    return s * d / floor_b(N) ** 2


def gumbel_scale(N):
    """a_N = N phi(b_N): near the top, x = a_N (score - b_N) is Gumbel, and the field of scores is a Poisson process
    with intensity e^{-x}; its j-th highest point is at -ln(E_1 + ... + E_j), E_i standard exponentials."""
    b = floor_b(N)
    return N * math.exp(-b * b / 2) / math.sqrt(2 * math.pi)


def margin(k, s, d, N):
    """Delta of the regime below the limit, in Gumbel units: (a_N b_N / 2) ln(R_control / R_dictionary) after the k
    planted picks. The dictionary has |r|^2 = (1 - s)(1 - k/d) left; the control has spent its k best noise atoms."""
    b, a = floor_b(N), gumbel_scale(N)
    return a * b / 2 * (-math.log(1 - s) - math.log(1 - k / d) + math.log(1 - k * (b * b + 1) / d))


def mean_occupancy_frozen(k, s, d, N):
    """E[occupancy] = (k + 1) / (1 - e^{-Delta}) below the limit, scores frozen (d large at fixed k). After the k
    planted picks the dictionary loses at its j-th noise step when Gamma_{k+j} < e^Delta Gamma'_j (two independent
    Poisson processes); W_j = Gamma_{k+j} - e^Delta Gamma'_j is a random walk with drift 1 - e^Delta started at
    Gamma_k, and Wald's identity with the exponential undershoot (mean e^Delta) gives E[J] = (k + e^Delta) / (e^Delta - 1)."""
    D = margin(k, s, d, N)
    return (k + 1) / (1 - math.exp(-D)) if D > 0 else float("inf")


_LEVELS = {}


def levels(v, K):
    """Gumbel location L_n (n = 1..K) of the top of a field that loses its top point at every step while every point
    diffuses (variance v and drift -v/2 per step, which leaves the Poisson field e^{-x} invariant). A removed point
    leaves a hole that diffuses away: e^{-L_n} = 1 + (expected holes above L_n). Frozen (v = 0): L_n = -ln n."""
    if (v, K) not in _LEVELS:
        L = np.zeros(K + 1)
        for n in range(2, K + 1):
            age = n - np.arange(1, n)
            x = L[n - 1]
            for _ in range(200):
                H = torch.special.ndtr(torch.tensor(-(x - L[1:n] + v * age / 2) / np.sqrt(v * age))).sum().item() if v else n - 1
                new = -math.log1p(H)
                if abs(new - x) < 1e-10:
                    break
                x = new
            L[n] = x
        _LEVELS[(v, K)] = L
    return _LEVELS[(v, K)]


def fluid_occupancy(k, s, d, N, memory=True, K=600):
    """Typical occupancy below the limit with the holes refilling (memory about 2 / v = 2 d / (a_N b_N)^2 steps):
    the first noise step j at which L_j - L_{k+j} < Delta, with the residuals of both pursuits followed step by step."""
    b, a = floor_b(N), gumbel_scale(N)
    L = levels(round(a * a * b * b / d, 6) if memory else 0.0, K)
    rho = lambda lev, step: (b + (lev + 0.5772) / a) ** 2 / (d - step)  # fraction of |r|^2 one noise pick explains
    RC = np.prod([1 - rho(L[l], l - 1) for l in range(1, k + 1)])
    RD = (1 - s) * (1 - k / d)
    for j in range(1, K - k):
        if L[j] - L[k + j] < a * b / 2 * math.log(RC / RD):
            return k + j
        RC, RD = RC * (1 - rho(L[k + j], k + j - 1)), RD * (1 - rho(L[j], k + j - 1))
    return float("nan")


def stop_at_first_step(k, s, d, N):
    """P(occupancy = 1): at step 1 both pursuits see the same y, so the dictionary loses iff its best score is below
    the control's. Its scores are N - k standard normals and k planted ones ~ N(mu, 1), mu = sqrt(s d / k):
    P = int Phi(x)^(N-k) Phi(x - mu)^k d[Phi(x)^N]. It tends to 1/2 far above the limit."""
    mu = math.sqrt(s * d / k)
    x = torch.linspace(-3, 12, 300001, dtype=torch.float64)
    logF = torch.special.log_ndtr(x)
    control = torch.exp(N * logF)
    dictionary = torch.exp((N - k) * logF + k * torch.special.log_ndtr(x - mu))
    return torch.trapz(dictionary * torch.gradient(control, spacing=(x,))[0], x).item()


# Reduced model
def sphere_scores(rng, shape, d):
    """sqrt(d) <u, e> for u uniform on the sphere of R^d: the exact law of a random atom's normalized score."""
    return (math.sqrt(d) * (2 * rng.beta((d - 1) / 2, (d - 1) / 2, size=shape) - 1)).astype(np.float32)


def reduced(rng, N, d, steps, T, k=0, s=1.0, refresh=True):
    """Gains (T, steps) in units of |y|^2 and, per step, whether a planted atom was picked (T, steps)."""
    xi = sphere_scores(rng, (T, N - k), d)  # unpicked noise atoms
    pl = rng.standard_normal((T, k), dtype=np.float32)  # planted atoms: normalized noise part of the score
    alive = np.ones((T, k), bool)
    R = np.ones(T)  # |r|^2 / |y|^2
    alpha = np.full(T, math.sqrt(s / k) if k else 0.0)  # planted amplitude left in the residual
    gains, planted = np.zeros((T, steps)), np.zeros((T, steps), bool)
    rows = np.arange(T)
    for K in range(steps):
        shrink = 1 - K / d  # |P_perp c|^2 for an atom against the K picked ones
        jn = xi.argmax(1)
        best = np.sqrt(R / d) * xi[rows, jn]  # raw score <c, r>
        choose = np.zeros(T, bool)
        if k:
            sp = (alpha * shrink)[:, None] + np.sqrt(np.clip(R - alpha**2 * shrink, 0, None) / d)[:, None] * pl
            sp = np.where(alive, sp, -np.inf)
            jp = sp.argmax(1)
            choose = sp[rows, jp] > best
            best = np.where(choose, sp[rows, jp], best)
            alive[rows[choose], jp[choose]] = False
        xi[rows[~choose], jn[~choose]] = -np.inf
        # nonnegative: a negative best adds nothing; an exhausted residual (s = 1) has nothing left to explain
        rho = np.where(R > 1e-12, np.clip(np.maximum(best, 0) ** 2 / (np.maximum(R, 1e-12) * shrink), 0, 1), 0.0)
        gains[:, K], planted[:, K] = R * rho, choose
        R = R * (1 - rho)
        alpha = np.where(choose, alpha, alpha * (1 - rho))
        if refresh:
            c, e = np.sqrt(1 - rho).astype(np.float32)[:, None], np.sqrt(rho).astype(np.float32)[:, None]
            dead = np.isneginf(xi)
            xi = c * xi - e * rng.standard_normal(xi.shape, dtype=np.float32)
            xi[dead] = -np.inf
            if k:
                pl = c * pl - e * rng.standard_normal(pl.shape, dtype=np.float32)
    return gains, planted


# Full pursuit on random atoms (decompose of jlens_open/jspace.py in float32, for the CPU)
def omp(X, atoms, steps):
    from jlens_open.jspace import _nnls
    n = len(X)
    R = X.clone()
    idx = torch.empty(n, 0, dtype=torch.long)
    coef = torch.empty(n, 0)
    res = []
    for _ in range(steps):
        scores = (R / R.norm(dim=1, keepdim=True).clamp(min=1e-6)) @ atoms.T
        if idx.shape[1]:
            scores.scatter_(1, idx, float("-inf"))
        idx = torch.cat([idx, scores.argmax(1, keepdim=True)], 1)
        A = atoms[idx]
        G = A @ A.transpose(1, 2)
        b = (A @ X[..., None]).squeeze(-1)
        coef = _nnls(G, b, torch.cat([coef, torch.zeros(n, 1)], 1))
        R = X - (coef[:, None, :] @ A).squeeze(1)
        res.append((R**2).sum(1))
    return idx, torch.stack(res, 1)


def random_unit(N, d, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(N, d, generator=g)
    return z / z.norm(dim=1, keepdim=True)


def full(N, d, steps, T, k, s, seed):
    """The experiment of recovery.py with a random dictionary, at any size: gains (T, steps) and planted picks."""
    g = torch.Generator().manual_seed(seed)
    atoms, control = random_unit(N, d, seed + 1), random_unit(N, d, seed + 2)
    idx = torch.rand(T, N, generator=g).topk(k, dim=1).indices
    w = atoms[idx].sum(1)
    z = torch.randn(T, d, generator=g)
    z *= (w.norm(dim=1) * math.sqrt((1 - s) / s) / z.norm(dim=1))[:, None]
    y = w + z
    picked, res = omp(y, atoms, steps)
    _, res_c = omp(y, control, steps)
    total = (y**2).sum(1, keepdim=True)
    gains = lambda r: (torch.cat([total, r[:, :-1]], 1) - r).numpy() / total.numpy()
    hit = (picked[:, :, None] == idx[:, None, :]).any(2).numpy()
    return gains(res), gains(res_c), hit


# Summaries
def summarize(g, gc, planted, k, steps):
    below = g < gc
    occ = np.where(below.any(1), below.argmax(1) + 1, steps + 1)
    return {OCC: np.median(occ), "occupancy (mean over trials)": occ.mean(), "share of trials with occupancy 1": (occ == 1).mean(),
            "occupancy 25th percentile": np.quantile(occ, 0.25), "occupancy 75th percentile": np.quantile(occ, 0.75),
            "trials censored": (occ > steps).mean(), "found in first k": planted[:, :k].sum(1).mean(),
            "found within all steps": planted.sum(1).mean(), "explained at k": g[:, :k].sum(1).mean(),
            "control explained at k": gc[:, :k].sum(1).mean()}


def cell(job):
    kind, size, k, s, T, seed, *long = job
    d, N = SIZES[size]
    steps = 3 * k + 60 if long and long[0] else k + max(10, k // 4)  # recovery.py's cap, or long enough to rarely bind
    t0 = time.time()
    torch.set_num_threads(int(os.environ.get("THREADS_PER_CELL", "1")))
    if kind == "full":
        g, gc, hit = full(N, d, steps, T, k, s, seed)
        out = summarize(g, gc, hit, k, steps)
    else:
        rng = np.random.default_rng(seed)
        refresh = kind == "reduced"
        g, planted = reduced(rng, N, d, steps, T, k, s, refresh)
        gc, _ = reduced(rng, N, d, steps, T, 0, 1.0, refresh)
        out = summarize(g, gc, planted, k, steps)
    return {"model": kind, "size": size, "d": d, "N": N, "share s": s, "k": k, "steps": steps, "trials": T,
            "k*": kstar(s, d, N)} | out | {"seconds": time.time() - t0}


def compare():
    """Data against the reduced model and the closed forms: tables, and the figure at Llama-3.1-8B's size."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pd.set_option("display.width", 250)
    res = lambda part: pd.read_csv(os.path.join(HERE, f"results/occupancy_theory_{part}.csv"))
    th, check, curves = res("reduced"), res("check"), res("curves")
    data = {}
    for name in ("llama3.1-8b", "gpt2-small"):
        r = pd.read_csv(os.path.join(HERE, f"results/{name}_recovery.csv"))
        if "noise" in r:  # results written before the noise column existed are isotropic
            r = r[r.noise == "isotropic"]
        data[name] = r[(r.dictionary == "random (isotropic)") & (r.weights == "equal")].assign(found=lambda t: t.recall * t.k)

    print("Reduced model against the GPU runs (random dictionary, isotropic noise, equal weights)")
    for name, r in data.items():
        m = r.merge(th[(th["size"] == name) & (th.model == "reduced")], on=["share s", "k"], suffixes=("", " (reduced)"))
        m = m[m["share s"] <= 0.2]
        e_occ, e_f = (m[OCC + " (reduced)"] - m[OCC]).abs(), (m["found in first k"] - m.found).abs()
        print(f"  {name}, s <= 0.2: median occupancy off by {e_occ.mean():.2f} on average (largest {e_occ.max():.0f}); "
              f"words found off by {e_f.mean():.2f} (largest {e_f.max():.2f})")
    w = check.pivot_table(index=["size", "share s", "k"], columns="model", values=[OCC, "found in first k"])
    e_occ, e_f = (w[(OCC, "reduced")] - w[(OCC, "full")]).abs(), (w[("found in first k", "reduced")] - w[("found in first k", "full")]).abs()
    print(f"Reduced model against the full pursuit at three new sizes ({len(w)} cells): median occupancy off by "
          f"{e_occ.mean():.2f} on average (largest {e_occ.max():.0f}); words found off by {e_f.mean():.2f} (largest {e_f.max():.2f})")

    print("\nBelow the limit: closed forms against the full pursuit (k <= k*/2)")
    rows = []
    for name, r in data.items():
        d, N = SIZES[name]
        rows += [(name, x["share s"], x.k, x[OCC], d, N) for _, x in r.iterrows() if x["share s"] <= 0.2 and x.k <= kstar(x["share s"], d, N) / 2]
    f = check[check.model == "full"]
    rows += [(x["size"], x["share s"], x.k, x[OCC], x.d, x.N) for _, x in f.iterrows() if x.k <= x["k*"] / 2]
    t = pd.DataFrame(rows, columns=["size", "share s", "k", "full pursuit", "d", "N"])
    t["fluid, holes refill"] = [fluid_occupancy(int(x.k), x["share s"], int(x.d), int(x.N)) for _, x in t.iterrows()]
    t["frozen mean (k+1)/(1-e^-Delta)"] = [mean_occupancy_frozen(int(x.k), x["share s"], int(x.d), int(x.N)) for _, x in t.iterrows()]
    print(t.drop(columns=["d", "N"]).round(1).to_string(index=False))

    print("\nFar above the limit: P(occupancy = 1), exact first-step integral against the reduced model (no cap)")
    c = curves[curves.k >= 5 * curves["k*"]]
    print(c.assign(predicted=[stop_at_first_step(int(x.k), x["share s"], int(x.d), int(x.N)) for _, x in c.iterrows()])
          [["share s", "k", "share of trials with occupancy 1", "predicted", OCC]].round(3).to_string(index=False))

    print("\nThe whole curve at Llama-3.1-8B's size (reduced model, no cap)")
    print(curves.pivot_table(index="k", columns="share s", values=[OCC, "found in first k", "trials censored"]).round(2).to_string())

    d, N = SIZES["llama3.1-8b"]
    fig, axs = plt.subplots(2, 3, figsize=(10, 6.6), sharex=True)
    for col, s in enumerate((0.05, 0.1, 0.2)):
        ks = kstar(s, d, N)
        g, u = data["llama3.1-8b"], curves[curves["share s"] == s].sort_values("k")
        g = g[g["share s"] == s]
        rcap = th[(th["size"] == "llama3.1-8b") & (th.model == "reduced") & (th["share s"] == s)].sort_values("k")
        below = range(1, int(ks / 2) + 1)  # where the closed form applies: the planted atoms are picked first
        ax = axs[0, col]
        ax.plot(u.k, u[OCC], color="#2a78d6", linewidth=2, label="reduced model, no step cap")
        ax.fill_between(u.k, u["occupancy 25th percentile"], u["occupancy 75th percentile"], color="#2a78d6", alpha=0.15, linewidth=0)
        ax.plot(rcap.k, rcap[OCC], color="#8c8c8c", linewidth=1.5, linestyle="--", label="reduced model, recovery.py's step cap")
        ax.plot(g.k, g[OCC], "o", color="black", markersize=6, label="full pursuit on GPU (experiment A)")
        ax.plot(below, [fluid_occupancy(k, s, d, N) for k in below], color="#e8743b", linewidth=2, linestyle=":", label="closed form, k <= k*/2")
        ax.plot([1, 300], [1, 300], color="black", linewidth=0.6, linestyle=(0, (1, 3)), label="occupancy = k")
        ax.set_title(f"s = {s:g}:  k* = s d / B_N = {ks:.1f}", fontsize=9)
        ax.set_ylabel("Occupancy, median over trials", fontsize=8)
        ax = axs[1, col]
        ax.plot(u.k, u["found in first k"], color="#2a78d6", linewidth=2, label="reduced model")
        ax.plot(g.k, g.found, "o", color="black", markersize=6, label="full pursuit on GPU (experiment A)")
        ax.plot([1, 300], [1, 300], color="black", linewidth=0.6, linestyle=(0, (1, 3)), label="all k found")
        ax.set_ylabel("Planted atoms among the first k picks", fontsize=8)
        ax.set_xlabel("Planted atoms k", fontsize=8)
        for ax in axs[:, col]:
            ax.axvline(ks, color="black", linewidth=0.8, linestyle="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(linestyle="--", alpha=0.4)
    axs[0, 0].legend(fontsize=6, loc="upper left", framealpha=0.9)
    axs[1, 0].legend(fontsize=6, loc="upper left")
    fig.suptitle(f"Random dictionary in isotropic noise at Llama-3.1-8B's size (d = {d}, N = {N:,}); dashed vertical line: k*", fontsize=9)
    plt.tight_layout()
    out = os.path.join(HERE, "figures/occupancy_theory.pdf")
    fig.savefig(out, bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["reduced", "check", "curves", "compare"], required=True)
    p.add_argument("--trials", type=int, default=256)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()
    if args.part == "compare":
        compare()
        raise SystemExit
    jobs = []
    if args.part == "reduced":
        for size in ("llama3.1-8b", "gpt2-small"):
            for kind in ("reduced", "frozen"):
                jobs += [(kind, size, k, s, args.trials, 1000 * i + j) for i, s in enumerate(SHARES_DATA) for j, k in enumerate(KS_DATA)]
    elif args.part == "curves":  # the whole occupancy curve at Llama-3.1-8B's size, without recovery.py's cap
        ks = (1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300)
        jobs += [("reduced", "llama3.1-8b", k, s, args.trials, 5000 + 1000 * i + j, True)
                 for i, s in enumerate(SHARES_CHECK) for j, k in enumerate(ks)]
    else:
        for size in [s for s in SIZES if s.startswith("check")]:
            for kind in ("reduced", "full"):
                trials = args.trials if kind == "reduced" else 128
                jobs += [(kind, size, k, s, trials, 1000 * i + j) for i, s in enumerate(SHARES_CHECK) for j, k in enumerate(KS_CHECK)]
    jobs.sort(key=lambda j: -j[2] * (SIZES[j[1]][1] if j[0] != "full" else SIZES[j[1]][0] * 50))  # longest first
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(cell, jobs):
            rows.append(r)
            print(f"{r['model']:8s} {r['size']:22s} s={r['share s']:<5} k={r['k']:<4} occ={r[OCC]:<6} found={r['found in first k']:.2f} ({r['seconds']:.0f} s)", flush=True)
    out = os.path.join(HERE, f"results/occupancy_theory_{args.part}.csv")
    pd.DataFrame(rows).sort_values(["size", "model", "share s", "k"]).to_csv(out, index=False, float_format="%.6g")
    print(f"Saved: {out}")
