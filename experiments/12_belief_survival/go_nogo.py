"""Go / no-go of experiment 12 on experiment 09's existing jump streams (CPU, no new model run).

Experiment 09 read 10 streams of 1000 numbers from N(300, 100) followed by 1000 from N(700, 100). For each layer and
lens, the lens's integer mean at the comma after number t (the "belief in number words") moves from the old level
toward the new one after the jump. We time that move: progress p(e) = (m(e) - before) / (after - before), with
before = mean over e in [-100, -1] and after = mean over e in [200, 299], and e50 = the first e >= 0 at which the
3-number running mean of p reaches 0.5. The same for the output's mean. 95% intervals: bootstrap over the 10
streams (2000 resamples; e50 recomputed on the resampled mean curve).

Rule fixed before running: experiment 12 goes ahead if, for the output and for at least half of the middle and late
layers whose move is large enough to time (|after - before| >= 100), the interval is no wider than e50 itself
(half-width <= 50% of e50) or no wider than 4 numbers. Otherwise the timing is too noisy and experiment 12 stops
before any GPU run.

    python experiments/12_belief_survival/go_nogo.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXP09_RAW  # noqa: E402

E = np.arange(-100, 300)
RNG = np.random.default_rng(0)


def e50(curve):
    """curve over E -> first e >= 0 where the 3-point running mean of progress reaches 0.5 (NaN if never)."""
    before = curve[(E >= -100) & (E < 0)].mean()
    after = curve[(E >= 200) & (E < 300)].mean()
    p = (curve - before) / (after - before)
    run = np.convolve(p, np.ones(3) / 3, mode="same")
    ok = np.where((E >= 0) & (run >= 0.5))[0]
    return E[ok[0]] if len(ok) else np.nan


def timing(M):
    """M (n_seq, len(E)) -> (e50 of the mean, 2.5% and 97.5% bootstrap quantiles, |after - before|)."""
    point = e50(M.mean(0))
    boots = [e50(M[RNG.integers(0, len(M), len(M))].mean(0)) for _ in range(2000)]
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    m = M.mean(0)
    return point, lo, hi, abs(m[E >= 200].mean() - m[E < 0].mean())


def main():
    lines, verdict = [], {}
    for model in ("llama3.1-8b", "olmo-3-32b"):
        R = os.path.join(EXP09_RAW, model)
        meta = json.load(open(os.path.join(R, "meta_switch.json")))
        bos, at = meta["bos"], meta["sequences"][0]["switch_at"]
        t = at + E
        com = bos + 2 * t + 1
        out = np.load(os.path.join(R, "behaviour_switch.npz"))["out_int"][:, t, 0]  # (10, len(E)) output mean
        p, lo, hi, size = timing(out)
        lines.append(f"\n## {model}\n\noutput mean: e50 = {p:.0f} [{lo:.0f}, {hi:.0f}], move {size:.0f}\n")
        ok_out = (hi - lo) / 2 <= max(0.5 * p, 4) if np.isfinite(p) else False
        lenses = [k for k in ("logit", "jlens", "jpp") if os.path.exists(os.path.join(R, f"switch_{k}_scalars.npy"))]
        n_ok = n_timed = 0
        lines.append("| lens | layer | e50 | 95% interval | move |\n|---|---|---|---|---|")
        for lens in lenses:
            S = np.load(os.path.join(R, f"switch_{lens}_scalars.npy"), mmap_mode="r")
            L = S.shape[2]
            for layer in range(L):
                M = np.asarray(S[:, com, layer, 1], float)
                p, lo, hi, size = timing(M)
                timed = size >= 100 and layer >= L // 3
                if timed:
                    n_timed += 1
                    n_ok += bool(np.isfinite(p) and (hi - lo) / 2 <= max(0.5 * p, 4))
                if layer % max(1, L // 16) == 0 or layer == L - 1:
                    lines.append(f"| {lens} | {layer} | {p:.0f} | [{lo:.0f}, {hi:.0f}] | {size:.0f} |")
        go = bool(ok_out and n_timed and n_ok >= n_timed / 2)
        verdict[model] = {"output_ok": bool(ok_out), "layers_timed": n_timed, "layers_ok": n_ok, "go": go}
        lines.append(f"\nMiddle and late layers with a move >= 100: {n_timed}; timed precisely: {n_ok}. "
                     f"Output timed precisely: {ok_out}. **{'GO' if go else 'NO-GO'}**")
    text = "# Experiment 12, go / no-go on experiment 09's jumps\n" + "\n".join(lines) + "\n"
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "go_nogo.md")
    open(path, "w").write(text)
    print(text)
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
