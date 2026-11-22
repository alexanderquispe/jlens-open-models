"""Stimuli of experiment 10, the ideal observer's values on them, and the design check (CPU, about a minute).

Every stream k has five versions (common.ARMS) built from the same standard normal draws z (default_rng(SEED + k)),
identical up to number TAU - 1; s = +1 for even k and -1 for odd k:
  control  x_t = MU0 + SIGMA0 z_t throughout
  jump     from TAU on, the level is MU0 + s DELTA
  outlier  only number TAU is moved, to the jump arm's value (so at TAU the two arms are the same number), then
           the stream goes on as control
  drift    the level slides from MU0 to MU0 + s DELTA over DRIFT numbers, starting at TAU (0.05 sd per number: no
           single number is surprising, but the belief has to move as far as in the jump arm)
  noise    from TAU on, the sd is NOISE times larger, same level (uncertainty grows, the level does not move)
Numbers are rounded and clipped to [10, 990].

Writes data/streams.jsonl (one stream per line: arm, seq, sign, numbers) and data/observer.npz (each observer value
of observer.KEYS as (arm, seq, t), for the default observer and the sensitivity settings), and prints the design
check: the correlation of the regressors after removing (seq, t) fixed effects (the twin comparison), in the window
of numbers TAU - 5 .. TAU + 40. The design is usable if surprisal and Bayesian surprise correlate below 0.5 there.

    python experiments/10_unlikely_vs_update/stimuli.py
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ARMS, CUTS, DATA, DELTA, DRIFT, HI, LEN, LO, MU0, N_SEQ, NOISE, SEED, SIGMA0, TAU  # noqa: E402
import observer  # noqa: E402

# Observer settings: the default and the sensitivity grid (h_mu, h_sig, rho)
SETTINGS = {"default": (observer.H_MU, observer.H_SIG, observer.RHO),
            "rho_low": (1 / 200, 1 / 500, 1 / 1000), "rho_high": (1 / 200, 1 / 500, 1 / 20),
            "h_low": (1 / 1000, 1 / 500, 1 / 100), "h_high": (1 / 50, 1 / 500, 1 / 100),
            "no_outliers": (1 / 200, 1 / 500, 0.0)}
WINDOW = (TAU - 5, TAU + 40)


def clip(x):
    return np.clip(np.round(x).astype(int), LO, HI)


def streams(n_seq=N_SEQ, length=LEN):
    out = []
    t = np.arange(length)
    for k in range(n_seq):
        z = np.random.default_rng(SEED + k).standard_normal(length)
        s = 1 if k % 2 == 0 else -1
        after = t >= TAU
        level = {
            "control": np.full(length, MU0, float),
            "jump": np.where(after, MU0 + s * DELTA, MU0).astype(float),
            "outlier": np.full(length, MU0, float),
            "drift": MU0 + s * DELTA * np.clip((t - TAU + 1) / DRIFT, 0, 1),
            "noise": np.full(length, MU0, float),
        }
        sd = {a: np.full(length, SIGMA0, float) for a in ARMS}
        sd["noise"] = np.where(after, NOISE * SIGMA0, SIGMA0).astype(float)
        for a in ARMS:
            x = clip(level[a] + sd[a] * z)
            if a == "outlier":
                x[TAU] = clip(MU0 + s * DELTA + SIGMA0 * z[TAU])
            out.append({"arm": a, "seq": k, "sign": s, "numbers": x.tolist()})
    return out


def within(v, by):
    """v minus its mean within groups `by` (the twin fixed effects)."""
    v = np.asarray(v, float)
    keys, inv = np.unique(by, axis=0, return_inverse=True)
    m = np.bincount(inv.ravel(), v) / np.bincount(inv.ravel())
    return v - m[inv.ravel()]


def design_table(obs, window=WINDOW):
    """Correlations of the regressors, (seq, t) fixed effects removed, over arms x seqs x window."""
    A, K, T = obs["surprisal"].shape
    t = np.arange(*window)
    level_pre = obs["level_after"][:, :, TAU - 1]
    regs = {
        "surprisal": obs["surprisal"][:, :, t],
        "bayes_kl": obs["bayes_kl"][:, :, t],
        "deviation": np.abs(obs["level_after"][:, :, t] - level_pre[:, :, None]) / SIGMA0,
        "uncertainty": obs["next_sd"][:, :, t],
    }
    kk, tt = np.meshgrid(np.arange(K), t, indexing="ij")
    by = np.stack([np.broadcast_to(kk, (A, K, len(t))).ravel(), np.broadcast_to(tt, (A, K, len(t))).ravel()], 1)
    W = {k: within(v.ravel(), by) for k, v in regs.items()}
    names = list(W)
    C = np.corrcoef(np.stack([W[k] for k in names]))
    return names, C



def level_arm():
    """The level-matched control added after the first run (2026-11-08): stream k at the jump arm's new level
    (MU0 + s DELTA) from the start, same draws. Jump - level after e = 100 compares the same current level with and
    without a history at MU0; written to separate files so the five-arm data stay as they are."""
    out = []
    for k in range(N_SEQ):
        z = np.random.default_rng(SEED + k).standard_normal(LEN)
        s = 1 if k % 2 == 0 else -1
        out.append({"arm": "level", "seq": k, "sign": s, "numbers": clip(MU0 + s * DELTA + SIGMA0 * z).tolist()})
    return out


def write_level(out_dir=DATA):
    S = level_arm()
    with open(os.path.join(out_dir, "streams_level.jsonl"), "w") as f:
        for s in S:
            f.write(json.dumps(s) + "\n")
    with open(os.path.join(out_dir, "cue_contexts_level.jsonl"), "w") as f:
        for s in S:
            for e in CUTS:
                f.write(json.dumps({"arm": "level", "seq": s["seq"], "sign": s["sign"], "e": e,
                                    "numbers": s["numbers"][: TAU + e + 1]}) + "\n")
    X = np.array([s["numbers"] for s in S])
    save = {}
    for name, (h_mu, h_sig, rho) in SETTINGS.items():
        obs, _ = observer.observe(X, h_mu=h_mu, h_sig=h_sig, rho=rho)
        for k, v in obs.items():
            save[f"{name}/{k}"] = v.astype(np.float32)[None]  # (1, seq, t), like one more arm
    np.savez_compressed(os.path.join(out_dir, "observer_level.npz"), **save)
    print(f"{out_dir}/streams_level.jsonl: {len(S)} streams; cue contexts and observer values written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DATA)
    parser.add_argument("--level-only", action="store_true", help="write only the level-matched arm's files")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.level_only:
        write_level(args.out)
        sys.exit(0)
    S = streams()
    with open(os.path.join(args.out, "streams.jsonl"), "w") as f:
        for s in S:
            f.write(json.dumps(s) + "\n")
    # Cue contexts: every stream cut after number TAU + e (e in CUTS), for run_cues.py
    with open(os.path.join(args.out, "cue_contexts.jsonl"), "w") as f:
        for s in S:
            for e in CUTS:
                f.write(json.dumps({"arm": s["arm"], "seq": s["seq"], "sign": s["sign"], "e": e,
                                    "numbers": s["numbers"][: TAU + e + 1]}) + "\n")
    X = np.array([s["numbers"] for s in S]).reshape(N_SEQ, len(ARMS), LEN).transpose(1, 0, 2)  # (arm, seq, t)
    for a in range(1, len(ARMS)):
        assert (X[a, :, :TAU] == X[0, :, :TAU]).all(), "arms must agree before the event"
    assert (X[ARMS.index("outlier"), :, TAU] == X[ARMS.index("jump"), :, TAU]).all()
    save = {}
    for name, (h_mu, h_sig, rho) in SETTINGS.items():
        obs, _ = observer.observe(X.reshape(-1, LEN), h_mu=h_mu, h_sig=h_sig, rho=rho)
        obs = {k: v.reshape(len(ARMS), N_SEQ, LEN) for k, v in obs.items()}
        for k, v in obs.items():
            save[f"{name}/{k}"] = v.astype(np.float32)
        names, C = design_table(obs)
        print(f"\nObserver {name}: h_mu = {h_mu:.4f}, h_sig = {h_sig:.4f}, rho = {rho:.4f}")
        print("  within-(seq, t) correlations, numbers TAU-5..TAU+40, all arms:")
        print("  " + " ".join(f"{n:>12}" for n in [""] + names))
        for n, row in zip(names, C):
            print("  " + f"{n:>12} " + " ".join(f"{c:12.3f}" for c in row))
        e = TAU
        for a, arm in enumerate(ARMS):
            o = {k: obs[k][a, :, e : e + 3].mean(0) for k in ("surprisal", "bayes_kl", "cpp", "opp")}
            print(f"  {arm:8s} e=0..2: surprisal {np.round(o['surprisal'], 2)}, KL {np.round(o['bayes_kl'], 2)}, "
                  f"cpp {np.round(o['cpp'], 2)}, outlier p {np.round(o['opp'], 2)}")
    np.savez_compressed(os.path.join(args.out, "observer.npz"), arms=np.array(ARMS), **save)
    print(f"\n{args.out}/streams.jsonl: {len(S)} streams; observer values in {args.out}/observer.npz")
    write_level(args.out)
