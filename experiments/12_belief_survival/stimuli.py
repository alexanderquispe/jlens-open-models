"""Stimuli of experiment 12 and the ideal observer's values on them (CPU, about a minute).

Writes (one stream per line: arm, seq, numbers, level, and the arm's parameters):
  data/jumps.jsonl        7 arms x N_SEQ streams of JUMP_LEN numbers: up and down jumps of each size in DELTAS at
                          number TAU, and a flat control at 500. Stream k of every arm uses the same draws
                          (default_rng(JUMP_SEED + k)).
  data/ramps_fast.jsonl   2 shapes (up-down, down-up) x N_SEQ, ramps of RAMPS["fast"] numbers
  data/ramps_slow.jsonl   the same with ramps of RAMPS["slow"] numbers (default_rng(RAMP_SEED + k) for both)
  data/observer.npz       the experiment-10 observer (default settings) on every stream: its values per number, and
                          its predictive distribution over 0..999 after numbers TAU - 1 .. TAU + 30 of the jumps
With --history (only these files):
  data/history_t{tau}.jsonl  for tau in HIST_TAUS: up and down jumps of each size in HIST_DELTAS after tau numbers
                          at the old level, then HIST_POST numbers at the new one (4 arms x N_SEQ streams). Stream k
                          uses the same draws just after the jump (default_rng(HIST_SEED + k)) and just before it (the
                          last tau of default_rng(HIST_SEED + 100 + k)'s draws) for every tau.
  data/observer_history.npz  the observer's values on them, keys "t{tau}/{key}"

    python experiments/12_belief_survival/stimuli.py
    python experiments/12_belief_survival/stimuli.py --history
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(os.path.dirname(HERE), "10_unlikely_vs_update"))  # after HERE: common is ours
import observer  # noqa: E402  (experiment 10's)
from common import (DATA, DELTAS, HI, HIGH, HIST_DELTAS, HIST_POST, HIST_SEED, HIST_TAUS, JUMP_LEN,  # noqa: E402
                    JUMP_SEED, LO, LOW, N_SEQ, PLATEAU, POST, PRE, RAMP_SEED, RAMPS, SIGMA, TAU)

JUMP_ARMS = [("flat", 0, 0)] + [(f"{'up' if s > 0 else 'down'}{dl}", dl, s) for dl in DELTAS for s in (1, -1)]


def clip(x):
    return np.clip(np.round(x).astype(int), LO, HI)


def jumps():
    out = []
    t = np.arange(JUMP_LEN)
    for k in range(N_SEQ):
        z = np.random.default_rng(JUMP_SEED + k).standard_normal(JUMP_LEN)
        for arm, dl, s in JUMP_ARMS:
            level = np.where(t < TAU, 500 - s * dl / 2, 500 + s * dl / 2)
            out.append({"arm": arm, "seq": k, "delta": dl, "sign": s, "level": level.tolist(),
                        "numbers": clip(level + SIGMA * z).tolist()})
    return out


def ramp_level(r, shape):
    a, b = (LOW, HIGH) if shape == "updown" else (HIGH, LOW)
    seg = [np.full(PRE, a, float), np.linspace(a, b, r + 1)[1:], np.full(PLATEAU, b, float),
           np.linspace(b, a, r + 1)[1:], np.full(POST, a, float)]
    return np.concatenate(seg)


def ramps(speed):
    r = RAMPS[speed]
    n = PRE + 2 * r + PLATEAU + POST
    out = []
    for k in range(N_SEQ):
        z = np.random.default_rng(RAMP_SEED + k).standard_normal(n)
        for shape in ("updown", "downup"):
            level = ramp_level(r, shape)
            out.append({"arm": f"{shape}_{speed}", "seq": k, "shape": shape, "speed": speed, "ramp": r,
                        "level": level.tolist(), "numbers": clip(level + SIGMA * z).tolist()})
    return out


def history(tau):
    out = []
    t = np.arange(tau + HIST_POST)
    for k in range(N_SEQ):
        z_pre = np.random.default_rng(HIST_SEED + 100 + k).standard_normal(max(HIST_TAUS))[-tau:]
        z_post = np.random.default_rng(HIST_SEED + k).standard_normal(HIST_POST)
        z = np.concatenate([z_pre, z_post])
        for dl in HIST_DELTAS:
            for s in (1, -1):
                level = np.where(t < tau, 500 - s * dl / 2, 500 + s * dl / 2)
                out.append({"arm": f"{'up' if s > 0 else 'down'}{dl}", "seq": k, "tau": tau, "delta": dl, "sign": s,
                            "level": level.tolist(), "numbers": clip(level + SIGMA * z).tolist()})
    return out


if __name__ == "__main__" and "--history" in sys.argv:
    save = {}
    H = {tau: history(tau) for tau in HIST_TAUS}
    for tau, S in H.items():  # the same draws around the jump for every history length
        a, b = H[max(HIST_TAUS)], S
        for x, y in zip(a, b):
            assert x["numbers"][-HIST_POST - min(HIST_TAUS):] == y["numbers"][-HIST_POST - min(HIST_TAUS):]
        with open(os.path.join(DATA, f"history_t{tau}.jsonl"), "w") as f:
            for s in S:
                f.write(json.dumps(s) + "\n")
        obs, _ = observer.observe(np.array([s["numbers"] for s in S]))
        for k, v in obs.items():
            save[f"t{tau}/{k}"] = v.astype(np.float32)
        print(f"history t{tau}: {len(S)} streams of {tau + HIST_POST} numbers")
    np.savez_compressed(os.path.join(DATA, "observer_history.npz"), **save)
    sys.exit(0)

if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    sets = {"jumps": jumps(), "ramps_fast": ramps("fast"), "ramps_slow": ramps("slow")}
    save = {}
    for name, S in sets.items():
        with open(os.path.join(DATA, f"{name}.jsonl"), "w") as f:
            for s in S:
                f.write(json.dumps(s) + "\n")
        X = np.array([s["numbers"] for s in S])
        dist_at = tuple(range(TAU - 1, TAU + 31)) if name == "jumps" else ()
        obs, dists = observer.observe(X, dist_at=dist_at)
        for k, v in obs.items():
            save[f"{name}/{k}"] = v.astype(np.float32)
        if dists:
            save[f"{name}/dist"] = np.stack([dists[t] for t in dist_at], 1).astype(np.float32)  # (n, 32, 1000)
            save[f"{name}/dist_t"] = np.array(dist_at)
        print(f"{name}: {len(S)} streams of {X.shape[1]} numbers")
    np.savez_compressed(os.path.join(DATA, "observer.npz"), **save)
    print(f"observer values in {DATA}/observer.npz")
