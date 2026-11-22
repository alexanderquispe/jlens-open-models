"""Contexts of experiment 11 (CPU, seconds): experiment 09's cue contexts, extended from 10 to 20 streams.

  switch   stream k of N(300, 100) (1000 numbers; the draws of Sarfati et al.'s generator, seed 18, so streams
           0..9 are experiment 09's), then e + 1 numbers drawn from N(700, 100) ("jump") or N(300, 100)
           ("control") with the same standard normal draws (default_rng(1000 + k)), e in CUTS. e = -1 is the
           first segment alone (both arms the same).
  outlier  stream k of N(500, 100) up to number t (the first number at or after 900 within 50 of 500), with
           number t kept ("typical"), set to 750 or 250 ("moderate", 2.5 sd) or 950 or 50 ("extreme", 4.5 sd;
           the side alternates with k).

Writes data/contexts.jsonl (block, seq, e / t, arm / level, numbers).

  switch_down  (--down; data/contexts_down.jsonl) the mirror block that completes a 2 x 2 design (history low /
           high x recent numbers low / high): stream k of N(700, 100) (the same standard normal draws as the
           N(300, 100) prefix, shifted by 400), then e + 1 numbers from N(300, 100) ("jump", down) or N(700, 100)
           ("control", the level-high stream) with the same tail draws. The down jump's tail is the up control's,
           and the level-high tail is the up jump's, so "changed" (history != recent) is the interaction and a
           reader of the recent numbers alone cannot produce it.

    python experiments/11_notice_report/stimuli.py
    python experiments/11_notice_report/stimuli.py --down
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SEED, LO, HI = 18, 10, 990
N_SEQ, PREFIX = 20, 1000
CUTS = (-1, 0, 1, 2, 4, 8, 32)
LEVELS = ("typical", "moderate", "extreme")


def draw(mu, sigma, n_seq=N_SEQ, length=PREFIX, seed=SEED):
    """Sarfati et al.'s generate_gaussian_numerical_sequences (as experiment 09's stimuli.draw)."""
    rng = np.random.default_rng(seed)
    return np.clip(np.round(rng.normal(loc=mu, scale=sigma, size=(n_seq, length))).astype(int), LO, HI)


def clip(x):
    return np.clip(np.round(x).astype(int), LO, HI)


def contexts():
    out = []
    first = draw(300, 100)
    for k in range(N_SEQ):
        z = np.random.default_rng(1000 + k).standard_normal(max(CUTS) + 1)
        for e in CUTS:
            for arm, mu in (("jump", 700), ("control", 300)):
                out.append({"block": "switch", "seq": k, "e": e, "arm": arm,
                            "numbers": first[k].tolist() + clip(mu + 100 * z[: e + 1]).tolist()})
    base = draw(500, 100)
    for k in range(N_SEQ):
        t = next(t for t in range(900, PREFIX) if abs(int(base[k, t]) - 500) <= 50)
        for level in LEVELS:
            last = int(base[k, t]) if level == "typical" else 500 + (1 if k % 2 == 0 else -1) * (250 if level == "moderate" else 450)
            out.append({"block": "outlier", "seq": k, "t": t, "level": level, "last": last,
                        "numbers": base[k, :t].tolist() + [last]})
    return out


def contexts_down():
    out = []
    first = draw(700, 100)
    for k in range(N_SEQ):
        z = np.random.default_rng(1000 + k).standard_normal(max(CUTS) + 1)
        for e in CUTS:
            for arm, mu in (("jump", 300), ("control", 700)):
                out.append({"block": "switch_down", "seq": k, "e": e, "arm": arm,
                            "numbers": first[k].tolist() + clip(mu + 100 * z[: e + 1]).tolist()})
    return out


if __name__ == "__main__" and "--down" in sys.argv:
    C = contexts_down()
    up = {(c["seq"], c["e"], c["arm"]): c["numbers"] for c in contexts() if c["block"] == "switch"}
    for c in C:  # the mirror: tails swap between the blocks, prefixes are the shifted draws
        k, e = c["seq"], c["e"]
        n_tail = e + 1
        other = up[(k, e, "control" if c["arm"] == "jump" else "jump")]
        assert c["numbers"][PREFIX:] == other[PREFIX:] and len(c["numbers"]) == PREFIX + n_tail
    with open(os.path.join(DATA, "contexts_down.jsonl"), "w") as f:
        for c in C:
            f.write(json.dumps(c) + "\n")
    print(f"{DATA}/contexts_down.jsonl: {len(C)} contexts")
    sys.exit(0)

if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    C = contexts()
    # streams 0..9 must be experiment 09's (its data/sequences.jsonl, when present)
    ref = os.path.join(os.path.dirname(HERE), "09_beliefs", "data", "sequences.jsonl")
    if os.path.exists(ref):
        s09 = {(s["cond"], s["seq"]): s["numbers"] for s in map(json.loads, open(ref)) if s["dataset"] == "stationary"}
        for k in range(10):
            assert draw(300, 100)[k].tolist() == s09[("m300_s100", k)]
            assert draw(500, 100)[k].tolist() == s09[("m500_s100", k)]
        print("streams 0..9 match experiment 09")
    with open(os.path.join(DATA, "contexts.jsonl"), "w") as f:
        for c in C:
            f.write(json.dumps(c) + "\n")
    print(f"{DATA}/contexts.jsonl: {len(C)} contexts")
