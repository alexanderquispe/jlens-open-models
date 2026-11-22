"""The stimuli of Sarfati et al. (2026), "The Shape of Beliefs", exactly as their code makes them.

Writes data/sequences.jsonl, one sequence per line:
  stationary  17 conditions (mu in 300..700 at sigma = 100; sigma in 10..200 at mu = 500) x 10 sequences of 1000
              numbers (their generate_sequences.py, seed 18 per condition)
  switch      the 10 sequences of N(300, 100) followed by the 10 of N(700, 100), sequence by sequence (their
              combined dataset "gaussian_m300_s100_l1000_n10+gaussian_m700_s100_l1000_n10"): 2000 numbers,
              the change at number 1000

tests.py checks that the per-condition files their code writes are reproduced byte for byte (SHA-256).

    python experiments/09_beliefs/stimuli.py
"""
import argparse
import json
import os

import numpy as np

from common import CONDITIONS, DATA, DELIM, HI, LEN, LO, N_SEQ, SEED, SWITCH, cond_name


def draw(mu, sigma, n_seq=N_SEQ, length=LEN, seed=SEED):
    """generate_gaussian_numerical_sequences of their generate_sequences.py: (n_seq, length) integers."""
    rng = np.random.default_rng(seed)
    samples = rng.normal(loc=mu, scale=sigma, size=(n_seq, length))
    return np.clip(np.round(samples).astype(int), LO, HI)


def to_text(numbers):
    """numerical_sequences_to_textual_prompts: "512,324,18," (a delimiter after the last number too)."""
    return DELIM.join(map(str, numbers)) + DELIM


def reference_jsonl(mu, sigma):
    """The bytes of their data/sequences/gaussian_m{mu}_s{sigma:03d}_l1000_n10.jsonl."""
    lines = [json.dumps({"sequence_id": f"seq_{p:03d}", "sequence_content": to_text(s)})
             for p, s in enumerate(draw(mu, sigma))]
    return ("\n".join(lines) + "\n").encode("utf-8")


def sequences():
    """All sequences as dicts: dataset, cond, seq, mu, sigma (the first segment's for the switch), numbers, text."""
    out = []
    for mu, sigma in CONDITIONS:
        for k, s in enumerate(draw(mu, sigma)):
            out.append({"dataset": "stationary", "cond": cond_name(mu, sigma), "seq": k, "mu": mu, "sigma": sigma,
                        "numbers": s.tolist(), "text": to_text(s)})
    (m1, s1), (m2, s2) = SWITCH
    for k, (a, b) in enumerate(zip(draw(m1, s1), draw(m2, s2))):
        nums = np.concatenate([a, b])
        out.append({"dataset": "switch", "cond": f"{cond_name(m1, s1)}+{cond_name(m2, s2)}", "seq": k, "mu": m1,
                    "sigma": s1, "mu2": m2, "sigma2": s2, "switch_at": len(a), "numbers": nums.tolist(),
                    "text": to_text(a) + to_text(b)})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(DATA, "sequences.jsonl"))
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    seqs = sequences()
    with open(args.out, "w") as f:
        for s in seqs:
            f.write(json.dumps(s) + "\n")
    n = {d: sum(s["dataset"] == d for s in seqs) for d in ("stationary", "switch")}
    print(f"{args.out}: {n['stationary']} stationary sequences ({len(CONDITIONS)} conditions x {N_SEQ}), "
          f"{n['switch']} switch sequences")
