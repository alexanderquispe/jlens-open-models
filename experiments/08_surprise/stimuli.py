"""Stimuli for experiment 08 (CPU only, no model).

Part A, change points (data/changepoint.csv). Each item is one draw of noise, shown in three arms:
  control  x_t = round(mu0 + sigma * eps_t)
  jump     the control numbers plus a level shift of +-delta*sigma from number tau on (a real change)
  outlier  the control numbers, except number tau, which equals the jump arm's (one wild value, no change)
The three arms are identical before tau, and jump and outlier are identical up to and including tau: at tau+1
the second number tells "it really changed" from "it was a fluke". sigma in {4, 8, 16} and delta in {2, 4, 8}
make the same jump very surprising in a calm sequence and ordinary in a noisy one. Numbers stay in [10, 989],
one token each in the Llama-3 tokenizer.

Part B, arithmetic twins (data/arith.jsonl). GSM8K test solutions (Cobbe et al. 2021), one equation "a op b = r"
per problem (op + or *, r < 1000), in four versions that differ only inside that equation:
  clean  as written
  near   r replaced by a close wrong value (last digits +-1 or +-2, same number of digits)
  far    r replaced by a distant wrong value (leading digit moved by 3 or more)
  swap   a and b swapped: still correct, a harmless edit that changes tokens
Every token before the edit is identical across the four versions, and every token after it too.

    python experiments/08_surprise/stimuli.py
    python experiments/08_surprise/stimuli.py --n-rep 2 --out-dir /tmp/smoke   # a small set for a smoke test
"""
import argparse
import json
import os
import re
import urllib.request

import numpy as np
import pandas as pd

from common import ARMS, DATA, PREFIX, SEP, T

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

parser = argparse.ArgumentParser()
parser.add_argument("--n-rep", type=int, default=40, help="items per (sigma, delta, direction) cell")
parser.add_argument("--sigmas", type=int, nargs="+", default=[4, 8, 16])
parser.add_argument("--deltas", type=int, nargs="+", default=[2, 4, 8])
parser.add_argument("--lo", type=int, default=10, help="smallest number allowed")
parser.add_argument("--hi", type=int, default=989, help="largest number allowed")
parser.add_argument("--tau", type=int, nargs=2, default=[12, 22], help="range of the change position (1-based)")
parser.add_argument("--max-arith", type=int, default=None, help="keep at most this many arithmetic items")
parser.add_argument("--out-dir", default=DATA)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
rng = np.random.default_rng(args.seed)


def changepoint_items():
    rows = []
    item = 0
    for sigma in args.sigmas:
        for delta in args.deltas:
            shift_abs = delta * sigma
            margin = 4 * sigma + shift_abs
            assert args.lo + margin < args.hi - margin, f"sigma={sigma}, delta={delta} do not fit in [lo, hi]"
            for direction in (1, -1):
                for _ in range(args.n_rep):
                    tau = int(rng.integers(args.tau[0], args.tau[1] + 1))
                    while True:  # redraw the rare sequence that leaves [lo, hi]
                        mu0 = int(rng.integers(args.lo + margin, args.hi - margin + 1))
                        control = np.rint(mu0 + sigma * rng.standard_normal(T)).astype(int)
                        jump = control.copy()
                        jump[tau - 1 :] += direction * shift_abs
                        outlier = control.copy()
                        outlier[tau - 1] = jump[tau - 1]
                        arms = {"control": control, "jump": jump, "outlier": outlier}
                        if all(a.min() >= args.lo and a.max() <= args.hi for a in arms.values()):
                            break
                    for arm in ARMS:
                        nums = arms[arm]
                        rows.append({"item": item, "arm": arm, "sigma": sigma, "delta": delta, "direction": direction,
                                     "tau": tau, "mu0": mu0, "shift": direction * shift_abs,
                                     "numbers": " ".join(map(str, nums)),
                                     "text": PREFIX + SEP.join(map(str, nums))})
                    item += 1
    return pd.DataFrame(rows)


# --- Part B ---------------------------------------------------------------------------------------------------
ANNOTATION = re.compile(r"<<([^<>]*)=([^<>]*)>>")
# A visible equation "a op b = r" (r possibly written $r); no $ before a or b, no decimals or thousands commas
EQUATION = re.compile(r"(?<![\d$])(?<!\d[.,])(\d+)(\s*)([+*])(\s*)(?<!\$)(\d+)(\s*)=(\s*)\$?(\d+)(?!\d)(?![.,]\d)")


def wrong_values(r, rng):
    """A near and a far wrong value for r, with the same number of digits."""
    n = len(str(r))
    near = [r + d for d in (-2, -1, 1, 2) if r + d >= 1 and len(str(r + d)) == n]
    lead = int(str(r)[0])
    far = [r + (L - lead) * 10 ** (n - 1) for L in range(1, 10) if abs(L - lead) >= 3]
    if n == 1:
        far = [L for L in range(1, 10) if abs(L - r) >= 4]
    return int(rng.choice(near)), int(rng.choice(far))


def arith_items():
    path = os.path.join(DATA, "gsm8k_test.jsonl")
    if not os.path.exists(path):
        urllib.request.urlretrieve(GSM8K_URL, path)
    problems = [json.loads(line) for line in open(path)]
    items = []
    for pid, prob in enumerate(problems):
        visible = ANNOTATION.sub("", prob["answer"])
        prefix = f"Question: {prob['question']}\nAnswer: "
        text = prefix + visible
        annotated = {(m.group(1).replace(" ", ""), m.group(2).strip()) for m in ANNOTATION.finditer(prob["answer"])}
        cands = []
        for m in EQUATION.finditer(visible):
            a, op, b, r = m.group(1), m.group(3), m.group(5), m.group(8)
            # the equation must be one the solution computes (either operand order) and r small enough
            if not ((f"{a}{op}{b}", r) in annotated or (f"{b}{op}{a}", r) in annotated):
                continue
            ra, rb, rr = int(a), int(b), int(r)
            if a == b or rr >= 1000 or rr < 1 or (ra + rb if op == "+" else ra * rb) != rr:
                continue
            cands.append(m)
        if not cands:
            continue
        m = cands[int(rng.integers(len(cands)))]
        off = len(prefix)
        a_s, a_e = off + m.start(1), off + m.end(1)
        b_s, b_e = off + m.start(5), off + m.end(5)
        r_s, r_e = off + m.start(8), off + m.end(8)
        r = int(m.group(8))
        near, far = wrong_values(r, rng)
        a, b = m.group(1), m.group(5)
        # swap: "a<sp>op<sp>b" -> "b<sp>op<sp>a"; the region from a's start to b's end keeps its length
        middle = text[a_e:b_s]
        swapped = b + middle + a
        twins = {
            "clean": text,
            "near": text[:r_s] + str(near) + text[r_e:],
            "far": text[:r_s] + str(far) + text[r_e:],
            "swap": text[:a_s] + swapped + text[b_e:],
        }
        assert len({len(t) for t in twins.values()}) == 1
        # the first later occurrence of r as a standalone number (identical text in every twin)
        reuse = re.search(rf"(?<![\d.,]){r}(?![\d])(?![.,]\d)", text[r_e:])
        items.append({
            "id": pid, "r": r, "near_value": near, "far_value": far, "a": int(a), "b": int(b), "op": m.group(3),
            "edit": {"clean": [r_s, r_e], "near": [r_s, r_e], "far": [r_s, r_e], "swap": [a_s, b_e]},
            "reuse": [r_e + reuse.start(), r_e + reuse.end()] if reuse else None,
            "texts": twins,
        })
    if args.max_arith:
        items = items[: args.max_arith]
    return items


df = changepoint_items()
df.to_csv(os.path.join(args.out_dir, "changepoint.csv"), index=False)
arith = arith_items()
with open(os.path.join(args.out_dir, "arith.jsonl"), "w") as f:
    for it in arith:
        f.write(json.dumps(it) + "\n")
print(f"Part A: {df.item.nunique()} items x {len(ARMS)} arms, T = {T} numbers "
      f"(values {df.numbers.str.split().explode().astype(int).min()}..{df.numbers.str.split().explode().astype(int).max()})")
print(f"Part B: {len(arith)} arithmetic items (of 1319 GSM8K test problems), "
      f"{sum(it['reuse'] is not None for it in arith)} with a later reuse of r")
print(f"Saved to {args.out_dir}")
