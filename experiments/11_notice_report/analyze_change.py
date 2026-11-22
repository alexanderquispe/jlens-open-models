"""Experiment 11, step 3 analysis (CPU, seconds): does injecting the pure change direction move the report?

Reads <raw>/<model>/inject_change_e4.csv.gz (inject_change.py) for every model found and writes
results/inject_change.csv (every model, format, layer, condition) and results/inject_change.md.

Score of a format = log P(the answer meaning "changed / surprising") - log P(the other answer); for "wait" the
surprise minus normal word sets; for "comma" the predicted mean and sd of the next number.
Per stream:
  natural change   (LH + HL - LL - HH) / 2 of the score with no injection (the report of change in the output)
  effect           score(injection) - score(none), averaged over the unchanged cells (LL, HH), the changed cells
                   (LH, HL), or all four
Mean, standard error and t over streams. A direction "reports change" if adding it moves every question toward
its changed answer (q_same has reversed polarity, q_ab / q_ba swap the options).

    python experiments/11_notice_report/analyze_change.py [--raw DIR] [--out DIR]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prompts  # noqa: E402

RAW = (os.path.join(os.environ["JLENS_SHARED"], "experiments", "11_notice_report") if os.environ.get("JLENS_SHARED")
       else os.path.join(HERE, "raw"))
QUESTIONS = ["q_surprise", "q_change", "q_same", "q_ab", "q_ba"]


def add_scores(df, model):
    """One or two score columns per row: (format, measure, value)."""
    spec = prompts.CHAT if prompts.is_chat(model) else prompts.BASE
    out = []
    for name, g in df.groupby("format"):
        kind, changed = spec[name][1], spec[name][2]
        if kind in ("yesno", "ab"):
            out.append(g.assign(measure="report", score=g[f"lp_{changed}"] - g[f"lp_{prompts.OPPOSITE[changed]}"]))
        elif kind == "words":
            out.append(g.assign(measure="surprise_words", score=g["set_surprise"] - g["set_normal"]))
        elif kind == "int":
            out.append(g.assign(measure="int_mean", score=g["int_mean"]))
            out.append(g.assign(measure="int_sd", score=g["int_sd"]))
    return pd.concat(out, ignore_index=True)


def stat(x):
    x = np.asarray(x, float)
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return x.mean(), se, (x.mean() / se if se and se > 0 else np.nan)


def analyze(model, df):
    df = add_scores(df, model)
    rows = []
    for (fmt, meas, layer), g in df.groupby(["format", "measure", "layer"]):
        p = g.pivot_table(index=["seq", "cell"], columns=["cond", "coef"], values="score")
        base = p[("none", 0.0)].unstack("cell")  # (seq, cell)
        nat = {"change": (base["LH"] + base["HL"] - base["LL"] - base["HH"]) / 2,
               "recent": (base["LH"] + base["HH"] - base["LL"] - base["HL"]) / 2,
               "history": (base["HL"] + base["HH"] - base["LL"] - base["LH"]) / 2}
        info = g.iloc[0]
        for cond, coef in p.columns:
            if cond == "none":
                continue
            eff = p[(cond, coef)].unstack("cell") - base
            for target, cells in (("unchanged", ["LL", "HH"]), ("changed", ["LH", "HL"]), ("all", ["LL", "LH", "HL", "HH"])):
                m, se, t = stat(eff[cells].mean(1))
                rows.append({"model": model, "format": fmt, "measure": meas, "layer": int(layer), "cond": cond,
                             "coef": coef, "cells": target, "effect": m, "se": se, "t": t, "n": len(eff),
                             "natural_change": nat["change"].mean(), "natural_change_se": stat(nat["change"])[1],
                             "natural_recent": nat["recent"].mean(), "natural_history": nat["history"].mean(),
                             "norm_change": info["norm_change"], "norm_recent": info["norm_recent"],
                             "norm_history": info["norm_history"], "cos_change_recent": info["cos_change_recent"]})
    return pd.DataFrame(rows)


def cell(r):
    return f"{r.effect:+.2f} ({r.t:+.1f})"


def report(model, res, md):
    layers = sorted(res.layer.unique())
    q = [f for f in QUESTIONS if f in set(res.format)]
    rep = res[(res.measure == "report") & res.format.isin(q)]
    md.append(f"\n## {model}\n")
    one = rep[rep.layer == layers[0]].drop_duplicates("format").set_index("format")
    md.append("Natural effects on the report score (no injection, nats; change = (LH + HL - LL - HH) / 2):\n")
    md.append("| question | change (se) | recent | history |\n|---|---|---|---|")
    for f in q:
        r = one.loc[f]
        md.append(f"| {f} | {r.natural_change:+.2f} ({r.natural_change_se:.2f}) | {r.natural_recent:+.2f} | {r.natural_history:+.2f} |")

    def table(title, cond, coef, cells):
        md.append(f"\n{title} Effect in nats (t over streams), by layer:\n")
        md.append("| question | " + " | ".join(f"L{l}" for l in layers) + " |\n|---|" + "---|" * len(layers))
        sub = rep[(rep.cond == cond) & (rep.coef == coef) & (rep.cells == cells)]
        for f in q:
            s = sub[sub.format == f].set_index("layer")
            md.append(f"| {f} | " + " | ".join(cell(s.loc[l]) for l in layers) + " |")
        ok = [l for l in layers if all((sub[(sub.format == f) & (sub.layer == l)].effect.iloc[0] > 0) == (coef > 0) for f in q)]
        side = '"changed"' if coef > 0 else '"unchanged"'
        names = ", ".join(f"L{l}" for l in ok) if ok else "none"
        md.append(f"\nLayers where all {len(q)} questions move toward {side}: {names}.")

    table("**+1 x change direction on the unchanged cells (LL, HH): the transplant.**", "change", 1.0, "unchanged")
    table("**+4 x change direction on the unchanged cells.**", "change", 4.0, "unchanged")
    table("**-1 x change direction on the changed cells (LH, HL): the removal.**", "change", -1.0, "changed")
    table("**Projection to the other class's mean, changed cells.**", "proj", 1.0, "changed")
    # controls: norm-matched random directions and the recent-numbers direction, all cells
    md.append("\n**Controls, all four cells** (mean effect on the report score over the questions, and the largest "
              "absolute effect on one question; nats):\n")
    md.append("| condition | " + " | ".join(f"L{l}" for l in layers) + " |\n|---|" + "---|" * len(layers))
    for label, cond, coef in (("+1 change", "change", 1.0), ("+4 change", "change", 4.0),
                              ("+4 random 0 (norm of 4 change)", "randm0", 4.0), ("+4 random 1", "randm1", 4.0),
                              ("+1 recent", "recent", 1.0), ("+4 recent", "recent", 4.0),
                              ("+1 history", "history", 1.0), ("+4 history", "history", 4.0)):
        s = rep[(rep.cond == cond) & (rep.coef == coef) & (rep.cells == "all")]
        md.append(f"| {label} | " + " | ".join(
            f"{s[s.layer == l].effect.mean():+.2f} (max {s[s.layer == l].effect.abs().max():.2f})" for l in layers) + " |")
    md.append("\nA direction that only shifts the answer bias moves q_ab and q_ba, or q_surprise and q_same, in "
              "opposite directions on this score, so its mean over the questions is near 0 while its largest effect "
              "is not.")
    # other readouts
    for meas, fmt, unit in (("surprise_words", "wait", "nats"), ("int_mean", "comma", "numbers"), ("int_sd", "comma", "numbers")):
        s = res[(res.measure == meas) & (res.format == fmt) & (res.cells == "all")]
        if s.empty:
            continue
        nat = s.iloc[0]
        md.append(f"\n**{meas} after \"{fmt}\"** ({unit}; natural change {nat.natural_change:+.2f}, recent "
                  f"{nat.natural_recent:+.2f}, history {nat.natural_history:+.2f}). Effect on all cells (t):\n")
        md.append("| condition | " + " | ".join(f"L{l}" for l in layers) + " |\n|---|" + "---|" * len(layers))
        for label, cond, coef in (("+1 change", "change", 1.0), ("+4 change", "change", 4.0), ("+1 recent", "recent", 1.0),
                                  ("+1 history", "history", 1.0), ("+4 random 0", "randm0", 4.0)):
            t = s[(s.cond == cond) & (s.coef == coef)].set_index("layer")
            md.append(f"| {label} | " + " | ".join(cell(t.loc[l]) for l in layers) + " |")
    one = res[res.format == q[0]].drop_duplicates("layer").set_index("layer")
    md.append(f"\nDirections at the tail of {q[0]} (mean over tail tokens): norm of change / recent / history, and "
              "cosine(change, recent): " + "; ".join(
                  f"L{l}: {one.loc[l].norm_change:.1f} / {one.loc[l].norm_recent:.1f} / {one.loc[l].norm_history:.1f}, "
                  f"{one.loc[l].cos_change_recent:+.2f}" for l in layers) + ".")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--e", type=int, default=4)
    a = ap.parse_args()
    md = ["# Experiment 11, step 3: injecting the pure change direction\n",
          "Cells: LL low throughout, LH up jump, HL down jump, HH high throughout (cut e = %d). The change direction "
          "is (LH + HL - LL - HH) / 2 of the residual stream at each tail token, fitted without the test stream. "
          "Report score = log P(changed answer) - log P(other answer)." % a.e]
    allres = []
    for model in sorted(os.listdir(a.raw)):
        f = os.path.join(a.raw, model, f"inject_change_e{a.e}.csv.gz")
        if not os.path.exists(f):
            continue
        res = analyze(model, pd.read_csv(f))
        info = json.load(open(f.replace(".csv.gz", "_info.json")))
        report(model, res, md)
        md.append(f"\n{len(info['seqs'])} streams; {info['seconds']} s on {info['gpu']}.")
        allres.append(res)
        print(f"{model}: {len(res)} rows", flush=True)
    assert allres, f"no inject_change_e{a.e}.csv.gz under {a.raw}"
    os.makedirs(a.out, exist_ok=True)
    pd.concat(allres).to_csv(os.path.join(a.out, "inject_change.csv"), index=False)
    open(os.path.join(a.out, "inject_change.md"), "w").write("\n".join(md) + "\n")
    print(f"{a.out}/inject_change.md")
