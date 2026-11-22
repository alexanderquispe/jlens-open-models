"""Lens readouts at the decision points of agent calls: tables and figures.

    python analyze.py CALLS_CSV ACTS_DIR [ACTS_DIR ...] --out results

CALLS_CSV is the output of labels.py, ACTS_DIR a folder written by replay.py. Positions: A before the reasoning,
B at its end, C at the end of the text (see replay.py). Lens 0 is the Jacobian lens, lens 1 the logit lens.

1. Replay check. The probability the replayed model gives to the token the agent actually wrote next.
2. Next tool. At B and C, for calls that go on to call a tool: the rank, in the lens readout of each block, of the
   word that names the first tool called, and how often that word is the most likely of the 14 tool words.
3. Act or not. At A, B and C: score = log P(act words) - log P(wait words) in the lens readout. AUC of the score
   for calls that change jobs against calls that do not, per block and averaged over thirds of the depth.
   Position C sits on different tokens (the end of a text, or the tag closing the reasoning), which early blocks
   tell apart, so C is also scored within the calls that wrote a text. For comparison with what the activations
   hold in any form: a difference-of-means direction fitted on the other agents' calls (leave one agent out).
4. Word sets by what the agent does. log P(set) - log P(control words), middle and late thirds.
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import wordsets

parser = argparse.ArgumentParser()
parser.add_argument("calls")
parser.add_argument("acts", nargs="+")
parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
POS, LENS = ["A", "B", "C"], ["J-lens", "logit lens"]
rng = np.random.default_rng(0)

# ---- load
labels = pd.read_csv(args.calls)
tokens = json.load(open(os.path.join(args.acts[0], "tokens.json")))
column = {t: i for i, t in enumerate(tokens["tokens"])}
metas, logps, ranks, hiddens = [], [], [], []
for folder in args.acts:
    for path in sorted(glob.glob(os.path.join(folder, "*.pt"))):
        if os.path.basename(path) == "head.pt":
            continue
        d = torch.load(path, weights_only=False)
        for m in d["meta"]:
            if "row" in m:
                metas.append(m | {"source": os.path.basename(folder.rstrip("/"))})
        if d["lens"]:
            logps.append(d["lens"]["logp"].float().numpy())
            ranks.append(d["lens"]["rank"].numpy())
            hiddens.append(d["H"].float().numpy())
logp, rank = np.concatenate(logps), np.concatenate(ranks)  # (calls, lens, position, block, token)
hidden = np.concatenate(hiddens)  # (calls, position, block, d)
meta = pd.DataFrame(metas).merge(labels, on=["source", "episode", "call"], how="left", validate="one_to_one")
assert len(meta) == len(logp) and meta.action.notna().all(), "calls without a label"
n_blocks = logp.shape[3]
thirds = {"early": range(0, n_blocks // 3), "middle": range(n_blocks // 3, 2 * n_blocks // 3),
          "late": range(2 * n_blocks // 3, n_blocks)}
md = [f"Calls replayed: {len(meta)} ({(meta.formation == 'team').sum()} team, {(meta.formation == 'coordinator').sum()} "
      f"coordinator), from {meta.episode.nunique()} episodes. Blocks: {n_blocks}.", ""]


def table(df, index=False):
    """A data frame as a Markdown table (pandas needs the tabulate package for this)."""
    if index:
        df = df.reset_index()
    df.columns = [" ".join(map(str, c)).strip() if isinstance(c, tuple) else str(c) for c in df.columns]
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def lse(x, axis=-1):
    m = x.max(axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis, keepdims=True))).squeeze(axis)


def set_logp(words):
    """log of the summed lens probability of the words' first tokens: (calls, lens, position, block)."""
    cols = sorted({column[t] for w in words for t in tokens["words"][w]})
    return lse(logp[..., cols])


def auc(pos, neg):
    """P(score of a positive > score of a negative), ties counted half."""
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return ((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def auc_ci(score, y, n=2000):
    """AUC with a 95% interval from resampling calls."""
    a = auc(score[y], score[~y])
    idx = rng.integers(0, len(y), (n, len(y)))
    boot = [auc(score[i][y[i]], score[i][~y[i]]) for i in idx]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return a, lo, hi


# ---- 1. replay check
written = np.exp(np.array(meta.written_logp.tolist()))
top1 = np.array([[t[k][0] == w[k] for k in range(3)] for t, w in zip(meta.top5, meta.written)])
same = (meta.n_prompt == meta.prompt_tokens_logged).mean()
check = pd.DataFrame({"position": POS, "mean probability of the written token": written.mean(0).round(3),
                      "written token is the model's top token": top1.mean(0).round(3)})
check.to_csv(os.path.join(args.out, "replay_check.csv"), index=False)
md += ["## 1. Replay check", "", f"Rebuilt prompts with the token count the server logged: {same:.1%}.", "",
       table(check), ""]

# ---- 2. next tool
tool_words = list(dict.fromkeys(wordsets.TOOLS.values()))
tool_cols = [[column[t] for t in tokens["words"][w]] for w in tool_words]
first_tool = meta.tools.fillna("").str.split().str[0]
has_tool = first_tool.isin(wordsets.TOOLS).to_numpy()
target = first_tool.map(wordsets.TOOLS)
rows = []
for li, lens in enumerate(LENS):
    for pi in (1, 2):
        per_tool = np.stack([logp[:, li, pi][..., c].max(-1) for c in tool_cols], -1)  # (calls, block, tools)
        guess = per_tool.argmax(-1)
        truth = np.array([tool_words.index(t) if h else -1 for t, h in zip(target, has_tool)])
        r = np.stack([rank[i, li, pi][:, [column[t] for t in tokens["words"][target[i]]]].min(-1)
                      for i in np.flatnonzero(has_tool)])  # (calls with a tool, block)
        for b in range(n_blocks):
            rows.append({"lens": lens, "position": POS[pi], "block": b, "median rank": np.median(r[:, b]),
                         "in top 10": (r[:, b] <= 10).mean(), "right tool of 14": (guess[has_tool, b] == truth[has_tool]).mean()})
tool = pd.DataFrame(rows)
tool.to_csv(os.path.join(args.out, "next_tool_by_block.csv"), index=False)
majority = target[has_tool].value_counts(normalize=True).iloc[0]
summary = []
for (lens, pos), g in tool.groupby(["lens", "position"]):
    for name, blocks in thirds.items():
        s = g[g.block.isin(blocks)]
        summary.append({"lens": lens, "position": pos, "third": name, "median rank": s["median rank"].median(),
                        "in top 10": round(s["in top 10"].mean(), 3), "right tool of 14": round(s["right tool of 14"].mean(), 3)})
summary = pd.DataFrame(summary)
summary.to_csv(os.path.join(args.out, "next_tool.csv"), index=False)
md += ["## 2. Next tool", "", f"Calls that go on to call a tool: {has_tool.sum()}. Always guessing the most common "
       f"first tool ({target[has_tool].value_counts().index[0]}) is right {majority:.1%} of the time.", "",
       table(summary), ""]
fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
for ax, pos in zip(axes, ("B", "C")):
    for lens, style in zip(LENS, ("-", "--")):
        g = tool[(tool.lens == lens) & (tool.position == pos)]
        ax.plot(g.block, g["right tool of 14"], style, label=lens)
    ax.axhline(majority, color="gray", linewidth=0.8, linestyle=":", label="most common tool")
    ax.set_title(f"position {pos}", fontsize=9)
    ax.set_xlabel("Block")
    ax.grid(linestyle="--", alpha=0.4)
axes[0].set_ylabel("first tool read correctly")
axes[0].legend(fontsize=7)
plt.tight_layout()
fig.savefig(os.path.join(args.out, "next_tool.png"), dpi=150)
plt.close(fig)

# ---- 3. act or not
score = set_logp(wordsets.SETS["act"]) - set_logp(wordsets.SETS["wait"])  # (calls, lens, position, block)
y = (meta.action == "change").to_numpy()
groups = {"all": np.ones(len(meta), bool), "team": (meta.formation == "team").to_numpy(),
          "coordinator": (meta.formation == "coordinator").to_numpy()}
rows, by_block = [], []
for gname, g in groups.items():
    if y[g].sum() < 3 or (~y[g]).sum() < 3:
        continue
    for li, lens in enumerate(LENS):
        for pi, pos in enumerate(POS):
            for b in range(n_blocks):
                by_block.append({"calls": gname, "lens": lens, "position": pos, "block": b,
                                 "auc": auc(score[g, li, pi, b][y[g]], score[g, li, pi, b][~y[g]])})
            for name, blocks in thirds.items():
                a, lo, hi = auc_ci(score[g, li, pi][:, list(blocks)].mean(1), y[g])
                rows.append({"calls": gname, "n change": int(y[g].sum()), "n other": int((~y[g]).sum()), "lens": lens,
                             "position": pos, "third": name, "auc": round(a, 3), "low": round(lo, 3), "high": round(hi, 3)})
act = pd.DataFrame(rows)
act.to_csv(os.path.join(args.out, "act_auc.csv"), index=False)
by_block = pd.DataFrame(by_block)
by_block.to_csv(os.path.join(args.out, "act_auc_by_block.csv"), index=False)
# A reference that needs no activations: was the agent's previous call an observation?
meta = meta.sort_values(["source", "episode", "agent", "call"])
previous = meta.groupby(["source", "episode", "agent"]).action.shift(1)
ref = auc((previous == "observe").to_numpy().astype(float)[meta.action.to_numpy() == "change"],
          (previous == "observe").to_numpy().astype(float)[meta.action.to_numpy() != "change"])
meta = meta.sort_index()
# C within the calls that wrote a text, and a probe on the activations themselves
has_text = np.array([p[2] != p[1] for p in meta.positions])
extra = []
g = has_text
if y[g].sum() >= 3 and (~y[g]).sum() >= 3:
    for li, lens in enumerate(LENS):
        for name, blocks in thirds.items():
            a, lo, hi = auc_ci(score[g, li, 2][:, list(blocks)].mean(1), y[g])
            extra.append({"score": f"{lens}, C, calls with a text", "n change": int(y[g].sum()),
                          "n other": int((~y[g]).sum()), "third": name, "auc": round(a, 3), "low": round(lo, 3), "high": round(hi, 3)})
a, lo, hi = auc_ci(has_text.astype(float), y)
extra.append({"score": "the call wrote a text (no activations)", "n change": int(y.sum()), "n other": int((~y).sum()),
              "third": "", "auc": round(a, 3), "low": round(lo, 3), "high": round(hi, 3)})
agent_id = meta.source + "/" + meta.episode + "/" + meta.agent.astype(str)
for pi, pos in enumerate(POS[:2]):
    for name, blocks in thirds.items():
        b = list(blocks)[len(blocks) // 2]
        x = hidden[:, pi, b]
        probe = np.full(len(meta), np.nan)
        for held in agent_id.unique():
            test = (agent_id == held).to_numpy()
            train = ~test
            if y[train].sum() < 2 or (~y[train]).sum() < 2:
                continue
            direction = x[train & y].mean(0) - x[train & ~y].mean(0)
            probe[test] = (x[test] - x[train].mean(0)) @ direction
        ok = ~np.isnan(probe)
        a, lo, hi = auc_ci(probe[ok], y[ok])
        extra.append({"score": f"difference of means on activations, {pos}, block {b}", "n change": int(y[ok].sum()),
                      "n other": int((~y[ok]).sum()), "third": name, "auc": round(a, 3), "low": round(lo, 3), "high": round(hi, 3)})
extra = pd.DataFrame(extra)
extra.to_csv(os.path.join(args.out, "act_auc_extra.csv"), index=False)
md += ["## 3. Act or not", "", "AUC of log P(act words) - log P(wait words) for calls that change jobs against the "
       "other calls, averaged over thirds of the depth, with 95% intervals from resampling calls. 0.5 is chance.",
       f"Reference without activations (the agent's previous call was an observation): AUC {ref:.3f}.", "",
       table(act), "", "Checks: position C within the calls that wrote a text, the text indicator alone, and a "
       "direction fitted on the activations of the other agents' calls.", "", table(extra), ""]
fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
for ax, pos in zip(axes, POS):
    for lens, style in zip(LENS, ("-", "--")):
        g = by_block[(by_block.calls == "all") & (by_block.lens == lens) & (by_block.position == pos)]
        ax.plot(g.block, g.auc, style, label=lens)
    ax.axhline(0.5, color="gray", linewidth=0.8)
    ax.set_title(f"position {pos}", fontsize=9)
    ax.set_xlabel("Block")
    ax.grid(linestyle="--", alpha=0.4)
axes[0].set_ylabel("AUC, change against other calls")
axes[0].legend(fontsize=7)
plt.tight_layout()
fig.savefig(os.path.join(args.out, "act_auc.png"), dpi=150)
plt.close(fig)

# ---- 3b. message or not (team calls)
control = set_logp(wordsets.SETS["control"])
team = (meta.formation == "team").to_numpy()
ym = (meta.action == "message").to_numpy()
rows = []
if ym[team].sum() >= 3:
    ask = set_logp(wordsets.SETS["ask"]) - control
    for li, lens in enumerate(LENS):
        for pi, pos in enumerate(POS[:2]):
            for name, blocks in thirds.items():
                a, lo, hi = auc_ci(ask[team, li, pi][:, list(blocks)].mean(1), ym[team])
                rows.append({"score": f"{lens}, ask words, {pos}", "third": name, "auc": round(a, 3), "low": round(lo, 3), "high": round(hi, 3)})
    for pi, pos in enumerate(POS[:2]):
        for name, blocks in thirds.items():
            b = list(blocks)[len(blocks) // 2]
            x = hidden[:, pi, b]
            probe = np.full(len(meta), np.nan)
            for held in agent_id[team].unique():
                test = (agent_id == held).to_numpy()
                train = team & ~test
                if ym[train].sum() < 2:
                    continue
                probe[test] = (x[test] - x[train].mean(0)) @ (x[train & ym].mean(0) - x[train & ~ym].mean(0))
            ok = ~np.isnan(probe)
            a, lo, hi = auc_ci(probe[ok], ym[ok])
            rows.append({"score": f"difference of means on activations, {pos}, block {b}", "third": name,
                         "auc": round(a, 3), "low": round(lo, 3), "high": round(hi, 3)})
    message = pd.DataFrame(rows)
    message.to_csv(os.path.join(args.out, "message_auc.csv"), index=False)
    md += ["## 3b. Message or not (team calls)", "", f"Calls whose action is a message to a peer ({int(ym[team].sum())}) "
           f"against the other team calls ({int((team & ~ym).sum())}). Lens score: log P(ask words) - log P(control words).",
           "", table(message), ""]

# ---- 3c. what the lens shows at the end of the reasoning: a few calls
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
    tops = []
    for folder in args.acts:
        for path in sorted(glob.glob(os.path.join(folder, "*.pt"))):
            if os.path.basename(path) != "head.pt":
                tops.append(torch.load(path, weights_only=False)["lens"]["top"].numpy())
    tops = np.concatenate(tops)  # (calls, lens, position, block, 20)
    rows = []
    for action in ("change", "message", "wait", "observe"):
        for i in meta.index[(meta.action == action) & (meta.formation == "team")][:3]:
            for b in (2 * n_blocks // 3 - 2, n_blocks - 8):
                rows.append({"action": action, "agent of": meta.principal[i], "call": meta.call[i], "block": b,
                             "J-lens top tokens at B": " ".join(repr(tok.decode([t])) for t in tops[i, 0, 1, b, :8])})
    examples = pd.DataFrame(rows)
    examples.to_csv(os.path.join(args.out, "examples_top_tokens.csv"), index=False)
    md += ["## 3c. Top lens tokens at the end of the reasoning", "", "Three team calls per action, Jacobian lens, "
           "position B. The next token the agent writes starts its text or its tool call.", "", table(examples), ""]
except Exception as error:
    print("examples skipped:", repr(error))

# ---- 4. word sets by action
rows = []
for name in ("act", "wait", "ask", "scarce", "fair", "people", "surprise"):
    s = set_logp(wordsets.SETS[name]) - control
    for pi, pos in enumerate(POS):
        for third in ("middle", "late"):
            v = s[:, 0, pi][:, list(thirds[third])].mean(1)  # Jacobian lens
            for (formation, action), idx in meta.groupby(["formation", "action"]).indices.items():
                rows.append({"set": name, "position": pos, "third": third, "formation": formation, "action": action,
                             "n": len(idx), "mean": round(float(v[idx].mean()), 3),
                             "se": round(float(v[idx].std(ddof=1) / np.sqrt(len(idx))), 3) if len(idx) > 1 else np.nan})
sets = pd.DataFrame(rows)
sets.to_csv(os.path.join(args.out, "sets_by_action.csv"), index=False)
wide = sets[(sets.position == "B") & (sets.third == "late")].pivot_table(index=["formation", "action", "n"], columns="set", values="mean")
md += ["## 4. Word sets by what the agent does", "", "Jacobian lens, position B, late third: mean of log P(set) - "
       "log P(control words). The other positions and the middle third are in sets_by_action.csv.", "",
       table(wide.astype(float).round(2), index=True), ""]

# ---- 5. changes to another user's jobs (team)
kind = np.where(meta.n_scale_peer > 0, "changes another user's jobs", "changes own jobs only")
changes = ((meta.action == "change") & (meta.formation == "team")).to_numpy()
if (changes & (meta.n_scale_peer > 0).to_numpy()).sum() >= 2:
    rows = []
    for pi, pos in enumerate(POS[:2]):
        for k in ("changes another user's jobs", "changes own jobs only"):
            idx = np.flatnonzero(changes & (kind == k))
            row = {"position": pos, "calls": k, "n": len(idx)}
            for name in ("people", "fair", "ask", "scarce", "act", "wait"):
                v = (set_logp(wordsets.SETS[name]) - control)[idx, 0, pi][:, list(thirds["late"])].mean(1)
                row[name] = f"{v.mean():.2f} ({v.std(ddof=1) / np.sqrt(len(v)):.2f})"
            rows.append(row)
    peer = pd.DataFrame(rows)
    peer.to_csv(os.path.join(args.out, "peer_changes.csv"), index=False)
    md += ["## 5. Changing another user's jobs", "", "Team calls that change jobs, split by whether any of the jobs "
           "belongs to another user. Jacobian lens, late third: mean (standard error) of log P(set) - log P(control "
           "words). Few calls: descriptive only.", "", table(peer), ""]

with open(os.path.join(args.out, "results.md"), "w") as f:
    f.write("\n".join(md))
print("\n".join(md))
