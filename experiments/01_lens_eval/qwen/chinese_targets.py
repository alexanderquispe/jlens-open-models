"""Experiment 01 rescored with each intermediate counted in English or in Chinese.

Same prompts, readout positions and pass@k as ../lens_eval.py. For four sets (association, multihop,
multilingual, typo) an intermediate also counts as found when the lens ranks its Chinese translation
(zh_intermediates.json, our translations) in the top k. A Chinese word counts through its first token only
when it is a single token or its first token has at least two CJK characters, so single common characters
(十 for 十二月) do not count. order-ops and poetry keep the English targets. Three target sets are scored:
English (reproduces ../results/<model>.csv), English or Chinese, and Chinese only.

    python experiments/01_lens_eval/qwen/chinese_targets.py --model qwen3-8b
"""
import argparse
import json
import os

import pandas as pd
import torch

from jlens_open import MODELS, LensModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), required=True)
parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: the first available)")
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")
CSV_OUT = os.path.join(HERE, f"results/chinese_targets_{args.model}.csv")
CSV_RANKS = os.path.join(HERE, f"results/chinese_targets_{args.model}_ranks.csv")
SETS = ["multihop", "multilingual", "order-ops", "poetry", "association", "typo"]
WITH_ZH = {"association", "multihop", "multilingual", "typo"}
with open(os.path.join(HERE, "zh_intermediates.json")) as f:
    ZH = json.load(f)

# Targets and readout positions as in ../lens_eval.py
NUMBER_WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four".split()
OPERATIONS = {
    "addition": ["+", "plus", "add", "addition", "sum"],
    "subtraction": ["-", "minus", "subtract", "subtraction"],
    "multiplication": ["*", "×", "times", "multiply", "multiplication"],
    "division": ["/", "÷", "divide", "divided", "division"],
    "squared": ["squared", "square", "²", "^"],
    "mod": ["mod", "%", "modulo", "remainder"],
}

lm = LensModel(args.model, args.device)
tok = lm.tok


def target_ids(word, eval_name):
    if eval_name != "order-ops":
        return lm.first_token(word)
    synonyms = OPERATIONS.get(word) or [word] + ([NUMBER_WORDS[int(word)]] if word.isdigit() and int(word) < len(NUMBER_WORDS) else [])
    ids = set()
    for s in synonyms:
        for w in (s, " " + s):
            t = tok.encode(w, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
    return ids


def encode_with_position(prompt, eval_name):
    ids = lm.encode(prompt.rstrip(" "))
    if eval_name != "poetry":
        return ids, len(ids) - 1
    return ids, max(i for i, t in enumerate(ids.tolist()) if "\n" in tok.decode([t]))


def is_cjk(s):
    return len(s) > 0 and all("一" <= c <= "鿿" for c in s)


def zh_ids(word):
    ids = set()
    for w in ZH.get(word, []):
        t = tok.encode(w, add_special_tokens=False)
        first = tok.decode([t[0]])
        if is_cjk(first) and (len(t) == 1 or len(first) >= 2):
            ids.add(t[0])
    return ids


rows = []
for eval_name in SETS:
    with open(os.path.join(DATA, f"lens-eval-{eval_name}.json")) as f:
        items = json.load(f)["items"]
    for item in items:
        ids, position = encode_with_position(item["prompt"], eval_name)
        hs, _ = lm.run(ids[None])
        for use_jacobian, lens in ((True, "J-lens"), (False, "logit lens")):
            logits = lm.unembed(torch.stack([lm.transport(hs[layer][0, position], layer, use_jacobian) for layer in range(lm.n_blocks)]))
            for word in item["intermediates"]:
                en = set(target_ids(word, eval_name))
                zh = zh_ids(word) if eval_name in WITH_ZH else set()
                for targets, label in ((en, "English"), (en | zh, "English or Chinese"), (zh, "Chinese only")):
                    if not targets:
                        continue
                    t = sorted(targets)
                    ranks = (logits[:, None, :] > logits[:, t][:, :, None]).sum(-1) + 1  # (layers, targets)
                    layer, j = divmod(ranks.argmin().item(), len(t))
                    rows.append({"eval": eval_name, "item": item["name"], "word": word, "lens": lens, "targets": label,
                                 "has Chinese token": bool(zh), "min_rank": ranks.min().item(), "best layer": layer,
                                 "best token": tok.decode([t[j]])})
    print(f"done: {eval_name} ({len(items)} items)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(CSV_RANKS, index=False)
tables = []
for label in ("English", "English or Chinese", "Chinese only"):
    d = df[df.targets == label].assign(**{"pass@10": lambda x: x.min_rank <= 10})
    t = d.groupby(["eval", "lens", "item"])["pass@10"].mean().groupby(["eval", "lens"]).mean().unstack("lens")
    t.loc["macro (6 evals)" if label != "Chinese only" else "macro (4 evals)"] = t.mean()
    tables.append(t.assign(targets=label).set_index("targets", append=True))
table = pd.concat(tables)
table.to_csv(CSV_OUT)

pd.set_option("display.width", 200)
coverage = df[(df.lens == "J-lens") & (df.targets == "English")].groupby("eval")["has Chinese token"].mean()
print(f"\n{args.model}: share of intermediates with a usable Chinese token\n{coverage.round(2).to_string()}")
print(f"\n{args.model}: pass@10 by target language")
print(table.round(3).to_string())
print(f"\nSaved: {CSV_OUT}\n       {CSV_RANKS}")
