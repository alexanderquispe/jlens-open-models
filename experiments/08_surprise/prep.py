"""Tokenization and validity checks of experiment 08 (needs only a tokenizer, so tests.py can run it on CPU).

Part A: every number must be one token, at the same token positions in the three arms, and the arms' tokens must
differ only at numbers whose values differ. Part B: the twins must have the same number of tokens; r (and each
wrong value) must be one token, the same position in clean, near and far, and the only token that differs there;
the swap must change only tokens inside the swapped region. Items that fail are dropped and counted.
"""
import numpy as np

from common import ARMS, PREFIX, SEP, SLOTS, T, TWINS


def tokenize(tok, text):
    """Token ids, with BOS first when the tokenizer has one (as LensModel.encode), and each token's character span."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc.input_ids), [tuple(o) for o in enc.offset_mapping]
    if any(a == b for a, b in offs):
        # The Llama-3 tokenizer under tokenizers 0.19 returns (start, start) for tokens that are whole vocabulary
        # words (no merges applied); their starts are right. Its byte-level tokens tile the text, so such a token
        # ends where the next one begins (tests.py checks that every span decodes to its token)
        starts = [a for a, _ in offs] + [len(text)]
        offs = [(a, starts[i + 1]) if a == b else (a, b) for i, (a, b) in enumerate(offs)]
    if tok.bos_token_id is not None:
        ids, offs = [tok.bos_token_id] + ids, [(0, 0)] + offs
    return ids, offs


def span_tokens(offs, start, end):
    """Indices of the tokens whose characters overlap [start, end)."""
    return [i for i, (a, b) in enumerate(offs) if b > a and a < end and b > start]


def integer_table(tok, n=1000):
    """{k: token id} for the integers 0..n-1 that are one token where a number stands in a Part A sequence."""
    table = {}
    for k in range(n):
        text = PREFIX + "1" + SEP + str(k)
        ids, offs = tokenize(tok, text)
        toks = span_tokens(offs, len(text) - len(str(k)), len(text))
        if len(toks) == 1:
            table[k] = ids[toks[0]]
    assert len(set(table.values())) == len(table), "two integers share a token"
    return table


def prepare_changepoint(df, tok, items):
    """{item: [(ids, P, numbers) per arm in ARMS order]} for the valid items, and the number dropped.
    P (T, 3): token positions of the slots num / sep / pred of each number (-1 where a slot does not exist)."""
    prepared, n_drop = {}, 0
    for item in items:
        sub = df[df.item == item].set_index("arm").loc[list(ARMS)]
        arms, ok = [], True
        for arm, row in sub.iterrows():
            nums = list(map(int, row.numbers.split()))
            ids, offs = tokenize(tok, row.text)
            spans, pos = [], len(PREFIX)
            for x in nums:
                assert row.text[pos : pos + len(str(x))] == str(x)
                spans.append((pos, pos + len(str(x))))
                pos += len(str(x)) + len(SEP)
            toks = [span_tokens(offs, a, b) for a, b in spans]
            if any(len(t) != 1 for t in toks):
                ok = False
                break
            num = [t[0] for t in toks]
            P = np.full((T, len(SLOTS)), -1)
            P[:, 0] = num
            P[:-1, 1] = np.array(num[:-1]) + 1
            P[:-1, 2] = np.array(num[1:]) - 1
            arms.append((ids, P, nums))
        if ok:
            base_ids, base_P, base_nums = np.array(arms[0][0]), arms[0][1], arms[0][2]
            for ids, P, nums in arms[1:]:
                if len(ids) != len(base_ids) or not (P == base_P).all():
                    ok = False
                    break
                differ = set(np.nonzero(np.array(ids) != base_ids)[0])
                if not differ <= {P[t, 0] for t in range(T) if nums[t] != base_nums[t]}:
                    ok = False
                    break
        if ok:
            prepared[item] = arms
        else:
            n_drop += 1
    return prepared, n_drop


def prepare_arith(items, tok, pre, post):
    """The valid items, each {"it", "ids" (per twin in TWINS order), "n", "edit" {twin: (first, last) token},
    "result" (token of r), "reuse" (token of the later r, or -1), "window" (token positions read)}; the drop
    reasons {reason: count}; and the longest window."""
    prepared, reasons, wmax = [], {}, 0

    def drop(why):
        reasons[why] = reasons.get(why, 0) + 1

    for it in items:
        tw = {name: tokenize(tok, it["texts"][name]) for name in TWINS}
        n = len(tw["clean"][0])
        if len({len(tw[name][0]) for name in TWINS}) != 1:
            drop("token count differs")
            continue
        # r (and each wrong value) must be one token, the same token position in clean, near and far
        r_tok = {name: span_tokens(tw[name][1], *it["edit"]["clean"]) for name in ("clean", "near", "far")}
        if any(len(v) != 1 for v in r_tok.values()) or len({v[0] for v in r_tok.values()}) != 1:
            drop("r not one token")
            continue
        clean = np.array(tw["clean"][0])
        edit, ok = {}, True
        for name in ("near", "far", "swap"):
            # the twins have the same number of tokens; the edited tokens must lie inside the edit region (the
            # swap keeps the region's length, so the text outside it is identical and aligned)
            differ = np.nonzero(np.array(tw[name][0]) != clean)[0]
            region = set(span_tokens(tw["clean"][1], *it["edit"][name])) | set(span_tokens(tw[name][1], *it["edit"][name]))
            if len(differ) == 0 or not set(differ) <= region or (name != "swap" and list(differ) != r_tok["clean"]):
                ok = False
                break
            edit[name] = (int(differ.min()), int(differ.max()))
        if not ok:
            drop("edit not confined to its tokens")
            continue
        edit["clean"] = edit["near"]
        result = edit["near"][0]
        reuse = -1
        if it["reuse"]:
            toks = span_tokens(tw["clean"][1], *it["reuse"])
            reuse = toks[0] if len(toks) == 1 else -1
        window = set(range(max(1, edit["swap"][0] - pre - 1), min(n, edit["near"][1] + post + 1)))
        if reuse > 0:
            window |= set(range(max(1, reuse - 3), min(n, reuse + 6)))
        window = sorted(window)
        wmax = max(wmax, len(window))
        prepared.append({"it": it, "ids": [tw[name][0] for name in TWINS], "n": n, "edit": edit, "result": result,
                         "reuse": reuse, "window": window})
    return prepared, reasons, wmax
