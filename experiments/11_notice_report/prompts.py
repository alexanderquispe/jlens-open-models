"""Prompt formats of experiment 11: how a context (a number stream) is turned into a question for a base model or an
instruct model, which answer tokens are read, and which answer means "the numbers changed / were surprising".

Base models read the stream as plain text, "412,388,...,705", followed by a cue (experiment 09's format), or inside
a two-example few-shot frame (both orders of the examples, to cancel a preference for the last example's answer).
Instruct models get the stream in a user message, through the model's chat template, and answer after the assistant
header. Questions with reversed polarity (q_same: "No" means changed) and both orders of a two-option question
(q_ab, q_ba) separate a report of the change from a bias toward one answer.
"""

# name: (cue after the stream, answer kind, the answer that means "changed / surprising")
BASE = {
    "comma": (",", "int", None),
    "wait": (",\n\nWait,", "words", None),
    "q_surprise": (",\n\nQ: Was the last number surprising? A:", "yesno", "yes"),
    "q_change": (",\n\nQ: Did the numbers change? A:", "yesno", "yes"),
    "q_same": (",\n\nQ: Do the last numbers look like the earlier ones? A:", "yesno", "no"),
    "q_ab": (",\n\nQ: Compared with the earlier numbers, the last numbers are (A) similar (B) different. A: (", "ab", "b"),
    "q_ba": (",\n\nQ: Compared with the earlier numbers, the last numbers are (A) different (B) similar. A: (", "ab", "a"),
}
_DEMO_YES = "Numbers: 52,48,51,49,50,53,47,50,49,51,95,97,94,96\nQ: Did the numbers change? A: Yes\n\n"
_DEMO_NO = "Numbers: 31,29,30,32,28,30,31,29,30,31,30,29,31,30\nQ: Did the numbers change? A: No\n\n"
FEWSHOT = {  # name: (text before the stream, cue after it, kind, changed answer)
    "fs_change": (_DEMO_YES + _DEMO_NO + "Numbers: ", "\nQ: Did the numbers change? A:", "yesno", "yes"),
    "fs_change_r": (_DEMO_NO + _DEMO_YES + "Numbers: ", "\nQ: Did the numbers change? A:", "yesno", "yes"),
}
_YN = " Answer with Yes or No only."
_AB = " Answer with A or B only."
CHAT = {  # name: (question after the stream in the user message, kind, changed answer)
    "q_surprise": ("Was the last number surprising, given the earlier numbers?" + _YN, "yesno", "yes"),
    "q_change": ("Did the numbers change?" + _YN, "yesno", "yes"),
    "q_same": ("Do the last numbers look like the earlier ones?" + _YN, "yesno", "no"),
    "q_ab": ("Compared with the earlier numbers, are the last numbers (A) similar or (B) different?" + _AB, "ab", "b"),
    "q_ba": ("Compared with the earlier numbers, are the last numbers (A) different or (B) similar?" + _AB, "ab", "a"),
}
CHAT_INTRO = "Here is a sequence of numbers:\n\n"
# Answer tokens: every variant that is a single token is read; an answer's log-probability is the log-sum-exp over
# its variants
ANSWERS = {"yes": ["Yes", " Yes", "yes", " yes", "YES"], "no": ["No", " No", "no", " no", "NO"],
           "a": ["A", " A"], "b": ["B", " B"]}
OPPOSITE = {"yes": "no", "no": "yes", "a": "b", "b": "a"}


def is_chat(model):
    return model.endswith("-it")


def formats(model):
    """[(name, family, kind, changed answer)]. Formats of one family share the text before the stream's end, so they
    are run on one KV cache."""
    if is_chat(model):
        return [(n, "chat", k, a) for n, (_, k, a) in CHAT.items()]
    return ([(n, "plain", k, a) for n, (_, k, a) in BASE.items()]
            + [(n, n, k, a) for n, (_, _, k, a) in FEWSHOT.items()])


def build(tok, model, numbers, name):
    """Token ids of the full prompt for a context and a format."""
    stream = ",".join(map(str, numbers))
    if is_chat(model):
        msg = [{"role": "user", "content": CHAT_INTRO + stream + "\n\n" + CHAT[name][0]}]
        text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        return tok(text, add_special_tokens=False).input_ids
    text = stream + BASE[name][0] if name in BASE else FEWSHOT[name][0] + stream + FEWSHOT[name][1]
    ids = tok(text).input_ids
    if tok.bos_token_id is not None and ids[0] != tok.bos_token_id:
        ids = [tok.bos_token_id] + ids
    return ids


def answer_ids(tok):
    """{answer: [single-token ids of its variants]} (duplicates removed)."""
    out = {}
    for a, variants in ANSWERS.items():
        ids = []
        for v in variants:
            t = tok.encode(v, add_special_tokens=False)
            if len(t) == 1 and t[0] not in ids:
                ids.append(t[0])
        out[a] = ids
    return out
