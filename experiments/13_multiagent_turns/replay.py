"""Replay the model calls of MAMUBench episodes through the weights and keep the activations at the decision points.

Input: the JSON lines of extract_calls.py (one line per call an agent made to the model). Each call is rebuilt as the
token sequence the server saw (chat template, tools, earlier turns), followed by what the model then wrote: its
reasoning, the tag that closes it, and its text. Three positions are read, at every block:

    A  the last prompt token ("<think>\\n"): before any reasoning for this call
    B  the last token of the tag closing the reasoning: the next token starts the text or the tool call
    C  the last token of the text (B when there is none): the next token starts a tool call or ends the turn

The prompts run to tens of thousands of tokens, so the forward pass goes in chunks with the key-value cache, and only
the three positions are kept. Before the episodes, the chunked pass is checked against a single pass on a short prompt.

Output, per episode: OUT/<episode>.pt with
    H         the block outputs (calls, 3, blocks, d), float16
    lens      for the Jacobian lens (index 0) and the logit lens (index 1), at every call, position and block:
              "logp" and "rank" of the first tokens of the words in wordsets.py (calls, 2, 3, blocks, tokens), and the
              20 most likely tokens "top" with their log probabilities "top_logp"
    meta      per call: the positions, whether the rebuilt prompt has the token count the server logged, and the
              model's own next-token distribution at A, B and C (top 5, and the probability of the token it wrote)
OUT/head.pt holds what the lens needs on a CPU (unembedding matrix, final norm's gain and epsilon), and
OUT/tokens.json the token ids of every word.

    sbatch hpc/gpu.sbatch env PYTHONPATH=. .venv-tf5/bin/python experiments/13_multiagent_turns/replay.py CALLS_DIR OUT_DIR
"""
import argparse
import glob
import json
import os
import sys
import time

import torch

from jlens_open import LensModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wordsets  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("calls")
parser.add_argument("out")
parser.add_argument("--model", default="qwen3.8-27b")
parser.add_argument("--chunk", type=int, default=4096)
parser.add_argument("--max-tokens", type=int, default=200_000, help="skip calls longer than this")
parser.add_argument("--limit", type=int, default=None, help="calls per episode (for a quick test)")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

lm = LensModel(args.model)
tok = lm.tok
print(f"{args.model}: {lm.n_blocks} blocks, d = {lm.d}, lens averaged over {lm.n_lens_prompts} prompts", flush=True)


def prompt_ids(row):
    """The prompt as the server tokenized it. The template wants tool-call arguments as mappings, not JSON text."""
    messages = json.loads(json.dumps(row["messages"]))
    for m in messages:
        for t in m.get("tool_calls") or []:
            if isinstance(t["function"]["arguments"], str):
                t["function"]["arguments"] = json.loads(t["function"]["arguments"])
    ids = tok.apply_chat_template(messages, tools=row["tools"], add_generation_prompt=True, tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def encode(text):
    return tok.encode(text, add_special_tokens=False)


def sequence(row):
    """Token ids, the positions A, B, C, and the token the model wrote after each (None when unknown)."""
    prompt = prompt_ids(row)
    reasoning = encode((row["reasoning"] or "").strip() + "\n</think>\n\n")
    text = encode((row["content"] or "").strip())
    ids = prompt + reasoning + text
    a, b = len(prompt) - 1, len(prompt) + len(reasoning) - 1
    c = len(ids) - 1
    if row["tool_calls"]:
        after_c = encode("\n\n<tool_call>" if text else "<tool_call>")[0]
    else:
        after_c = tok.convert_tokens_to_ids("<|im_end|>")
    after_b = text[0] if text else after_c
    return ids, [a, b, c], [reasoning[0], after_b, after_c], len(prompt)


@torch.no_grad()
def forward(ids, positions, chunk):
    """Block outputs (len(positions), blocks, d) in float32 and the logits (len(positions), vocab) at `positions`.
    chunk=None runs a single pass without the cache."""
    wanted = torch.tensor(positions)
    hidden = torch.zeros(len(positions), lm.n_blocks, lm.d)
    logits = torch.zeros(len(positions), lm.model.lm_head.weight.shape[0])
    offset = 0

    def hook(layer):
        def f(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            for k, p in enumerate(positions):
                if offset <= p < offset + h.shape[1]:
                    hidden[k, layer] = h[0, p - offset].float().cpu()
        return f

    handles = [b.register_forward_hook(hook(i)) for i, b in enumerate(lm.blocks)]
    try:
        ids = torch.tensor(ids)[None].to(lm.device)
        past = None
        for start in range(0, ids.shape[1], chunk or ids.shape[1]):
            piece = ids[:, start:start + (chunk or ids.shape[1])]
            offset = start
            inside = (wanted >= start) & (wanted < start + piece.shape[1])
            if chunk is None and HEADLESS:
                # Logits at every position of a 100k-token prompt take tens of GB: run the decoder alone and
                # unembed the wanted positions only
                last = lm.decoder(input_ids=piece, use_cache=False).last_hidden_state
                for k in inside.nonzero().flatten().tolist():
                    logits[k] = lm.model.lm_head(last[0, positions[k] - start]).float().cpu()
                del last
                continue
            out = lm.model(input_ids=piece, past_key_values=past, use_cache=chunk is not None)
            past = out.past_key_values if chunk is not None else None
            for k in inside.nonzero().flatten().tolist():
                logits[k] = out.logits[0, positions[k] - start].float().cpu()
            del out
    finally:
        for h in handles:
            h.remove()
    return hidden, logits


WORD_IDS = {w: sorted(lm.first_token(w)) for w in wordsets.words()}
TOKENS = sorted({t for ids in WORD_IDS.values() for t in ids})
with open(os.path.join(args.out, "tokens.json"), "w") as f:
    json.dump({"words": WORD_IDS, "tokens": TOKENS, "decoded": [tok.decode([t]) for t in TOKENS]}, f)


@torch.no_grad()
def lens_readout(hidden):
    """hidden (positions, blocks, d) -> logp, rank (2, positions, blocks, tokens) and top, top_logp
    (2, positions, blocks, 20); index 0 of the first axis is the Jacobian lens, index 1 the logit lens."""
    ids = torch.tensor(TOKENS, device=lm.device)
    out = {"logp": [], "rank": [], "top": [], "top_logp": []}
    for use_jacobian in (True, False):
        per_position = {k: [] for k in out}
        for k in range(hidden.shape[0]):
            x = torch.stack([lm.transport(hidden[k, layer].to(lm.device), layer, use_jacobian) for layer in range(lm.n_blocks)])
            logp = lm.unembed(x).log_softmax(-1)  # (blocks, vocab)
            top = logp.topk(20)
            per_position["logp"].append(logp[:, ids].cpu())
            rank = [((logp[:, None, :] > logp[:, part][:, :, None]).sum(-1) + 1).cpu() for part in ids.split(32)]
            per_position["rank"].append(torch.cat(rank, 1))
            per_position["top"].append(top.indices.cpu())
            per_position["top_logp"].append(top.values.cpu())
        for name in out:
            out[name].append(torch.stack(per_position[name]))
    return {name: torch.stack(v) for name, v in out.items()}


# The chunked pass has to give what a single pass gives
check = encode("The quick brown fox jumps over the lazy dog. " * 40)[:300]
HEADLESS = False
h_one, l_one = forward(check, [150, 299], None)
try:
    HEADLESS = True
    h_dec, l_dec = forward(check, [150, 299], None)
    head_gap = (l_dec - l_one).abs().max().item()
    HEADLESS = head_gap < 0.05 and (h_dec - h_one).abs().max().item() < 1e-3
    print(f"decoder alone against the full model: largest logit gap {head_gap:.4f}; used: {HEADLESS}", flush=True)
except Exception as error:  # keep the slower route that is known to work
    HEADLESS = False
    print(f"decoder alone failed ({error!r}): using the full model", flush=True)
h_cut, l_cut = forward(check, [150, 299], 64)
gap = ((h_cut - h_one).norm(dim=-1) / h_one.norm(dim=-1)).max().item()
same_top = (l_one.argmax(-1) == l_cut.argmax(-1)).all().item()
print(f"chunked against single pass: largest relative gap {gap:.4f}, same top token {same_top}", flush=True)
chunk = args.chunk if gap < 0.05 and same_top else None
if chunk is None:
    print("the chunked pass differs: using single passes without the cache", flush=True)

head = os.path.join(args.out, "head.pt")
if not os.path.exists(head):
    torch.save({"W": lm.model.lm_head.weight.detach().to(torch.float16).cpu(), "gain": lm.gain.float().cpu(),
                "eps": getattr(lm.norm, "eps", getattr(lm.norm, "variance_epsilon", None)),
                "norm": type(lm.norm).__name__, "softcap": lm.softcap}, head)

for path in sorted(glob.glob(os.path.join(args.calls, "*.jsonl"))):
    episode = os.path.basename(path)[:-len(".jsonl")]
    out = os.path.join(args.out, episode + ".pt")
    rows = [json.loads(line) for line in open(path)][:args.limit]
    H, meta, L = [], [], []
    part = out + ".part"  # progress within the episode, so that a job that runs out of time loses little
    if os.path.exists(out):
        saved = torch.load(out, weights_only=False)
        if len(saved["meta"]) >= len(rows) and not any("skipped" in m for m in saved["meta"]):
            print(f"{episode}: done already", flush=True)
            continue
        H, meta = list(saved["H"]), saved["meta"]
        L = [{name: v[i] for name, v in saved["lens"].items()} for i in range(len(H))]
    elif os.path.exists(part):
        saved = torch.load(part, weights_only=False)
        H, meta, L = saved["H"], saved["meta"], saved["L"]
    meta = [m for m in meta if "row" in m]  # calls skipped before (out of memory) are tried again
    done = {m["call"] for m in meta}
    if done:
        print(f"{episode}: {len(done)} calls done before", flush=True)
    last_save = time.time()
    for row in rows:
        if row["call"] in done:
            continue
        t0 = time.time()
        ids, positions, written, n_prompt = sequence(row)
        info = {"episode": episode, "call": row["call"], "n_prompt": n_prompt, "n_tokens": len(ids),
                "prompt_tokens_logged": row["prompt_tokens"], "positions": positions, "written": written}
        if len(ids) > args.max_tokens:
            info["skipped"] = "too long"
            meta.append(info)
            continue
        try:
            hidden, logits = forward(ids, positions, chunk)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            info["skipped"] = "out of memory"
            meta.append(info)
            print(f"{episode} call {row['call']}: out of memory at {len(ids)} tokens", flush=True)
            continue
        logp = logits.log_softmax(-1)
        info["top5"] = logp.topk(5).indices.tolist()
        info["top5_logp"] = logp.topk(5).values.tolist()
        info["written_logp"] = [logp[k, w].item() for k, w in enumerate(written)]
        info["row"] = len(H)
        H.append(hidden.to(torch.float16))
        L.append(lens_readout(hidden))
        meta.append(info)
        print(f"{episode} call {row['call']}: {len(ids)} tokens (prompt {n_prompt}, logged {row['prompt_tokens']}), "
              f"{time.time() - t0:.1f} s, wrote {[tok.decode([w]) for w in written]}, "
              f"p = {[round(float(torch.tensor(x).exp()), 3) for x in info['written_logp']]}", flush=True)
        if time.time() - last_save > 240:
            torch.save({"H": H, "meta": meta, "L": L}, part + ".tmp")
            os.replace(part + ".tmp", part)
            last_save = time.time()
    tmp = out + ".tmp"
    kinds = {"logp": torch.float16, "top_logp": torch.float16, "rank": torch.int32, "top": torch.int32}
    lens = {name: torch.stack([x[name].to(kind) for x in L]) for name, kind in kinds.items()} if L else {}
    H = [h.to(torch.float16) for h in H]
    torch.save({"H": torch.stack(H) if H else torch.zeros(0, 3, lm.n_blocks, lm.d, dtype=torch.float16),
                "lens": lens, "meta": meta}, tmp)
    os.replace(tmp, out)
    if os.path.exists(part):
        os.remove(part)
    print(f"Saved: {out} ({len(H)} of {len(rows)} calls)", flush=True)
