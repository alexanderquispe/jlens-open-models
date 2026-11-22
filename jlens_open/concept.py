"""Concept Lens (Gong, Huang and Wang 2026, "Verbalizing Multi-Token Concepts in LLMs", arXiv:2608.31084) on LensModel.

Written from the authors' code (github.com/XijieGo/c-lens at f1a6811, src/bindlens), which is not imported or run
here: the method's constants and steps are theirs, the implementation is ours. One readout of the hidden states h_l
at the last token of a prompt, for each source layer l (Llama-3.1-8B: 10-30):

  READ   the lens logits of h_l, unembed(J_l h_l); the first 100 tokens, in three rank bands (1-20, 21-50, 51-100,
         beam widths 12, 6, 6), keeping word-initial lexical tokens (at most 20 per band) as roots.
  BIND   the model itself completes each root after a scaffold that contains the prompt,
         "when model is answering the cloze below:\\n{prompt} .\\nit will think a concept:", by a beam search over
         lexical tokens (nucleus 0.85 of the top 10 at each step, at most 4 tokens). A beam of 2 or more tokens
         becomes a candidate when an end token ("." " ." "\\n" ".\\n" EOS) is in the nucleus and no intra-word fragment
         beats the best end token. Per layer: the best 25 by proposer score, minus longer phrases whose token prefix
         scores at least as high.
  CARRY  each candidate C is read in a carrier, "Remember the following concept in your mind. Concept:" + C +
         ". Fact: The capital of France is Paris.", and its block outputs at the carrier layer (27 of 32) on the
         suffix tokens are pooled: v_C = sum_p w_p (h_p - mu_p) + b, with w >= 0 summing to 1. The reader (mu, w, b)
         is fitted in closed form on 300 one-token concepts whose target is the lens direction W_U[c] J_carrier:
         mu_p and b are the means, w_p the mean cosine of (h_p - mu_p) with (target - b), clipped at 0.
  TRANSPORT  v_C is moved to source layer s by ridge regression on the lens: v_s = M_s v_C, with
         M_s = (J_s^T J_s + lam I)^-1 J_s^T J_c and lam = 0.01 |J_s|_F^2 / d.
  SCORE  with N the final norm (gain included) and q = N(J_s h_s): a candidate scores cos(N(J_s v_s), q) and a
         vocabulary token cos(W_U[w], q), both times the median |W_U[w]|; candidates and all tokens are ranked
         together.

Differences from the authors' code, none of which changes what is computed beyond floating point:
  - the transport maps are kept in float32 (theirs are applied in float64);
  - every row of a BIND frontier is decoded from one shared KV cache of the scaffold (theirs also caches; the rows
    are batched differently);
  - v_C is cached by surface string across readouts (it depends on the string alone).
The scaffold, carrier and readout are text-level, so BIND needs transformers >= 5 (.venv-tf5), whose cache classes
this module fills directly.
"""
import hashlib
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field

import torch

CARRIER_PREFIX = "Remember the following concept in your mind. Concept:"
CARRIER_SUFFIX = ". Fact: The capital of France is Paris."
DECODE_TEMPLATE = "when model is answering the cloze below:\n{cloze} .\nit will think a concept:"
ROOT_BANDS = ((1, 20, 12), (21, 50, 6), (51, 100, 6))  # (first rank, last rank, beam width), 1-indexed, inclusive
SEED_K = 20
TOP_P = 0.85
EXPAND_K = 10
MAX_TOKENS = 4
SELECT_TOP_K = 25
RIDGE = 1e-2
CALIBRATION_URL = ("https://raw.githubusercontent.com/XijieGo/c-lens/f1a6811e7c20c880025eb368144d9fe17815a14a/"
                   "data/singleton_calibration_300.json")
CALIBRATION_SHA256 = "ecd0d39079af05f768cc01402c5e8f300be037218dfa3068bbd4e8bd78a11e90"


def default_layers(n_blocks):
    """Source layers and carrier layer: the authors' 10-30 and 27 for Llama-3.1-8B (32 blocks), scaled to depth."""
    return list(range(round(10 * n_blocks / 32), round(30 * n_blocks / 32) + 1)), round(27 * n_blocks / 32)


def calibration_concepts(path):
    """The authors' 300 calibration concepts (data/singleton_calibration_300.json), downloaded once to `path`."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(CALIBRATION_URL, timeout=60) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
    with open(path, "rb") as f:
        data = f.read()
    assert hashlib.sha256(data).hexdigest() == CALIBRATION_SHA256, f"{path} differs from the authors' file"
    return json.loads(data)["concepts"]


def transport_map(Js, Jc, ridge=RIDGE):
    """M = (Js^T Js + lam I)^-1 Js^T Jc in float64, lam = ridge |Js|_F^2 / d: v_s = M v_c keeps the lens effect."""
    Js, Jc = Js.double(), Jc.double()
    d = Js.shape[1]
    lam = ridge * (Js**2).sum() / d
    chol, info = torch.linalg.cholesky_ex(Js.T @ Js + lam * torch.eye(d, dtype=torch.float64, device=Js.device))
    assert int(info) == 0, "ridge system not positive definite"
    return torch.cholesky_solve(Js.T @ Jc, chol)


def positive_pool(X, Y):
    """The authors' positive-pool reader. X (n, P, d) suffix states, Y (n, d) target directions.
    Returns means (P, d), weights (P,) >= 0 summing to 1, bias (d,) and the mean paired cosine of the fit."""
    X, Y = X.float(), Y.float()
    means, bias = X.mean(0), Y.mean(0)
    Xc, Yc = X - means, Y - bias
    cos = (Xc * Yc[:, None]).sum(-1) / (Xc.norm(dim=-1) * Yc.norm(dim=-1)[:, None]).clamp_min(1e-12)
    w = cos.mean(0).clamp_min(0)
    w = w / w.sum().clamp_min(1e-12) if float(w.sum()) > 0 else torch.full_like(w, 1 / len(w))
    pooled = (Xc * w[None, :, None]).sum(1)
    paired = ((pooled * Yc).sum(-1) / (pooled.norm(dim=-1) * Yc.norm(dim=-1)).clamp_min(1e-12)).mean()
    return means, w, bias, float(paired)


def nucleus(lp, top_p=TOP_P, top_k=EXPAND_K):
    """Per row of log-probs: the top_k (index, logprob) pairs and how many to keep (cumulative probability < top_p,
    plus one; at least 1)."""
    v, i = lp.topk(min(top_k, lp.shape[1]), dim=1)
    keep = ((v.exp().cumsum(1) < top_p).sum(1) + 1).clamp(1, v.shape[1])
    return i, v, keep


def prefix_dominance(cands):
    """cands: [(score, ids, text)]. Drop a candidate when a shorter one whose ids are its prefix scores no lower."""
    return [c for c in cands if not any(len(o[1]) < len(c[1]) and c[1][: len(o[1])] == o[1] and o[0] >= c[0]
                                        for o in cands if o is not c)]


def normalise(text):
    text = re.sub(r"\s+", " ", str(text)).strip().strip("\"'`")
    return text.rstrip(". ").casefold()


def valid_phrase(text):
    return bool(text) and text[0].isalpha() and all(c.isalpha() or c in " '-" for c in text)


def _word(s):
    return bool(s) and s[0].isalpha() and all(c.isalpha() or c in "'-" for c in s)


@dataclass
class Group:
    layer: int
    width: int
    beams: list
    done: dict = field(default_factory=dict)  # normalised text -> (score, ids, surface)


class ConceptLens:
    def __init__(self, lm, J=None, source_layers=None, carrier_layer=None, kv_budget_gb=16.0):
        """lm: a LensModel. J: {layer: (d, d) float32} lens matrices (default lm.J; pass the J++ lens to read with it)."""
        self.lm, self.tok, self.dev = lm, lm.tok, lm.device
        self.J = lm.J if J is None else J
        src, car = default_layers(lm.n_blocks)
        self.src = list(source_layers) if source_layers is not None else src
        self.car = carrier_layer if carrier_layer is not None else car
        assert all(l in self.J for l in self.src + [self.car]), "the lens has no J at some source or carrier layer"
        self.kv_budget = kv_budget_gb * 2**30
        self.W = lm.model.lm_head.weight  # (V, d), model dtype
        self.V = self.W.shape[0]
        self.wn = self.W.float().norm(dim=1)
        self.med = self.wn.median()
        self._tables()
        self._transport_maps()
        self.reader = None
        self.vC = {}  # surface -> v_C (d,), or None when the carrier does not align
        self._surface = {}
        config = getattr(lm.model.config, "text_config", lm.model.config)
        self.window = getattr(config, "sliding_window", None)

    # ---------------------------------------------------------------------------------------------- tables
    def _tables(self):
        """Lexical, word-initial lexical and intra-word fragment tokens, and the end tokens (bind.py of the authors)."""
        lex, init, frag = [], [], []
        n_tok = len(self.tok)
        for i in range(self.V):
            if i >= n_tok:
                continue
            raw = self.tok.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            surf = self.tok.decode([i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if _word(surf.replace("\n", " ").strip().replace(" ", "")):
                lex.append(i)
            if raw and raw[0].isspace() and _word(raw.strip()):
                init.append(i)
            if raw and not raw[0].isspace() and _word(raw):
                frag.append(i)
        self.lex = torch.tensor(lex, device=self.dev)
        self.word_initial = set(init)
        self.frag = torch.tensor(frag, device=self.dev)
        ends = []
        for text in (".", " .", "\n", ".\n"):
            ids = self.tok.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                ends.append(ids[0])
        if self.tok.eos_token_id is not None:
            ends.append(self.tok.eos_token_id)
        self.end = torch.tensor(sorted(set(ends)), device=self.dev)
        self.end_set = set(self.end.tolist())
        self.term = torch.cat([self.lex, self.end])

    @torch.no_grad()
    def _transport_maps(self):
        self.M = {s: transport_map(self.J[s], self.J[self.car]).float() for s in self.src if s != self.car}
        if self.dev == "cuda":
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------------------------- helpers
    def N(self, x):
        """The final norm, gain included, in the model's dtype (the authors' SCORE normalization)."""
        return self.lm.norm(x.to(self.lm.norm.weight.device, self.lm.dtype)).float().to(self.dev)

    def lens_logits(self, h, layer):
        return self.lm.unembed(h @ self.J[layer].T)

    def encode(self, text, offsets=False):
        enc = self.tok(text, add_special_tokens=True, return_offsets_mapping=offsets)
        ids = list(enc.input_ids)
        offs = list(enc.offset_mapping) if offsets else None
        bos = self.tok.bos_token_id
        if bos is not None and ids[0] != bos:
            ids = [bos] + ids
            offs = [(0, 0)] + offs if offsets else None
        return (ids, offs) if offsets else ids

    def surface(self, ids):
        ids = tuple(ids)
        if ids not in self._surface:
            self._surface[ids] = (self.tok.decode(list(ids), skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                  .replace("\n", " ").strip())
        return self._surface[ids]

    @torch.no_grad()
    def block_outputs(self, batch_ids, layers):
        """{layer: (B, T, d) float32} block outputs for a batch of equal-length id lists."""
        out = {}

        def hook(l):
            def f(module, inputs, output):
                out[l] = (output[0] if isinstance(output, tuple) else output).float().to(self.dev)
            return f

        hs = [self.lm.blocks[l].register_forward_hook(hook(l)) for l in layers]
        try:
            self.lm.model(torch.tensor(batch_ids, device=self.dev), use_cache=False, logits_to_keep=1)
        finally:
            for h in hs:
                h.remove()
        return out

    # ---------------------------------------------------------------------------------------------- CARRY
    def grid(self, concept):
        """Token ids of the carrier and the suffix span, or None if the concept/suffix boundary does not align."""
        prefix, suffix = CARRIER_PREFIX, CARRIER_SUFFIX
        ids, offs = self.encode(prefix + concept + suffix, offsets=True)
        c0, c1 = len(prefix), len(prefix) + len(concept)
        cidx = [i for i, (a, b) in enumerate(offs) if b > a and b > c0 and a < c1]
        sidx = [i for i, (a, b) in enumerate(offs) if b > a and a >= c1]
        if not cidx or any(offs[i][1] > c1 for i in cidx) or not sidx:
            return None
        if sidx != list(range(sidx[0], sidx[-1] + 1)) or sidx[-1] != len(ids) - 1:
            return None
        return {"ids": ids, "concept_ids": [ids[i] for i in cidx], "suffix_ids": [ids[i] for i in sidx]}

    @torch.no_grad()
    def _suffix_states(self, grids, batch):
        """(n, P, d) carrier-layer block outputs on the suffix tokens, in the order of `grids`."""
        order = sorted(range(len(grids)), key=lambda i: len(grids[i]["ids"]))
        out = [None] * len(grids)
        s = 0
        while s < len(order):
            n = len(grids[order[s]]["ids"])
            e = s
            while e < len(order) and e - s < batch and len(grids[order[e]]["ids"]) == n:
                e += 1
            idx = order[s:e]
            P = len(grids[idx[0]]["suffix_ids"])
            h = self.block_outputs([grids[i]["ids"] for i in idx], [self.car])[self.car][:, -P:]
            for j, i in enumerate(idx):
                out[i] = h[j]
            s = e
        return torch.stack(out)

    @torch.no_grad()
    def fit_reader(self, concepts, batch=8):
        """The positive-pool reader (fit_positive_pool_reader of the authors' carry.py), on their 300 concepts."""
        grids = [self.grid(c) for c in concepts]
        aligned = [g for g in grids if g is not None]
        single = [g for g in aligned if len(g["concept_ids"]) == 1]
        assert single, "no one-token calibration concept aligns in the carrier"
        suffix = single[0]["suffix_ids"]
        single = [g for g in single if g["suffix_ids"] == suffix]
        X = self._suffix_states(single, batch)  # (n, P, d)
        ids = torch.tensor([g["concept_ids"][0] for g in single], device=self.dev)
        Y = self.W[ids].float() @ self.J[self.car]  # lens direction of each concept token (no norm gain, as theirs)
        means, w, bias, paired = positive_pool(X, Y)
        self.reader = {"means": means, "weights": w, "bias": bias, "suffix_ids": suffix}
        self.vC = {}
        return {"requested": len(concepts), "aligned": len(aligned), "used": len(single),
                "suffix_tokens": len(suffix), "paired_cosine": paired, "weights": w.tolist()}

    @torch.no_grad()
    def carry(self, surfaces, batch=16):
        """Fill self.vC for the surfaces not seen yet."""
        new = [s for s in dict.fromkeys(surfaces) if s not in self.vC]
        grids, keep = [], []
        for s in new:
            g = self.grid(s)
            if g is None or g["suffix_ids"] != self.reader["suffix_ids"]:
                self.vC[s] = None
            else:
                grids.append(g)
                keep.append(s)
        if not grids:
            return
        X = self._suffix_states(grids, batch)
        r = self.reader
        v = ((X - r["means"]) * r["weights"][None, :, None]).sum(1) + r["bias"]
        for s, vi in zip(keep, v):
            self.vC[s] = vi

    # ---------------------------------------------------------------------------------------------- BIND
    @torch.no_grad()
    def _prefill(self, ids):
        out = self.lm.model(torch.tensor([ids], device=self.dev), use_cache=True, logits_to_keep=1)
        kv = [(layer.keys, layer.values) for layer in out.past_key_values.layers]
        return kv, out.logits[0, -1].float()

    @torch.no_grad()
    def _next_logits(self, kv, seqs):
        """(len(seqs), V) next-token logits after the cached scaffold followed by each seq (all the same length)."""
        from transformers.cache_utils import DynamicCache, DynamicLayer

        n = kv[0][0].shape[2]
        per_token = sum(K[0, :, :1].numel() + V[0, :, :1].numel() for K, V in kv) * kv[0][0].element_size()
        k = len(seqs[0])
        if self.window:
            assert n + k <= self.window, "the scaffold is longer than the sliding window"
        b_max = max(1, int(self.kv_budget // (per_token * (n + k))))
        rows = []
        for s in range(0, len(seqs), b_max):
            chunk = seqs[s : s + b_max]
            b = len(chunk)
            cache = DynamicCache()
            for K, V in kv:
                layer = DynamicLayer()
                layer.lazy_initialization(K, V)
                layer.keys, layer.values = K.expand(b, -1, -1, -1), V.expand(b, -1, -1, -1)
                cache.layers.append(layer)
            out = self.lm.model(torch.tensor(chunk, device=self.dev), past_key_values=cache, use_cache=True,
                                logits_to_keep=1)
            rows.append(out.logits[:, -1].float())
            del cache, out
        return torch.cat(rows)

    @torch.no_grad()
    def propose(self, cloze, lens_logits):
        """{layer: [(surface, ids, proposer score)]} candidates (BIND and SELECT of the authors)."""
        ctx = self.encode(DECODE_TEMPLATE.format(cloze=cloze.rstrip()))
        kv, z0 = self._prefill(ctx)
        lp0 = torch.full((self.V,), float("-inf"), device=self.dev)
        lp0[self.lex] = torch.log_softmax(z0[self.lex], 0)
        lp0 = lp0.cpu()
        groups = []
        for l in self.src:
            top = lens_logits[l].topk(100).indices.tolist()
            for a, b, width in ROOT_BANDS:
                roots = [t for t in top[a - 1 : b] if t in self.word_initial][:SEED_K]
                groups.append(Group(l, width, [((t,), float(lp0[t])) for t in roots]))
        lex, term = self.lex.cpu(), self.term.cpu()
        for depth in range(MAX_TOKENS):  # frontier lengths 1 .. MAX_TOKENS
            seqs = list(dict.fromkeys(seq for g in groups for seq, _ in g.beams))
            if not seqs:
                break
            Z = self._next_logits(kv, seqs)
            row = {seq: i for i, seq in enumerate(seqs)}
            ends = {}
            if depth >= 1:  # beams of 2+ tokens: the terminal check
                ti, tv, tk = nucleus(torch.log_softmax(Z[:, self.term], 1))
                veto = (Z[:, self.frag].max(1).values > Z[:, self.end].max(1).values) if len(self.frag) else None
                ti, tv, tk = term[ti.cpu()], tv.cpu(), tk.cpu()
                veto = veto.cpu() if veto is not None else torch.zeros(len(seqs), dtype=torch.bool)
                for seq, r in row.items():
                    e = [float(tv[r, c]) for c in range(int(tk[r])) if int(ti[r, c]) in self.end_set]
                    if e and not bool(veto[r]):
                        ends[seq] = max(e)
            if depth < MAX_TOKENS - 1:
                ai, av, ak = nucleus(torch.log_softmax(Z[:, self.lex], 1))
                ai, av, ak = lex[ai.cpu()], av.cpu(), ak.cpu()
            for g in groups:
                kids = []
                for seq, s in g.beams:
                    if seq in ends:
                        text = self.surface(seq)
                        if valid_phrase(text):
                            key = normalise(text)
                            score = s + ends[seq]
                            if key not in g.done or score > g.done[key][0]:
                                g.done[key] = (score, seq, text)
                    if depth < MAX_TOKENS - 1:
                        r = row[seq]
                        for c in range(int(ak[r])):
                            t = int(ai[r, c])
                            new = seq + (t,)
                            if t in seq or not valid_phrase(self.surface(new)):
                                continue
                            kids.append((new, s + float(av[r, c])))
                if depth < MAX_TOKENS - 1:
                    kids.sort(key=lambda x: -x[1])
                    g.beams = kids[: g.width]
            del Z
        del kv
        out = {}
        for l in self.src:
            merged = {}
            for g in groups:
                if g.layer != l:
                    continue
                for key, (score, ids, text) in g.done.items():
                    if key not in merged or score > merged[key][0]:
                        merged[key] = (score, ids, text)
            top = sorted(merged.values(), key=lambda x: -x[0])[:SELECT_TOP_K]
            out[l] = [(text, ids, score) for score, ids, text in prefix_dominance(top)]
        return out

    # ---------------------------------------------------------------------------------------------- readout
    @torch.no_grad()
    def readout(self, cloze, H, n_tokens=10):
        """Concept Lens at the last token of `cloze`. H: {layer: (d,) float32 block output there}.

        Returns {layer: {"concepts": [(surface, joint rank, score, proposer score)] sorted by rank,
                         "tokens": [(token id, score)] the n_tokens best vocabulary tokens}}.
        """
        assert self.reader is not None, "fit_reader first"
        L = {l: self.lens_logits(H[l][None], l)[0] for l in self.src}
        cands = self.propose(cloze, L)
        self.carry([c[0] for l in self.src for c in cands[l]])
        Wb = self.W
        out = {}
        for l in self.src:
            q = self.N(H[l][None] @ self.J[l].T)[0]
            sv = (Wb @ q.to(Wb.dtype)).float() / self.wn * self.med
            ts, ti = sv.topk(n_tokens)
            cl = [c for c in cands[l] if self.vC.get(c[0]) is not None]
            rec = {"tokens": list(zip(ti.tolist(), ts.tolist())), "concepts": []}
            if cl:
                v = torch.stack([self.vC[c[0]] for c in cl])
                if l != self.car:
                    v = v @ self.M[l].T
                dC = self.N(v @ self.J[l].T)
                sC = (dC / dC.norm(dim=1, keepdim=True)) @ q * self.med
                rank = 1 + (sv[None, :] > sC[:, None]).sum(1) + (sC[None, :] > sC[:, None]).sum(1)
                rec["concepts"] = sorted(((c[0], int(r), float(s), c[2]) for c, r, s in zip(cl, rank.tolist(), sC)),
                                         key=lambda x: x[1])
            out[l] = rec
        return out
