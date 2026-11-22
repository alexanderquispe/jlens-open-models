"""Tests of experiment 09 (CPU, no model). From the repo root:

    .venv-tf5/bin/python experiments/09_beliefs/tests.py                           # stimuli, Concept Lens parts, analysis
    .venv-tf5/bin/python experiments/09_beliefs/tests.py --tokenizer llama3.1-8b   # also the tokenizer checks (HF cache)

  stimuli     the 17 condition files that the paper's code writes are reproduced byte for byte (SHA-256 of their
              data/sequences/*.jsonl at github.com/raphael-goodfire/shape-of-beliefs); the switch sequences are the
              N(300) and N(700) sequences end to end
  concept     the Concept Lens pieces: the ridge transport solves its normal equations and keeps the lens effect
              when J is invertible; the positive-pool reader puts its weight on the suffix position that carries the
              target; the nucleus rule and prefix dominance on hand-made cases
  analysis    analyze.py on synthetic outputs with planted effects: a word that rises with mu, one that rises with
              sigma and one that appears after the switch are found first; a lens whose integer mean equals mu from
              some layer on gives r = 1 there; the behaviour go / no-go
  tokenizer   (with --tokenizer) every integer is one token; the streams tokenize as [BOS] n , n , ...; a stream cut
              after any comma tokenizes as a prefix of the whole; the Concept Lens carrier aligns for the authors'
              calibration concepts (downloaded once)
"""
import argparse
import hashlib
import os
import sys
import tempfile
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(os.path.dirname(HERE)))
from common import CONDITIONS, E_BINS, MUS, MU0, SIGMA0, SIGMAS, T_BINS, cond_name  # noqa: E402
from stimuli import draw, reference_jsonl, sequences  # noqa: E402

# SHA-256 of data/sequences/gaussian_m{mu}_s{sigma:03d}_l1000_n10.jsonl in the paper's repository
PAPER_SHA256 = {
    "m300_s100": "29a681d19d0ba13fd11d07afe4fae2017b585453a086ea4dca5bd082f31998ee",
    "m350_s100": "2c765e468c1fc6f01b6d6017bc15d2c4248c1b063a90a0dea3416fde7684faf8",
    "m400_s100": "0b44c8c373e59db046299cb73c11faf0770eaf14cdca5b0f90d3553942398f37",
    "m450_s100": "20db01970289610963e16cf291cc89b7737e56af798d0535a682fd2c69958e77",
    "m500_s010": "ed35f90d0ad1ca70c5b391b696d4f9325aae5e12b11b47bc92086b8559679e29",
    "m500_s020": "ec4dfb54372916e15d1bcf6cad602e2de812e0c3e8ee9ce40520a38f6f992628",
    "m500_s030": "4cdad037eb24cd6e4df0a4af45dadb5b98c1f44e39ce878c8acb04255cbcb508",
    "m500_s050": "b27d73acb3ebf6a6845473aa984c2d2ac58824bee71f320d04b108665a233174",
    "m500_s080": "e78d7a84cc252a606485307d80b0ba31463006aad671709108dd08b882f973a6",
    "m500_s100": "7a40c9020bbe958bfd5493bf2fe20496ada63273f4da6d6db434a776499c0b01",
    "m500_s120": "e4ec047a561913cb00d2dffa1fe72ded66b3c480cc9ca99f1e133cd77295b16a",
    "m500_s150": "5092422d7cac519f1eb9d0090b92a5689cdcd7ef3b0156cc0a7c3ce3df756a6d",
    "m500_s200": "9fa90adf2fe9aa805024cbb00614c2d34cafa9ea6f6cb7ff29bba9be50c819e1",
    "m550_s100": "34a4164d473449d38a938646283f4c6df7141fe4729e0447a412c70cb9ff936a",
    "m600_s100": "5c2267308fe39144062a452a1d1c164f1681c2dfb6c8f6396bd927b0829b82cc",
    "m650_s100": "68aebc8674d85b892903e6502dfab9201ad619848e777c797e2c39a9a086b3c6",
    "m700_s100": "848c8c080705402cadc03b0a0d4d4e2a5e84d28cf620dfc5e3df1ed57e81311b",
}


# ---------------------------------------------------------------------------------------------------------- stimuli
def test_stimuli_match_paper():
    assert len(CONDITIONS) == 17
    for mu, sigma in CONDITIONS:
        got = hashlib.sha256(reference_jsonl(mu, sigma)).hexdigest()
        assert got == PAPER_SHA256[cond_name(mu, sigma)], f"{cond_name(mu, sigma)} differs from the paper's file"


def test_sequences():
    seqs = sequences()
    st = [s for s in seqs if s["dataset"] == "stationary"]
    sw = [s for s in seqs if s["dataset"] == "switch"]
    assert len(st) == 170 and len(sw) == 10
    a, b = draw(300, 100), draw(700, 100)
    for s in sw:
        k = s["seq"]
        assert s["numbers"] == a[k].tolist() + b[k].tolist() and s["switch_at"] == 1000
        assert s["text"] == ",".join(map(str, s["numbers"])) + ","
    for s in st:
        assert len(s["numbers"]) == 1000 and min(s["numbers"]) >= 10 and max(s["numbers"]) <= 990
    # one seed per condition: the conditions share their standard normal draws
    x, y = draw(500, 100).astype(float), draw(500, 50).astype(float)
    inner = (x > 10) & (x < 990)
    assert np.abs((x[inner] - 500) / 2 + 500 - y[inner]).max() <= 1.0


# ---------------------------------------------------------------------------------------------------------- concept
def test_transport():
    from jlens_open.concept import RIDGE, transport_map

    g = torch.Generator().manual_seed(0)
    d = 24
    Js, Jc = torch.randn(d, d, generator=g, dtype=torch.float64), torch.randn(d, d, generator=g, dtype=torch.float64)
    M = transport_map(Js, Jc)
    lam = RIDGE * (Js**2).sum() / d
    assert torch.allclose((Js.T @ Js + lam * torch.eye(d, dtype=torch.float64)) @ M, Js.T @ Jc, atol=1e-9)
    # with no ridge and an invertible Js, the map is exact: Js M v = Jc v
    M0 = transport_map(Js, Jc, ridge=0.0)
    v = torch.randn(d, generator=g, dtype=torch.float64)
    assert torch.allclose(Js @ (M0 @ v), Jc @ v, atol=1e-7)


def test_positive_pool():
    from jlens_open.concept import positive_pool

    g = torch.Generator().manual_seed(1)
    n, P, d = 200, 6, 32
    Y = torch.randn(n, d, generator=g)
    X = torch.randn(n, P, d, generator=g)
    X[:, 3] = 2 * Y + 0.1 * torch.randn(n, d, generator=g) + 5  # position 3 carries the target (scaled, offset)
    means, w, bias, paired = positive_pool(X, Y)
    assert abs(float(w.sum()) - 1) < 1e-6 and (w >= 0).all()
    assert int(w.argmax()) == 3 and float(w[3]) > 0.8, w
    assert paired > 0.9
    assert torch.allclose(bias, Y.mean(0))


def test_nucleus_and_dominance():
    from jlens_open.concept import normalise, nucleus, prefix_dominance, valid_phrase

    p = torch.tensor([[0.5, 0.3, 0.1, 0.05, 0.05], [0.9, 0.05, 0.03, 0.01, 0.01], [0.2] * 5])
    i, v, keep = nucleus(p.log(), top_p=0.85, top_k=4)
    # cumulative 0.5, 0.8, 0.9 -> two below 0.85, keep 3; 0.9 -> none below, keep 1; 0.2, 0.4, 0.6, 0.8 -> keep 4 (cap)
    assert keep.tolist() == [3, 1, 4] and i[0, :3].tolist() == [0, 1, 2]
    c = [(-1.0, (5, 6), "black hole"), (-0.5, (5,), "black"), (-2.0, (5, 6, 7), "black hole sun"), (-0.2, (8, 9), "x y")]
    kept = [x[2] for x in prefix_dominance(c)]
    assert kept == ["black", "x y"], kept
    c = [(-1.0, (5, 6), "black hole"), (-1.5, (5,), "black")]
    assert [x[2] for x in prefix_dominance(c)] == ["black hole", "black"]
    assert normalise("  Normal  Distribution. ") == "normal distribution" and normalise("'black hole'") == "black hole"
    assert valid_phrase("normal distribution") and not valid_phrase("3 sigma") and not valid_phrase("a.b")


# ---------------------------------------------------------------------------------------------------------- analysis
def synthetic_raw(root, L=6, V=60, T=600, n_seq=2, seed=0):
    """Fake run_model.py outputs with planted effects, in root/<model>/. Returns (disp, cls)."""
    import json

    rng = np.random.default_rng(seed)
    out = os.path.join(root, "fake")
    os.makedirs(out, exist_ok=True)
    disp = np.array([f"w{i}" for i in range(V)], dtype=object)
    cls = np.array(["word"] * V)
    cls[:10] = "integer"
    disp[:10] = [str(100 * i) for i in range(10)]
    conds = [cond_name(m, s) for m, s in CONDITIONS]
    nb = len(T_BINS) - 1
    lenses = ["logit", "jlens"]
    seqs = [{"cond": c, "seq": k, "mu": m, "sigma": s} for c, (m, s) in zip(conds, CONDITIONS) for k in range(n_seq)]
    N, n_tok = len(seqs), 1 + 2 * T
    for k in lenses:
        C = rng.normal(-12, 1, (L, len(conds) * nb, V)).astype(np.float32)
        for i, (m, s) in enumerate(CONDITIONS):
            for b in range(nb):
                C[:, i * nb + b, 20] = -3 + (m - 500) / 100   # w20 rises with mu
                C[:, i * nb + b, 21] = -3 + 3 * np.log(s / 100)   # w21 rises with sigma
                C[:, i * nb + b, 22] = -3 + (4 if b == nb - 1 else 0)  # w22 rises late
        np.save(os.path.join(out, f"stationary_{k}_cells.npy"), C.astype(np.float16))
        Cs = rng.normal(-12, 1, (L, len(E_BINS) - 1, V)).astype(np.float32)
        Cs[:, :, 23] = -3
        for e in (1, 2, 3, 4):
            Cs[:, E_BINS.index(e), 23] = 3  # w23 jumps right after the switch
        np.save(os.path.join(out, f"switch_{k}_cells.npy"), Cs.astype(np.float16))
        S = np.zeros((N, n_tok, L, 3), np.float32)
        S[..., 0] = 0.5
        S[..., 2] = 50
        for j, q in enumerate(seqs):
            S[j, :, :, 1] = 500 + rng.normal(0, 30, (n_tok, L))
            S[j, :, 3:, 1] = q["mu"]                            # from layer 3 on the lens's integer mean is mu
            S[j, :, 3:, 2] = q["sigma"]
        np.save(os.path.join(out, f"stationary_{k}_scalars.npy"), S)
        TOP = rng.integers(0, V, (N, n_tok, L, 20)).astype(np.int32)
        TOP[..., 0] = 33                                            # w33 always first
        np.save(os.path.join(out, f"stationary_{k}_top.npy"), TOP)
        np.save(os.path.join(out, f"stationary_{k}_lex.npy"), rng.normal(0, 1, (N, n_tok, L, 8)).astype(np.float16))
    out_int = np.zeros((N, T, 5), np.float32)
    for j, q in enumerate(seqs):
        out_int[j, :, 0] = q["mu"] + rng.normal(0, 3, T)
        out_int[j, :, 1] = q["sigma"] * 1.1
    np.savez(os.path.join(out, "behaviour_stationary.npz"), out_int=out_int, out_logp=np.zeros((N, n_tok - 1)),
             out_snap=np.zeros((N, 1, 1000)), snap_t=np.array([0]))
    with open(os.path.join(out, "meta_stationary.json"), "w") as f:
        json.dump({"T": T, "bos": 1, "sequences": seqs}, f)
    return out, disp, cls, L


def test_analysis_synthetic():
    import analyze

    with tempfile.TemporaryDirectory() as root:
        raw, disp, cls, L = synthetic_raw(root)
        ct = analyze.contrasts(raw, ["logit", "jlens"], L, disp, cls, has_switch=True)
        first = lambda c, d: ct[(ct.contrast == c) & (ct.direction == d) & (ct["class"] == "word") & (ct["rank"] == 1)]  # noqa: E731
        assert set(first("mu 700 vs 300 (late)", "up").token) == {"w20"}
        assert np.allclose(first("mu 700 vs 300 (late)", "up").delta, 4.0, atol=0.01)
        assert set(first("sigma 200 vs 10 (late)", "up").token) == {"w21"}
        assert np.allclose(first("sigma 200 vs 10 (late)", "up").delta, 3 * np.log(20), atol=0.02)
        assert set(first("late (t >= 512) vs early (8 <= t < 32)", "up").token) == {"w22"}
        assert set(first("switch: e in [1, 8) vs [-100, -10)", "up").token) == {"w23"}
        import json

        meta = json.load(open(os.path.join(raw, "meta_stationary.json")))
        lb = analyze.lens_belief(raw, ["logit", "jlens"], meta, L)
        assert (lb[lb.layer >= 3].r_mu > 0.999).all() and (lb[lb.layer < 3].r_mu.abs() < 0.9).all()
        assert np.allclose(lb[lb.layer >= 3].slope_mu, 1) and np.allclose(lb[lb.layer >= 3].slope_logsigma, 1)
        wf = analyze.word_frequency(raw, ["jlens"], meta, L, disp, cls)
        top = wf[(wf["class"] == "word") & (wf["rank"] == 1)]
        assert set(top.token) == {"w33"} and np.allclose(top.frac_in_top20, 1 + 19 / 60, atol=0.05)
        R = dict(np.load(os.path.join(raw, "behaviour_stationary.npz")))
        beh, v = analyze.behaviour(R, meta)
        assert v["go"] and abs(v["mean_vs_mu_slope"] - 1) < 0.02 and abs(v["logsd_vs_logsigma_slope"] - 1) < 1e-6
        tw = analyze.top_words(raw, ["jlens"], L, disp, cls)
        assert len(tw) and set(tw.layers) == {"early", "middle", "late"}


# ---------------------------------------------------------------------------------------------------------- tokenizer
def tokenizer_checks(model):
    from transformers import AutoTokenizer

    from common import RAW
    from jlens_open import MODELS

    tok = AutoTokenizer.from_pretrained(MODELS[model]["hf"])
    ints = [tok.encode(str(k), add_special_tokens=False) for k in range(1000)]
    assert all(len(t) == 1 for t in ints), "some integer is not one token"
    comma = tok.encode(",", add_special_tokens=False)
    assert len(comma) == 1
    bos = tok.bos_token_id

    def enc(text):
        ids = tok(text).input_ids
        return [bos] + ids if bos is not None and ids[0] != bos else ids

    for s in [x for x in sequences() if x["seq"] == 0][:3] + [x for x in sequences() if x["dataset"] == "switch"][:1]:
        want = ([bos] if bos is not None else []) + [i for x in s["numbers"] for i in (ints[x][0], comma[0])]
        ids = enc(s["text"])
        assert ids == want, f"{s['cond']}: the stream does not tokenize as [BOS] n , n , ..."
        for t in (0, 1, 7, 127, len(s["numbers"]) - 1):
            cut = ",".join(map(str, s["numbers"][: t + 1])) + ","
            assert enc(cut) == ids[: len(enc(cut))] and len(enc(cut)) == (1 if bos is not None else 0) + 2 * t + 2
    print(f"  {model}: 1000 integers one token each; streams and their cuts tokenize as expected")

    from jlens_open.concept import CARRIER_PREFIX, CARRIER_SUFFIX, calibration_concepts

    concepts = calibration_concepts(os.path.join(RAW, "clens_singleton_calibration_300.json"))
    n_one, n_fail, suffix = 0, 0, set()
    for c in concepts:
        enc_ = tok(CARRIER_PREFIX + c + CARRIER_SUFFIX, add_special_tokens=True, return_offsets_mapping=True)
        offs = enc_.offset_mapping
        c0, c1 = len(CARRIER_PREFIX), len(CARRIER_PREFIX) + len(c)
        cidx = [i for i, (a, b) in enumerate(offs) if b > a and b > c0 and a < c1]
        sidx = [i for i, (a, b) in enumerate(offs) if b > a and a >= c1]
        if not cidx or not sidx or any(offs[i][1] > c1 for i in cidx):
            n_fail += 1
            continue
        n_one += len(cidx) == 1
        suffix.add(tuple(enc_.input_ids[i] for i in sidx))
    print(f"  carrier: {len(concepts)} calibration concepts, {n_fail} fail to align, {n_one} are one token; "
          f"{len(suffix)} distinct suffix tokenization(s) of {len(next(iter(suffix)))} tokens")
    assert n_fail == 0 and n_one >= 50 and len(suffix) == 1  # the reader is fit on the one-token concepts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=None, help="a model key of jlens_open.MODELS (needs the HF cache)")
    args = parser.parse_args()
    tests = [test_stimuli_match_paper, test_sequences, test_transport, test_positive_pool, test_nucleus_and_dominance,
             test_analysis_synthetic]
    if args.tokenizer:
        tests.append(lambda: tokenizer_checks(args.tokenizer))
    failed = 0
    for t in tests:
        name = getattr(t, "__name__", "tokenizer")
        try:
            t()
            print(f"ok    {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
