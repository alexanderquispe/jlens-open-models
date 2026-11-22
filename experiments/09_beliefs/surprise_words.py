"""CPU. Usage: python experiments/09_beliefs/surprise_words.py (after run_model.py for both models).

Do surprise / change words move with surprise in experiment 09's raw outputs?
A. switch: z(group) - z(control) at the first new number, vs its pre-switch fluctuation (per layer).
B. stationary: correlation, across positions, between the surprisal of number t and z(group) - z(control).
C. switch: rank of single surprise/change words in the full vocabulary, before vs at the switch."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import GROUPS, E_BINS, RAW  # noqa: E402
gi = {g: i for i, g in enumerate(GROUPS)}
SHOW = ["surprise", "change", "stats", "high", "low", "wide"]
WORDS = [" surprise", " unexpected", " sudden", " suddenly", " strange", " wait", " change", " changed", " jump",
         " shift", " switch", " new", " different", " anomaly", " outlier", " error", " wrong", " but", " however"]

for model in ("llama3.1-8b", "olmo-3-32b"):
    R = os.path.join(RAW, model)
    info = json.load(open(os.path.join(R, "run_info.json")))
    lenses = info["lenses"]
    print(f"\n==================== {model}  ({info['n_blocks']} layers)")

    # ---------- A. switch: per-sequence change at the switch, specific to the group (minus the mean change of
    # the other non-control groups), t over the 10 sequences; best layer by t.
    ms = json.load(open(os.path.join(R, "meta_switch.json")))
    bos, T = ms["bos"], ms["T"]
    t = np.arange(T); e = t - 1000
    num, com = bos + 2 * t, bos + 2 * t + 1
    pre = (e >= -100) & (e < -10)
    others = [gi[g] for g in GROUPS if g != "control"]
    print("\nA. switch: change in z(group) vs e in [-100,-10), minus the mean change of the other groups;"
          " delta in z units, t over 10 sequences, best layer by |t|")
    for k in lenses:
        X = np.asarray(np.load(os.path.join(R, f"switch_{k}_lex.npy"), mmap_mode="r"), dtype=np.float32)
        out = []
        for g in ("surprise", "change", "high", "stats"):
            j = gi[g]
            oth = [o for o in others if o != j]
            res = []
            for name, P, sel in (("num e=0", num, e == 0), ("com e=0", com, e == 0), ("com e=1..3", com, (e >= 1) & (e <= 3))):
                d = X[:, P[sel]].mean(1) - X[:, P[pre]].mean(1)  # (N, L, G)
                spec = d[..., j] - d[..., oth].mean(-1)  # (N, L)
                m, se = spec.mean(0), spec.std(0, ddof=1) / np.sqrt(len(spec)) + 1e-9
                tt = m / se; l = int(np.argmax(np.abs(tt)))
                res.append(f"{name}: {m[l]:+.2f} (t {tt[l]:+5.1f}, L{l:02d})")
            out.append(f"    {g:9s} " + "   ".join(res))
        print(f"  {k}:"); print("\n".join(out))

    # ---------- B. stationary: surprisal of each number vs group readout
    mst = json.load(open(os.path.join(R, "meta_stationary.json")))
    T = mst["T"]; t = np.arange(32, T)
    num, com = mst["bos"] + 2 * t, mst["bos"] + 2 * t + 1
    B = np.load(os.path.join(R, "behaviour_stationary.npz"))
    s = -B["out_logp"][:, 2 * t]  # surprisal of number t (token 2t+1 predicted at 2t), (N, len(t))
    s = (s - s.mean()) / s.std()
    print("\nB. stationary (t >= 32): corr(surprisal of number t, z(group) - z(control)), best layer by |r|,"
          " at the number itself and at the comma after it")
    for k in lenses:
        X = np.asarray(np.load(os.path.join(R, f"stationary_{k}_lex.npy"), mmap_mode="r"), dtype=np.float32)
        D = X - X[..., gi["control"]][..., None]
        out = []
        for g in SHOW:
            j = gi[g]
            res = []
            for pos, name in ((num, "num"), (com, "com")):
                y = D[:, pos][..., j]  # (N, t, L)
                y = (y - y.mean((0, 1))) / (y.std((0, 1)) + 1e-6)
                r = (y * s[..., None]).mean((0, 1))
                l = int(np.argmax(np.abs(r))); res.append(f"{name} r={r[l]:+.3f} (L{l:02d})")
            out.append(f"    {g:9s} " + "   ".join(res))
        print(f"  {k}:"); print("\n".join(out))

    # ---------- C. ranks of single words in the switch cells
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(info["hf"])
    ids = {w: tok.encode(w, add_special_tokens=False) for w in WORDS}
    ids = {w: v[0] for w, v in ids.items() if len(v) == 1}
    b_pre, b0, b1 = E_BINS.index(-100), E_BINS.index(0), E_BINS.index(1)
    print(f"\nC. switch cells: best (lowest) rank over layers of each word among {len(tok)} tokens:"
          " e in [-100,-10) -> e = 0 -> e = 1")
    for k in lenses:
        C = np.load(os.path.join(R, f"switch_{k}_cells.npy"), mmap_mode="r")
        L = C.shape[0]
        rk = {w: [[], [], []] for w in ids}
        for l in range(L):
            for bi, b in enumerate((b_pre, b0, b1)):
                v = np.asarray(C[l, b], dtype=np.float32)
                for w, i in ids.items():
                    rk[w][bi].append(int((v > v[i]).sum()) + 1)
        line = []
        for w in ids:
            a = [min(x) for x in rk[w]]
            line.append(f"{w.strip()} {a[0]}->{a[1]}->{a[2]}")
        print(f"  {k}: " + "; ".join(line))
