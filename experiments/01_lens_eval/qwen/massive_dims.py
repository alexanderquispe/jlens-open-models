"""Is the J-lens's calibration set by the massive-activation dimensions of the final residual stream?

The final RMSNorm divides every coordinate by the RMS of the vector, so if a few coordinates of h_final are
huge and J_l h_l gets them wrong, every logit is rescaled. For six layers, on wikitext (positions 16..127):
  - the singular values of J_l and the share of its top output singular vector u_1 on the 2 dims with the
    largest mean |h_final|;
  - KL(p_final || p_lens) and whether the model's top token is in the lens's top 10, for the logit lens and
    the J-lens, with the transported vector as is, with its 2 massive coordinates set to their mean over
    positions in h_final (a fixed correction), and set to their true values at each position (oracle).

    python experiments/01_lens_eval/qwen/massive_dims.py --model qwen3-8b
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
CSV_OUT = os.path.join(HERE, f"results/massive_dims_{args.model}.csv")
CSV_SPECTRUM = os.path.join(HERE, f"results/massive_dims_{args.model}_spectrum.csv")

lm = LensModel(args.model, args.device)
L = lm.n_blocks
with open(os.path.join(HERE, "../../03_jspace/data/wikitext_sample.json")) as f:
    paragraphs = json.load(f)["paragraphs"]
text_ids = torch.stack([lm.encode(p)[:128] for p in paragraphs if len(lm.encode(p)) >= 128][:16])
hs, _ = lm.run(text_ids)
acts = [h[:, 16:].reshape(-1, lm.d) for h in hs]
del hs
final = acts[L - 1]
magnitude = final.abs().mean(0)
massive = magnitude.argsort(descending=True)[:2]
share = ((final[:, massive] ** 2).sum() / (final**2).sum()).item()
print(f"{args.model}: massive dims of h_final {massive.tolist()} (mean |h| {magnitude[massive].round().tolist()}, "
      f"median {magnitude.median():.2f}); share of |h_final|^2 on them {share:.3f}", flush=True)
final_logp = torch.cat([torch.log_softmax(lm.unembed(final[s : s + 256]), -1).cpu() for s in range(0, len(final), 256)])


def score(x):
    kl, in10 = 0.0, 0
    for s in range(0, len(x), 256):
        logp = torch.log_softmax(lm.unembed(x[s : s + 256]), -1).cpu()
        f = final_logp[s : s + 256]
        kl += (f.exp() * (f - logp)).sum().item()
        in10 += (logp.topk(10, -1).indices == f.argmax(-1)[:, None]).any(-1).sum().item()
    return kl / len(x), in10 / len(x)


rows, spectrum = [], []
for layer in [round(f * (L - 1)) for f in (0.25, 0.4, 0.5, 0.6, 0.7, 0.8)]:
    U, S, _ = torch.linalg.svd(lm.J[layer].cpu())
    energy = S**2 / (S**2).sum()
    spectrum.append({"layer": layer, "s_1": S[0].item(), "s_2": S[1].item(), "energy share of s_1": energy[0].item(),
                     "share of u_1 on the massive dims": (U[massive.cpu(), 0] ** 2).sum().item()})
    for lens, x in (("logit lens", acts[layer]), ("J-lens", lm.transport(acts[layer], layer))):
        for fix in ("as is", "massive dims = mean of h_final", "massive dims = true h_final (oracle)"):
            y = x.clone()
            if fix == "massive dims = mean of h_final":
                y[:, massive] = final[:, massive].mean(0)
            elif fix == "massive dims = true h_final (oracle)":
                y[:, massive] = final[:, massive]
            kl, in10 = score(y)
            rows.append({"layer": layer, "lens": lens, "massive dims": fix, "KL": kl, "final top-1 in lens top-10": in10})
    print(f"done: layer {layer}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(CSV_OUT, index=False)
spectrum = pd.DataFrame(spectrum).set_index("layer")
spectrum.to_csv(CSV_SPECTRUM)
pd.set_option("display.width", 200)
print(f"\n{args.model}: singular values of J\n{spectrum.round(2).to_string()}")
for metric in ("KL", "final top-1 in lens top-10"):
    print(f"\n{args.model}: {metric}")
    print(df.pivot_table(index=["lens", "massive dims"], columns="layer", values=metric).round(2).to_string())
print(f"\nSaved: {CSV_OUT}\n       {CSV_SPECTRUM}")
