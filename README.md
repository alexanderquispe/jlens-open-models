# Jacobian lens on open-weight models

Experiments with the Jacobian lens (J-lens) of Gurnee et al. (2026) on open-weight models, using the lenses
Neuronpedia fitted with Anthropic's reference code. The J-lens reads the residual stream after block *l*
through the average Jacobian to the last block,

$$\mathrm{lens}_l(h) = W_U \,\mathrm{norm}(J_l h), \qquad J_l = \mathbb{E}\left[\partial h_\mathrm{final} / \partial h_l\right],$$

and becomes the logit lens when *J_l = I*.

| Experiment | Question | Main result (Llama-3.1-8B) | Other models (Llama-3.1-8B-Instruct, Qwen3-8B, OLMo-3-7B) |
|---|---|---|---|
| [01 · Lens quality](experiments/01_lens_eval) | Does the J-lens surface intermediate concepts the logit lens misses? (replication) | pass@10 0.414 vs 0.277 on six prompt sets; the gap is in layers 15–23 | Holds for Instruct (0.421 vs 0.318) and OLMo (0.354 vs 0.292); reversed on Qwen (0.246 vs 0.291), whose J-lens reads many intermediates in Chinese (with Chinese targets: 0.304 vs 0.314) |
| [02 · Weekday circle](experiments/02_weekday_circle) | What does the J-lens see of the weekday circle of Engels et al. (2025)? | The circle moves the answer at the day token in layers 0–17, where the day lens directions do nothing; from layer 18 the answer is at the last token, where the lens reads it | Same in all three: circle 0.17–0.42, day lens span ≤ 0.02 |
| [03 · J-space](experiments/03_jspace) | How much of the residual stream do sparse combinations of J-lens vectors explain, and what breaks when they are ablated? | 7–11% of the variance (k = 10). Ablating it in layers 4–19 drops two-hop recall to 0.60–0.73; random directions with the same covariance remove the same variance and raise the loss as much, but recall stays at 0.90–0.98 | Instruct: same (J-lens 0.79–0.90 vs control 0.96–1.00 in layers 4–15). Qwen: weaker, the control also hurts recall (layers 12–23: 0.41–0.56 vs 0.75–0.88). OLMo: same as Llama, smaller (layers 4–23: 0.67–0.94 vs 0.95–0.99) |
| [04 · Multihop swap](experiments/04_multihop_swap) | Is the intermediate entity the J-lens reads (*Lima → Peru → Spanish*) causally used? | Swapping the country's J-lens vector moves the answer in 59–80% of trials at the capital token (logit lens ≤ 14%, random 0%); the country enters the J-lens span at layer 5 | J-lens swap 0.76–0.90 in all three. Qwen's logit lens is nearly as causal (0.50–0.73); in OLMo the countries' natural difference barely lies in their J-lens span (≤ 0.10) |
| [05 · Cyclic irreps](experiments/05_cyclic_irreps) | Which Fourier frequencies of the month representation carry "two months from January"? | The k = 1 circle moves 13–15% of answers, k = {1, 2} 40%, all frequencies 94–98%: the frequencies act jointly; the month-name lens span does nothing | Same in all three (k = 1 ≤ 0.24, all k 0.59–0.93, lens span ≤ 0.02). For the 7 days (Llama) the circle does more: 0.39–0.46 of 0.61–0.72 |
| [06 · False context](experiments/06_false_context) | With false few-shot labels, is the truth in the workspace before the model says the falsehood? | A race: on prompts answered falsely the truth is in the workspace when it forms before the override (continent 0.87 at layer 9 vs baseline 0.44), not when it forms after (language, animal) | Same race in all three. Qwen overrides later and shows the truth in 82–95% for continent, sentiment and language; OLMo shows a brief continent window at layer 11 (0.96 vs 0.50) |

## Layout

```
jlens_open/          shared code: models and lenses (models.py), Fourier tools (fourier.py), downloads (download.py)
experiments/NN_*/    one folder per experiment: script, README, data, results, figures (see experiments/README.md)
hpc/                 Slurm scripts and instructions for the Caltech cluster
run_all.sh           every experiment and configuration for one model
```

## Quick start

```bash
uv sync                    # creates .venv with the pinned versions (or: python -m venv .venv && pip install -e .)
source .venv/bin/activate
huggingface-cli login      # Llama-3.1-8B is gated: accept Meta's license on its HF page first

bash run_all.sh gpt2-small     # ~12 min on an M5 Max
bash run_all.sh llama3.1-8b    # ~4.5 h on an M5 Max (LAYERS=::2 patches every second layer in 04 and 05)
```

The device is picked automatically (CUDA, then MPS, then CPU) or set with `--device`. For the cluster, see
[hpc/README.md](hpc/README.md).

## Models

Registered in `jlens_open/models.py` (`MODELS`): `gpt2-small`, `llama3.1-8b`, `llama3.1-8b-it`, `qwen3-8b`,
`olmo-3-7b`, `qwen3-32b` and `olmo-3-32b`, plus two with lenses we fitted ourselves (see below): `gpt2-small-fit`
and `mistral-7b`. The last four need `transformers` 5, kept for now in a second environment
next to the pinned one:

```bash
uv venv --python 3.11 .venv-tf5
uv pip install --python .venv-tf5/bin/python torch==2.8.0 transformers==5.11.0 numpy==1.26.4 pandas==2.2.2 matplotlib==3.9.2
PYTHONPATH=. .venv-tf5/bin/python experiments/01_lens_eval/lens_eval.py --model qwen3-8b
```

A model is added with one
entry: its Hugging Face id, the path of its lens in
[`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens), and the dtype the lens was fit
in. `LensModel` handles GPT-2-style and Llama-style decoders.

As of 2026-10-25, Neuronpedia has J-lenses (fit on wikitext) for GPT-2 small, Pythia-70M, Llama 3.1 8B
(base and Instruct), Llama 3.3 70B Instruct, Gemma 2 (2B, 9B, 27B), Gemma 3 (270M to 27B), Gemma 4 (E2B, E4B,
31B), Qwen2.5 7B Instruct, Qwen3 (1.7B to 32B), Qwen3.5 (0.8B to 27B), Qwen3.6 27B, OLMo 3 (7B, 32B) and
gpt-oss-20b, and a pile-10k lens for DeepSeek-V4-Flash. Some models also have J++ lenses, including
Qwen3.5-122B-A10B. Most of the newer models need a newer `transformers` than the one pinned here (see open tasks).

### Fitting a lens

For a model Neuronpedia does not cover, `jlens_open/fit.py` fits the J-lens with the estimator of the reference
code and the recipe of Neuronpedia's lenses: WikiText-103 prompts cut to 128 tokens, Jacobians averaged over
positions 16 and later, and a stop once one more prompt moves the mean by less than 0.2% (or at 1000 prompts). The
model's entry points its lens to `$JLENS_SHARED/lenses/...`, where the fit writes it, in Neuronpedia's format:

```bash
sbatch --gres=gpu:nvidia_h200:1 --mem=192G hpc/gpu.sbatch env PYTHON=.venv-tf5/bin/python PYTHONPATH=. \
  python -m jlens_open.fit mistral-7b --dim-batch 128 --max-prompts 500
```

Each prompt takes one forward pass and d / dim_batch backward passes; a checkpoint every 25 prompts lets an
interrupted fit resume.

- **GPT-2 small, as a check** (`gpt2-small-fit`): 359 prompts in fp32 until the 0.2% stop, a few minutes. Per
  layer, cosine 0.996–0.9999 with Neuronpedia's lens; the remaining difference (about 10%) is close to what
  sampling different prompts alone gives (about 7%). Experiment 01: pass@10 0.137 with our lens, 0.139 with theirs.
- **Mistral-7B v0.1** (`mistral-7b`, not on Neuronpedia): 18 s per prompt on one H200 (about $5 for 500 prompts).
  After 500 prompts one more prompt still moved the mean by 0.7%, so we stopped there rather than run to 1000
  (Neuronpedia's Llama-3.1-8B lens has 539 prompts, its Llama-3.3-70B lens 125). Experiment 01: pass@10 0.470 vs
  0.444 for the logit lens (multihop 0.780 vs 0.629). Swapping the country's J-lens vector at the capital
  token (04) moves the answer in 64% of trials (logit lens 58%, random 1%). All six experiments: 37 min on one H200.

## Working together

- `main` always runs. Work on a branch and open a pull request; someone else reviews it before merging.
- Each experiment lives in its own folder (see [experiments/README.md](experiments/README.md)). Changes to
  `jlens_open/` affect everyone: after one, run `bash run_all.sh gpt2-small` (12 min) and check
  `git status experiments/`. On the same machine, rerunning GPT-2 gives byte-identical CSVs and figures, so any
  changed file is a changed result and the PR should say why. On a GPU, numbers can differ in the last digits
  (we saw 1e-16 on Llama); anything larger is a real change.
- Never commit model weights, lenses, activations or tokens. Large files go to `$JLENS_SHARED` on the cluster.
- The environment is pinned in `pyproject.toml` and `uv.lock`. Add dependencies with `uv add`.

## Open tasks

1. **Newer `transformers`.** Qwen3, Qwen3.5/3.6, Gemma 3/4 and OLMo 3 need a newer version than 4.43.3 (Neuronpedia
   fit its lenses with 5.x). Checked so far with 5.11 (in `.venv-tf5`): experiment 01 gives byte-identical CSVs
   for GPT-2 small and Llama-3.1-8B. Moving the pin to 5.x would leave one environment.
2. **Several GPUs.** Load models with `device_map="auto"` and keep each J on its block's device, for 70B and
   larger (Llama 3.3 70B, Qwen3.5-122B-A10B).
3. **J++ lenses** (Ayonrinde and Lindsey), available for several models; their file format is not handled yet.
4. ~~**Fitting lenses**~~ Done: `jlens_open/fit.py` (see Models).

## Notes

- The lenses were fit with BOS forced and without the first 16 positions; every prompt here starts with BOS.
- On GPT-2, `LensModel` gives logits identical to Anthropic's `jlens` package.
- Llama runs in bf16, the dtype its lens was fit in. Transformers 4.43.3 applies Llama 3.1's RoPE correctly
  (θ = 500000 with llama3 scaling). TransformerLens 2.3 does not, so it is not used here.
- Results in `experiments/*/results` were produced on one NVIDIA L40S (Caltech cluster), with every layer patched
  in 04 and 05. The numbers quoted in the READMEs come from earlier runs on an Apple M5 Max (MPS) and differ
  slightly from these files.

## References

- W. Gurnee et al. (2026). *Verbalizable Representations Form a Global Workspace in Language Models.*
  [transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace).
  Reference code: [github.com/anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens).
- J. Engels, E. J. Michaud, I. Liao, W. Gurnee, M. Tegmark (2025). *Not All Language Model Features Are
  One-Dimensionally Linear.* ICLR 2025, [arXiv:2405.14860](https://arxiv.org/abs/2405.14860).
  Code: [github.com/JoshEngels/MultiDimensionalFeatures](https://github.com/JoshEngels/MultiDimensionalFeatures).
- Pre-fitted lenses: [huggingface.co/neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens).

## License

Code: MIT. The evaluation sets in `experiments/01_lens_eval/data` come from `anthropics/jacobian-lens`
(Apache-2.0, license included there); the lenses from `neuronpedia/jacobian-lens` (Apache-2.0). Llama 3.1 is
under Meta's Llama 3.1 Community License.
