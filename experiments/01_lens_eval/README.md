# 01 · Lens quality: J-lens vs logit lens

**Question.** Does the Jacobian lens surface intermediate concepts that the logit lens misses? This replicates
the lens evaluation of Gurnee et al. (2026) with Neuronpedia's pre-fitted lenses.

```bash
python experiments/01_lens_eval/lens_eval.py --model llama3.1-8b   # ~1 min on an M5 Max after loading
```

Outputs: `results/<model>.csv` (pass@1 and pass@10 per prompt set), `results/<model>_by_layer.csv`
(fraction of intermediates in the top 10 at each layer), `figures/<model>.{pdf,png}`.

## Method

Six prompt sets from the reference code (`data/`, copied from `anthropics/jacobian-lens` at commit `581d398`,
Apache-2.0): multihop, multilingual, order-ops, poetry, association, typo. Each item names intermediate concepts
that never appear in the prompt, e.g. "Brazil" in *"The ocean on the coast of the country where Carnival is most
famously celebrated is the"*.

For each intermediate, the rank in the lens readout over the full vocabulary is taken at its best layer; pass@k
is the fraction of intermediates with rank ≤ k, averaged over items. The readout is at the last prompt token,
except for poetry, which reads at the newline ending line 1 (as in the reference code's
`data/evaluations/README.md`). A word's rank is the best over the first tokens of "word" and " word"; for
order-ops, over single-token synonyms (digits and number words, symbols and operation words; these lists are
ours).

## Results

**Llama-3.1-8B** (`results/llama3.1-8b.csv`):

| Prompt set | pass@1 J-lens | pass@1 logit lens | pass@10 J-lens | pass@10 logit lens |
|---|---|---|---|---|
| multihop | 0.301 | 0.237 | 0.613 | 0.505 |
| multilingual | 0.271 | 0.273 | 0.535 | 0.432 |
| order-ops | 0.127 | 0.009 | 0.427 | 0.145 |
| poetry | 0.010 | 0.031 | 0.061 | 0.041 |
| association | 0.039 | 0.039 | 0.137 | 0.098 |
| typo | 0.271 | 0.135 | 0.708 | 0.438 |
| **macro** | **0.170** | **0.121** | **0.414** | **0.277** |

The J-lens advantage is in layers 15–23 (peak at layer 21: 0.25 vs 0.19 of intermediates in the top 10).

![Lens eval by layer, Llama-3.1-8B](figures/llama3.1-8b.png)

**GPT-2 small:** both lenses are near the floor (macro pass@10 0.139 J-lens vs 0.171 logit lens).

## Other models

pass@10, J-lens / logit lens (`results/<model>.csv`; Qwen3-8B and OLMo-3-7B with transformers 5.11):

| Prompt set | Llama-3.1-8B | Llama-3.1-8B-Instruct | Qwen3-8B | OLMo-3-7B |
|---|---|---|---|---|
| multihop | 0.613 / 0.505 | 0.570 / 0.586 | 0.511 / 0.516 | 0.532 / 0.514 |
| multilingual | 0.535 / 0.432 | 0.523 / 0.425 | 0.432 / 0.437 | 0.465 / 0.467 |
| order-ops | 0.427 / 0.145 | 0.345 / 0.118 | 0.327 / 0.191 | 0.382 / 0.200 |
| poetry | 0.061 / 0.041 | 0.122 / 0.163 | 0.000 / 0.000 | 0.031 / 0.041 |
| association | 0.137 / 0.098 | 0.196 / 0.127 | 0.000 / 0.020 | 0.039 / 0.059 |
| typo | 0.708 / 0.438 | 0.771 / 0.490 | 0.208 / 0.583 | 0.677 / 0.469 |
| **macro** | **0.414 / 0.277** | **0.421 / 0.318** | **0.246 / 0.291** | **0.354 / 0.292** |

- The J-lens advantage replicates on Llama-Instruct and OLMo, carried by `order-ops` and `typo`. On multihop it
  is at most 0.02 in every model except base Llama (0.11).
- **Qwen3-8B is the exception**: the J-lens loses on macro, mostly on `typo` (0.208 vs 0.583). Most of the gap
  comes from language: Qwen's J-lens reads many intermediates in Chinese (next section).

## Why the J-lens loses on Qwen3-8B

The lens matches the model and our code applies it correctly. Qwen's J-lens reads many intermediates in Chinese,
which English targets miss; on `typo` it reads "misspelling" rather than the corrected word; and its KL is
inflated by two massive dimensions of the residual stream. Scripts and outputs in `qwen/` (transformers 5.11):

```bash
PYTHONPATH=. .venv-tf5/bin/python experiments/01_lens_eval/qwen/chinese_targets.py --model qwen3-8b  # ~1 min
# also lens_fidelity.py, jacobian_check.py, massive_dims.py, top_tokens.py (--set typo); ~7 min in all
```

**The stored J is the model's Jacobian** (`jacobian_check.py`). We recompute rows of J with autograd, using the
reference estimator (cotangent at every position from 16 on) on six wikitext paragraphs and 16 random directions.
Cosine with the stored J: 0.66, 0.83, 0.94 at layers 9, 18, 27 of Qwen; 0.76, 0.89, 0.97 at layers 8, 16, 24 of
Llama. Six paragraphs give a noisy estimate, so early layers are lower in both. Using the J of layer l−1 or l+1
does not help (`lens_fidelity.py`): on wikitext, all three variants have KL 14–22 to the model's output in
layers 4–21, against 9–12.5 for the logit lens.

**Qwen's J-lens reads intermediates in Chinese** (`chinese_targets.py`). We rescored association, multihop,
multilingual and typo with Chinese translations of the intermediates (`qwen/zh_intermediates.json`, 363 words;
ours, not checked by a native speaker). A Chinese word counts when it is one token, or when its first token has
at least two characters (so 十 alone does not count for 十二月). pass@10, J-lens / logit lens:

| Targets | Qwen3-8B | Llama-3.1-8B | OLMo-3-7B |
|---|---|---|---|
| English (table above) | 0.246 / 0.291 | 0.414 / 0.277 | 0.354 / 0.292 |
| English or Chinese | **0.304 / 0.314** | 0.414 / 0.279 | 0.355 / 0.294 |
| Chinese only (4 sets) | 0.281 / 0.249 | 0.032 / 0.127 | 0.010 / 0.077 |

- On Qwen, multihop turns in the J-lens's favour (0.565 vs 0.516; Chinese only 0.392 vs 0.230). Intermediates
  the J-lens finds only in Chinese include 火星 for Mars (*mars-color*, rank 8 at layer 24), 中国 for China
  (*super-populous-capital*, layer 10) and 十月 for October (*holiday-halloween-monthnum*). In *amazon-language*,
  巴西 (Brazil) is its rank-2 token at layer 15.
- Llama's and OLMo's J-lenses almost never rank the Chinese words, so the gain on Qwen does not come from simply
  adding targets.
- The macro gap shrinks from −0.045 to −0.010. What remains is `typo` (0.479 vs 0.646), along with association and
  multilingual (0.000 vs 0.039, 0.453 vs 0.491).

**On `typo`, Qwen's J-lens reads "misspelling"** (`results/top_tokens_qwen3-8b_typo.txt`). At the misspelled
word, its top tokens in layers 16–23 are other common misspellings (' wich', ' seperate', ' alot', ' adress',
' teh') rather than the corrected word.

**Two massive dimensions inflate the KL** (`massive_dims.py`).
- Two coordinates of Qwen's final residual stream (2276 and 1838, mean |h| 782 and 664, against a median of 9)
  hold 58% of its squared norm; in Llama the top two hold 4%.
- Qwen's J has one dominant singular direction, with 41–69% of the squared Frobenius norm in layers 9–21 (Llama:
  at most 15%). Its output vector lies 81–84% on those two coordinates.
- Setting the two coordinates of J_l h_l to their mean in h_final halves the J-lens KL. At layer 18 it drops from
  17.9 to 9.9, close to the logit lens's 10.2.
- The same correction leaves the top 10 unchanged: the model's next token is in it 2% of the time either way.

So the final RMSNorm explains the poor calibration, but not the ranking. On wikitext, Qwen's J-lens has the
model's next token in its top 10 at most 5% of the time up to layer 21 (logit lens: up to 16%). Llama's J-lens
beats the logit lens there in layers 5–19.
