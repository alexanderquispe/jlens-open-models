"""Models and their Jacobian lenses.

Jacobian lens (Gurnee et al. 2026, "Verbalizable Representations Form a Global Workspace in
Language Models"; reference code github.com/anthropics/jacobian-lens):

    lens_l(h) = unembed(J_l h),   J_l = E[d h_final / d h_l]

h_l is the output of block l and h_final the output of the last block, before the final norm.
The lenses are the ones Neuronpedia fitted with the reference code (neuronpedia/jacobian-lens on
the HF Hub). As in the reference code (`force_bos=True`), every prompt starts with BOS when the
tokenizer has one (Qwen's has none).
"""
import json

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

LENS_REPO = "neuronpedia/jacobian-lens"
# Keys follow Neuronpedia's folder names; "lens" is the file inside LENS_REPO
MODELS = {
    "gpt2-small": {
        "hf": "gpt2",
        "lens": "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt",
        "dtype": torch.float32,
    },
    # Lens fitted in bf16, so the model runs in bf16 too (16 GB instead of 32 GB)
    "llama3.1-8b": {
        "hf": "meta-llama/Llama-3.1-8B",
        "lens": "llama3.1-8b/jlens/Salesforce-wikitext/Llama-3.1-8B_jacobian_lens.pt",
        "dtype": torch.bfloat16,
    },
    "llama3.1-8b-it": {
        "hf": "meta-llama/Llama-3.1-8B-Instruct",
        "lens": "llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt",
        "dtype": torch.bfloat16,
    },
    # The rest need transformers >= 5 (results here: 5.11)
    "qwen3-8b": {
        "hf": "Qwen/Qwen3-8B",
        "lens": "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
        "dtype": torch.bfloat16,
    },
    "olmo-3-7b": {
        "hf": "allenai/Olmo-3-1025-7B",
        "lens": "olmo-3-1025-7b/jlens/Salesforce-wikitext/Olmo-3-1025-7B_jacobian_lens.pt",
        "dtype": torch.bfloat16,
    },
    # 65 GB of weights in bf16: one H200 (141 GB); an 80 GB H100 is too tight with the lens and activations
    "qwen3-32b": {
        "hf": "Qwen/Qwen3-32B",
        "lens": "qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt",
        "dtype": torch.bfloat16,
    },
    "olmo-3-32b": {
        "hf": "allenai/Olmo-3-1125-32B",
        "lens": "olmo-3-1125-32b/jlens/Salesforce-wikitext/Olmo-3-1125-32B_jacobian_lens.pt",
        "dtype": torch.bfloat16,
    },
    # Also needs transformers 5: its tokenizer.json is too new for the tokenizers that 4.43 pins. 141 GB in bf16,
    # plus 21 GB of lens in float32: two H200s (LensModel splits the blocks when it sees several GPUs)
    "llama3.3-70b-it": {
        "hf": "meta-llama/Llama-3.3-70B-Instruct",
        "lens": "llama3.3-70b-it/jlens/Salesforce-wikitext/Llama-3.3-70B-Instruct_jacobian_lens.pt",
        "dtype": torch.bfloat16,
    },
    "qwen3-4b": {
        "hf": "Qwen/Qwen3-4B",
        "lens": "qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt",
        "dtype": torch.bfloat16,
    },
    # J++ lenses (Ayonrinde and Lindsey 2026): same file format and readout as the J-lens, unembed(h @ J_l^T) on the
    # block output, fit with Jacobian filtering. A "-jpp" model runs every experiment with the J++ lens in place
    # of the J-lens (the columns labelled "J-lens" then hold the J++ lens)
    "qwen3-4b-jpp": {
        "hf": "Qwen/Qwen3-4B",
        "lens": "qwen3-4b/jpp/Salesforce-wikitext/Qwen3-4B_jpp_lens.pt",
        "dtype": torch.bfloat16,
    },
    "olmo-3-32b-jpp": {
        "hf": "allenai/Olmo-3-1125-32B",
        "lens": "olmo-3-1125-32b/jpp/Salesforce-wikitext/Olmo-3-1125-32B_jpp_lens.pt",
        "dtype": torch.bfloat16,
    },
    # Newer architectures (one H200 each): Gemma 4 is multimodal (text decoder in model.language_model) with a
    # final logit softcap; Qwen3.6 mixes linear and full attention, and its final norm's gain is 1 + weight
    "gemma-4-31b": {
        "hf": "google/gemma-4-31B",
        "lens": "gemma-4-31b/jlens/Salesforce-wikitext/gemma-4-31B_jacobian_lens.pt",
        "dtype": torch.bfloat16,
    },
    "gemma-4-31b-jpp": {
        "hf": "google/gemma-4-31B",
        "lens": "gemma-4-31b/jpp/Salesforce-wikitext/gemma-4-31B_jpp_lens.pt",
        "dtype": torch.bfloat16,
    },
    "qwen3.6-27b": {
        "hf": "Qwen/Qwen3.6-27B",
        "lens": "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
        "dtype": torch.bfloat16,
    },
    "qwen3.6-27b-jpp": {
        "hf": "Qwen/Qwen3.6-27B",
        "lens": "qwen3.6-27b/jpp/Salesforce-wikitext/Qwen3.6-27B_jpp_lens.pt",
        "dtype": torch.bfloat16,
    },
    # Gemma 2 (final logit softcap) and Gemma 3 (multimodal, like Gemma 4); both have the 1 + weight norm gain.
    # Gemma 2's weights are stored in float32 (109 GB to download) and loaded in bf16
    "gemma-2-27b": {
        "hf": "google/gemma-2-27b",
        "lens": "gemma-2-27b/jlens/Salesforce-wikitext/gemma-2-27b_jacobian_lens.pt",
        "dtype": torch.bfloat16,
    },
    "gemma-3-27b": {
        "hf": "google/gemma-3-27b-pt",
        "lens": "gemma-3-27b/jlens/Salesforce-wikitext/gemma-3-27b-pt_jacobian_lens.pt",
        "dtype": torch.bfloat16,
    },
    # Mixture of experts (256 experts, 10B active), with only a J++ lens. 250 GB in bf16: three H200s
    "qwen3.5-122b-a10b-jpp": {
        "hf": "Qwen/Qwen3.5-122B-A10B",
        "lens": "qwen3.5-122b-a10b/jpp/Salesforce-wikitext/Qwen3.5-122B-A10B_jpp_lens.pt",
        "dtype": torch.bfloat16,
    },
}


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def max_memory(hf_id, n_gpus):
    """Weights per GPU for device_map="auto". The first GPU takes half a share, since it also holds the lenses
    and the activations the experiments keep; the last takes what is left."""
    with open(hf_hub_download(hf_id, "model.safetensors.index.json")) as f:
        share = json.load(f)["metadata"]["total_size"] / (n_gpus - 0.5)
    mem = {i: int(share * (0.5 if i == 0 else 1)) for i in range(n_gpus - 1)}
    mem[n_gpus - 1] = int(0.9 * torch.cuda.get_device_properties(n_gpus - 1).total_memory)
    return mem


@torch.no_grad()
def norm_gain(norm):
    """The gain g of the final norm, norm(x) = x / rms(x) * g: its weight, or 1 + weight for the zero-centred
    RMSNorm of Qwen3.5 (and Gemma 2/3). Told apart by normalizing a vector of ones, whose RMS is 1."""
    w = norm.weight.float()
    if "RMSNorm" not in type(norm).__name__:
        return w
    probe = norm(torch.ones_like(norm.weight)).float()
    return w if (probe - w).abs().max() < (probe - 1 - w).abs().max() else 1 + w


class LensModel:
    def __init__(self, name, device=None):
        spec = MODELS[name]
        self.name, self.dtype = name, spec["dtype"]
        self.device = device or default_device()
        self.tok = AutoTokenizer.from_pretrained(spec["hf"])
        n_gpus = torch.cuda.device_count() if self.device == "cuda" else 1
        if n_gpus > 1:
            # Blocks split across the GPUs; activations, lenses and logits stay on the first (self.device)
            self.model = AutoModelForCausalLM.from_pretrained(
                spec["hf"], torch_dtype=self.dtype, device_map="auto", max_memory=max_memory(spec["hf"], n_gpus)
            ).eval()
            placed = set(self.model.hf_device_map.values())
            assert placed <= set(range(n_gpus)), f"weights offloaded off the GPUs: {placed}"
            self.model.register_forward_hook(lambda module, inputs, out: setattr(out, "logits", out.logits.to(self.device)))
        else:
            self.model = AutoModelForCausalLM.from_pretrained(spec["hf"], torch_dtype=self.dtype).to(self.device).eval()
        # GPT-2 style (transformer.h, ln_f) or Llama style (model.layers, norm); multimodal checkpoints (Gemma 4)
        # keep the Llama-style text decoder in model.language_model
        decoder = self.model.transformer if hasattr(self.model, "transformer") else self.model.model
        decoder = getattr(decoder, "language_model", decoder)
        self.blocks = decoder.h if hasattr(decoder, "h") else decoder.layers
        self.norm = decoder.ln_f if hasattr(decoder, "ln_f") else decoder.norm
        self.n_blocks = len(self.blocks)
        self.d = self.norm.weight.shape[0]
        self.gain = norm_gain(self.norm).to(self.device)
        config = getattr(self.model.config, "text_config", self.model.config)
        self.softcap = getattr(config, "final_logit_softcapping", None)  # Gemma: logits = c tanh(logits / c)
        if n_gpus > 1:
            print(f"{name}: blocks per GPU", [sum(next(b.parameters()).device.index == i for b in self.blocks) for i in range(n_gpus)])
        lens = torch.load(hf_hub_download(LENS_REPO, spec["lens"]), weights_only=True)
        # Blocks 0..n-2 have a J; the last block's output is h_final itself (J = I)
        self.J = {layer: m.float().to(self.device) for layer, m in lens["J"].items()}
        self.n_lens_prompts = lens["n_prompts"]

    def encode(self, text):
        ids = self.tok(text).input_ids
        bos = self.tok.bos_token_id
        if bos is not None and ids[0] != bos:
            ids = [bos] + ids
        return torch.tensor(ids)

    def first_token(self, word):
        """First token id of `word` and of `" " + word` (the two ways a word can start)."""
        return {self.tok.encode(w, add_special_tokens=False)[0] for w in (word, " " + word)}

    @torch.no_grad()
    def run(self, input_ids, deltas=None, positions=None, store=True):
        """Block outputs [n_blocks] of (batch, seq, d) in float32, and the last-position logits.

        deltas: optional {layer: (batch, d) tensor} added after that block at `positions`
        (one index per row; default the last position). store=False skips the block outputs
        (returns None for them), which makes patched runs faster.
        """
        outputs = {}
        rows = torch.arange(len(input_ids))
        positions = torch.full((len(input_ids),), input_ids.shape[1] - 1) if positions is None else torch.as_tensor(positions)

        def hook(layer):
            def f(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if deltas is not None and layer in deltas:
                    hidden = hidden.clone()
                    hidden[rows, positions] += deltas[layer].to(hidden.device, hidden.dtype)
                if store:
                    outputs[layer] = hidden.to(self.device).float()
                if isinstance(output, tuple):
                    return (hidden,) + tuple(output[1:])
                return hidden

            return f

        handles = [b.register_forward_hook(hook(i)) for i, b in enumerate(self.blocks)]
        try:
            logits = self.model(input_ids.to(self.device), use_cache=False).logits[:, -1].float()
        finally:
            for h in handles:
                h.remove()
        return ([outputs[i] for i in range(self.n_blocks)] if store else None), logits

    @torch.no_grad()
    def unembed(self, x):
        logits = self.model.lm_head(self.norm(x.to(self.norm.weight.device, self.dtype)))
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits.float().to(self.device)

    def transport(self, x, layer, use_jacobian=True):
        """J_l x (the lens); the identity for the logit lens or for the last block."""
        if not use_jacobian or layer not in self.J:
            return x
        return x @ self.J[layer].T

    @torch.no_grad()
    def lens_directions(self, token_ids, layer):
        """(len(token_ids), d) residual directions at `layer` that raise each token's lens logit
        (rows of W_U diag(g) J_l, with g the final norm's gain)."""
        w = self.model.lm_head.weight[token_ids].float().to(self.device) * self.gain
        if layer in self.J:
            w = w @ self.J[layer]
        return w.cpu().numpy()
