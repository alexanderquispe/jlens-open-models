"""Models and their Jacobian lenses.

Jacobian lens (Gurnee et al. 2026, "Verbalizable Representations Form a Global Workspace in
Language Models"; reference code github.com/anthropics/jacobian-lens):

    lens_l(h) = unembed(J_l h),   J_l = E[d h_final / d h_l]

h_l is the output of block l and h_final the output of the last block, before the final norm.
The lenses are the ones Neuronpedia fitted with the reference code (neuronpedia/jacobian-lens on
the HF Hub). As in the reference code (`force_bos=True`), every prompt starts with BOS when the
tokenizer has one (Qwen's has none).
"""
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
    # The next two need transformers >= 5 (results here: 5.11)
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
}


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class LensModel:
    def __init__(self, name, device=None):
        spec = MODELS[name]
        self.name, self.dtype = name, spec["dtype"]
        self.device = device or default_device()
        self.tok = AutoTokenizer.from_pretrained(spec["hf"])
        self.model = AutoModelForCausalLM.from_pretrained(spec["hf"], torch_dtype=self.dtype).to(self.device).eval()
        # GPT-2 style (transformer.h, ln_f) or Llama style (model.layers, norm)
        decoder = self.model.transformer if hasattr(self.model, "transformer") else self.model.model
        self.blocks = decoder.h if hasattr(decoder, "h") else decoder.layers
        self.norm = decoder.ln_f if hasattr(decoder, "ln_f") else decoder.norm
        self.n_blocks = len(self.blocks)
        self.d = self.model.config.hidden_size
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
    def run(self, input_ids, deltas=None, positions=None):
        """Block outputs [n_blocks] of (batch, seq, d) in float32, and the last-position logits.

        deltas: optional {layer: (batch, d) tensor} added after that block at `positions`
        (one index per row; default the last position).
        """
        store = {}
        rows = torch.arange(len(input_ids))
        positions = torch.full((len(input_ids),), input_ids.shape[1] - 1) if positions is None else torch.as_tensor(positions)

        def hook(layer):
            def f(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if deltas is not None and layer in deltas:
                    hidden = hidden.clone()
                    hidden[rows, positions] += deltas[layer].to(hidden.dtype)
                store[layer] = hidden.float()
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
        return [store[i] for i in range(self.n_blocks)], logits

    @torch.no_grad()
    def unembed(self, x):
        return self.model.lm_head(self.norm(x.to(self.dtype))).float()

    def transport(self, x, layer, use_jacobian=True):
        """J_l x (the lens); the identity for the logit lens or for the last block."""
        if not use_jacobian or layer not in self.J:
            return x
        return x @ self.J[layer].T

    @torch.no_grad()
    def lens_directions(self, token_ids, layer):
        """(len(token_ids), d) residual directions at `layer` that raise each token's lens logit
        (rows of W_U diag(g) J_l, with g the final norm's gain)."""
        w = self.model.lm_head.weight[token_ids].float() * self.norm.weight.float()
        if layer in self.J:
            w = w @ self.J[layer]
        return w.cpu().numpy()
