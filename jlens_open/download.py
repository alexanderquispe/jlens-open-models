"""Download models and their lenses into the Hugging Face cache, so that jobs can run offline.

    python -m jlens_open.download llama3.1-8b gpt2-small
"""
import argparse

from huggingface_hub import hf_hub_download, snapshot_download

from .models import LENS_REPO, MODELS

parser = argparse.ArgumentParser(description="Download models and their lenses into the Hugging Face cache.")
parser.add_argument("models", nargs="+", choices=list(MODELS))
args = parser.parse_args()

for name in args.models:
    spec = MODELS[name]
    # Weights, config and tokenizer only: Meta's original checkpoint (original/*.pth) would add another 16 GB
    path = snapshot_download(spec["hf"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    print(f"{name}: model in {path}")
    print(f"{name}: lens in {hf_hub_download(LENS_REPO, spec['lens'])}")
