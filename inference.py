"""Simple inference utility for Megalodon and Llama checkpoints."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Union

import numpy as np
import torch

from megalodon_enwik8 import Llama, MegalodonLM

# A non-enwik8-ish default prompt with key info early in the text to sanity-check
# whether multi-chunk context is used during generation.
DEFAULT_PROMPT = (
    "Journal entry: The coastal city of Zephyria just opened the emerald bay lighthouse. "
    "Key fact: The secret passphrase is LUMA-4242 and the ferry departs at dawn. "
    "Locals brew citrus tea with basil and mint, and the streets are paved with cobalt tiles. "
    "A traveling cartographer sketched the harbor from the old fortress, noting two cranes and a rusted tram. "
    "Visitors should watch for the sea glass markets"
)


def load_model(
    checkpoint_path: Path, device: torch.device
) -> Union[MegalodonLM, Llama]:
    """Load model weights from a training checkpoint.

    Automatically detects model type (Megalodon or Llama) from saved config.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    model_type = str(cfg.get("model", "llama")).lower()

    if model_type == "megalodon":
        import megalodon

        print(f"megalodon-hf version: {megalodon.__version__}")
        model = MegalodonLM(
            vocab_size=cfg.get("num_tokens", 256),
            model_dim=cfg.get("model_dim", 384),
            num_layers=cfg.get("num_layers", 6),
            num_heads=cfg.get("num_heads", 3),
            z_dim=cfg.get("z_dim", 192),
            value_dim=cfg.get("value_dim", 384),
            ffn_hidden_dim=cfg.get("ffn_hidden_dim", 1024),
            cema_ndim=cfg.get("cema_ndim", 8),
            chunk_size=int(cfg.get("chunk_size", cfg.get("seq_len", 512))),
            norm_num_groups=cfg.get("norm_num_groups", 32),
            dropout=cfg.get("dropout", 0.0),
            attention_dropout=cfg.get("attention_dropout", 0.0),
            hidden_dropout=cfg.get("hidden_dropout", 0.0),
            swiglu=bool(cfg.get("swiglu", True)),
            rescale_nffn=bool(cfg.get("rescale_nffn", False)),
            scale_emb=bool(cfg.get("scale_emb", False)),
            share_emb=bool(cfg.get("share_emb", False)),
            rope_base=cfg.get("rope_base"),
            init_mode=cfg.get("init_mode", "he"),
            gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
        ).to(device)
    elif model_type == "llama":
        model = Llama(
            num_tokens=cfg.get("num_tokens", 256),
            dim=cfg.get("dim", 512),
            depth=cfg.get("depth", 16),
            heads=cfg.get("heads", 8),
            dim_head=cfg.get("dim_head", 64),
            tied_embedding=cfg.get("tied_embedding", True),
            ffn_dim_multiplier=cfg.get("ffn_dim_multiplier"),
            flash_attn=bool(cfg.get("flash_attn", False)),
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {model_type} model from {checkpoint_path}")
    return model


def encode_prompt(text: str, device: torch.device) -> torch.Tensor:
    """Encode text into a uint8/long tensor for character-level Megalodon."""
    arr = np.frombuffer(text.encode("utf-8", "replace"), dtype=np.uint8)
    tokens = torch.from_numpy(arr.astype(np.int64)).unsqueeze(0).to(device)
    return tokens


def read_prompt(args) -> str:
    """Resolve prompt text from args."""
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    if args.prompt_gzip:
        with gzip.open(args.prompt_gzip, "rb") as f:
            raw = f.read(args.prompt_len)
        return raw.decode("utf-8", "replace")
    if args.prompt:
        return args.prompt
    return DEFAULT_PROMPT


def get_parser():
    parser = argparse.ArgumentParser(
        description="Megalodon inference helper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to training checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt string to condition on (overrides default)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Read prompt from a UTF-8 text file",
    )
    parser.add_argument(
        "--prompt-gzip",
        type=str,
        default=None,
        help="Read prompt bytes from a gzip file (useful for enwik8).",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=384,
        help="Prompt length when reading from gzip (bytes).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.1,
        help="Min-p filtering threshold.",
    )
    return parser


def main():
    args = get_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), device)

    prompt_text = read_prompt(args)
    prompt = encode_prompt(prompt_text, device)

    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            min_p=args.min_p,
        )

    prompt_len = prompt.shape[-1]
    gen_len = generated.shape[-1]
    gen_text = "".join(chr(max(32, c)) for c in generated[0].cpu().numpy())

    print(f"Device: {device}")
    print(f"Prompt length: {prompt_len} | Generated length: {gen_len}")
    if len(prompt_text) < 1000:
        print("\nPROMPT:\n", prompt_text)
    else:
        print("\nPROMPT (first 200 chars):\n", prompt_text[:200])
    print("\nGENERATED:\n", gen_text)


if __name__ == "__main__":
    main()
