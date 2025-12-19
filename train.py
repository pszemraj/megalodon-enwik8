"""Simple training script for language models."""

import argparse
import gzip
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from decoder_pytorch import Llama, get_optimal_device, model_summary
from decoder_pytorch import MegalodonLM


# Data utilities
def cycle(loader):
    """Cycle through a dataloader infinitely."""
    while True:
        for data in loader:
            yield data


class SequenceDataset(Dataset):
    """Simple dataset for sequence modeling."""

    def __init__(self, data: torch.Tensor, seq_len: int):
        """
        Initialize dataset.

        :param torch.Tensor data: data tensor
        :param int seq_len: maximum sequence length
        """
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return max(1, (self.data.size(0) - self.seq_len) // 2)

    def __getitem__(self, idx):
        # Random sampling for better coverage
        start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,)).item()
        return self.data[start : start + self.seq_len + 1].long()


def load_data(data_path: str, train_split: float = 0.9):
    """Load character-level data from gzip file."""
    with gzip.open(data_path) as f:
        data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8).copy()

    train_size = int(len(data) * train_split)
    train_data = torch.from_numpy(data[:train_size])
    val_data = torch.from_numpy(data[train_size:])

    return train_data, val_data


def train(config_path: str, resume_checkpoint: Optional[str] = None):
    """Main training function."""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Setup device
    device, device_type, amp_dtype = get_optimal_device()
    print(f"Device: {device}")

    device_caps = {
        "supports_fused_optimizer": device_type == "cuda",
        "supports_flash_attn": device_type in ("cuda", "mps"),
    }

    # Setup autocast context
    use_autocast = bool(config.get("use_autocast", True))

    def autocast_context():
        if use_autocast:
            return torch.autocast(device_type=device_type, dtype=amp_dtype)
        return nullcontext()

    print(
        f"Mixed precision: {'enabled' if use_autocast else 'disabled'}"
        f"{f' ({amp_dtype})' if use_autocast else ' (full fp32)'}"
    )

    if config.get("seed"):
        torch.manual_seed(config["seed"])

    # Load data
    train_data, val_data = load_data(config["data_path"])
    train_dataset = SequenceDataset(train_data, config["seq_len"])
    val_dataset = SequenceDataset(val_data, config["seq_len"])
    train_tokens_total = train_data.numel()

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    train_loader = cycle(train_loader)
    val_loader = cycle(val_loader)

    # Create model
    model_type = str(config.get("model", "llama")).lower()

    if model_type == "llama":
        flash_attn_requested = config.get("flash_attn")
        if flash_attn_requested is None:
            flash_attn_requested = device_caps["supports_flash_attn"]
        elif flash_attn_requested and not device_caps["supports_flash_attn"]:
            print(
                "Warning: flash attention requested but not supported on this device; using standard attention."
            )
            flash_attn_requested = False

        model = Llama(
            num_tokens=config.get("num_tokens", 256),
            dim=config.get("dim", 512),
            depth=config.get("depth", 16),
            heads=config.get("heads", 8),
            dim_head=config.get("dim_head", 64),
            tied_embedding=config.get("tied_embedding", True),
            ffn_dim_multiplier=config.get("ffn_dim_multiplier"),
            flash_attn=bool(flash_attn_requested),
        ).to(device)
    elif model_type in ("megalodon", "mega"):
        seq_len = config["seq_len"]
        chunk_size = int(config.get("chunk_size", seq_len))
        if seq_len > chunk_size and seq_len % chunk_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be <= chunk_size ({chunk_size}) or divisible by it for Megalodon."
            )

        model = MegalodonLM(
            vocab_size=config.get("num_tokens", 256),
            model_dim=config.get("model_dim", 384),
            num_layers=config.get("num_layers", 6),
            num_heads=config.get("num_heads", 3),
            z_dim=config.get("z_dim", 192),
            value_dim=config.get("value_dim", 384),
            ffn_hidden_dim=config.get("ffn_hidden_dim", 1024),
            cema_ndim=config.get("cema_ndim", 8),
            chunk_size=chunk_size,
            norm_num_groups=config.get("norm_num_groups", 32),
            dropout=config.get("dropout", 0.0),
            attention_dropout=config.get("attention_dropout", 0.0),
            hidden_dropout=config.get("hidden_dropout", 0.0),
            swiglu=bool(config.get("swiglu", True)),
            rescale_nffn=bool(config.get("rescale_nffn", False)),
            scale_emb=bool(config.get("scale_emb", False)),
            share_emb=bool(config.get("share_emb", False)),
            rope_base=config.get("rope_base"),
            init_mode=config.get("init_mode", "he"),
            gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type '{model_type}'")

    model_summary(model, max_depth=3, show_param_shapes=True)

    # Optimizer
    # Fused optimizer is available for CUDA only (not MPS or CPU)
    has_complex_params = any(p.is_complex() for p in model.parameters())
    fused_opt = device_caps["supports_fused_optimizer"] and not has_complex_params
    if device_caps["supports_fused_optimizer"] and not fused_opt:
        print("Disabling fused optimizer because the model has complex parameters.")

    optimizer = Adam(
        model.parameters(),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 0.0),
        fused=fused_opt,
    )

    # Training state
    step = 0

    # Resume if specified
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint.get("step", 0)
        print(f"Resumed from step {step}")

    # Compile model if requested
    if config.get("compile", False):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Create output directory
    run_dir = Path(config.get("run_dir", "runs"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    num_batches = config.get("num_batches", 100000)
    grad_accum_every = config.get("grad_accum_every", 4)
    validate_every = config.get("validate_every", 100)
    val_batches = config.get("val_batches", 50)
    save_every = config.get("save_every", 1000)
    generate_every = config.get("generate_every", 500)
    generate_prompt_len = config.get("generate_prompt_len", 128)
    generate_length = config.get("generate_length", 256)

    pbar = tqdm(range(num_batches), desc="training")
    cumulative_tokens = 0

    for _ in pbar:
        # Validation runs before the training step; tokens_seen counts training tokens only.
        if step % validate_every == 0:
            model.eval()
            val_loss_sum = 0
            val_tokens = 0
            for _ in range(val_batches):
                data = next(val_loader).to(device)
                inputs = data[:, :-1]
                targets = data[:, 1:]
                with torch.no_grad(), autocast_context():
                    logits = model(inputs, return_loss=False)
                    loss_unreduced = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        reduction="none",
                    )
                    val_loss_sum += loss_unreduced.sum().item()
                    val_tokens += targets.numel()

            val_loss = val_loss_sum / val_tokens
            data_passes = cumulative_tokens / max(1, train_tokens_total)
            tqdm.write(
                f"Step {step} | Val loss: {val_loss:.4f} | tokens_seen={cumulative_tokens} | passes={data_passes:.3f}x"
            )
            # Log metrics
            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n")

        model.train()

        # Training step with gradient accumulation
        # Accumulate raw losses and token counts for correct normalization.
        # Critical pitfall: normalizing each micro-batch separately leads to
        # incorrect gradients. Sum first, divide once at the end.
        total_loss_sum = 0
        total_tokens = 0
        loss_accumulator = None  # Keep gradient graph alive
        optimizer.zero_grad()

        for _ in range(grad_accum_every):
            data = next(train_loader).to(device)
            inputs = data[:, :-1]
            targets = data[:, 1:]

            with autocast_context():
                logits = model(inputs, return_loss=False)
                # Compute unnormalized loss
                loss_unreduced = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="none",
                )
                loss_sum = loss_unreduced.sum()
                num_tokens = targets.numel()

                # Accumulate raw loss tensors
                if loss_accumulator is None:
                    loss_accumulator = loss_sum
                else:
                    loss_accumulator = loss_accumulator + loss_sum

                total_loss_sum += loss_sum.item()
                total_tokens += num_tokens

        # Normalize ONCE by total tokens across all micro-batches
        loss = loss_accumulator / total_tokens
        loss.backward()

        # Compute final normalized loss for display
        avg_loss = total_loss_sum / total_tokens
        # Gradient clipping
        if config.get("grad_clip_norm"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])

        optimizer.step()

        # Project EMA parameters back into stable region (Megalodon only)
        if hasattr(model, "project_ema_parameters"):
            model.project_ema_parameters()

        cumulative_tokens += total_tokens
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # Generation
        if step % generate_every == 0 and step > 0:
            model.eval()
            data = next(val_loader).to(device)
            prompt = data[:1, :generate_prompt_len]

            with torch.no_grad():
                generated = model.generate(
                    prompt,
                    max_length=generate_length,
                    temperature=config.get("temperature", 1.0),
                    min_p=config.get("min_p", 0.1),
                )

            # Decode if character-level
            if config.get("num_tokens", 256) <= 256:
                prompt_text = "".join(
                    [chr(max(32, c)) for c in prompt[0].cpu().numpy()]
                )
                gen_text = "".join(
                    [chr(max(32, c)) for c in generated[0].cpu().numpy()]
                )
                tqdm.write(f"\n{'=' * 50} Step {step} {'=' * 50}")
                tqdm.write(f"Prompt: {prompt_text}")
                tqdm.write(f"Generated: {gen_text}")

        # Save checkpoint
        if step % save_every == 0 and step > 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": config,
            }
            torch.save(checkpoint, run_dir / f"checkpoint_{step}.pt")
            tqdm.write(f"Saved checkpoint at step {step}")

        step += 1

    # Final save
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config,
    }
    torch.save(checkpoint, run_dir / "final.pt")
    print(f"\nTraining complete! Final checkpoint saved to {run_dir}/final.pt")


def get_parser() -> argparse.ArgumentParser:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Train language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/simple.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    return parser


def run():
    """Entry point for training script."""
    args = get_parser().parse_args()
    train(args.config, args.resume)


if __name__ == "__main__":
    run()
