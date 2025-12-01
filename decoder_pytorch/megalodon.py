"""Lightweight wrapper to train Megalodon on character data."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import gumbel_sample, min_p_filter

try:
    from megalodon.configuration_megalodon import MegalodonConfig
    from megalodon.modeling_megalodon import MegalodonForCausalLM
except ImportError as err:  # pragma: no cover - env guard
    raise ImportError(
        "Unable to import `megalodon`. Install the megalodon-hf package in the "
        "active environment (e.g., `pip install -e .` from the repo root) before "
        "using MegalodonLM."
    ) from err


class MegalodonLM(nn.Module):
    """HF-style Megalodon LM with a Llama-like interface used by train.py."""

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        z_dim: int,
        value_dim: int,
        ffn_hidden_dim: int,
        cema_ndim: int,
        chunk_size: int,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        swiglu: bool = True,
        rescale_nffn: bool = False,
        scale_emb: bool = False,
        share_emb: bool = False,
        rope_base: Optional[float] = None,
        init_mode: str = "he",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        config = MegalodonConfig(
            vocab_size=vocab_size,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            z_dim=z_dim,
            value_dim=value_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            cema_ndim=cema_ndim,
            chunk_size=chunk_size,
            norm_num_groups=norm_num_groups,
            dropout=dropout,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            swiglu=swiglu,
            rescale_nffn=rescale_nffn,
            scale_emb=scale_emb,
            share_emb=share_emb,
            rope_base=rope_base,
            init_mode=init_mode,
            gradient_checkpointing=gradient_checkpointing,
            pad_token_id=0,
        )
        self.model = MegalodonForCausalLM(config)
        self.config = config

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.config.vocab_size

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        min_p: float = 0.1,
    ) -> torch.Tensor:
        """Autoregressive sampling loop with min-p filtering."""
        out = prompt.clone()
        attn_mask = torch.ones_like(out, dtype=torch.long, device=out.device)

        for _ in range(max_length):
            logits = self(
                out,
                mask=attn_mask,
                return_loss=False,
            )[:, -1]

            logits = min_p_filter(logits, min_p=min_p)
            sample = gumbel_sample(logits, temperature=temperature, dim=-1)
            out = torch.cat((out, sample), dim=-1)
            attn_mask = torch.ones_like(out, dtype=torch.long, device=out.device)

        return out[:, prompt.shape[-1] :]

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> torch.Tensor:
        """Forward pass matching the template's expectations."""
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]
        attn_mask = mask
        if attn_mask is None:
            attn_mask = torch.ones_like(x, dtype=torch.long, device=x.device)

        outputs = self.model(
            input_ids=x,
            attention_mask=attn_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits

        if not return_loss:
            return logits

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        return loss
