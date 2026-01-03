"""Lightweight wrapper to train Megalodon on character data."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from megalodon.configuration_megalodon import MegalodonConfig
from megalodon.modeling_megalodon import MegalodonForCausalLM

from .utils import gumbel_sample, min_p_filter


class MegalodonLM(nn.Module):
    """Simplified Megalodon wrapper matching the Llama interface in this repo.

    Wraps megalodon-hf's MegalodonForCausalLM with a training-friendly API:
    - forward(x, return_loss=True) for training loops
    - generate() with min-p sampling for inference

    Key differences from Llama:
    - Uses CEMA (Complex EMA) for O(n) sequence modeling
    - Chunk-based attention (chunk_size must divide seq_len or seq_len <= chunk_size)
    - Requires bfloat16 or float32 (no float16 support)

    See megalodon-hf docs for architecture details.
    """

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

    @property
    def model_dim(self) -> int:
        """Return model dimension (for API compatibility with Llama)."""
        return self.config.model_dim

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        filter_thres: float = 0.9,  # unused, for API compat with Llama
        min_p: float = 0.1,
    ) -> torch.Tensor:
        """Autoregressive sampling loop with min-p filtering."""
        del filter_thres  # unused, accepted for API compatibility
        out = prompt.clone()
        attn_mask = torch.ones_like(out, dtype=torch.long, device=out.device)

        outputs = self.model(
            input_ids=out,
            attention_mask=attn_mask,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1]
        cache = outputs.past_key_values

        for _ in range(max_length):
            logits = min_p_filter(logits, min_p=min_p)
            sample = gumbel_sample(logits, temperature=temperature, dim=-1)
            out = torch.cat((out, sample), dim=-1)

            step_mask = torch.ones_like(sample, dtype=torch.long, device=out.device)
            outputs = self.model(
                input_ids=sample,
                attention_mask=step_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )

            logits = outputs.logits[:, -1]
            cache = outputs.past_key_values

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
