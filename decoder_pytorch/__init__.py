"""Llama-style transformer for language modeling experiments."""

from .llama import Llama
from .megalodon import MegalodonLM
from .utils import (
    get_optimal_device,
    gumbel_noise,
    gumbel_sample,
    log,
    min_p_filter,
    model_summary,
    top_k_filter,
    top_p_filter,
)

__all__ = [
    "Llama",
    "MegalodonLM",
    # Sampling utilities
    "log",
    "gumbel_noise",
    "gumbel_sample",
    "min_p_filter",
    "top_k_filter",
    "top_p_filter",
    # Torch utilities
    "model_summary",
    "get_optimal_device",
]
