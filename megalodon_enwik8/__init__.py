"""Megalodon vs Transformer comparison on character-level language modeling.

This package provides:
- MegalodonLM: Wrapper for megalodon-hf with a simple training interface
- Llama: Standard transformer baseline for comparison

See README.md for usage and RESULTS.md for experimental results.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

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
    "__version__",
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
