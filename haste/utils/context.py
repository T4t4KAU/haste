"""Context management module."""

from dataclasses import dataclass
from threading import local
import torch


@dataclass
class Context:
    """Context class for storing inference context information.
    
    This class holds various context information used during model inference,
    such as sequence lengths, slot mappings, and block tables.
    """
    is_prefill: bool = False  # Whether this is a prefill operation
    is_jit: bool = False  # Whether this is a JIT execution
    cu_seqlens_q: torch.Tensor | None = None  # Cumulative sequence lengths for queries
    cu_seqlens_k: torch.Tensor | None = None  # Cumulative sequence lengths for keys
    max_seqlen_q: int = 0  # Maximum sequence length for queries
    max_seqlen_k: int = 0  # Maximum sequence length for keys
    slot_mapping: torch.Tensor | None = None  # Slot mapping for KV cache
    context_lens: torch.Tensor | None = None  # Context lengths
    block_tables: torch.Tensor | None = None  # Block tables for KV cache management

_STATE = local()  # Thread-local storage for context


def get_context():
    """Get the current context.
    
    Returns:
        Context: Current context object
    """
    return getattr(_STATE, "context", Context())


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, is_jit=False):
    """Set the current context.
    
    Args:
        is_prefill (bool): Whether this is a prefill operation
        cu_seqlens_q (torch.Tensor | None, optional): Cumulative sequence lengths for queries. Defaults to None.
        cu_seqlens_k (torch.Tensor | None, optional): Cumulative sequence lengths for keys. Defaults to None.
        max_seqlen_q (int, optional): Maximum sequence length for queries. Defaults to 0.
        max_seqlen_k (int, optional): Maximum sequence length for keys. Defaults to 0.
        slot_mapping (torch.Tensor | None, optional): Slot mapping for KV cache. Defaults to None.
        context_lens (torch.Tensor | None, optional): Context lengths. Defaults to None.
        block_tables (torch.Tensor | None, optional): Block tables for KV cache management. Defaults to None.
        is_jit (bool, optional): Whether this is a JIT execution. Defaults to False.
    """
    _STATE.context = Context(
        is_prefill,
        is_jit,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
    )


def reset_context():
    """Reset the current context to default values."""
    _STATE.context = Context()
