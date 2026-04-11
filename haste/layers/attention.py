"""Attention mechanism implementation with support for CPU and GPU."""

import torch
from torch import nn
from torch.nn import functional as F

import triton
import triton.language as tl

from haste.utils.context import get_context
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except Exception:  # pragma: no cover - optional dependency
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None

if triton is not None:
    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        """Triton kernel for storing KV cache.
        
        Args:
            key_ptr: Pointer to key tensor
            key_stride: Stride of key tensor
            value_ptr: Pointer to value tensor
            value_stride: Stride of value tensor
            k_cache_ptr: Pointer to key cache
            v_cache_ptr: Pointer to value cache
            slot_mapping_ptr: Pointer to slot mapping
            D: Dimensionality of the embeddings
        """
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot.to(tl.int64) * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


def _expand_kv_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Expand key/value heads to match query heads.
    
    Args:
        x (torch.Tensor): Input tensor with shape [seq_len, kv_heads, head_dim]
        num_heads (int): Number of query heads
        
    Returns:
        torch.Tensor: Expanded tensor with shape [seq_len, num_heads, head_dim]
    """
    kv_heads = x.size(1)
    if kv_heads == num_heads:
        return x

    assert num_heads % kv_heads == 0, (
        f"num_heads ({num_heads}) must be divisible by num_kv_heads ({kv_heads})"
    )
    return x.repeat_interleave(num_heads // kv_heads, dim=1)


def _cpu_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None,
    causal: bool,
    prefix_len: int = 0,
) -> torch.Tensor:
    """CPU implementation of scaled dot-product attention.
    
    Args:
        q (torch.Tensor): Query tensor with shape [seq_len_q, num_heads, head_dim]
        k (torch.Tensor): Key tensor with shape [seq_len_k, num_kv_heads, head_dim]
        v (torch.Tensor): Value tensor with shape [seq_len_k, num_kv_heads, head_dim]
        softmax_scale (float | None): Scale factor for softmax
        causal (bool): Whether to use causal attention
        prefix_len (int, optional): Length of prefix. Defaults to 0.
        
    Returns:
        torch.Tensor: Attention output with shape [seq_len_q, num_heads, head_dim]
    """
    if q.numel() == 0:
        return torch.empty_like(q)

    if k.size(0) == 0:
        return torch.zeros_like(q)

    q_float = q.float().permute(1, 0, 2).unsqueeze(0)
    k_float = _expand_kv_heads(k.float(), q.size(1)).permute(1, 0, 2).unsqueeze(0)
    v_float = _expand_kv_heads(v.float(), q.size(1)).permute(1, 0, 2).unsqueeze(0)

    attn_mask = None
    if causal:
        q_positions = torch.arange(q.size(0), device=q.device) + prefix_len
        k_positions = torch.arange(k.size(0), device=q.device)
        attn_mask = (k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)).view(1, 1, q.size(0), k.size(0))

    output = F.scaled_dot_product_attention(
        q_float,
        k_float,
        v_float,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=softmax_scale if softmax_scale is not None else None,
    )
    return output.squeeze(0).permute(1, 0, 2).to(dtype=q.dtype)


def _flatten_cache(cache: torch.Tensor) -> torch.Tensor:
    """Flatten cache tensor to 3D.
    
    Args:
        cache (torch.Tensor): Cache tensor with shape [num_blocks, block_size, num_heads, head_dim] or [seq_len, num_heads, head_dim]
        
    Returns:
        torch.Tensor: Flattened cache with shape [seq_len, num_heads, head_dim]
    """
    if cache.dim() == 4:
        return cache.view(-1, cache.size(-2), cache.size(-1))
    if cache.dim() == 3:
        return cache
    raise ValueError(f"Unsupported cache rank: {cache.dim()}")


def _gather_cache_sequence(cache: torch.Tensor, cache_len: int, block_table: torch.Tensor | None) -> torch.Tensor:
    """Gather cache sequence based on block table.
    
    Args:
        cache (torch.Tensor): Cache tensor
        cache_len (int): Length of cache to gather
        block_table (torch.Tensor | None): Block table for gathering
        
    Returns:
        torch.Tensor: Gathered cache sequence
    """
    if cache.dim() == 3:
        return cache[:cache_len]

    if block_table is None:
        return cache.view(-1, cache.size(-2), cache.size(-1))[:cache_len]

    block_size = cache.size(1)
    num_blocks = (cache_len + block_size - 1) // block_size
    block_ids = block_table[:num_blocks].to(dtype=torch.long, device=cache.device)
    valid_block_ids = block_ids[block_ids >= 0]
    gathered = cache.index_select(0, valid_block_ids).reshape(-1, cache.size(-2), cache.size(-1))
    return gathered[:cache_len]


def cpu_flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
):
    """CPU implementation of flash attention for variable-length sequences.
    
    Args:
        q (torch.Tensor): Query tensor
        k (torch.Tensor): Key tensor
        v (torch.Tensor): Value tensor
        max_seqlen_q (int): Maximum sequence length for queries
        cu_seqlens_q (torch.Tensor): Cumulative sequence lengths for queries
        max_seqlen_k (int): Maximum sequence length for keys
        cu_seqlens_k (torch.Tensor): Cumulative sequence lengths for keys
        softmax_scale (float | None, optional): Scale factor for softmax. Defaults to None.
        causal (bool, optional): Whether to use causal attention. Defaults to False.
        
    Returns:
        torch.Tensor: Attention output
    """
    del max_seqlen_q, max_seqlen_k

    output = torch.empty_like(q)
    batch_size = cu_seqlens_q.numel() - 1

    for batch_idx in range(batch_size):
        q_start = int(cu_seqlens_q[batch_idx].item())
        q_end = int(cu_seqlens_q[batch_idx + 1].item())
        k_start = int(cu_seqlens_k[batch_idx].item())
        k_end = int(cu_seqlens_k[batch_idx + 1].item())

        q_seq = q[q_start:q_end]
        k_seq = k[k_start:k_end]
        v_seq = v[k_start:k_end]
        prefix_len = max(k_seq.size(0) - q_seq.size(0), 0) if causal else 0
        output[q_start:q_end] = _cpu_attention(
            q_seq,
            k_seq,
            v_seq,
            softmax_scale=softmax_scale,
            causal=causal,
            prefix_len=prefix_len,
        )

    return output


def cpu_flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    cu_seqlens_q: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
):
    """CPU implementation of flash attention with KV cache.
    
    Args:
        q (torch.Tensor): Query tensor
        k_cache (torch.Tensor): Key cache
        v_cache (torch.Tensor): Value cache
        cache_seqlens (torch.Tensor): Cache sequence lengths
        softmax_scale (float | None, optional): Scale factor for softmax. Defaults to None.
        causal (bool, optional): Whether to use causal attention. Defaults to False.
        cu_seqlens_q (torch.Tensor | None, optional): Cumulative sequence lengths for queries. Defaults to None.
        max_seqlen_q (int | None, optional): Maximum sequence length for queries. Defaults to None.
        
    Returns:
        torch.Tensor: Attention output
    """
    del max_seqlen_q

    context = get_context()
    block_tables = context.block_tables

    if cu_seqlens_q is None:
        if q.dim() == 4:
            batch_size, seqlen_q = q.size(0), q.size(1)
            flat_q = q.reshape(-1, q.size(-2), q.size(-1))
        elif q.dim() == 3:
            batch_size, seqlen_q = q.size(0), 1
            flat_q = q
        else:
            raise ValueError(f"Unsupported q rank without cu_seqlens_q: {q.dim()}")

        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * seqlen_q,
            seqlen_q,
            dtype=torch.int32,
            device=q.device,
        )
        reshape_output = q.dim() == 4
    else:
        flat_q = q
        batch_size = cu_seqlens_q.numel() - 1
        seqlen_q = None
        reshape_output = False

    output = torch.empty_like(flat_q)
    for batch_idx in range(batch_size):
        q_start = int(cu_seqlens_q[batch_idx].item())
        q_end = int(cu_seqlens_q[batch_idx + 1].item())
        cache_len = int(cache_seqlens[batch_idx].item())
        block_table = None if block_tables is None else block_tables[batch_idx]

        q_seq = flat_q[q_start:q_end]
        k_seq = _gather_cache_sequence(k_cache, cache_len, block_table)
        v_seq = _gather_cache_sequence(v_cache, cache_len, block_table)
        prefix_len = max(cache_len - q_seq.size(0), 0) if causal else 0

        output[q_start:q_end] = _cpu_attention(
            q_seq,
            k_seq,
            v_seq,
            softmax_scale=softmax_scale,
            causal=causal,
            prefix_len=prefix_len,
        )

    if reshape_output:
        return output.view(batch_size, seqlen_q, q.size(-2), q.size(-1))
    return output


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """Store key and value tensors into KV cache.
    
    Args:
        key (torch.Tensor): Key tensor
        value (torch.Tensor): Value tensor
        k_cache (torch.Tensor): Key cache
        v_cache (torch.Tensor): Value cache
        slot_mapping (torch.Tensor): Slot mapping for cache storage
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    
    # Check if slot_mapping has the same number of elements as N
    assert slot_mapping.numel() == N,\
        f"slot_mapping.numel() ({slot_mapping.numel()}) != N ({N})"
    
    # Check if running on CPU, use Python loop if so
    if key.device.type == 'cpu':
        flat_k_cache = _flatten_cache(k_cache)
        flat_v_cache = _flatten_cache(v_cache)
        valid = (slot_mapping >= 0) & (slot_mapping < flat_k_cache.size(0))
        if valid.any():
            slot_idx = slot_mapping[valid].to(dtype=torch.long)
            flat_k_cache.index_copy_(0, slot_idx, key[valid].to(dtype=flat_k_cache.dtype))
            flat_v_cache.index_copy_(0, slot_idx, value[valid].to(dtype=flat_v_cache.dtype))
    else:
        assert triton is not None, "Triton is required for GPU KV cache writes"
        # Use Triton kernel on GPU
        store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

class Attention(nn.Module):
    """Attention module with support for KV cache and speculative decoding.
    
    This class implements multi-head attention with support for KV caching,
    speculative decoding, and both CPU and GPU execution.
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        draft: bool = False,
        speculate: bool = False,
        draft_async: bool = False,
        F: int = 1,
        K: int = 1,
    ):
        """Initialize the Attention module.
        
        Args:
            num_heads (int): Number of attention heads
            head_dim (int): Dimension of each attention head
            scale (float): Scale factor for attention scores
            num_kv_heads (int): Number of key/value heads
            draft (bool, optional): Whether this is a draft model. Defaults to False.
            speculate (bool, optional): Whether to use speculative decoding. Defaults to False.
            draft_async (bool, optional): Whether to use async draft mode. Defaults to False.
            F (int, optional): Async fan-out. Defaults to 1.
            K (int, optional): Speculation length. Defaults to 1.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.draft = draft
        self.speculate = speculate
        self.draft_async = draft_async
        self.prefill_wrappers = {}
        self.F = F  # async_fan_out
        self.K = K  # speculate_k
        self.only_prefill_wrapper = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Forward pass of the Attention module.
        
        Args:
            q (torch.Tensor): Query tensor
            k (torch.Tensor): Key tensor
            v (torch.Tensor): Value tensor
            
        Returns:
            torch.Tensor: Attention output
        """
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        k_cache, v_cache = self.k_cache, self.v_cache

        context = get_context()
        if (
            self.k_cache.numel()
            and self.v_cache.numel()
            and context.slot_mapping is not None
            and context.slot_mapping.numel() == k.size(0)
        ):
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache

            k, v = k.view(-1, self.num_kv_heads, self.head_dim), v.view(-1, self.num_kv_heads, self.head_dim)
            
            # Choose between flash_attn and CPU version based on device
            if q.device.type == 'cuda':
                assert flash_attn_varlen_func is not None, "flash_attn is required for CUDA prefill"
                o = flash_attn_varlen_func(q, k, v,
                                            max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                            max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                            softmax_scale=self.scale, causal=True)
            else:
                o = cpu_flash_attn_varlen_func(q, k, v,
                                            max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                            max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                            softmax_scale=self.scale, causal=True)

        else:
            # verify/glue decode: multi-query with cu_seqlens_q (K+1 or variable per seq)
            verify_or_glue = (
                self.speculate and context.cu_seqlens_q is not None
            )
            decode = not verify_or_glue
            tree_decode = (
                decode and self.speculate and self.draft and self.draft_async
                and not context.is_jit
            )

            if verify_or_glue:
                assert context.context_lens is not None
                # Choose between flash_attn and CPU version based on device
                if q.device.type == 'cuda':
                    assert flash_attn_with_kvcache is not None, "flash_attn is required for CUDA decode"
                    o = flash_attn_with_kvcache(q, k_cache, v_cache,
                                            cache_seqlens=context.context_lens,
                                            softmax_scale=self.scale, causal=True,
                                            cu_seqlens_q=context.cu_seqlens_q, max_seqlen_q=context.max_seqlen_q,
                                            )
                else:
                    o = cpu_flash_attn_with_kvcache(q, k_cache, v_cache,
                                            cache_seqlens=context.context_lens,
                                            softmax_scale=self.scale, causal=True,
                                            cu_seqlens_q=context.cu_seqlens_q, max_seqlen_q=context.max_seqlen_q,
                                            )

            elif tree_decode:
                if self.only_prefill_wrapper is not None:
                    prefill_wrapper = self.only_prefill_wrapper
                else:
                    mq_len = self.F * (self.K+1)
                    bs = q.shape[0] // mq_len
                    wrapper_bs = None
                    for available_bs in sorted(self.prefill_wrappers.keys()):
                        if available_bs >= bs:
                            wrapper_bs = available_bs
                            break
                    prefill_wrapper = self.prefill_wrappers[wrapper_bs]
                o = prefill_wrapper.run(q, (self.k_cache, self.v_cache))
            else: # single query decode
                q = q.unsqueeze(1)
                # Choose between flash_attn and CPU version based on device
                if q.device.type == 'cuda':
                    assert flash_attn_with_kvcache is not None, "flash_attn is required for CUDA decode"
                    o = flash_attn_with_kvcache(q, k_cache, v_cache,
                                                cache_seqlens=context.context_lens,
                                                softmax_scale=self.scale, causal=True,
                                                )
                else:
                    o = cpu_flash_attn_with_kvcache(q, k_cache, v_cache,
                                                cache_seqlens=context.context_lens,
                                                softmax_scale=self.scale, causal=True,
                                                )

        o = o.view(-1, self.num_heads * self.head_dim)
        
        return o
