"""Rotary Embedding module."""

from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to input tensor.
    
    Args:
        x (torch.Tensor): Input tensor
        cos (torch.Tensor): Cosine embeddings
        sin (torch.Tensor): Sine embeddings
        
    Returns:
        torch.Tensor: Tensor with rotary embeddings applied
    """
    cos = cos.unsqueeze(-2).to(x.device)
    sin = sin.unsqueeze(-2).to(x.device)
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE).
    
    This class implements the RoPE positional encoding for transformer models.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        """Initialize the RotaryEmbedding module.
        
        Args:
            head_size (int): Size of each attention head
            rotary_dim (int): Dimension of rotary embeddings
            max_position_embeddings (int): Maximum position embedding
            base (float): Base value for calculating inverse frequency
        """
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # print(f"head_size: {head_size}, rotary_dim: {rotary_dim}, max_position_embeddings: {max_position_embeddings}, base: {base}")
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the RotaryEmbedding module.
        
        Args:
            positions (torch.Tensor): Position indices
            query (torch.Tensor): Query tensor
            key (torch.Tensor): Key tensor
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Query and key tensors with rotary embeddings applied
        """
        # Handle case where query is empty
        if query.numel() == 0:
            return query, key
        num_tokens = positions.size(0)
        # Ensure positions and cos_sin_cache are on the same device
        positions = positions.to(self.cos_sin_cache.device)
        cos_sin = self.cos_sin_cache[positions]
        # cos_sin = torch.ones_like(cos_sin)
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape

        num_q_heads = query.shape[-1] // self.head_size
        query = query.view(num_tokens, num_q_heads, self.head_size)
        query = apply_rotary_emb(query, cos, sin).view(query_shape)
        key_shape = key.shape
        num_k_heads = key.shape[-1] // self.head_size
        key = key.view(num_tokens, num_k_heads, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """Get rotary embedding instance with caching.
    
    Args:
        head_size (int): Size of each attention head
        rotary_dim (int): Dimension of rotary embeddings
        max_position (int): Maximum position embedding
        base (float): Base value for calculating inverse frequency
        rope_scaling (dict | None, optional): Rope scaling configuration. Defaults to None.
        
    Returns:
        RotaryEmbedding: Instance of RotaryEmbedding
    """
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
