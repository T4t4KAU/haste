"""Rotary Embedding module."""

import math
import warnings

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
        rope_scaling: dict | None = None,
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
        self.rotary_dim = rotary_dim
        self.base = float(base)
        self.rope_scaling = _normalize_rope_scaling(
            rope_scaling,
            max_position_embeddings=max_position_embeddings,
            base=base,
        )
        self.max_seq_len_cached = 0
        self.attention_scaling = 1.0
        self.register_buffer("cos_sin_cache", torch.empty(0), persistent=False)
        self._build_cache(max_position_embeddings)

    def _compute_default_inv_freq(
        self,
        *,
        base: float,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        return 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=device) / self.rotary_dim))

    def _compute_inv_freq(
        self,
        seq_len: int,
        *,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, float]:
        rope_scaling = self.rope_scaling
        if rope_scaling is None:
            return self._compute_default_inv_freq(base=self.base, device=device), 1.0

        rope_type = str(rope_scaling.get("rope_type", "default")).lower()
        factor = rope_scaling.get("factor")
        if factor is not None:
            factor = float(factor)

        if rope_type == "default":
            return self._compute_default_inv_freq(base=self.base, device=device), 1.0

        if rope_type == "linear":
            inv_freq = self._compute_default_inv_freq(base=self.base, device=device)
            if factor is not None and factor > 0:
                inv_freq /= factor
            return inv_freq, 1.0

        if rope_type in {"dynamic", "dynamic_ntk"}:
            original_max = int(rope_scaling.get("original_max_position_embeddings") or seq_len)
            seq_len = max(int(seq_len), original_max)
            scaled_base = self.base
            if factor is not None and factor > 0 and self.rotary_dim > 2:
                scaled_base = self.base * (
                    (factor * seq_len / original_max) - (factor - 1.0)
                ) ** (self.rotary_dim / (self.rotary_dim - 2))
            return self._compute_default_inv_freq(base=scaled_base, device=device), 1.0

        if rope_type == "yarn":
            original_max = int(rope_scaling.get("original_max_position_embeddings") or seq_len)
            attention_factor = rope_scaling.get("attention_factor")
            if attention_factor is None:
                effective_factor = factor
                if effective_factor is None or effective_factor <= 0:
                    effective_factor = max(float(seq_len) / float(original_max), 1.0)
                attention_factor = 1.0 if effective_factor <= 1.0 else (0.1 * math.log(effective_factor) + 1.0)

            beta_fast = float(rope_scaling.get("beta_fast") or 32.0)
            beta_slow = float(rope_scaling.get("beta_slow") or 1.0)
            truncate = bool(rope_scaling.get("truncate", True))
            pos_freqs = self.base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=device) / self.rotary_dim)
            inv_freq_extrapolation = 1.0 / pos_freqs
            effective_factor = factor if factor is not None and factor > 0 else max(float(seq_len) / float(original_max), 1.0)
            inv_freq_interpolation = 1.0 / (effective_factor * pos_freqs)

            def find_correction_dim(num_rotations: float) -> float:
                return (
                    self.rotary_dim * math.log(original_max / (num_rotations * 2 * math.pi))
                ) / (2 * math.log(self.base))

            def find_correction_range(low_rot: float, high_rot: float) -> tuple[float, float]:
                low = find_correction_dim(low_rot)
                high = find_correction_dim(high_rot)
                if truncate:
                    low = math.floor(low)
                    high = math.ceil(high)
                return max(low, 0), min(high, self.rotary_dim - 1)

            def linear_ramp_factor(min_pos: float, max_pos: float) -> torch.Tensor:
                if min_pos == max_pos:
                    max_pos += 1e-3
                linear = (torch.arange(self.rotary_dim // 2, dtype=torch.float32, device=device) - min_pos) / (max_pos - min_pos)
                return torch.clamp(linear, 0, 1)

            low, high = find_correction_range(beta_fast, beta_slow)
            inv_freq_extrapolation_factor = 1.0 - linear_ramp_factor(low, high)
            inv_freq = (
                inv_freq_interpolation * (1.0 - inv_freq_extrapolation_factor)
                + inv_freq_extrapolation * inv_freq_extrapolation_factor
            )
            return inv_freq, float(attention_factor)

        if rope_type == "longrope":
            original_max = int(rope_scaling.get("original_max_position_embeddings") or seq_len)
            long_factor = rope_scaling.get("long_factor")
            short_factor = rope_scaling.get("short_factor")
            if long_factor is None or short_factor is None:
                warnings.warn(
                    "longrope configuration is missing short_factor/long_factor; "
                    "falling back to default RoPE.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._compute_default_inv_freq(base=self.base, device=device), 1.0
            scale_factors = long_factor if seq_len > original_max else short_factor
            scale_factors_t = torch.tensor(scale_factors, dtype=torch.float32, device=device)
            inv_freq_shape = torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=device) / self.rotary_dim
            inv_freq = 1.0 / (scale_factors_t * (self.base ** inv_freq_shape))
            attention_factor = rope_scaling.get("attention_factor")
            if attention_factor is None:
                effective_factor = factor
                if effective_factor is None:
                    effective_factor = max(float(seq_len) / float(original_max), 1.0)
                attention_factor = 1.0 if effective_factor <= 1.0 else math.sqrt(
                    1.0 + math.log(effective_factor) / math.log(original_max)
                )
            return inv_freq, float(attention_factor)

        if rope_type == "llama3":
            inv_freq = self._compute_default_inv_freq(base=self.base, device=device)
            effective_factor = float(factor or 1.0)
            low_freq_factor = float(rope_scaling["low_freq_factor"])
            high_freq_factor = float(rope_scaling["high_freq_factor"])
            original_max = int(rope_scaling["original_max_position_embeddings"])

            low_freq_wavelen = original_max / low_freq_factor
            high_freq_wavelen = original_max / high_freq_factor

            wavelen = 2 * math.pi / inv_freq
            inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / effective_factor, inv_freq)
            smooth_factor = (original_max / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smoothed_inv_freq = (1.0 - smooth_factor) * inv_freq_llama / effective_factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
            inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
            return inv_freq_llama, 1.0

        warnings.warn(
            f"Unsupported rope_scaling type '{rope_type}', falling back to default RoPE.",
            RuntimeWarning,
            stacklevel=2,
        )
        return self._compute_default_inv_freq(base=self.base, device=device), 1.0

    def _build_cache(self, seq_len: int) -> None:
        device = self.cos_sin_cache.device if self.cos_sin_cache.numel() else None
        inv_freq, self.attention_scaling = self._compute_inv_freq(seq_len, device=device)
        t = torch.arange(seq_len, dtype=torch.float, device=inv_freq.device)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling
        cache = torch.cat((cos, sin), dim=-1)
        self.cos_sin_cache = cache
        self.max_seq_len_cached = seq_len

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
        max_position = int(positions.max().item()) + 1
        if max_position > self.max_seq_len_cached:
            self._build_cache(max_position)
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


def _normalize_rope_scaling(
    rope_scaling: dict | None,
    *,
    max_position_embeddings: int,
    base: float,
) -> dict | None:
    if rope_scaling is None:
        return None
    if not isinstance(rope_scaling, dict):
        raise TypeError(f"rope_scaling must be a dict or None, got {type(rope_scaling).__name__}")

    normalized = dict(rope_scaling)
    rope_type = normalized.get("rope_type", normalized.get("type", "default"))
    normalized["rope_type"] = str(rope_type).lower()
    normalized.setdefault("rope_theta", float(normalized.get("base", base)))
    normalized.setdefault("original_max_position_embeddings", max_position_embeddings)
    return normalized


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
) -> RotaryEmbedding:
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
    return RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position,
        base=base,
        rope_scaling=rope_scaling,
    )
