"""Sampling parameters module."""

from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Sampling parameters for text generation."""
    temperature: float = 1.0  # Sampling temperature
    draft_temperature: float | None = None  # Sampling temperature for draft model
    max_new_tokens: int = 256  # Maximum number of new tokens to generate
    ignore_eos: bool = False  # Whether to ignore end-of-sequence token
