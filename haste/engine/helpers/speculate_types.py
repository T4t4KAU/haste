"""Speculate types and base classes."""

from dataclasses import dataclass
import torch
from haste.engine.sequence import Sequence
from abc import ABC, abstractmethod


@dataclass
class SpeculateResult:
    """Result of speculation."""
    speculations: torch.Tensor  # Speculated tokens
    logits_q: torch.Tensor | None  # Logits from query
    cache_hits: torch.Tensor | None = None  # Cache hit indicators
    from_q_mask: torch.Tensor | None = None  # Mask indicating which tokens are from query
    lookahead: int | None = None  # Lookahead value


@dataclass
class VerifyResult:
    """Result of verification."""
    new_suffixes: list[list[int]]  # New suffixes after verification
    recovery_tokens: list[int]  # Recovery tokens
    outcome_keys: list[tuple[int, int, int]] | None = None  # Outcome keys


class SpeculatorBase(ABC):
    """Base class for speculators."""
    def __init__(self, lookahead: int, device: torch.device):
        """Initialize speculator.
        
        Args:
            lookahead (int): Number of tokens to speculate
            device (torch.device): Device to use
        """
        self.lookahead = lookahead
        self.device = device

    @abstractmethod
    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        """Prefill speculator.
        
        Args:
            seqs (list[Sequence]): List of sequences
            verify_result (VerifyResult): Verification result
            
        Returns:
            SpeculateResult: Speculation result
        """
        pass

    @abstractmethod
    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        """Speculate tokens.
        
        Args:
            seqs (list[Sequence]): List of sequences
            verify_result (VerifyResult): Verification result
            
        Returns:
            SpeculateResult: Speculation result
        """
        pass


class VerifierBase(ABC):
    """Base class for verifiers."""
    def __init__(self, lookahead: int, device: torch.device):
        """Initialize verifier.
        
        Args:
            lookahead (int): Number of tokens to verify
            device (torch.device): Device to use
        """
        self.lookahead = lookahead
        self.device = device

    @abstractmethod
    def prefill(self, seqs: list[Sequence], eagle: bool = False) -> VerifyResult:
        """Prefill verifier.
        
        Args:
            seqs (list[Sequence]): List of sequences
            eagle (bool): Whether to use eagle mode
            
        Returns:
            VerifyResult: Verification result
        """
        pass

    @abstractmethod
    def verify(self, seqs: list[Sequence], speculate_result: SpeculateResult, eagle: bool = False) -> VerifyResult:
        """Verify speculated tokens.
        
        Args:
            seqs (list[Sequence]): List of sequences
            speculate_result (SpeculateResult): Speculation result
            eagle (bool): Whether to use eagle mode
            
        Returns:
            VerifyResult: Verification result
        """
        pass
