"""Synchronous speculator for LLM inference."""

import torch

from haste.engine.sequence import Sequence
from haste.engine.model_runner import ModelRunner
from haste.engine.helpers.speculate_types import SpeculateResult, VerifyResult, SpeculatorBase

class SpeculatorSync(SpeculatorBase):
    """Synchronous speculator for LLM inference."""

    def __init__(self, lookahead: int, device: torch.device, draft_model_runner: ModelRunner):
        """Initialize synchronous speculator.
        
        Args:
            lookahead (int): Number of tokens to speculate
            device (torch.device): Device to run on
            draft_model_runner (ModelRunner): Draft model runner
        """
        super().__init__(lookahead, device)
        self.draft_model_runner = draft_model_runner

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        """Prefill operation for synchronous speculator.
        
        Args:
            seqs (list[Sequence]): List of sequences
            verify_result (VerifyResult): Verify result
            
        Returns:
            SpeculateResult: Speculate result
        """
        self.draft_model_runner.run(seqs, True)

        if len(seqs) > 0:
            print(f"[PREFILL] seq0 prompt_len={seqs[0].num_prompt_tokens} recovery={seqs[0].recovery_token_id}", flush=True)

        return SpeculateResult([], [], lookahead=self.lookahead)

    # Speculative inference
    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        """Speculate operation for synchronous speculator.
        
        Args:
            seqs (list[Sequence]): List of sequences
            verify_result (VerifyResult): Verify result
            
        Returns:
            SpeculateResult: Speculate result
        """
        batch_size = len(seqs)

        # Allocate memory for speculative tokens
        speculations = torch.zeros(
            batch_size,
            self.lookahead + 1,
            dtype=torch.int64,
            device=self.device,
        )

        logits_q = []

        # Batch write to GPU
        recovery_tokens = []
        for i, seq in enumerate(seqs):
            if seq.recovery_token_id is None:
                raise ValueError(f"recovery_token_id is None for seq {i}")
            recovery_tokens.append(seq.recovery_token_id)
            seq.append_token(seq.recovery_token_id)

        speculations[:, 0] = torch.tensor(recovery_tokens, dtype=torch.int64, device=self.device)

        for k in range(self.lookahead + 1):
            token_ids, step_logits_q = self.draft_model_runner.run(seqs, False, True, True)
            for s in seqs:
                s.num_draft_cached_tokens += 1

            if k == self.lookahead:
                break

            logits_q.append(step_logits_q)

            for i, (seq, token_id) in enumerate(zip(seqs, token_ids)):
                seq.append_token(token_id)

            speculations[:, k + 1] = torch.tensor(token_ids, dtype=torch.int64, device=self.device)

        logits_q = torch.stack(logits_q, dim=1)  # (batch_size, lookahead, vocab_size)

        return SpeculateResult(speculations, logits_q, lookahead=self.lookahead)
