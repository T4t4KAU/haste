"""Asynchronous speculator for LLM inference."""

import torch
from transformers import AutoTokenizer

from haste.engine.draft_runner import DraftRunner
from haste.engine.helpers.speculate_types import SpeculateResult, SpeculatorBase, VerifyResult
from haste.engine.sequence import Sequence
from haste.utils.misc import decode_tokens


class SpeculatorAsync(SpeculatorBase):
    """Asynchronous speculator for LLM inference."""

    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        draft_model_runner: DraftRunner,
        tokenizer: AutoTokenizer,
        verbose: bool = False,
    ):
        """Initialize asynchronous speculator.
        
        Args:
            lookahead (int): Number of tokens to speculate
            device (torch.device): Device to run on
            draft_model_runner (DraftRunner): Draft model runner
            tokenizer (AutoTokenizer): Tokenizer
            verbose (bool): Whether to enable verbose logging
        """
        super().__init__(lookahead, device)
        self.draft_model_runner = draft_model_runner
        self.tokenizer = tokenizer
        self.verbose = verbose

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        """Prefill operation for asynchronous speculator.
        
        Args:
            seqs (list[Sequence]): List of sequences
            verify_result (VerifyResult): Verify result
            
        Returns:
            SpeculateResult: Speculate result
        """
        del verify_result
        self.draft_model_runner.submit_prefill(seqs)
        return SpeculateResult([], [])

    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        """Speculate operation for asynchronous speculator.
        
        Args:
            seqs (list[Sequence]): List of sequences
            verify_result (VerifyResult): Verify result
            
        Returns:
            SpeculateResult: Speculate result
        """
        del verify_result
        response = self.draft_model_runner.request_speculation(seqs)

        if self.verbose and seqs:
            print(f"[async_speculator] batch={len(seqs)} cache_hits={int(response.cache_hits.sum().item())}/{len(seqs)}", flush=True)

        speculation_rows = response.speculations.tolist()
        for seq, row in zip(seqs, speculation_rows):
            seq.token_ids.extend(row)
            seq.num_tokens += len(row)
            seq.last_token = row[-1]
            seq.num_draft_cached_tokens += len(row)

        if self.verbose:
            for idx, row in enumerate(speculation_rows[:2]):
                print(f"[async_speculator] seq{idx} -> {decode_tokens(row, self.tokenizer)}", flush=True)

        return SpeculateResult(
            speculations=response.speculations,
            logits_q=response.logits_q,
            cache_hits=response.cache_hits,
            from_q_mask=response.from_q_mask,
            lookahead=response.lookahead,
        )

    def report_verify_feedback(
        self,
        verify_elapsed: float,
        batch_size: int,
        accepted_fraction: float,
    ) -> None:
        """Report verify feedback to draft runner.
        
        Args:
            verify_elapsed (float): Verify elapsed time
            batch_size (int): Batch size
            accepted_fraction (float): Accepted fraction
        """
        self.draft_model_runner.report_verify_feedback(
            verify_elapsed,
            batch_size,
            accepted_fraction,
        )
