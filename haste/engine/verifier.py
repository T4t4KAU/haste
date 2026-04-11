"""Verifier for LLM speculative decoding."""

from time import perf_counter

import torch
from transformers import AutoTokenizer

from haste.engine.helpers.speculate_types import (
    SpeculateResult,
    VerifyResult,
    VerifierBase,
)
from haste.engine.model_runner import ModelRunner
from haste.engine.sequence import Sequence
from haste.utils.verify import verify


class Verifier(VerifierBase):
    """Verifier for LLM speculative decoding."""

    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        target_model_runner: ModelRunner,
        sampler_x: float | None = None,
        async_fan_out: int | None = None,
        jit_speculate: bool = False,
        tokenizer: AutoTokenizer | None = None,
        metrics: dict | None = None,
    ):
        """Initialize verifier.
        
        Args:
            lookahead (int): Number of tokens to verify
            device (torch.device): Device to run on
            target_model_runner (ModelRunner): Target model runner
            sampler_x (float | None): Sampler x parameter
            async_fan_out (int | None): Asynchronous fan-out
            jit_speculate (bool): Whether to use JIT speculation
            tokenizer (AutoTokenizer | None): Tokenizer
            metrics (dict | None): Metrics dictionary
        """
        super().__init__(lookahead=lookahead, device=device)
        self.target_model_runner = target_model_runner
        self.sampler_x = sampler_x
        self.async_fan_out = async_fan_out
        self.jit_speculate = jit_speculate
        self.tokenizer = tokenizer
        self.metrics = metrics

    def prefill(self, seqs: list[Sequence], eagle: bool = False) -> VerifyResult:
        """Prefill operation for verifier.
        
        Args:
            seqs (list[Sequence]): List of sequences
            eagle (bool): Eagle mode flag
            
        Returns:
            VerifyResult: Verify result
        """
        del eagle
        token_ids = self.target_model_runner.run(seqs, True)
        for seq, token_id in zip(seqs, token_ids):
            seq.recovery_token_id = token_id

        return VerifyResult([], [seq.recovery_token_id for seq in seqs])

    def verify(
        self,
        seqs: list[Sequence],
        speculate_result: SpeculateResult,
        eagle: bool = False,
    ) -> VerifyResult:
        """Verify speculative tokens.
        
        Args:
            seqs (list[Sequence]): List of sequences
            speculate_result (SpeculateResult): Speculate result
            eagle (bool): Eagle mode flag
            
        Returns:
            VerifyResult: Verify result
        """
        del eagle

        batch_size = len(seqs)
        lookahead = speculate_result.lookahead if speculate_result.lookahead is not None else self.lookahead
        t0 = perf_counter()
        logits_p_flat = self.target_model_runner.run(
            seqs,
            False,
            False,
            True,
            verify_lookahead=lookahead,
        )

        for seq in seqs:
            seq.num_cached_tokens += lookahead + 1

        logits_p = logits_p_flat.view(batch_size, lookahead + 1, -1)
        speculations = speculate_result.speculations.to(self.device)
        verify_mask = speculate_result.from_q_mask
        if verify_mask is not None:
            verify_mask = verify_mask.to(self.device)

        temperatures_target = torch.tensor(
            [seq.temperature for seq in seqs],
            dtype=torch.float32,
            device=self.device,
        )
        temperatures_draft = torch.tensor(
            [
                seq.draft_temperature if seq.draft_temperature is not None else seq.temperature
                for seq in seqs
            ],
            dtype=torch.float32,
            device=self.device,
        )
        needs_q_logits = bool(((temperatures_target > 0) | (temperatures_draft > 0)).any().item())
        logits_q = None
        if needs_q_logits:
            if speculate_result.logits_q is None:
                raise RuntimeError("Verifier requires draft logits for non-greedy verification, but none were provided.")
            logits_q = speculate_result.logits_q.to(self.device)

        new_suffixes, recovery_tokens = verify(
            logits_p=logits_p,
            logits_q=logits_q,
            speculations=speculations,
            temperatures_target=temperatures_target,
            temperatures_draft=temperatures_draft,
            cache_hits=verify_mask,
            sampler_x=self.sampler_x,
            async_fan_out=self.async_fan_out,
            jit_speculate=self.jit_speculate,
        )

        if self.metrics is not None:
            self.metrics["target_verify_times"].append(perf_counter() - t0)
            self.metrics["accepted_suffix_lens_with_recovery"].extend(len(s) for s in new_suffixes)
            if speculate_result.cache_hits is not None:
                cache_hits_cpu = speculate_result.cache_hits.to(torch.float32).cpu()
                self.metrics["cache_hits"].append(cache_hits_cpu.mean().item())
                for hit, suffix in zip(cache_hits_cpu.tolist(), new_suffixes):
                    if hit >= 0.5:
                        self.metrics["accepted_suffix_lens_on_hit"].append(len(suffix))
                    else:
                        self.metrics["accepted_suffix_lens_on_miss"].append(len(suffix))

        outcome_keys = [
            (seq.seq_id, len(new_suffix) - 1, recovery_token)
            for seq, new_suffix, recovery_token in zip(seqs, new_suffixes, recovery_tokens)
        ]
        return VerifyResult(
            new_suffixes=new_suffixes,
            recovery_tokens=recovery_tokens,
            outcome_keys=outcome_keys,
        )
