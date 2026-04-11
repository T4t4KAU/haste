"""Inference step implementations for LLM inference."""

from abc import ABC, abstractmethod
from time import perf_counter
from transformers import AutoTokenizer

from haste.engine.model_runner import ModelRunner
from haste.engine.sequence import Sequence, SequenceStatus
from haste.engine.scheduler import Scheduler
from haste.engine.helpers.speculate_types import SpeculatorBase, VerifierBase, VerifyResult
from haste.utils.misc import decode_tokens

class InferenceStep(ABC):
    """Abstract base class for inference steps."""

    def __init__(self, scheduler: Scheduler):
        """Initialize inference step.
        
        Args:
            scheduler (Scheduler): Scheduler
        """
        self.scheduler = scheduler
    
    @abstractmethod
    def decode(self, seqs: list[Sequence]) -> int:
        """Decode sequences.
        
        Args:
            seqs (list[Sequence]): List of sequences
            
        Returns:
            int: Number of tokens processed
        """
        pass
    
    @abstractmethod
    def prefill(self, seqs: list[Sequence]) -> int:
        """Prefill sequences.
        
        Args:
            seqs (list[Sequence]): List of sequences
            
        Returns:
            int: Number of tokens processed
        """
        pass

class AutoRegressiveStep(InferenceStep):
    """Auto-regressive inference step."""

    def __init__(
        self,
        scheduler: Scheduler,
        model_runner: ModelRunner,
        tokenizer: AutoTokenizer,
        metrics: dict | None = None,
    ):
        """Initialize auto-regressive step.
        
        Args:
            scheduler (Scheduler): Scheduler
            model_runner (ModelRunner): Model runner
            tokenizer (AutoTokenizer): Tokenizer
            metrics (dict | None): Metrics dictionary
        """
        super().__init__(scheduler)
        self.model_runner = model_runner
        self.tokenizer = tokenizer
        self.metrics = metrics
    
    def step(self, seqs: list[Sequence], is_prefill: bool) -> int:
        """Runs a single step of inference on the given sequences.
        
        Args:
            seqs (list[Sequence]): List of sequences
            is_prefill (bool): Whether this is a prefill step
            
        Returns:
            int: Number of tokens processed
        """
        if __debug__:
            print(f'[auto_regressive_step] is_prefill={is_prefill}', flush=True)
        
        token_ids = self.model_runner.run(seqs, is_prefill)
        
        if __debug__:
            decoded_tokens = decode_tokens(token_ids, self.tokenizer)
            print(f'[auto_regressive_step] decoded_tokens={decoded_tokens}', flush=True)
        
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        
        return len(seqs) if not is_prefill else sum(len(seq) for seq in seqs)
       
    def prefill(self, seqs: list[Sequence]) -> int:
        """Prefills the given sequences with the model's prefix.
        
        Args:
            seqs (list[Sequence]): List of sequences
            
        Returns:
            int: Number of tokens processed
        """
        return self.step(seqs, is_prefill=True)
    
    def decode(self, seqs: list[Sequence]) -> int:
        """Decodes the given sequences.
        
        Args:
            seqs (list[Sequence]): List of sequences
            
        Returns:
            int: Number of tokens processed
        """
        return self.step(seqs, is_prefill=False)
    

class SpecDecodeStep(InferenceStep):
    """Speculative decoding step."""

    def __init__(
        self, 
        scheduler: Scheduler, 
        speculator: SpeculatorBase, 
        verifier: VerifierBase, 
        tokenizer: AutoTokenizer, 
        async_spec: bool,
        metrics: dict | None = None,
    ):
        """Initialize speculative decoding step.
        
        Args:
            scheduler (Scheduler): Scheduler
            speculator (SpeculatorBase): Speculator
            verifier (VerifierBase): Verifier
            tokenizer (AutoTokenizer): Tokenizer
            async_spec (bool): Whether to use asynchronous speculation
            metrics (dict | None): Metrics dictionary
        """
        super().__init__(scheduler)
        self.speculator = speculator
        self.verifier = verifier
        self.tokenizer = tokenizer
        self.async_spec = async_spec
        self.metrics = metrics
        
    def prefill(self, seqs: list[Sequence]) -> int:
        """Prefills the given sequences with the model's prefix.
        
        Args:
            seqs (list[Sequence]): List of sequences
            
        Returns:
            int: Number of tokens processed
        """

        speculator_time = 0.0
        verifier_time = 0.0
        if self.async_spec:
            empty_verify_result = VerifyResult([], [])
            t0 = perf_counter()
            self.speculator.prefill(seqs, empty_verify_result)
            speculator_time = perf_counter() - t0
            t1 = perf_counter()
            verify_result = self.verifier.prefill(seqs)
            verifier_time = perf_counter() - t1
        else:
            t0 = perf_counter()
            verify_result = self.verifier.prefill(seqs)
            verifier_time = perf_counter() - t0
            t1 = perf_counter()
            self.speculator.prefill(seqs, verify_result)
            speculator_time = perf_counter() - t1

        if self.metrics is not None:
            self.metrics["prefill_speculator_times"].append(speculator_time)
            self.metrics["prefill_verifier_times"].append(verifier_time)
        
        for seq in seqs:
            assert seq.recovery_token_id is not None
            seq.num_cached_tokens = seq.num_prompt_tokens
            seq.num_draft_cached_tokens = seq.num_prompt_tokens
        
        return sum(len(seq) for seq in seqs)
    
    def decode(self, seqs: list[Sequence]) -> int:
        """Decodes the given sequences using speculative decoding.
        
        Args:
            seqs (list[Sequence]): List of sequences
            
        Returns:
            int: Number of tokens processed
        """
        # Save original sequence states
        saved = [(len(seq.token_ids), seq.num_tokens, seq.last_token, 
                  seq.num_draft_cached_tokens, seq.num_cached_tokens) for seq in seqs]
        in_verify_result = VerifyResult(
            new_suffixes=[],
            recovery_tokens=[],
        )
        
        #### STEP 1: SPECULATE ####
        speculate_start = perf_counter()
        speculate_result = self.speculator.speculate(seqs, in_verify_result)
        speculate_elapsed = perf_counter() - speculate_start
        if self.metrics is not None:
            self.metrics["speculate_times"].append(speculate_elapsed)
            effective_lookahead = speculate_result.lookahead if speculate_result.lookahead is not None else self.speculator.lookahead
            self.metrics["effective_lookaheads"].extend([effective_lookahead] * len(seqs))
        
        if __debug__:
            speculations = speculate_result.speculations
            print(f'[speculator_decode] speculations={speculations}', flush=True)
            speculations_list = speculations.tolist()
            
            for i, speculation in enumerate(speculations_list):
                decoded_tokens = decode_tokens(speculation, self.tokenizer)
                print(f'[speculator_decode]  speculation {i}: {decoded_tokens}', flush=True)

        #### STEP 2: VERIFY ####
        verify_start = perf_counter()
        verify_result = self.verifier.verify(seqs, speculate_result)
        verify_elapsed = perf_counter() - verify_start
        if self.metrics is not None:
            self.metrics["verify_times"].append(verify_elapsed)
        if hasattr(self.speculator, "report_verify_feedback"):
            effective_lookahead = speculate_result.lookahead if speculate_result.lookahead is not None else self.speculator.lookahead
            accepted_spec_tokens = sum(max(0, len(new_suffix) - 1) for new_suffix in verify_result.new_suffixes)
            accepted_fraction = accepted_spec_tokens / max(1, len(seqs) * effective_lookahead)
            self.speculator.report_verify_feedback(verify_elapsed, len(seqs), accepted_fraction)

        
        if __debug__:
            recovery_tokens = verify_result.recovery_tokens
            new_suffixes = verify_result.new_suffixes
            for i, new_suffix in enumerate(new_suffixes):
                decoded_tokens = decode_tokens(new_suffix + [recovery_tokens[i]], self.tokenizer)
                print(f'[speculator_decode] verification {i}: {decoded_tokens}', flush=True)
        
        # Restore sequence states, undoing changes from speculation and verification
        rollback_start = perf_counter()
        for seq, (orig_len, orig_nt, orig_lt, orig_ndc, orig_nct) in zip(seqs, saved):
            del seq.token_ids[orig_len:]
            seq.num_tokens = orig_nt
            seq.last_token = orig_lt
            seq.num_draft_cached_tokens = orig_ndc
            seq.num_cached_tokens = orig_nct
        rollback_elapsed = perf_counter() - rollback_start
        if self.metrics is not None:
            self.metrics["rollback_times"].append(rollback_elapsed)
        
        
        #### STEP 3: POSTPROCESS ####
        postprocess_start = perf_counter()
        self.scheduler.postprocess_speculate(
            seqs,
            verify_result.new_suffixes,
            verify_result.recovery_tokens,
        )
        postprocess_elapsed = perf_counter() - postprocess_start
        if self.metrics is not None:
            self.metrics["postprocess_times"].append(postprocess_elapsed)
        
        return sum(len(seq) for seq in verify_result.new_suffixes)
        
