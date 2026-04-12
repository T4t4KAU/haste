"""LLM Engine module for handling language model inference with speculative decoding."""

from dataclasses import fields
from time import perf_counter

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from haste.config import Config
from haste.engine.draft_runner import DraftRunner
from haste.engine.model_runner import ModelRunner
from haste.engine.scheduler import Scheduler
from haste.engine.sequence import Sequence
from haste.engine.speculator_async import SpeculatorAsync
from haste.engine.speculator_sync import SpeculatorSync
from haste.engine.step import AutoRegressiveStep, InferenceStep, SpecDecodeStep
from haste.engine.verifier import Verifier
from haste.sampling_params import SamplingParams
from haste.utils.misc import infer_model_family
from haste.utils.profiling import fresh_metrics, reset_metrics, summarize_numeric_series

# Initialize metrics dictionary for performance tracking
METRICS = fresh_metrics()


class LLMEngine:
    """LLM Engine class for managing language model inference with optional speculative decoding.
    
    This class handles the entire inference process, including model loading, request scheduling,
    and generating responses. It supports both regular autoregressive decoding and speculative decoding
    for improved performance.
    """
    
    def __init__(self, model: str, **kwargs):
        """Initialize the LLM Engine with the specified model.
        
        Args:
            model (str): Path or name of the target language model
            **kwargs: Additional configuration parameters
        """
        # Extract configuration fields from kwargs
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model=model, **config_kwargs)
        
        # Set block size for sequence management
        Sequence.block_size = self.config.kvcache_block_size

        # Validate model family compatibility if speculative decoding is enabled
        if self.config.speculate:
            target_family = infer_model_family(self.config.model)
            draft_family = infer_model_family(self.config.draft_model)
            assert target_family == draft_family, "ERROR: target and draft model families must match"

        # Initialize target model runner
        self.model_runner = ModelRunner(
            self.config,
            event=None,
            device=self.config.target_device,
            is_draft=False,
        )

        # Initialize draft model runner if speculative decoding is enabled
        self.draft_runner = None
        self.draft_config = None
        if self.config.speculate:
            self.draft_runner = DraftRunner(self.config)
            self.draft_config = self.draft_runner.draft_config

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
            use_fast=True,
            local_files_only=True,
        )
        
        # Set EOS token ID from tokenizer
        self.config.eos = self.tokenizer.eos_token_id
        
        # Initialize scheduler
        self.scheduler = Scheduler(self.config, draft_config=self.draft_config)

    def exit(self):
        """Clean up resources and shut down runners."""
        if self.draft_runner is not None:
            self.draft_runner.shutdown()
        if self.model_runner is not None:
            self.model_runner.shutdown()

    def shutdown(self):
        """Alias for exit method."""
        self.exit()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """Add a new inference request to the scheduler.
        
        Args:
            prompt (str | list[int]): Input prompt as string or token IDs
            sampling_params (SamplingParams): Sampling parameters for generation
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        self.scheduler.add(Sequence(prompt, sampling_params))

    def step(self, infer_step: InferenceStep):
        """Execute a single inference step.
        
        Args:
            infer_step (InferenceStep): Inference step to execute
            
        Returns:
            list[tuple[int, list[int]]]: List of (sequence ID, completion token IDs) for finished sequences
        """
        # Measure scheduling time
        schedule_start = perf_counter()
        seqs, is_prefill = self.scheduler.schedule()
        schedule_elapsed = perf_counter() - schedule_start
        
        # Execute inference
        start = perf_counter()
        ttl_tokens = infer_step.prefill(seqs) if is_prefill else infer_step.decode(seqs)
        infer_elapsed = perf_counter() - start
        elapsed = schedule_elapsed + infer_elapsed

        # Update metrics
        METRICS["scheduler_times"].append(schedule_elapsed)
        METRICS["target_step_times"].append(elapsed)

        if is_prefill:
            METRICS["prefill_total_time"] += elapsed
            METRICS["prefill_total_tokens"] += ttl_tokens
            METRICS["prefill_step_times"].append(elapsed)
            METRICS["prefill_batch_sizes"].append(len(seqs))
            METRICS["prefill_step_tokens"].append(ttl_tokens)
        else:
            METRICS["decode_total_time"] += elapsed
            METRICS["decode_total_tokens"] += ttl_tokens
            METRICS["decode_step_times"].append(elapsed)
            METRICS["decode_batch_sizes"].append(len(seqs))
            METRICS["decode_step_tokens"].append(ttl_tokens)

        # Collect finished sequences
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs

    def is_finished(self):
        """Check if all requests have been completed.
        
        Returns:
            bool: True if all requests are finished, False otherwise
        """
        return self.scheduler.is_finished()

    def create_inference_step(self) -> InferenceStep:
        """Create an appropriate inference step based on configuration.
        
        Returns:
            InferenceStep: Inference step instance
        """
        if not self.config.speculate:
            # Create regular autoregressive step
            return AutoRegressiveStep(
                self.scheduler,
                self.model_runner,
                self.tokenizer,
                metrics=METRICS,
            )

        # Create speculative decoding components
        if self.config.draft_async:
            speculator = SpeculatorAsync(
                lookahead=self.config.speculate_k,
                device=self.config.draft_device,
                draft_model_runner=self.draft_runner,
                tokenizer=self.tokenizer,
                verbose=self.config.verbose,
            )
        else:
            speculator = SpeculatorSync(
                lookahead=self.config.speculate_k,
                device=self.config.draft_device,
                draft_model_runner=self.draft_runner,
            )

        verifier = Verifier(
            lookahead=self.config.speculate_k,
            device=self.config.target_device,
            target_model_runner=self.model_runner,
            sampler_x=self.config.sampler_x,
            async_fan_out=self.config.async_fan_out,
            jit_speculate=self.config.jit_speculate,
            tokenizer=self.tokenizer,
            metrics=METRICS,
        )
        
        # Create speculative decode step
        return SpecDecodeStep(
            scheduler=self.scheduler,
            speculator=speculator,
            verifier=verifier,
            tokenizer=self.tokenizer,
            async_spec=self.config.draft_async,
            metrics=METRICS,
        )

    def _collect_runner_profiles(self) -> dict:
        """Collect performance profiles from runners.
        
        Returns:
            dict: Dictionary of runner profiles
        """
        profiles = {
            "target": self.model_runner.profile_summary(),
        }
        if self.draft_runner is not None:
            profiles["draft_model"] = self.draft_runner.profile_summary()
            profiles["draft_worker"] = self.draft_runner.worker_profile_summary()
        return profiles

    def generate(
        self,
        prompts: list[str] | list[list[int]] | str | list[int],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        stream_callback=None,
        log_metrics: bool = True,
    ):
        """Generate responses for the given prompts.
        
        Args:
            prompts (list[str] | list[list[int]] | str | list[int]): Input prompts
            sampling_params (SamplingParams | list[SamplingParams]): Sampling parameters
            use_tqdm (bool, optional): Whether to use progress bar. Defaults to True.
            stream_callback (callable, optional): Callback for streaming output. Defaults to None.
            log_metrics (bool, optional): Whether to print metrics after generation. Defaults to True.
            
        Returns:
            tuple[list[dict], dict]: List of generated outputs and metrics
        """
        # Reset metrics
        reset_metrics(METRICS)
        METRICS["engine_wall_time"] = 0.0

        # Normalize prompts to list format
        if isinstance(prompts, (str, list)) and prompts and isinstance(prompts[0], int):
            prompts = [prompts]
        elif isinstance(prompts, str):
            prompts = [prompts]

        # Normalize sampling params to list format
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # Reset profiles
        self.model_runner.reset_profile()
        if self.draft_runner is not None:
            self.draft_runner.reset_profile()

        # Add requests
        METRICS["num_requests"] = len(prompts)
        for prompt, params in zip(prompts, sampling_params):
            self.add_request(prompt, params)

        # Create inference step
        inference_step = self.create_inference_step()
        outputs = {}
        stream_lengths = {}
        generate_start = perf_counter()

        # Initialize progress bar
        progress = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True) if use_tqdm else None
        step_idx = 0
        max_steps = self.config.max_steps if self.config.max_steps is not None else float("inf")

        # Main inference loop
        while not self.is_finished() and step_idx < max_steps:
            step_idx += 1
            step_outputs = self.step(inference_step)

            # Handle streaming callback
            if stream_callback is not None:
                for seq in self.scheduler.running:
                    cur = seq.num_completion_tokens
                    prev = stream_lengths.get(seq.seq_id, 0)
                    if cur > prev:
                        stream_callback(seq.seq_id, seq.completion_token_ids[prev:cur])
                        stream_lengths[seq.seq_id] = cur

            # Process finished sequences
            for seq_id, token_ids in step_outputs:
                outputs[seq_id] = token_ids
                if stream_callback is not None:
                    prev = stream_lengths.get(seq_id, 0)
                    if len(token_ids) > prev:
                        stream_callback(seq_id, token_ids[prev:])
                if progress is not None:
                    progress.update(1)

        # Close progress bar
        if progress is not None:
            progress.close()

        # Update metrics
        METRICS["engine_wall_time"] = perf_counter() - generate_start
        METRICS["completed_requests"] = len(outputs)
        METRICS["num_engine_steps"] = step_idx
        METRICS["runner_profiles"] = self._collect_runner_profiles()

        # Format outputs
        ordered = [outputs[seq_id] for seq_id in sorted(outputs)]
        formatted = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in ordered]

        # Log metrics if not streaming
        if stream_callback is None and log_metrics:
            self.log_metrics()
        return formatted, METRICS

    def log_metrics(self):
        """Log performance metrics."""
        if METRICS["prefill_total_time"] > 0:
            print(
                f"Final Prefill Throughput: {int(METRICS['prefill_total_tokens'] / METRICS['prefill_total_time'])}tok/s",
                flush=True,
            )
        if METRICS["decode_total_time"] > 0:
            print(
                f"Final Decode Throughput: {int(METRICS['decode_total_tokens'] / METRICS['decode_total_time'])}tok/s",
                flush=True,
            )

        if self.config.speculate and METRICS["accepted_suffix_lens_with_recovery"]:
            total = sum(METRICS["accepted_suffix_lens_with_recovery"])
            steps = len(METRICS["accepted_suffix_lens_with_recovery"])
            print(f"[metrics] Avg Tokens per step (incl recovery): {total / steps:.2f}", flush=True)
            accepted = total - steps
            effective_lookaheads = METRICS.get("effective_lookaheads", [])
            lookahead_budget = sum(effective_lookaheads)
            if not lookahead_budget:
                lookahead_budget = steps * self.config.speculate_k
            print(
                f"[metrics] Avg Fraction of Speculated Tokens Accepted: {accepted / max(1, lookahead_budget):.2f}",
                flush=True,
            )
            print(
                f"[metrics] Avg target time per full step (ms): {sum(METRICS['target_step_times']) * 1000 / len(METRICS['target_step_times']):.2f}",
                flush=True,
            )
            if METRICS["target_verify_times"]:
                print(
                    f"[metrics] Avg target verify time (ms): {sum(METRICS['target_verify_times']) * 1000 / len(METRICS['target_verify_times']):.2f}",
                    flush=True,
                )
            if self.config.draft_async and METRICS["cache_hits"]:
                print(
                    f"[metrics] Avg Cache Hits: {sum(METRICS['cache_hits']) / len(METRICS['cache_hits']):.2f}",
                    flush=True,
                )
            if self.config.draft_async and self.config.async_auto_tune:
                draft_worker = METRICS.get("runner_profiles", {}).get("draft_worker", {})
                if draft_worker and self.config.verbose:
                    avg_k = draft_worker.get("effective_lookahead", {}).get("mean")
                    avg_f = draft_worker.get("effective_fan_out_cap", {}).get("mean")
                    final_k = draft_worker.get("final_effective_k")
                    final_f = draft_worker.get("final_effective_f")
                    avg_k_text = f"{avg_k:.2f}" if avg_k is not None else "n/a"
                    avg_f_text = f"{avg_f:.2f}" if avg_f is not None else "n/a"
                    print(
                        "[auto_tune_kf] "
                        f"static_k={self.config.speculate_k} static_f={self.config.async_fan_out} "
                        f"final_k={final_k} final_f={final_f} "
                        f"avg_k={avg_k_text} avg_f={avg_f_text}",
                        flush=True,
                    )
            if METRICS["speculate_times"] and METRICS["verify_times"]:
                speculate = summarize_numeric_series(METRICS["speculate_times"], scale=1000.0)
                verify = summarize_numeric_series(METRICS["verify_times"], scale=1000.0)
                post = summarize_numeric_series(METRICS["postprocess_times"], scale=1000.0)
                print(
                    f"[profile] Decode step ms: speculate={speculate.get('mean', 0.0):.2f} "
                    f"verify={verify.get('mean', 0.0):.2f} postprocess={post.get('mean', 0.0):.2f}",
                    flush=True,
                )
