"""Model runner for LLM inference."""

import psutil
import torch
from time import perf_counter

from multiprocessing.synchronize import Event
from transformers import AutoConfig, AutoTokenizer

from haste.config import Config
from haste.engine.helpers.runner_helpers import (
    prepare_block_tables_from_seqs,
    prepare_decode_tensors_from_seqs,
    prepare_prefill_tensors_from_seqs,
)
from haste.engine.sequence import Sequence
from haste.layers.sampler import Sampler
from haste.models.qwen3 import Qwen3ForCausalLM
from haste.utils.context import reset_context, set_context
from haste.utils.loader import load_model
from haste.utils.profiling import build_runner_profile_summary


class ModelRunner:
    """Model runner for LLM inference."""

    def __init__(
        self,
        config: Config,
        event: Event | list[Event] | None,
        device: torch.device,
        is_draft: bool = False,
        init_q=None,
    ):
        """Initialize model runner.
        
        Args:
            config (Config): Engine configuration
            event (Event | list[Event] | None): Event for process synchronization
            device (torch.device): Device to run model on
            is_draft (bool): Whether this is a draft model
            init_q (Queue | None): Queue for initialization
        """
        self.config = config
        self.verbose = config.verbose
        self.device = torch.device(device)
        self.is_draft = is_draft
        self.event = event
        self._exiting = False

        self.model_path = config.draft_model if is_draft else config.model
        self.hf_config = config.draft_hf_config if is_draft else config.target_hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.graphs: dict[str, dict[int, torch.cuda.CUDAGraph]] = {}  # CUDA graphs for decode
        self.graph_vars: dict[str, dict[str, torch.Tensor]] = {}  # Variables for CUDA graphs
        self.graph_bs_list: dict[str, list[int]] = {}  # Batch sizes for CUDA graphs

        if is_draft:
            if config.draft_hf_config.torch_dtype != config.target_hf_config.torch_dtype:
                if self.verbose:
                    print(
                        f"Warning: casting draft dtype {config.draft_hf_config.torch_dtype} "
                        f"to target dtype {config.target_hf_config.torch_dtype}",
                        flush=True,
                    )
                config.draft_hf_config.torch_dtype = config.target_hf_config.torch_dtype

            assert config.draft_hf_config.vocab_size == config.target_hf_config.vocab_size, (
                "ERROR in ModelRunner: target/draft vocab sizes must match"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,
            local_files_only=True,
        )
        self.max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        default_dtype = torch.get_default_dtype()
        if self.device.type == "cuda":
            torch.set_default_device(self.device.type if self.device.index is None else f"cuda:{self.device.index}")
            torch.set_default_dtype(self.hf_config.torch_dtype)

        self.model = self._build_model(self.hf_config)
        load_model(self.model, self.model_path)
        self.sampler = Sampler(sampler_x=config.sampler_x, async_fan_out=config.async_fan_out)
        ModelRunner.reset_profile(self)

        self.warmup_model()
        if self.is_draft:
            self.allocate_kv_cache_draft()
        else:
            self.allocate_kv_cache_target()

        if init_q is not None:
            init_q.put(self.config.num_kvcache_blocks)
            init_q.close()

        if self.device.type == "cuda" and not self.enforce_eager and not self.config.draft_async:
            self._maybe_capture_decode_cudagraph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        ModelRunner.reset_profile(self)

    def reset_profile(self):
        """Reset profile statistics."""
        self._profile = {
            f"{mode}_{suffix}": []
            for mode in ("prefill", "decode", "verify")
            for suffix in (
                "prepare_times",
                "model_times",
                "sample_times",
                "total_times",
                "batch_sizes",
                "input_tokens",
            )
        }
        self._profile["transfer_h2d_times"] = []
        self._profile["transfer_d2h_times"] = []
        self._profile["transfer_h2d_bytes"] = []
        self._profile["transfer_d2h_bytes"] = []
        self._pending_cuda_transfer_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event, int]] = []

    def _queue_transfer_event(
        self,
        direction: str,
        start_event: torch.cuda.Event,
        end_event: torch.cuda.Event,
        num_bytes: int,
    ) -> None:
        """Queue a CUDA transfer event for deferred aggregation."""
        if self.device.type != "cuda":
            return
        self._pending_cuda_transfer_events.append((direction, start_event, end_event, int(num_bytes)))

    def _flush_transfer_events(self) -> None:
        """Flush queued CUDA transfer events into scalar profile series."""
        if self.device.type != "cuda" or not self._pending_cuda_transfer_events:
            return
        torch.cuda.synchronize(self.device)
        pending = self._pending_cuda_transfer_events
        self._pending_cuda_transfer_events = []
        for direction, start_event, end_event, num_bytes in pending:
            elapsed_sec = max(0.0, start_event.elapsed_time(end_event) / 1000.0)
            self._profile[f"transfer_{direction}_times"].append(elapsed_sec)
            self._profile[f"transfer_{direction}_bytes"].append(num_bytes)

    def _move_tensor_to_device(self, tensor: torch.Tensor, *, non_blocking: bool = False) -> torch.Tensor:
        """Move a tensor to the runner device and record CPU->GPU transfer time."""
        if tensor.device == self.device:
            return tensor
        if self.device.type == "cuda" and tensor.device.type == "cpu":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            moved = tensor.to(self.device, non_blocking=non_blocking)
            end_event.record()
            self._queue_transfer_event("h2d", start_event, end_event, tensor.numel() * tensor.element_size())
            return moved
        return tensor.to(self.device, non_blocking=non_blocking)

    def _tensor_to_list(self, tensor: torch.Tensor) -> list:
        """Convert a tensor to a Python list and record GPU->CPU transfer time."""
        if tensor.device.type != "cuda":
            return tensor.tolist()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        values = tensor.tolist()
        end_event.record()
        self._queue_transfer_event("d2h", start_event, end_event, tensor.numel() * tensor.element_size())
        return values

    def _record_profile(
        self,
        mode: str,
        *,
        batch_size: int,
        input_tokens: int,
        prepare_time: float,
        model_time: float,
        sample_time: float,
        total_time: float,
    ):
        """Record profile statistics.
        
        Args:
            mode (str): Mode (prefill, decode, verify)
            batch_size (int): Batch size
            input_tokens (int): Number of input tokens
            prepare_time (float): Preparation time
            model_time (float): Model execution time
            sample_time (float): Sampling time
            total_time (float): Total time
        """
        self._profile[f"{mode}_prepare_times"].append(prepare_time)
        self._profile[f"{mode}_model_times"].append(model_time)
        self._profile[f"{mode}_sample_times"].append(sample_time)
        self._profile[f"{mode}_total_times"].append(total_time)
        self._profile[f"{mode}_batch_sizes"].append(batch_size)
        self._profile[f"{mode}_input_tokens"].append(input_tokens)

    def profile_summary(self) -> dict:
        """Get profile summary.
        
        Returns:
            dict: Profile summary
        """
        self._flush_transfer_events()
        return build_runner_profile_summary(
            self._profile,
            device=str(self.device),
            is_draft=self.is_draft,
        )

    def _build_model(self, hf_config: AutoConfig):
        """Build model.
        
        Args:
            hf_config (AutoConfig): Hugging Face config
            
        Returns:
            nn.Module: Built model
        """
        if hf_config.model_type == "qwen3":
            from haste.models.qwen3 import Qwen3ForCausalLM
            model = Qwen3ForCausalLM(
                config=hf_config,
                draft=self.is_draft,
                speculate=self.config.speculate,
                spec_k=self.config.speculate_k,
                async_fan_out=self.config.async_fan_out,
                draft_async=self.config.draft_async,
                auto_tune_kf=self.config.async_auto_tune,
            )
        elif hf_config.model_type in ["smollm2", "smol_lm2"]:
            from haste.models.smollm2 import SmolLM2ForCausalLM
            model = SmolLM2ForCausalLM(
                config=hf_config,
                draft=self.is_draft,
                speculate=self.config.speculate,
                spec_k=self.config.speculate_k,
                async_fan_out=self.config.async_fan_out,
                draft_async=self.config.draft_async,
                auto_tune_kf=self.config.async_auto_tune,
            )
        elif hf_config.model_type == "llama":
            from haste.models.llama3_2 import Llama3_2ForCausalLM
            model = Llama3_2ForCausalLM(
                config=hf_config,
                draft=self.is_draft,
                speculate=self.config.speculate,
                spec_k=self.config.speculate_k,
                async_fan_out=self.config.async_fan_out,
                draft_async=self.config.draft_async,
                auto_tune_kf=self.config.async_auto_tune,
            )
        else:
            raise ValueError(f"Unsupported model type: {hf_config.model_type}")

        return model.to(self.device)

    def shutdown(self):
        """Shutdown model runner."""
        if self._exiting:
            return
        self._exiting = True
        self.graphs.clear()
        self.graph_vars.clear()
        self.graph_bs_list.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def warmup_model(self):
        """Warmup model."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(max(1, num_seqs))]
        self.run(seqs, True)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _maybe_capture_decode_cudagraph(self):
        """Capture CUDA graph for decode."""
        try:
            graph_vars, graphs, graph_bs_list = self.capture_decode_cudagraph()
        except torch.OutOfMemoryError:
            if self.verbose:
                print("[model_runner] CUDA graph capture ran out of memory, falling back to eager decode.", flush=True)
            self.graph_vars.pop("decode", None)
            self.graphs.pop("decode", None)
            self.graph_bs_list.pop("decode", None)
            torch.cuda.empty_cache()
            return
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" not in message and "cuda error" not in message:
                raise
            if self.verbose:
                print(
                    f"[model_runner] CUDA graph capture failed ({exc}); falling back to eager decode.",
                    flush=True,
                )
            self.graph_vars.pop("decode", None)
            self.graphs.pop("decode", None)
            self.graph_bs_list.pop("decode", None)
            torch.cuda.empty_cache()
            return

        self.graph_vars["decode"] = graph_vars
        self.graphs["decode"] = graphs
        self.graph_bs_list["decode"] = graph_bs_list

    def allocate_kv_cache_draft(self):
        """Allocate KV cache for draft model."""
        free, _ = torch.cuda.mem_get_info(device=self.device)
        num_kv_heads = self.hf_config.num_key_value_heads
        block_bytes = (
            2
            * self.hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * self.hf_config.head_dim
            * self.hf_config.torch_dtype.itemsize
        )
        usable_bytes = free * self.config.gpu_memory_utilization
        self.config.num_kvcache_blocks = int(usable_bytes) // block_bytes
        assert self.config.num_kvcache_blocks > 0, "ERROR in ModelRunner: draft KV cache does not fit on GPU"

        self.kv_cache = torch.zeros(
            2,
            self.hf_config.num_hidden_layers,
            self.config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            self.hf_config.head_dim,
            device=self.device,
            dtype=self.hf_config.torch_dtype,
        )
        self._bind_kv_cache()

    def allocate_kv_cache_target(self):
        """Allocate KV cache for target model."""
        mem = psutil.virtual_memory()
        free = mem.available
        num_kv_heads = self.hf_config.num_key_value_heads
        block_bytes = (
            2
            * self.hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * self.hf_config.head_dim
            * self.hf_config.torch_dtype.itemsize
        )
        usable_bytes = free * self.config.cpu_memory_utilization
        self.config.num_kvcache_blocks = int(usable_bytes) // block_bytes
        assert self.config.num_kvcache_blocks > 0, "ERROR in ModelRunner: target KV cache does not fit in system RAM"

        self.kv_cache = torch.zeros(
            2,
            self.hf_config.num_hidden_layers,
            self.config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            self.hf_config.head_dim,
            device=self.device,
            dtype=self.hf_config.torch_dtype,
        )
        self._bind_kv_cache()

    def _bind_kv_cache(self):
        """Bind KV cache to model layers."""
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_prefill(self, seqs: list[Sequence]):
        """Prepare tensors for prefill."""
        input_ids, positions, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping = (
            prepare_prefill_tensors_from_seqs(
                seqs,
                self.block_size,
                self.is_draft,
                device=self.device,
                transfer_recorder=self._queue_transfer_event,
            )
        )

        block_tables = None
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = prepare_block_tables_from_seqs(
                seqs,
                self.is_draft,
                device=self.device,
                transfer_recorder=self._queue_transfer_event,
            )

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping if slot_mapping.numel() > 0 else None,
            block_tables=block_tables,
            context_lens=None,
        )
        return input_ids, positions

    def prepare_decode(
        self,
        seqs: list[Sequence],
        verify: bool = False,
        verify_lookahead: int | None = None,
    ):
        """Prepare tensors for decode."""
        lookahead = self.config.speculate_k if verify_lookahead is None else verify_lookahead
        input_ids, positions, slot_mapping, context_lens = prepare_decode_tensors_from_seqs(
            seqs,
            self.block_size,
            self.is_draft,
            verify,
            lookahead if verify else -1,
            device=self.device,
            transfer_recorder=self._queue_transfer_event,
        )
        block_tables = prepare_block_tables_from_seqs(
            seqs,
            self.is_draft,
            device=self.device,
            transfer_recorder=self._queue_transfer_event,
        )

        if verify:
            seqlen_q = torch.full(
                (len(seqs),),
                lookahead + 1,
                dtype=torch.int32,
                device=self.device,
            )
            cu_seqlens_q = torch.zeros(len(seqs) + 1, dtype=torch.int32, device=self.device)
            cu_seqlens_q[1:] = torch.cumsum(seqlen_q, dim=0)
            set_context(
                is_prefill=False,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=lookahead + 1,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
            )
        else:
            set_context(
                is_prefill=False,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
            )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """Prepare sampling parameters."""
        temperatures = []
        for seq in seqs:
            if self.is_draft and seq.draft_temperature is not None:
                temperatures.append(seq.draft_temperature)
            else:
                temperatures.append(seq.temperature)

        pin_memory = self.device.type == "cuda"
        return self._move_tensor_to_device(
            torch.tensor(
                temperatures,
                dtype=torch.float32,
                pin_memory=pin_memory,
            ),
            non_blocking=pin_memory,
        )

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, last_only: bool = True):
        """Run model."""
        if (
            not is_prefill
            and last_only
            and self.device.type == "cuda"
            and not self.enforce_eager
            and "decode" in self.graphs
        ):
            return self.run_decode_cudagraph(input_ids, positions, last_only, self.graph_vars["decode"])

        outputs = self.model(input_ids, positions)
        return self.model.compute_logits(outputs, last_only)

    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        last_only: bool = True,
        draft_return_logits: bool = False,
        verify_lookahead: int | None = None,
    ):
        """Run model on sequences."""
        mode = "prefill" if is_prefill else ("decode" if last_only else "verify")
        batch_size = len(seqs)
        run_start = perf_counter()
        prepare_time = 0.0
        model_time = 0.0
        sample_time = 0.0
        input_tokens = 0
        result = None

        try:
            prepare_start = perf_counter()
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(
                seqs,
                verify=not last_only,
                verify_lookahead=verify_lookahead,
            )
            prepare_time = perf_counter() - prepare_start
            input_tokens = int(input_ids.numel())

            sample_prepare_start = perf_counter()
            temperatures = self.prepare_sample(seqs) if last_only else None
            prepare_time += perf_counter() - sample_prepare_start

            model_start = perf_counter()
            logits = self.run_model(input_ids, positions, is_prefill=is_prefill, last_only=last_only)
            model_time = perf_counter() - model_start

            if not last_only:
                result = logits
            else:
                sample_start = perf_counter()
                token_ids = self._tensor_to_list(self.sampler(logits, temperatures))
                sample_time = perf_counter() - sample_start
                result = (token_ids, logits) if draft_return_logits else token_ids
        finally:
            total_time = perf_counter() - run_start
            self._record_profile(
                mode,
                batch_size=batch_size,
                input_tokens=input_tokens,
                prepare_time=prepare_time,
                model_time=model_time,
                sample_time=sample_time,
                total_time=total_time,
            )
            reset_context()

        return result

    @torch.inference_mode()
    def _forward_last_logits_from_token_batches_chunk(self, token_batches: list[list[int]]) -> torch.Tensor:
        """Forward last logits from token batches chunk."""
        input_ids = []
        positions = []
        cu_seqlens = [0]
        max_len = 0

        for tokens in token_batches:
            input_ids.extend(tokens)
            positions.extend(range(len(tokens)))
            cu_seqlens.append(cu_seqlens[-1] + len(tokens))
            max_len = max(max_len, len(tokens))

        pin_memory = self.device.type == "cuda"
        input_ids_t = self._move_tensor_to_device(
            torch.tensor(input_ids, dtype=torch.int64, pin_memory=pin_memory),
            non_blocking=pin_memory,
        )
        positions_t = self._move_tensor_to_device(
            torch.tensor(positions, dtype=torch.int64, pin_memory=pin_memory),
            non_blocking=pin_memory,
        )
        cu_seqlens_t = self._move_tensor_to_device(
            torch.tensor(cu_seqlens, dtype=torch.int32, pin_memory=pin_memory),
            non_blocking=pin_memory,
        )

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_t,
            cu_seqlens_k=cu_seqlens_t,
            max_seqlen_q=max_len,
            max_seqlen_k=max_len,
            slot_mapping=None,
            context_lens=None,
            block_tables=None,
        )
        logits = self.run_model(input_ids_t, positions_t, is_prefill=True, last_only=True)
        reset_context()
        return logits

    @torch.inference_mode()
    def forward_last_logits_from_token_batches(self, token_batches: list[list[int]]) -> torch.Tensor:
        """Forward last logits from token batches."""
        if not token_batches:
            return torch.empty(0, self.hf_config.vocab_size, dtype=self.hf_config.torch_dtype, device=self.device)

        max_chunk_tokens = max(1, self.config.max_num_batched_tokens)
        if self.is_draft and self.config.draft_async and self.device.type == "cuda":
            # Async draft-side stateless speculation can fan out into many long
            # branches. Keep the internal prefill micro-batches smaller than the
            # public scheduling budget so we preserve GPU headroom on consumer cards.
            max_chunk_tokens = min(max_chunk_tokens, 2048)
        logits_chunks = []
        chunk: list[list[int]] = []
        chunk_tokens = 0

        for tokens in token_batches:
            token_count = len(tokens)
            if chunk and chunk_tokens + token_count > max_chunk_tokens:
                logits_chunks.append(self._forward_last_logits_from_token_batches_chunk(chunk))
                chunk = []
                chunk_tokens = 0

            chunk.append(tokens)
            chunk_tokens += token_count

            # A single oversized request still needs to make progress.
            if chunk_tokens >= max_chunk_tokens:
                logits_chunks.append(self._forward_last_logits_from_token_batches_chunk(chunk))
                chunk = []
                chunk_tokens = 0

        if chunk:
            logits_chunks.append(self._forward_last_logits_from_token_batches_chunk(chunk))

        return torch.cat(logits_chunks, dim=0)

    @torch.inference_mode()
    def speculate_stateless_batch(
        self,
        token_batches: list[list[int]],
        recovery_tokens: list[int],
        temperatures: list[float],
        lookahead: int,
        return_step_logits: bool = True,
        return_all_logits: bool = True,
        fork_counts: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Speculate stateless batch."""
        current_batches = [tokens + [recovery] for tokens, recovery in zip(token_batches, recovery_tokens)]
        batch_size = len(current_batches)
        speculations = torch.empty(batch_size, lookahead + 1, dtype=torch.int64, device=self.device)
        speculations[:, 0] = torch.tensor(recovery_tokens, dtype=torch.int64, device=self.device)
        temp_t = torch.tensor(temperatures, dtype=torch.float32, device=self.device)

        step_logits = [] if return_step_logits else None
        all_bonus_logits = [] if return_all_logits else None
        fork_tokens = [] if fork_counts is not None else None
        for step in range(lookahead + 1):
            logits = self.forward_last_logits_from_token_batches(current_batches)
            if all_bonus_logits is not None:
                all_bonus_logits.append(logits)

            if step == lookahead:
                if fork_tokens is not None and fork_counts[step] > 0:
                    topk = torch.topk(
                        logits,
                        min(logits.size(-1), fork_counts[step]),
                        dim=-1,
                    ).indices
                    fork_tokens.append(topk[:, : fork_counts[step]])
                break

            if step_logits is not None:
                step_logits.append(logits)
            next_tokens = self._tensor_to_list(self.sampler(logits, temp_t))
            next_tokens_t = torch.tensor(next_tokens, dtype=torch.int64, device=self.device)
            speculations[:, step + 1] = next_tokens_t

            if fork_tokens is not None and fork_counts[step] > 0:
                topk = torch.topk(
                    logits,
                    min(logits.size(-1), fork_counts[step] + 1),
                    dim=-1,
                ).indices
                filtered = topk.masked_select(topk != next_tokens_t.unsqueeze(1)).view(batch_size, -1)
                fork_tokens.append(filtered[:, : fork_counts[step]])

            for tokens, token_id in zip(current_batches, next_tokens):
                tokens.append(token_id)

        step_logits_t = torch.stack(step_logits, dim=1) if step_logits is not None else None
        all_bonus_logits_t = torch.stack(all_bonus_logits, dim=1) if all_bonus_logits is not None else None
        fork_tokens_t = torch.cat(fork_tokens, dim=1) if fork_tokens is not None else None
        return speculations, step_logits_t, all_bonus_logits_t, fork_tokens_t

    def run_decode_cudagraph(self, input_ids, positions, last_only, graph_vars):
        """Run decode using CUDA graph."""
        del last_only
        from haste.utils.context import get_context

        context = get_context()
        flat_batch_size = input_ids.size(0)
        graph_bs = next(bs for bs in self.graph_bs_list["decode"] if bs >= flat_batch_size)
        graph = self.graphs["decode"][graph_bs]
        vars_for_graph = graph_vars[graph_bs]

        for key, value in vars_for_graph.items():
            if key != "outputs":
                value.zero_()

        if graph_bs > flat_batch_size:
            pad = graph_bs - flat_batch_size
            device = input_ids.device
            input_ids = torch.cat([input_ids, torch.zeros(pad, dtype=input_ids.dtype, device=device)], dim=0)
            positions = torch.cat([positions, torch.zeros(pad, dtype=positions.dtype, device=device)], dim=0)
            slot_mapping = torch.cat(
                [
                    context.slot_mapping,
                    torch.full((pad,), -1, dtype=context.slot_mapping.dtype, device=device),
                ],
                dim=0,
            )
            context_lens = torch.cat(
                [
                    context.context_lens,
                    context.context_lens[-1:].expand(pad).contiguous(),
                ],
                dim=0,
            )
            block_tables = None
            if context.block_tables is not None:
                block_tables = torch.cat(
                    [
                        context.block_tables,
                        context.block_tables[-1:].expand(pad, -1).contiguous(),
                    ],
                    dim=0,
                )
        else:
            slot_mapping = context.slot_mapping
            context_lens = context.context_lens
            block_tables = context.block_tables

        vars_for_graph["input_ids"][:graph_bs] = input_ids
        vars_for_graph["positions"][:graph_bs] = positions
        vars_for_graph["slot_mapping"][:graph_bs] = slot_mapping
        vars_for_graph["context_lens"][:graph_bs] = context_lens
        if block_tables is not None:
            vars_for_graph["block_tables"][:graph_bs, :block_tables.size(1)] = block_tables

        graph.replay()
        outputs = vars_for_graph["outputs"][:flat_batch_size]
        return self.model.compute_logits(outputs, True)

    @torch.inference_mode()
    def capture_decode_cudagraph(self):
        """Capture CUDA graph for decode."""
        max_bs = min(self.config.max_num_seqs + 1, 512)
        graph_bs_list = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        if max_bs not in graph_bs_list:
            graph_bs_list.append(max_bs)
        graph_bs_list = sorted(set(bs for bs in graph_bs_list if bs <= max_bs))

        graphs = {}
        graph_vars = {}
        hidden = self.hf_config.hidden_size

        for bs in graph_bs_list:
            input_ids = torch.zeros(bs, dtype=torch.int64, device=self.device)
            positions = torch.zeros(bs, dtype=torch.int64, device=self.device)
            slot_mapping = torch.zeros(bs, dtype=torch.int32, device=self.device)
            context_lens = torch.ones(bs, dtype=torch.int32, device=self.device)
            block_tables = torch.zeros(bs, self.max_num_blocks, dtype=torch.int32, device=self.device)
            outputs = torch.zeros(bs, hidden, dtype=self.hf_config.torch_dtype, device=self.device)
            try:
                set_context(
                    is_prefill=False,
                    slot_mapping=slot_mapping,
                    context_lens=context_lens,
                    block_tables=block_tables,
                )
                outputs[:] = self.model(input_ids, positions)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs[:] = self.model(input_ids, positions)
            finally:
                reset_context()

            graphs[bs] = graph
            graph_vars[bs] = {
                "input_ids": input_ids,
                "positions": positions,
                "slot_mapping": slot_mapping,
                "context_lens": context_lens,
                "block_tables": block_tables,
                "outputs": outputs,
            }

        return graph_vars, graphs, graph_bs_list
