"""Draft runner for speculative decoding."""

import dataclasses
import os
import queue
import time
from dataclasses import dataclass
from threading import Lock, Thread

import torch

from haste.config import Config
from haste.engine.model_runner import ModelRunner
from haste.engine.sequence import Sequence
from haste.utils.async_helpers.async_spec_helpers import get_forked_recovery_tokens_from_logits
from haste.utils.profiling import build_draft_worker_profile_summary


@dataclass
class DraftRequest:
    """Draft request data class."""
    kind: str  # Request kind: 'prefill', 'speculate', or 'shutdown'
    seqs: list[Sequence] | None = None  # List of sequences
    response_q: queue.Queue | None = None  # Response queue


@dataclass
class DraftResponse:
    """Draft response data class."""
    speculations: torch.Tensor  # Speculated tokens
    logits_q: torch.Tensor | None  # Logits from query
    cache_hits: torch.Tensor  # Cache hit indicators
    from_q_mask: torch.Tensor  # Mask indicating which tokens are from query
    lookahead: int  # Lookahead value
    fan_out_cap: int  # Fan-out capacity
    serve_ms: float = 0.0  # Worker serve time for this request in ms


@dataclass
class CachedDraftState:
    """Cached draft state data class."""
    speculation: torch.Tensor  # Speculated tokens
    fork_recovery_tokens: torch.Tensor  # Fork recovery tokens
    lookahead: int  # Lookahead value
    fork_width: int  # Fork width
    gpu_speculation: torch.Tensor | None = None  # Optional GPU speculation cache


@dataclass
class AutoTuneState:
    """Auto-tune state data class."""
    stage: str  # Current stage: 'search_k', 'search_f', or 'steady'
    trial_k: int  # Trial lookahead value
    trial_f: int  # Trial fan-out value
    settled_k: int  # Settled lookahead value
    settled_f: int  # Settled fan-out value
    best_hidden_k: int  # Best hidden lookahead value
    best_hidden_f: int  # Best hidden fan-out value
    best_k_score: float = -1.0  # Best lookahead score
    best_f_score: float = -1.0  # Best fan-out score
    trial_observations: int = 0  # Number of trial observations
    stable_steps: int = 0  # Number of stable steps
    ema_verify_ms: float = 0.0  # EMA of verification time in ms
    ema_populate_ms: float = 0.0  # EMA of populate time in ms
    ema_wait_ms: float = 0.0  # EMA of wait time in ms
    ema_serve_ms: float = 0.0  # EMA of serve time in ms
    ema_exposed_ms: float = 0.0  # EMA of exposed wait in ms
    ema_cache_hit_rate: float = 0.0  # EMA of cache hit rate
    ema_accept_fraction: float = 0.0  # EMA of accept fraction
    probe_baseline_k: int = 1  # Baseline K before a steady-state reprobe
    probe_baseline_f: int = 1  # Baseline F before a steady-state reprobe
    probe_baseline_score: float = -1.0  # Baseline throughput score before reprobe


class DraftRunner(ModelRunner):
    """Draft runner for speculative decoding."""
    
    @classmethod
    def create_draft_config(cls, config: Config) -> Config:
        """Create draft configuration from base configuration.
        
        Args:
            config (Config): Base configuration
            
        Returns:
            Config: Draft configuration
        """
        return dataclasses.replace(
            config,
            model=config.draft_model,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=config.enforce_eager,
        )

    def __init__(self, config: Config, init_q=None):
        """Initialize draft runner.
        
        Args:
            config (Config): Configuration
            init_q: Initialization queue
        """
        self.draft_config = self.create_draft_config(config)
        self.is_draft = True
        self.draft_async = config.draft_async
        super().__init__(self.draft_config, event=None, device=config.draft_device, is_draft=True, init_q=init_q)

        self._tree_cache: dict[tuple[int, int, int], CachedDraftState] = {}  # Tree cache for draft states
        self._draft_step_times: list[float] = []  # Draft step times
        self._fan_out_batch_hint = 0  # Fan-out batch hint
        self._controller_lock = Lock()  # Controller lock
        self._runtime_lookahead_cap = config.speculate_k  # Runtime lookahead capacity
        self._runtime_fan_out_cap = config.async_fan_out  # Runtime fan-out capacity
        self._auto_tune_state: AutoTuneState | None = None  # Auto-tune state
        self._last_logged_policy: tuple[str, int, int] | None = None  # Last logged policy
        self._last_request_wait_ms = 0.0  # Last request wait time in ms
        self._last_request_serve_ms = 0.0  # Last request serve time in ms
        self._last_exposed_wait_ms = 0.0  # Last exposed wait time in ms
        self._last_cache_hit_rate = 0.0  # Last cache hit rate
        self._last_accept_fraction = 0.0  # Last accept fraction
        self._request_queue: queue.Queue | None = None  # Request queue
        self._worker: Thread | None = None  # Worker thread
        self._worker_error: Exception | None = None  # Worker error
        self.reset_profile()

        if self.draft_async:
            self._request_queue = queue.Queue()
            self._worker = Thread(
                target=self._draft_loop,
                name="haste-draft-worker",
                daemon=True,
            )
            self._worker.start()

    def reset_profile(self):
        """Reset profile data."""
        super().reset_profile()
        self._draft_step_times = []
        self._reset_runtime_policy(log_init=False)
        self._worker_profile = {
            "request_wait_times": [],
            "exposed_wait_times": [],
            "worker_total_times": [],
            "worker_serve_times": [],
            "cache_populate_times": [],
            "populate_branch_counts": [],
            "fast_populate_flags": [],
            "request_batch_sizes": [],
            "worker_batch_sizes": [],
            "cache_hit_rates": [],
            "tree_cache_sizes": [],
            "effective_lookaheads": [],
            "effective_fan_out_caps": [],
        }

    def _maybe_pin_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pin CPU tensors when CUDA is available to speed host/device transfers."""
        if self.device.type != "cuda" or tensor.device.type != "cpu":
            return tensor
        if tensor.is_pinned():
            return tensor
        return tensor.pin_memory()

    def _gpu_cache_budget_bytes(self) -> int:
        """Get GPU cache budget in bytes for async draft caching."""
        if self.device.type != "cuda":
            return 0
        env_mb = os.environ.get("HASTE_ASYNC_GPU_CACHE_MB")
        if env_mb:
            try:
                return max(0, int(env_mb)) * 1024 * 1024
            except ValueError:
                return 0
        return 256 * 1024 * 1024

    def _should_keep_gpu_cache(self, tensor: torch.Tensor) -> bool:
        """Heuristic to keep a GPU copy of speculation cache."""
        budget = self._gpu_cache_budget_bytes()
        if budget <= 0:
            return False
        bytes_needed = tensor.numel() * tensor.element_size()
        return bytes_needed <= budget

    def _to_cpu_pinned(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        """Move CUDA tensors to pinned CPU memory for non-blocking transfers."""
        if tensor is None:
            return None
        if tensor.device.type == "cpu":
            return self._maybe_pin_cpu(tensor)
        cpu_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
        cpu_tensor.copy_(tensor, non_blocking=False)
        return cpu_tensor

    def worker_profile_summary(self) -> dict:
        """Get worker profile summary.
        
        Returns:
            dict: Worker profile summary
        """
        summary = build_draft_worker_profile_summary(self._worker_profile, device=str(self.device))
        summary["auto_tune_enabled"] = self.config.async_auto_tune
        summary["static_speculate_k"] = self.config.speculate_k
        summary["static_async_fan_out"] = self.config.async_fan_out
        summary["final_effective_k"] = self._runtime_lookahead_cap
        summary["final_effective_f"] = self._runtime_fan_out_cap
        summary["auto_tune_stage"] = self._auto_tune_state.stage if self._auto_tune_state is not None else "disabled"
        return summary

    def shutdown(self):
        """Shutdown draft runner."""
        if self._exiting:
            return

        if self.draft_async and self._worker is not None and self._request_queue is not None:
            self._request_queue.put(DraftRequest(kind="shutdown"))
            self._worker.join()
            self._worker = None

        if self.draft_async and self._draft_step_times:
            avg_ms = sum(self._draft_step_times) * 1000 / len(self._draft_step_times)
            print(f"[metrics] Avg draft step time (ms): {avg_ms:.2f}", flush=True)

        super().shutdown()

    def _auto_tune_bounds(self) -> tuple[int, int, int, int]:
        """Get auto-tune bounds.
        
        Returns:
            tuple[int, int, int, int]: Min and max values for k and f
        """
        max_k = self.config.speculate_k if self.config.async_auto_tune_max_k is None else self.config.async_auto_tune_max_k
        max_f = self.config.async_fan_out if self.config.async_auto_tune_max_f is None else self.config.async_auto_tune_max_f
        max_k = max(1, min(self.config.speculate_k, max_k))
        max_f = max(1, min(self.config.async_fan_out, max_f))
        min_k = max(1, min(self.config.async_auto_tune_min_k, max_k))
        min_f = max(1, min(self.config.async_auto_tune_min_f, max_f))
        return min_k, max_k, min_f, max_f

    def _reset_runtime_policy(self, log_init: bool = True):
        """Reset runtime policy.
        
        Args:
            log_init (bool): Whether to log initialization
        """
        self._fan_out_batch_hint = 0
        self._last_logged_policy = None
        self._last_request_wait_ms = 0.0
        self._last_request_serve_ms = 0.0
        self._last_exposed_wait_ms = 0.0
        self._last_cache_hit_rate = 0.0
        self._last_accept_fraction = 0.0
        if self.config.async_auto_tune:
            min_k, max_k, min_f, max_f = self._auto_tune_bounds()
            stage = "search_k" if max_k > min_k else ("search_f" if max_f > min_f else "steady")
            self._runtime_lookahead_cap = min_k
            self._runtime_fan_out_cap = min_f
            self._auto_tune_state = AutoTuneState(
                stage=stage,
                trial_k=min_k,
                trial_f=min_f,
                settled_k=min_k,
                settled_f=min_f,
                best_hidden_k=min_k,
                best_hidden_f=min_f,
            )
            if log_init:
                self._log_auto_tune_status("init")
        else:
            self._runtime_lookahead_cap = self.config.speculate_k
            self._runtime_fan_out_cap = self.config.async_fan_out
            self._auto_tune_state = None

    def submit_prefill(self, seqs: list[Sequence]):
        """Submit prefill request.
        
        Args:
            seqs (list[Sequence]): List of sequences
        """
        self._reset_runtime_policy(log_init=True)
        if not self.draft_async:
            self.run(seqs, True)
            return

        assert self._request_queue is not None
        self._request_queue.put(
            DraftRequest(
                kind="prefill",
                seqs=[seq.clone_spec() for seq in seqs],
            )
        )

    def request_speculation(self, seqs: list[Sequence]) -> DraftResponse:
        """Request speculation.
        
        Args:
            seqs (list[Sequence]): List of sequences
            
        Returns:
            DraftResponse: Draft response
        """
        if not self.draft_async:
            raise RuntimeError("request_speculation() is only valid in draft_async mode")
        if self._worker_error is not None:
            raise RuntimeError("draft worker failed") from self._worker_error

        assert self._request_queue is not None
        response_q: queue.Queue = queue.Queue(maxsize=1)
        wait_start = time.perf_counter()
        self._request_queue.put(
            DraftRequest(
                kind="speculate",
                seqs=[seq.clone_spec() for seq in seqs],
                response_q=response_q,
            )
        )
        result = response_q.get()
        wait_elapsed = time.perf_counter() - wait_start
        self._worker_profile["request_wait_times"].append(wait_elapsed)
        self._worker_profile["request_batch_sizes"].append(len(seqs))
        self._last_request_wait_ms = wait_elapsed * 1000.0
        if isinstance(result, Exception):
            self._worker_error = result
            raise RuntimeError("draft worker failed") from result
        self._last_request_serve_ms = max(0.0, float(getattr(result, "serve_ms", 0.0)))
        self._last_exposed_wait_ms = max(0.0, self._last_request_wait_ms - self._last_request_serve_ms)
        self._worker_profile["exposed_wait_times"].append(self._last_exposed_wait_ms / 1000.0)
        cache_hit_rate = result.cache_hits.to(torch.float32).mean().item()
        self._last_cache_hit_rate = cache_hit_rate
        self._worker_profile["cache_hit_rates"].append(cache_hit_rate)
        self._worker_profile["effective_lookaheads"].append(result.lookahead)
        self._worker_profile["effective_fan_out_caps"].append(result.fan_out_cap)
        return result

    @staticmethod
    def _needs_hit_logits(seq: Sequence) -> bool:
        """Check if hit logits are needed.
        
        Args:
            seq (Sequence): Sequence
            
        Returns:
            bool: Whether hit logits are needed
        """
        draft_temperature = seq.draft_temperature if seq.draft_temperature is not None else seq.temperature
        return seq.temperature > 0 or draft_temperature > 0

    @classmethod
    def _needs_response_logits(cls, seqs: list[Sequence]) -> bool:
        """Check if response logits are needed.
        
        Args:
            seqs (list[Sequence]): List of sequences
            
        Returns:
            bool: Whether response logits are needed
        """
        return any(cls._needs_hit_logits(seq) for seq in seqs)

    def _current_batch_hint(self, batch_size: int) -> int:
        """Get current batch hint.
        
        Args:
            batch_size (int): Batch size
            
        Returns:
            int: Current batch hint
        """
        self._fan_out_batch_hint = max(self._fan_out_batch_hint, batch_size)
        return self._fan_out_batch_hint

    def _effective_lookahead(self, batch_size: int) -> int:
        """Get effective lookahead.
        
        Args:
            batch_size (int): Batch size
            
        Returns:
            int: Effective lookahead
        """
        self._current_batch_hint(batch_size)
        if not self.config.async_auto_tune:
            return self.config.speculate_k
        _, max_k, _, _ = self._auto_tune_bounds()
        return max(1, min(self._runtime_lookahead_cap, max_k))

    def _base_fan_out_cap(self, batch_size: int) -> int:
        """Get base fan-out capacity.
        
        Args:
            batch_size (int): Batch size
            
        Returns:
            int: Base fan-out capacity
        """
        del batch_size
        if self.config.async_auto_tune:
            return self._auto_tune_bounds()[3]
        return self.config.async_fan_out

    def _effective_fan_out_lists(self, batch_size: int, lookahead: int | None = None) -> tuple[list[int], list[int]]:
        """Get effective fan-out lists.
        
        Args:
            batch_size (int): Batch size
            lookahead (int | None): Lookahead value
            
        Returns:
            tuple[list[int], list[int]]: Fan-out lists for cache hits and misses
        """
        fan_out_list = self.config.fan_out_list
        fan_out_list_miss = self.config.fan_out_list_miss
        assert fan_out_list is not None
        assert fan_out_list_miss is not None
        self._current_batch_hint(batch_size)
        base_hit = fan_out_list[: lookahead + 1] if lookahead is not None else fan_out_list
        base_miss = fan_out_list_miss[: lookahead + 1] if lookahead is not None else fan_out_list_miss
        if not self.config.async_auto_tune:
            return list(base_hit), list(base_miss)

        runtime_cap = min(self._runtime_fan_out_cap, self._base_fan_out_cap(batch_size))
        return (
            [min(count, runtime_cap) for count in base_hit],
            [min(count, runtime_cap) for count in base_miss],
        )

    @staticmethod
    def _ema(previous: float, value: float, alpha: float, initialized: bool) -> float:
        """Calculate exponential moving average.
        
        Args:
            previous (float): Previous value
            value (float): Current value
            alpha (float): Smoothing factor
            initialized (bool): Whether EMA is initialized
            
        Returns:
            float: Updated EMA value
        """
        if not initialized:
            return value
        return previous * (1.0 - alpha) + value * alpha

    def _reset_auto_tune_observations(self, state: AutoTuneState) -> None:
        """Reset auto-tune observations.
        
        Args:
            state (AutoTuneState): Auto-tune state
        """
        state.trial_observations = 0
        state.ema_verify_ms = 0.0
        state.ema_populate_ms = 0.0
        state.ema_wait_ms = 0.0
        state.ema_serve_ms = 0.0
        state.ema_exposed_ms = 0.0
        state.ema_cache_hit_rate = 0.0
        state.ema_accept_fraction = 0.0

    def _begin_auto_tune_probe(self, state: AutoTuneState, stage: str, trial_k: int, trial_f: int) -> None:
        """Begin auto-tune probe.
        
        Args:
            state (AutoTuneState): Auto-tune state
            stage (str): Stage
            trial_k (int): Trial lookahead value
            trial_f (int): Trial fan-out value
        """
        state.stage = stage
        state.trial_k = trial_k
        state.trial_f = trial_f
        self._runtime_lookahead_cap = trial_k
        self._runtime_fan_out_cap = trial_f
        self._reset_auto_tune_observations(state)
        self._log_auto_tune_status("probe")

    def _settle_auto_tune(self, state: AutoTuneState, settled_k: int, settled_f: int) -> None:
        """Settle auto-tune.
        
        Args:
            state (AutoTuneState): Auto-tune state
            settled_k (int): Settled lookahead value
            settled_f (int): Settled fan-out value
        """
        state.stage = "steady"
        state.trial_k = settled_k
        state.trial_f = settled_f
        state.settled_k = settled_k
        state.settled_f = settled_f
        state.best_hidden_k = settled_k
        state.best_hidden_f = settled_f
        state.stable_steps = 0
        self._runtime_lookahead_cap = settled_k
        self._runtime_fan_out_cap = settled_f
        self._reset_auto_tune_observations(state)
        self._log_auto_tune_status("settled")

    def _begin_runtime_reprobe(
        self,
        state: AutoTuneState,
        stage: str,
        trial_k: int,
        trial_f: int,
        baseline_score: float,
    ) -> None:
        """Probe a slightly more aggressive steady-state policy and revert if it loses."""
        state.probe_baseline_k = self._runtime_lookahead_cap
        state.probe_baseline_f = self._runtime_fan_out_cap
        state.probe_baseline_score = baseline_score
        self._begin_auto_tune_probe(state, stage, trial_k, trial_f)

    def _log_auto_tune_status(self, reason: str) -> None:
        """Log auto-tune status.
        
        Args:
            reason (str): Reason for logging
        """
        if not self.config.async_auto_tune or not getattr(self.config, "verbose", False):
            return
        stage = self._auto_tune_state.stage if self._auto_tune_state is not None else "disabled"
        snapshot = (stage, self._runtime_lookahead_cap, self._runtime_fan_out_cap)
        last_logged_policy = getattr(self, "_last_logged_policy", None)
        if reason != "init" and snapshot == last_logged_policy:
            return
        self._last_logged_policy = snapshot
        print(
            "[auto_tune_kf] "
            f"reason={reason} stage={stage} "
            f"static_k={self.config.speculate_k} static_f={self.config.async_fan_out} "
            f"effective_k={self._runtime_lookahead_cap} effective_f={self._runtime_fan_out_cap}",
            flush=True,
        )

    def _auto_tune_hidden(self, state: AutoTuneState) -> bool:
        """Check if auto-tune is hidden.
        
        Args:
            state (AutoTuneState): Auto-tune state
            
        Returns:
            bool: Whether auto-tune is hidden
        """
        verify_budget = max(1.0, state.ema_verify_ms * self.config.async_auto_tune_margin)
        exposed_budget = max(2.0, state.ema_verify_ms * self.config.async_auto_tune_wait_ratio)
        return state.ema_populate_ms <= verify_budget and state.ema_exposed_ms <= exposed_budget

    def _auto_tune_underloaded(self, state: AutoTuneState) -> bool:
        """Check if auto-tune is underloaded.
        
        Args:
            state (AutoTuneState): Auto-tune state
            
        Returns:
            bool: Whether auto-tune is underloaded
        """
        verify_budget = max(1.0, state.ema_verify_ms * self.config.async_auto_tune_underfill_ratio)
        exposed_budget = max(1.0, state.ema_verify_ms * self.config.async_auto_tune_wait_ratio * 0.5)
        return state.ema_populate_ms <= verify_budget and state.ema_exposed_ms <= exposed_budget

    def _step_throughput_score(self, state: AutoTuneState) -> float:
        """Estimate generation throughput for the current runtime policy."""
        yielded_tokens = 1.0 + state.ema_accept_fraction * state.trial_k
        step_ms = max(1.0, state.ema_verify_ms + state.ema_wait_ms)
        return yielded_tokens / step_ms

    def _next_k_probe(self, state: AutoTuneState, max_k: int) -> int:
        """Choose the next K probe, ramping faster when overlap headroom is abundant."""
        if state.trial_k >= max_k:
            return max_k

        deep_underloaded = (
            state.trial_k >= 2
            and state.ema_populate_ms <= max(1.0, state.ema_verify_ms * 0.4)
            and state.ema_exposed_ms <= 1.0
            and state.ema_accept_fraction >= self.config.async_auto_tune_accept_floor
        )
        step = 2 if deep_underloaded else 1
        return min(max_k, state.trial_k + step)

    def _lookahead_score(self, state: AutoTuneState) -> float:
        """Calculate lookahead score.
        
        Args:
            state (AutoTuneState): Auto-tune state
            
        Returns:
            float: Lookahead score
        """
        return self._step_throughput_score(state)

    def _fan_out_score(self, state: AutoTuneState) -> float:
        """Calculate fan-out score.
        
        Args:
            state (AutoTuneState): Auto-tune state
            
        Returns:
            float: Fan-out score
        """
        return self._step_throughput_score(state)

    def report_verify_feedback(
        self,
        verify_elapsed_s: float,
        batch_size: int,
        accepted_fraction: float,
    ) -> None:
        """Report verify feedback.
        
        Args:
            verify_elapsed_s (float): Verify elapsed time in seconds
            batch_size (int): Batch size
            accepted_fraction (float): Accepted fraction
        """
        if not self.config.async_auto_tune:
            return

        self._last_accept_fraction = accepted_fraction
        verify_ms = verify_elapsed_s * 1000.0
        if verify_ms <= 0:
            return

        recent_populates = self._worker_profile["cache_populate_times"]
        populate_ms = 1000.0 * recent_populates[-1] if recent_populates else 0.0
        wait_ms = self._last_request_wait_ms
        serve_ms = self._last_request_serve_ms
        exposed_ms = self._last_exposed_wait_ms
        cache_hit_rate = self._last_cache_hit_rate
        state = self._auto_tune_state
        if state is None:
            return

        min_k, max_k, min_f, max_f = self._auto_tune_bounds()
        alpha = self.config.async_auto_tune_ema_alpha

        with self._controller_lock:
            initialized = state.trial_observations > 0
            state.ema_verify_ms = self._ema(state.ema_verify_ms, verify_ms, alpha, initialized)
            state.ema_populate_ms = self._ema(state.ema_populate_ms, populate_ms, alpha, initialized)
            state.ema_wait_ms = self._ema(state.ema_wait_ms, wait_ms, alpha, initialized)
            state.ema_serve_ms = self._ema(state.ema_serve_ms, serve_ms, alpha, initialized)
            state.ema_exposed_ms = self._ema(state.ema_exposed_ms, exposed_ms, alpha, initialized)
            state.ema_cache_hit_rate = self._ema(state.ema_cache_hit_rate, cache_hit_rate, alpha, initialized)
            state.ema_accept_fraction = self._ema(state.ema_accept_fraction, accepted_fraction, alpha, initialized)
            state.trial_observations += 1

            hidden = self._auto_tune_hidden(state)
            probe_ready = state.trial_observations >= self.config.async_auto_tune_probe_steps
            accept_ready = (
                state.trial_k <= min_k
                or state.ema_accept_fraction >= self.config.async_auto_tune_accept_floor
            )

            if state.stage == "search_k":
                if not probe_ready:
                    return

                k_score = self._lookahead_score(state)
                score_floor = state.best_k_score * (1.0 - self.config.async_auto_tune_score_tolerance)

                if hidden and accept_ready and (state.best_k_score < 0.0 or k_score >= score_floor):
                    if k_score > state.best_k_score:
                        state.best_k_score = k_score
                        state.best_hidden_k = state.trial_k
                    state.settled_k = state.trial_k
                    if state.trial_k < max_k:
                        self._begin_auto_tune_probe(state, "search_k", self._next_k_probe(state, max_k), min_f)
                        return

                settled_k = max(min_k, state.best_hidden_k)
                state.settled_k = settled_k
                state.best_hidden_f = min_f
                state.best_f_score = -1.0
                if max_f > min_f:
                    self._begin_auto_tune_probe(state, "search_f", settled_k, min_f)
                else:
                    self._settle_auto_tune(state, settled_k, min_f)
                return

            if state.stage == "search_f":
                if not probe_ready:
                    return

                f_score = self._fan_out_score(state)
                score_floor = state.best_f_score * (1.0 - self.config.async_auto_tune_score_tolerance)

                if hidden and (state.best_f_score < 0.0 or f_score >= score_floor):
                    if f_score > state.best_f_score:
                        state.best_f_score = f_score
                        state.best_hidden_f = state.trial_f
                    if state.trial_f < max_f:
                        self._begin_auto_tune_probe(state, "search_f", state.settled_k, state.trial_f + 1)
                        return

                settled_f = max(min_f, state.best_hidden_f)
                self._settle_auto_tune(state, state.settled_k, settled_f)
                return

            if state.stage in {"reprobe_k", "reprobe_f"}:
                if not probe_ready:
                    return

                probe_score = self._step_throughput_score(state)
                baseline_floor = state.probe_baseline_score * (1.0 - self.config.async_auto_tune_score_tolerance)
                if hidden and (state.probe_baseline_score < 0.0 or probe_score >= baseline_floor):
                    self._settle_auto_tune(state, state.trial_k, state.trial_f)
                else:
                    self._settle_auto_tune(state, state.probe_baseline_k, state.probe_baseline_f)
                return

            exposed_wait = state.ema_exposed_ms > max(2.0, state.ema_verify_ms * self.config.async_auto_tune_wait_ratio)
            overloaded = (not hidden) or exposed_wait
            underloaded = self._auto_tune_underloaded(state)

            if overloaded:
                state.stable_steps = 0
                old_k = self._runtime_lookahead_cap
                old_f = self._runtime_fan_out_cap
                if self._runtime_fan_out_cap > min_f and (
                    state.ema_exposed_ms > 0.0
                    or state.ema_cache_hit_rate > self.config.async_auto_tune_cache_hit_target
                    or state.ema_populate_ms > state.ema_verify_ms
                ):
                    self._runtime_fan_out_cap -= 1
                elif self._runtime_lookahead_cap > min_k:
                    self._runtime_lookahead_cap -= 1
                elif self._runtime_fan_out_cap > min_f:
                    self._runtime_fan_out_cap -= 1
                state.settled_k = self._runtime_lookahead_cap
                state.settled_f = self._runtime_fan_out_cap
                state.trial_k = self._runtime_lookahead_cap
                state.trial_f = self._runtime_fan_out_cap
                state.best_hidden_k = self._runtime_lookahead_cap
                state.best_hidden_f = self._runtime_fan_out_cap
                self._reset_auto_tune_observations(state)
                if self._runtime_lookahead_cap != old_k or self._runtime_fan_out_cap != old_f:
                    self._log_auto_tune_status("shrink")
            else:
                state.stable_steps += 1
                if underloaded and state.stable_steps >= self.config.async_auto_tune_reprobe_interval:
                    baseline_score = self._step_throughput_score(state)
                    if (
                        self._runtime_lookahead_cap < max_k
                        and state.ema_accept_fraction >= self.config.async_auto_tune_accept_floor
                    ):
                        self._begin_runtime_reprobe(
                            state,
                            "reprobe_k",
                            self._runtime_lookahead_cap + 1,
                            self._runtime_fan_out_cap,
                            baseline_score,
                        )
                        return
                    elif (
                        self._runtime_fan_out_cap < max_f
                        and state.ema_cache_hit_rate < self.config.async_auto_tune_cache_hit_target
                    ):
                        self._begin_runtime_reprobe(
                            state,
                            "reprobe_f",
                            self._runtime_lookahead_cap,
                            self._runtime_fan_out_cap + 1,
                            baseline_score,
                        )
                        return

    @torch.inference_mode()
    def _trace_speculation_paths(
        self,
        token_batches: list[list[int]],
        speculations: torch.Tensor,
        lookahead: int,
    ) -> torch.Tensor:
        """Replay a known speculation path to recover the per-step logits.
        
        Args:
            token_batches (list[list[int]]): Token batches
            speculations (torch.Tensor): Speculated tokens
            lookahead (int): Lookahead value
            
        Returns:
            torch.Tensor: Logits
        """
        spec_rows = speculations.tolist()
        current_batches = [tokens + [row[0]] for tokens, row in zip(token_batches, spec_rows)]
        all_logits = []

        for step in range(lookahead + 1):
            logits = self.forward_last_logits_from_token_batches(current_batches)
            all_logits.append(logits)
            if step == lookahead:
                break

            next_tokens = [row[step + 1] for row in spec_rows]
            for tokens, token_id in zip(current_batches, next_tokens):
                tokens.append(token_id)

        return torch.stack(all_logits, dim=1)

    @torch.inference_mode()
    def _speculate_low_latency_miss_batch(
        self,
        token_batches: list[list[int]],
        recovery_tokens: list[int],
        temperatures: list[float],
        lookahead: int,
        backup_steps: int,
        fork_counts: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cheap greedy backup for large-batch cache misses.

        We pay for a small fixed number of draft forward passes, then reuse the
        last draft step for the remaining speculative slots. This keeps the
        miss path much cheaper than a full just-in-time fallback without
        collapsing the speculation cache quality to pure one-step guesses.
        """
        batch_size = len(token_batches)
        current_batches = [tokens + [recovery] for tokens, recovery in zip(token_batches, recovery_tokens)]
        temp_t = torch.tensor(temperatures, dtype=torch.float32, device=self.device)

        speculations = torch.empty(batch_size, lookahead + 1, dtype=torch.int64, device=self.device)
        speculations[:, 0] = torch.tensor(recovery_tokens, dtype=torch.int64, device=self.device)
        logits_history: list[torch.Tensor] = []
        last_tokens = speculations[:, 0]
        steps_to_run = min(max(1, backup_steps), lookahead)

        for step in range(steps_to_run + 1):
            logits = self.forward_last_logits_from_token_batches(current_batches)
            logits_history.append(logits)
            if step == steps_to_run:
                break

            next_tokens = self.sampler(logits, temp_t).to(torch.int64)
            speculations[:, step + 1] = next_tokens
            last_tokens = next_tokens
            next_tokens_list = next_tokens.tolist()
            for tokens, token_id in zip(current_batches, next_tokens_list):
                tokens.append(token_id)

        for pad_idx in range(steps_to_run + 1, lookahead + 1):
            speculations[:, pad_idx] = last_tokens

        max_count = max(fork_counts, default=0)
        if max_count == 0:
            return speculations, torch.empty(batch_size, 0, dtype=torch.int64, device=self.device)

        fork_tokens = []
        for accepted_count, count in enumerate(fork_counts):
            if count <= 0:
                continue
            logits = logits_history[min(accepted_count, len(logits_history) - 1)]
            topk = torch.topk(logits, min(logits.size(-1), count), dim=-1).indices
            fork_tokens.append(topk[:, :count])
        return speculations, torch.cat(fork_tokens, dim=1)

    def _should_use_fast_populate(
        self,
        *,
        batch_size: int,
        lookahead: int,
        branch_count: int,
        branch_temperatures: list[float],
    ) -> bool:
        """Check if fast populate should be used.
        
        Args:
            batch_size (int): Batch size
            lookahead (int): Lookahead value
            branch_count (int): Branch count
            branch_temperatures (list[float]): Branch temperatures
            
        Returns:
            bool: Whether fast populate should be used
        """
        if lookahead <= 0 or branch_count <= 0:
            return False
        if any(temperature != 0 for temperature in branch_temperatures):
            return False
        if self._current_batch_hint(batch_size) < self.config.async_fast_populate_threshold:
            return False
        return branch_count >= max(1, batch_size * self.config.async_fast_populate_branch_factor)

    @torch.inference_mode()
    def _serve_speculation_request(
        self,
        seqs: list[Sequence],
    ) -> tuple[DraftResponse, list[int], torch.Tensor | None, dict[int, torch.Tensor]]:
        """Serve speculation request.
        
        Args:
            seqs (list[Sequence]): List of sequences
            
        Returns:
            tuple[DraftResponse, list[int], torch.Tensor | None, dict[int, torch.Tensor]]: 
                Draft response, computed indices, computed logits, and fork overrides
        """
        batch_size = len(seqs)
        lookahead = self._effective_lookahead(batch_size)
        vocab_size = self.hf_config.vocab_size
        dtype = self.hf_config.torch_dtype
        need_response_logits = self._needs_response_logits(seqs)
        fan_out_list, fan_out_list_miss = self._effective_fan_out_lists(batch_size, lookahead)

        speculations = torch.empty(batch_size, lookahead + 1, dtype=torch.int64, device=self.device)
        logits_q = None
        if need_response_logits:
            logits_q = torch.zeros(batch_size, lookahead, vocab_size, dtype=dtype, device=self.device)
        cache_hits = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        keys = [
            (seq.seq_id, seq.last_spec_step_accepted_len - 1, seq.recovery_token_id)
            for seq in seqs
        ]
        token_batches = [list(seq.token_ids) for seq in seqs]
        temperatures = [
            seq.draft_temperature if seq.draft_temperature is not None else seq.temperature
            for seq in seqs
        ]
        recovery_tokens = [seq.recovery_token_id for seq in seqs]

        hit_trace_indices: list[int] = []
        hit_trace_specs_cpu: list[torch.Tensor] = []
        hit_fork_overrides: dict[int, torch.Tensor] = {}
        miss_indices: list[int] = []

        for idx, key in enumerate(keys):
            cached_state = self._tree_cache.get(key)
            if cached_state is None:
                miss_indices.append(idx)
                continue

            if cached_state.lookahead < lookahead:
                miss_indices.append(idx)
                continue

            cache_hits[idx] = True
            if cached_state.gpu_speculation is not None:
                speculations[idx] = cached_state.gpu_speculation[: lookahead + 1]
            else:
                speculations[idx] = cached_state.speculation[: lookahead + 1].to(self.device, non_blocking=True)

            expected_width = sum(fan_out_list)
            if self._needs_hit_logits(seqs[idx]) or cached_state.fork_width < expected_width:
                hit_trace_indices.append(idx)
                hit_trace_specs_cpu.append(cached_state.speculation[: lookahead + 1])
            else:
                hit_fork_overrides[idx] = cached_state.fork_recovery_tokens[:expected_width]

        computed_indices: list[int] = []
        computed_logits: list[torch.Tensor] = []
        if miss_indices:
            if need_response_logits:
                miss_specs, miss_logits_q, miss_all_logits, _ = self.speculate_stateless_batch(
                    [token_batches[idx] for idx in miss_indices],
                    [recovery_tokens[idx] for idx in miss_indices],
                    [temperatures[idx] for idx in miss_indices],
                    lookahead,
                )
            else:
                use_fast_backup = (
                    self._current_batch_hint(batch_size) >= self.config.async_fast_miss_threshold
                    and lookahead > 0
                )
                if use_fast_backup:
                    miss_specs, miss_fork_tokens = self._speculate_low_latency_miss_batch(
                        [token_batches[idx] for idx in miss_indices],
                        [recovery_tokens[idx] for idx in miss_indices],
                        [temperatures[idx] for idx in miss_indices],
                        lookahead,
                        min(lookahead, max(1, self.config.async_fast_miss_steps)),
                        fan_out_list_miss,
                    )
                else:
                    miss_specs, _, _, miss_fork_tokens = self.speculate_stateless_batch(
                        [token_batches[idx] for idx in miss_indices],
                        [recovery_tokens[idx] for idx in miss_indices],
                        [temperatures[idx] for idx in miss_indices],
                        lookahead,
                        return_step_logits=False,
                        return_all_logits=False,
                        fork_counts=fan_out_list_miss,
                    )
                miss_logits_q = None
                miss_all_logits = None
                assert miss_fork_tokens is not None
                for local_idx, batch_idx in enumerate(miss_indices):
                    hit_fork_overrides[batch_idx] = self._to_cpu_pinned(miss_fork_tokens[local_idx])

            miss_index_t = torch.tensor(miss_indices, dtype=torch.long, device=self.device)
            speculations[miss_index_t] = miss_specs
            if logits_q is not None:
                assert miss_logits_q is not None
                logits_q[miss_index_t] = miss_logits_q
            if miss_all_logits is not None:
                computed_indices.extend(miss_indices)
                computed_logits.append(miss_all_logits)

        if hit_trace_indices:
            hit_specs = torch.stack(hit_trace_specs_cpu, dim=0)
            hit_all_logits = self._trace_speculation_paths(
                [token_batches[idx] for idx in hit_trace_indices],
                hit_specs,
                lookahead,
            )
            hit_index_t = torch.tensor(hit_trace_indices, dtype=torch.long, device=self.device)
            if logits_q is not None:
                logits_q[hit_index_t] = hit_all_logits[:, :lookahead, :]
            computed_indices.extend(hit_trace_indices)
            computed_logits.append(hit_all_logits)

        response = DraftResponse(
            speculations=self._to_cpu_pinned(speculations),
            logits_q=self._to_cpu_pinned(logits_q),
            cache_hits=self._to_cpu_pinned(cache_hits),
            from_q_mask=torch.ones(batch_size, dtype=torch.bool),
            lookahead=lookahead,
            fan_out_cap=max(max(fan_out_list, default=0), max(fan_out_list_miss, default=0)),
        )
        combined_logits = torch.cat(computed_logits, dim=0) if computed_logits else None
        return response, computed_indices, combined_logits, hit_fork_overrides

    @torch.inference_mode()
    def _populate_next_cache(
        self,
        seqs: list[Sequence],
        response: DraftResponse,
        computed_indices: list[int],
        computed_all_logits: torch.Tensor | None,
        fork_overrides: dict[int, torch.Tensor],
    ) -> tuple[int, bool]:
        """Populate next cache.
        
        Args:
            seqs (list[Sequence]): List of sequences
            response (DraftResponse): Draft response
            computed_indices (list[int]): Computed indices
            computed_all_logits (torch.Tensor | None): Computed all logits
            fork_overrides (dict[int, torch.Tensor]): Fork overrides
            
        Returns:
            tuple[int, bool]: Number of branches and whether fast populate was used
        """
        if not seqs:
            self._tree_cache.clear()
            return 0, False

        lookahead = response.lookahead
        fan_out_list, fan_out_list_miss = self._effective_fan_out_lists(len(seqs), lookahead)

        forked_recovery_tokens_by_idx: dict[int, list[int]] = {
            batch_idx: tokens.tolist()
            for batch_idx, tokens in fork_overrides.items()
        }

        if computed_indices:
            assert computed_all_logits is not None
            computed_index_t = torch.tensor(computed_indices, dtype=torch.long, device=self.device)
            computed_cache_hits = response.cache_hits.to(self.device)[computed_index_t]
            computed_speculations = response.speculations.to(self.device)[computed_index_t]
            computed_forked_recovery_tokens = get_forked_recovery_tokens_from_logits(
                self.config,
                computed_all_logits,
                computed_cache_hits,
                computed_speculations,
                tokenizer=self.tokenizer,
                fan_out_list=fan_out_list,
                fan_out_list_miss=fan_out_list_miss,
                lookahead=lookahead,
            )
            for local_idx, batch_idx in enumerate(computed_indices):
                forked_recovery_tokens_by_idx[batch_idx] = computed_forked_recovery_tokens[local_idx].tolist()

        branch_token_batches: list[list[int]] = []
        branch_recovery_tokens: list[int] = []
        branch_temperatures: list[float] = []
        branch_keys: list[tuple[int, int, int]] = []

        for batch_idx, seq in enumerate(seqs):
            spec_row = response.speculations[batch_idx].tolist()
            candidate_tokens = forked_recovery_tokens_by_idx[batch_idx]
            counts = fan_out_list if response.cache_hits[batch_idx].item() else fan_out_list_miss
            temperature = seq.draft_temperature if seq.draft_temperature is not None else seq.temperature

            offset = 0
            for accepted_count, fan_out in enumerate(counts):
                accepted_prefix = spec_row[:accepted_count + 1]
                base_tokens = list(seq.token_ids) + accepted_prefix
                for _ in range(fan_out):
                    recovery_token = candidate_tokens[offset]
                    offset += 1
                    branch_keys.append((seq.seq_id, accepted_count, recovery_token))
                    branch_token_batches.append(base_tokens)
                    branch_recovery_tokens.append(recovery_token)
                    branch_temperatures.append(temperature)

        if not branch_keys:
            self._tree_cache.clear()
            return 0, False

        use_fast_populate = self._should_use_fast_populate(
            batch_size=len(seqs),
            lookahead=lookahead,
            branch_count=len(branch_keys),
            branch_temperatures=branch_temperatures,
        )

        if use_fast_populate:
            branch_specs, branch_fork_recovery_tokens = self._speculate_low_latency_miss_batch(
                branch_token_batches,
                branch_recovery_tokens,
                branch_temperatures,
                lookahead,
                min(lookahead, max(1, self.config.async_fast_populate_steps)),
                fan_out_list,
            )
            branch_fork_recovery_tokens = self._maybe_pin_cpu(branch_fork_recovery_tokens.cpu())
        elif all(temperature == 0 for temperature in branch_temperatures):
            branch_specs, _, _, branch_fork_recovery_tokens = self.speculate_stateless_batch(
                branch_token_batches,
                branch_recovery_tokens,
                branch_temperatures,
                lookahead,
                return_step_logits=False,
                return_all_logits=False,
                fork_counts=fan_out_list,
            )
            assert branch_fork_recovery_tokens is not None
            branch_fork_recovery_tokens = self._maybe_pin_cpu(branch_fork_recovery_tokens.cpu())
        else:
            branch_specs, _, branch_all_logits, _ = self.speculate_stateless_batch(
                branch_token_batches,
                branch_recovery_tokens,
                branch_temperatures,
                lookahead,
                return_step_logits=False,
            )
            assert branch_all_logits is not None
            branch_cache_hits = torch.ones(len(branch_keys), dtype=torch.bool, device=self.device)
            branch_fork_recovery_tokens = get_forked_recovery_tokens_from_logits(
                self.config,
                branch_all_logits,
                branch_cache_hits,
                branch_specs,
                tokenizer=self.tokenizer,
                fan_out_list=fan_out_list,
                fan_out_list_miss=fan_out_list_miss,
                lookahead=lookahead,
            )
            branch_fork_recovery_tokens = self._maybe_pin_cpu(branch_fork_recovery_tokens.cpu())

        branch_specs_cpu = self._maybe_pin_cpu(branch_specs.cpu())
        gpu_cache = branch_specs if self._should_keep_gpu_cache(branch_specs) else None
        self._tree_cache = {
            key: CachedDraftState(
                speculation=branch_specs_cpu[idx].clone(),
                fork_recovery_tokens=self._maybe_pin_cpu(branch_fork_recovery_tokens[idx].clone()),
                lookahead=lookahead,
                fork_width=int(branch_fork_recovery_tokens[idx].numel()),
                gpu_speculation=gpu_cache[idx].clone() if gpu_cache is not None else None,
            )
            for idx, key in enumerate(branch_keys)
        }
        return len(branch_keys), use_fast_populate

    def _draft_loop(self):
        """Draft loop for asynchronous operation."""
        assert self._request_queue is not None
        if self.device.type == "cuda":
            cuda_index = 0 if self.device.index is None else self.device.index
            torch.cuda.set_device(cuda_index)

        while True:
            request: DraftRequest = self._request_queue.get()

            if request.kind == "shutdown":
                break

            if request.kind == "prefill":
                self._tree_cache.clear()
                continue

            if request.kind != "speculate":
                raise RuntimeError(f"Unknown draft request kind: {request.kind}")

            assert request.seqs is not None
            assert request.response_q is not None

            start = time.perf_counter()
            try:
                serve_start = time.perf_counter()
                response, computed_indices, computed_all_logits, fork_overrides = self._serve_speculation_request(
                    request.seqs
                )
                serve_elapsed = time.perf_counter() - serve_start
                response.serve_ms = serve_elapsed * 1000.0
                request.response_q.put(response)
                populate_start = time.perf_counter()
                populate_branch_count, used_fast_populate = self._populate_next_cache(
                    request.seqs,
                    response,
                    computed_indices,
                    computed_all_logits,
                    fork_overrides,
                )
                populate_elapsed = time.perf_counter() - populate_start
                total_elapsed = time.perf_counter() - start
                self._draft_step_times.append(total_elapsed)
                self._worker_profile["worker_total_times"].append(total_elapsed)
                self._worker_profile["worker_serve_times"].append(serve_elapsed)
                self._worker_profile["cache_populate_times"].append(populate_elapsed)
                self._worker_profile["populate_branch_counts"].append(populate_branch_count)
                self._worker_profile["fast_populate_flags"].append(1 if used_fast_populate else 0)
                self._worker_profile["worker_batch_sizes"].append(len(request.seqs))
                self._worker_profile["tree_cache_sizes"].append(len(self._tree_cache))
            except Exception as exc:  # pragma: no cover - surfaced to caller
                self._worker_error = exc
                request.response_q.put(exc)
                break
