"""Configuration module for Haste LLM engine."""

import os
from dataclasses import dataclass
from transformers import AutoConfig
import torch

from haste.utils.misc import resolve_pretrained_path


@dataclass
class Config:
    """Configuration class for Haste LLM engine."""
    model: str  # Path to the target model
    max_num_batched_tokens: int = 16384  # Maximum number of batched tokens
    max_num_seqs: int = 512  # Maximum number of concurrent sequences
    max_model_len: int = 4096  # Maximum model sequence length
    
    draft_device: torch.device = torch.device("cuda")  # Device for draft model
    target_device: torch.device = torch.device("cpu")  # Device for target model
    
    gpu_memory_utilization: float = 0.7  # GPU memory utilization factor
    cpu_memory_utilization: float = 0.5  # CPU memory utilization factor
    
    enforce_eager: bool = False  # Whether to enforce eager mode
    target_hf_config: AutoConfig | None = None  # Hugging Face config for target model
    eos: int = -1  # End-of-sequence token
    kvcache_block_size: int = 256  # KV cache block size
    num_kvcache_blocks: int = -1  # Number of KV cache blocks
    
    # speculator config
    draft_hf_config: AutoConfig | None = None  # Hugging Face config for draft model
    draft_model: str | None = None  # Path to the draft model
    speculate: bool = False  # Whether to use speculative decoding
    speculate_k: int = 1  # Number of tokens to speculate
    draft_async: bool = False  # Whether to use asynchronous draft model
    
    async_fan_out: int = 3  # Number of async draft runners
    fan_out_list: list[int] | None = None  # List of fan-out values
    fan_out_list_miss: list[int] | None = None  # List of fan-out values for cache misses
    sampler_x: float | None = None  # Sampler X parameter
    jit_speculate: bool = False  # Whether to use JIT for speculation
    async_auto_tune: bool = False  # Whether to auto-tune async parameters
    async_auto_tune_min_k: int = 1  # Minimum speculative lookahead for auto-tune
    async_auto_tune_max_k: int | None = None  # Maximum speculative lookahead for auto-tune
    async_auto_tune_min_f: int = 1  # Minimum fan-out for auto-tune
    async_auto_tune_max_f: int | None = None  # Maximum fan-out for auto-tune
    async_auto_tune_probe_steps: int = 2  # Number of probe steps for auto-tune
    async_auto_tune_reprobe_interval: int = 16  # Interval between reprobes for auto-tune
    async_auto_tune_margin: float = 0.95  # Margin for auto-tune
    async_auto_tune_wait_ratio: float = 0.08  # Wait ratio for auto-tune
    async_auto_tune_underfill_ratio: float = 0.55  # Underfill ratio for auto-tune
    async_auto_tune_accept_floor: float = 0.35  # Acceptance floor for auto-tune
    async_auto_tune_cache_hit_target: float = 0.7  # Cache hit target for auto-tune
    async_auto_tune_ema_alpha: float = 0.35  # EMA alpha for auto-tune
    async_auto_tune_score_tolerance: float = 0.05  # Score tolerance for auto-tune
    async_adaptive_lookahead: bool = True  # Whether to use adaptive lookahead
    async_mid_batch_threshold: int = 4  # Mid-batch threshold for adaptive lookahead
    async_large_batch_threshold: int = 6  # Large-batch threshold for adaptive lookahead
    async_mid_batch_k: int | None = 5  # Lookahead for mid-sized batches
    async_large_batch_k: int | None = 4  # Lookahead for large batches
    async_fast_miss_threshold: int = 4  # Threshold for fast miss detection
    async_fast_miss_steps: int = 2  # Steps for fast miss detection
    async_fast_populate_threshold: int = 4  # Threshold for fast populate
    async_fast_populate_steps: int = 2  # Steps for fast populate
    async_fast_populate_branch_factor: int = 2  # Branch factor for fast populate
    
    # debugging config
    verbose: bool = False  # Whether to enable verbose logging
    debug_mode: bool = False  # Whether to enable debug mode
    max_steps: int | None = None  # Maximum number of steps
    
    fan_out_t: int = 1  # Fan-out for target model
    fan_out_t_miss: int = 1  # Fan-out for target model on cache misses

    @property
    def max_blocks(self):
        """Calculate maximum number of KV cache blocks."""
        return (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size
    
    def __post_init__(self):
        """Initialize configuration after creation."""
        model = resolve_pretrained_path(self.model)
        self.model = model
        assert os.path.isdir(model)

        self.target_hf_config = AutoConfig.from_pretrained(
            model,
            local_files_only=True,
        )
        
        self.max_model_len = min(self.max_model_len, self.target_hf_config.max_position_embeddings)
        
        # If speculation is enabled
        if self.speculate:
            assert self.draft_model is not None, "ERROR in Config: draft_model must be provided when speculate is True"
            draft_model = resolve_pretrained_path(self.draft_model)
            self.draft_model = draft_model
            assert os.path.isdir(draft_model), f"ERROR in Config: draft_model path {draft_model} does not exist"

            self.draft_hf_config = AutoConfig.from_pretrained(
                draft_model,
                local_files_only=True,
            )
            
            self.max_model_len = min(self.max_model_len, self.draft_hf_config.max_position_embeddings)
            
            # If async speculation is enabled
            if self.draft_async:
                auto_tune_max_k = self.speculate_k if self.async_auto_tune_max_k is None else self.async_auto_tune_max_k
                self.async_auto_tune_max_k = max(1, min(self.speculate_k, auto_tune_max_k))
                self.async_auto_tune_min_k = max(1, min(self.async_auto_tune_min_k, self.async_auto_tune_max_k))

                auto_tune_max_f = self.async_fan_out if self.async_auto_tune_max_f is None else self.async_auto_tune_max_f
                self.async_auto_tune_max_f = max(1, min(self.async_fan_out, auto_tune_max_f))
                self.async_auto_tune_min_f = max(1, min(self.async_auto_tune_min_f, self.async_auto_tune_max_f))

                if self.fan_out_list is None:
                    self.fan_out_list = [self.async_fan_out] * (self.speculate_k + 1)
                
                if self.fan_out_list_miss is None:
                    self.fan_out_list_miss = list(self.fan_out_list)
                
                assert sum(self.fan_out_list_miss) == sum(self.fan_out_list), "ERROR in Config: fan_out_list_miss must be the same as fan_out_list"
                self.MQ_LEN = sum(self.fan_out_list)
                    
        assert self.max_num_batched_tokens >= self.max_model_len, "ERROR in Config: max_num_batched_tokens must be at least max_model_len"
