# Haste: Hardware-Aware Asynchronous Speculative Decoding for Low-VRAM Heterogeneous Inference

Haste is a high-performance LLM inference engine that implements asynchronous speculative decoding to accelerate text generation.

## Installation

### Using Conda

1. Create a conda environment:
   ```bash
   conda create -n dl python=3.12
   conda activate dl
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### example.py

The `example.py` script demonstrates basic usage of the Haste engine with a few sample prompts and prints a compact profiling summary.

#### Command Line Arguments

- `--target-model-path`: Path to the target model (required)
- `--draft-model-path`: Path to the draft model (required)
- `--max-new-tokens`: Maximum number of new tokens to generate (default: 128)
- `--temperature`: Sampling temperature for target model (default: 0.0)
- `--draft-temperature`: Sampling temperature for draft model (default: 0.0)
- `--speculate-k`: Number of tokens to speculate (default: 7)
- `--async-fan-out`: Number of async draft runners (default: 3)
- `--auto-tune-kf`: Dynamically search and adjust speculative lookahead/fan-out at runtime
- `--profile-output`: Optional JSON file for the profiling report
- `--include-raw-metrics`: Include raw per-step metric series in the profiling report

#### Example Usage

```bash
python -O example.py \
    --target-model-path /path/to/Qwen3-32B \
    --draft-model-path /path/to/Qwen3-0.6B \
    --auto-tune-kf \
    --profile-output outputs/example_profile.json
```

Example with local ModelScope directories:

```bash
python -O example.py \
    --target-model-path /path/to/models/Qwen3-32B \
    --draft-model-path /path/to/models/Qwen3-0.6B
```

### bench.py

The `bench.py` script benchmarks Haste on real prompts and now emits both throughput metrics and a stage-level profiling breakdown.

#### Command Line Arguments

- `--target-model-path`: Path to the target model (required)
- `--draft-model-path`: Path to the draft model (required)
- `--dataset-root`: Root directory for datasets 
- `--datasets`: Comma-separated dataset names under dataset-root, or `all` (default: `alpaca,gsm8k,humaneval,mt_bench_1,qa,sum`)
- `--dataset-file`: Optional explicit JSONL file. Overrides `--datasets`
- `--prompt-limit`: Maximum number of prompts to benchmark (default: 32)
- `--max-num-seqs`: Maximum number of concurrent sequences (default: 0, uses hardware-based default)
- `--max-num-batched-tokens`: Maximum number of batched tokens for the scheduler (default: 4096)
- `--max-new-tokens`: Maximum number of new tokens to generate (default: 128)
- `--temperature`: Sampling temperature for target model (default: 0.0)
- `--draft-temperature`: Sampling temperature for draft model (default: 0.0)
- `--speculate-k`: Number of tokens to speculate (default: 7)
- `--async-fan-out`: Number of async draft runners (default: 3)
- `--auto-tune-kf`: Dynamically search and adjust speculative lookahead/fan-out at runtime
- `--max-model-len`: Maximum model length (default: 4096)
- `--warmup-runs`: Number of warmup runs (default: 1)
- `--turn-index`: Which item from `turns` to use when a prompt file is multi-turn (default: 0)
- `--join-turns`: Join every element in `turns` into one prompt instead of using `--turn-index`
- `--shuffle`: Shuffle prompts before applying `--prompt-limit`
- `--seed`: Random seed for shuffling (default: 0)
- `--enforce-eager`: Disable CUDA graph capture and force eager mode
- `--use-tqdm`: Show generation progress bars
- `--profile-output`: Optional JSON file for the full profiling report
- `--include-raw-metrics`: Include raw per-step metric series in the profiling report

#### Example Usage

```bash
python -O bench.py \
    --target-model-path /path/to/Qwen3-32B \
    --draft-model-path /path/to/Qwen3-0.6B \
    --dataset-root /path/to/datasets \
    --datasets alpaca,gsm8k \
    --prompt-limit 16 \
    --max-num-seqs 8 \
    --max-new-tokens 128 \
    --auto-tune-kf \
    --profile-output outputs/bench_profile.json
```

Example with local ModelScope directories:

```bash
python -O bench.py \
    --target-model-path /path/to/models/Qwen3-32B \
    --draft-model-path /path/to/models/Qwen3-0.6B \
    --dataset-root ./datasets \
    --datasets alpaca,gsm8k \
    --prompt-limit 20 \
    --max-num-seqs 8 \
    --max-new-tokens 128 \
    --auto-tune-kf
```

## Output

Both scripts generate detailed metrics about the inference process, including:

- Prefill throughput (tokens/second)
- Decode throughput (tokens/second)
- Overall throughput (tokens/second)
- Average cache hit rate
- Speculative token acceptance rate
- Mean / p50 / p95 stage latency for scheduling, speculation, verification, rollback, and post-processing
- Target runner and draft worker timing breakdowns
- Wall time
- Total processed tokens
- Requested new tokens
- Generated new tokens

The optional profiling JSON contains the same summary in machine-readable form, and `--include-raw-metrics` additionally stores the raw per-step time series for offline analysis.

## Profiling

Haste supports two profiling modes:

- Console profiling: prints a compact latency and throughput summary after generation.
- JSON profiling: saves the full profiling report to a file with `--profile-output`.

### Quick Profiling With `example.py`

Use `example.py` when you want a fast sanity check on a small fixed prompt set:

```bash
python -O example.py \
    --target-model-path /path/to/Qwen3-32B \
    --draft-model-path /path/to/Qwen3-0.6B \
    --profile-output outputs/example_profile.json
```

If you also want raw per-step series for offline analysis:

```bash
python -O example.py \
    --target-model-path /path/to/Qwen3-32B \
    --draft-model-path /path/to/Qwen3-0.6B \
    --profile-output outputs/example_profile_raw.json \
    --include-raw-metrics
```

### Benchmark Profiling With `bench.py`

Use `bench.py` for dataset-level throughput and profiling:

```bash
python -O bench.py \
    --target-model-path /path/to/Qwen3-32B \
    --draft-model-path /path/to/Qwen3-0.6B \
    --dataset-root /path/to/datasets \
    --datasets alpaca,gsm8k \
    --prompt-limit 32 \
    --max-num-seqs 8 \
    --max-new-tokens 128 \
    --profile-output outputs/bench_profile.json
```

For a smaller smoke test:

```bash
python -O bench.py \
    --target-model-path /path/to/Qwen3-32B \
    --draft-model-path /path/to/Qwen3-0.6B \
    --dataset-root /path/to/datasets \
    --datasets alpaca \
    --prompt-limit 1 \
    --max-num-seqs 1 \
    --max-new-tokens 4 \
    --warmup-runs 0 \
    --profile-output outputs/bench_smoke_profile.json
```

### What The Profiler Reports

The profiling summary is organized into these parts:

- `throughput`: prefill, decode, overall, and generation throughput
- `cache`: asynchronous cache hit statistics
- `acceptance`: speculative acceptance statistics
- `stages`: engine-level timing for scheduler, speculate, verify, rollback, and postprocess
- `runners.target`: target runner timing split into prepare, model, sample, and total
- `runners.draft_model`: draft model timing split into prepare, model, sample, and total
- `runners.draft_worker`: async worker timing such as request wait, serve, and cache population

Latency summaries are reported as `mean`, `p50`, `p95`, and `max`.

### JSON Report Notes

The JSON report produced by `--profile-output` is intended for scripting and plotting.

- Use the top-level `throughput` field for aggregate performance comparisons.
- Use `stages` to locate engine-side bottlenecks.
- Use `runners.draft_worker` to check whether async speed is being limited by queue wait or cache population.
- Use `--include-raw-metrics` only when you need detailed traces, because it makes the report larger.

## Models

Haste currently supports Qwen3 models. You can use any Qwen3 model as the target model, and a smaller Qwen3 model as the draft model for speculative decoding.
