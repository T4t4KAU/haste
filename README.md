# Haste: Hardware-Aware Asynchronous Speculative Decoding for Low-VRAM Heterogeneous Inference

Haste is a high-performance LLM inference engine that implements asynchronous speculative decoding to accelerate text generation. It leverages CPU-GPU collaboration to enable efficient inference on edge devices with limited VRAM.

## Key Features

- **CPU-GPU Collaborative Inference**: Uses CPU for large target models and GPU for smaller draft models, optimizing resource utilization
- **Asynchronous Speculative Decoding**: GPU speculates next tokens while CPU verifies, maximizing hardware utilization
- **Low VRAM Requirements**: Designed for edge devices with limited GPU memory
- **Auto-tuning**: Dynamically adjusts speculative parameters for optimal performance
- **Progress Bar Support**: Visual generation progress with `--use-tqdm` parameter
- **Multi-Model Support**: Now supports Qwen2.5, Qwen3, Llama3.2, and SmolLM2 models

## Installation

### Using Conda

1. Create a conda environment:
   ```bash
   conda create -n haste python=3.12
   conda activate haste
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
- `--mode`: Decoding mode: `ar`, `spec_sync`, or `spec_async` (default: `spec_async`)
- `--draft-model-path`: Path to the draft model. Required for `spec_sync` and `spec_async`
- `--max-new-tokens`: Maximum number of new tokens to generate (default: 128)
- `--temperature`: Sampling temperature for target model (default: 0.0)
- `--draft-temperature`: Sampling temperature for draft model (default: 0.0)
- `--speculate-k`: Number of tokens to speculate (default: 1)
- `--async-fan-out`: Number of async draft runners (default: 1)
- `--auto-tune-kf`: Dynamically search and adjust speculative lookahead/fan-out at runtime
- `--verbose`: Enable detailed runtime logs, including auto-tune K/F transitions
- `--profile-output`: Optional JSON file for the profiling report
- `--include-raw-metrics`: Include raw per-step metric series in the profiling report
- `--use-tqdm`: Show generation progress bars

#### Example Usage

```bash
python -O example.py \
    --mode spec_async \
    --target-model-path /path/to/Qwen3-4B \
    --draft-model-path /path/to/Qwen3-0.6B \
    --auto-tune-kf \
    --use-tqdm \
    --profile-output outputs/example_profile.json
```

Using Llama3.2 models:

```bash
python -O example.py \
    --mode spec_async \
    --target-model-path /path/to/Llama-3.2-3B \
    --draft-model-path /path/to/Llama-3.2-1B \
    --auto-tune-kf \
    --use-tqdm
```

Using SmolLM2 models:

```bash
python -O example.py \
    --mode spec_async \
    --target-model-path /path/to/SmolLM2-1.7B \
    --draft-model-path /path/to/SmolLM2-135M \
    --auto-tune-kf \
    --use-tqdm
```

Autoregressive comparison:

```bash
python -O example.py \
    --mode ar \
    --target-model-path /path/to/models/Qwen3-4B
```

Synchronous speculative decoding comparison:

```bash
python -O example.py \
    --mode spec_sync \
    --target-model-path /path/to/models/Qwen3-4B \
    --draft-model-path /path/to/models/Qwen3-0.6B
```

### bench.py

The `bench.py` script benchmarks Haste on real prompts and now emits both throughput metrics and a stage-level profiling breakdown.

#### Command Line Arguments

- `--target-model-path`: Path to the target model (required)
- `--mode`: Decoding mode: `ar`, `spec_sync`, or `spec_async` (default: `spec_async`)
- `--draft-model-path`: Path to the draft model. Required for `spec_sync` and `spec_async`
- `--dataset-root`: Root directory for datasets 
- `--datasets`: Comma-separated dataset names under dataset-root, or `all` (default: `alpaca,gsm8k,humaneval,mt_bench_1,qa,sum`)
- `--dataset-file`: Optional explicit JSONL file. Overrides `--datasets`
- `--prompt-limit`: Maximum number of prompts to benchmark (default: 32)
- `--max-num-seqs`: Maximum number of concurrent sequences (default: 0, uses hardware-based default)
- `--max-num-batched-tokens`: Maximum number of batched tokens for the scheduler (default: 4096)
- `--max-new-tokens`: Maximum number of new tokens to generate (default: 128)
- `--temperature`: Sampling temperature for target model (default: 0.0)
- `--draft-temperature`: Sampling temperature for draft model (default: 0.0)
- `--speculate-k`: Number of tokens to speculate (default: 1)
- `--async-fan-out`: Number of async draft runners (default: 1)
- `--auto-tune-kf`: Dynamically search and adjust speculative lookahead/fan-out at runtime
- `--verbose`: Enable detailed runtime logs, including auto-tune K/F transitions
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
    --mode spec_async \
    --target-model-path /path/to/Qwen3-4B \
    --draft-model-path /path/to/Qwen3-0.6B \
    --dataset-root /path/to/datasets \
    --datasets alpaca,gsm8k \
    --prompt-limit 16 \
    --max-num-seqs 8 \
    --max-new-tokens 128 \
    --auto-tune-kf \
    --use-tqdm \
    --profile-output outputs/bench_profile.json
```

### server.py

The `server.py` script starts a lightweight HTTP service around a single loaded Haste engine instance. It uses only the Python standard library for the web layer, so there are no extra serving dependencies beyond the core runtime stack.

#### Command Line Arguments

- `--host`: Host to bind the server (default: `0.0.0.0`)
- `--port`: Port to bind the server (default: `8000`)
- `--mode`: Decoding mode: `ar`, `spec_sync`, or `spec_async` (default: `spec_async`)
- `--target-model-path`: Path to the target model (required)
- `--draft-model-path`: Path to the draft model. Required for `spec_sync` and `spec_async`
- `--max-num-seqs`: Maximum number of concurrent sequences (default: `32`)
- `--max-num-batched-tokens`: Maximum number of batched tokens for the scheduler (default: `4096`)
- `--max-model-len`: Maximum model length (default: `4096`)
- `--default-max-new-tokens`: Default generation length when the request does not override it (default: `128`)
- `--speculate-k`: Number of tokens to speculate (default: `1`)
- `--async-fan-out`: Number of async draft runners (default: `1`)
- `--auto-tune-kf`: Dynamically search and adjust speculative lookahead/fan-out at runtime
- `--enforce-eager`: Disable CUDA graph capture and force eager mode
- `--verbose`: Enable verbose runtime logs

## Output

Both scripts generate detailed metrics about the inference process, including:

- Prefill throughput (tokens/second)
- Decode throughput (tokens/second)
- Overall throughput (tokens/second)
- Average cache hit rate for `spec_async`
- Speculative token acceptance rate for `spec_sync` and `spec_async`
- Mean / p50 / p95 stage latency for scheduling, speculation, verification, rollback, and post-processing
- Target runner timing breakdowns
- Draft worker timing breakdowns for `spec_async`
- Wall time
- Total processed tokens
- Requested new tokens
- Generated new tokens

The optional profiling JSON contains the same summary in machine-readable form, and `--include-raw-metrics` additionally stores the raw per-step time series for offline analysis.

## Profiling

Haste supports two profiling modes:

- Console profiling: prints a compact latency and throughput summary after generation.
- JSON profiling: saves the full profiling report to a file with `--profile-output`.

## Models

Haste currently supports the following models:

- **Qwen2.5**: Tested with Qwen2.5-3B-Instruct (target) and Qwen2.5-0.5B-Instruct (draft)
- **Qwen3**: Any Qwen3 model as target, smaller Qwen3 as draft
- **Llama3.2**: Tested with Llama-3.2-3B (target) and Llama-3.2-1B (draft)
- **SmolLM2**: Tested with SmolLM2-1.7B (target) and SmolLM2-135M (draft)

## Use Cases

Haste is particularly well-suited for:

- **Edge Devices**: Low-VRAM GPUs (e.g., consumer laptops, small form-factor PCs)
- **Heterogeneous Environments**: Systems with both CPU and GPU resources
- **Real-time Applications**: Where low latency is critical
- **Cost-Constrained Deployments**: Maximizing existing hardware utilization

## How It Works

1. **Asynchronous Speculation**: GPU runs a small draft model to speculate next tokens
2. **CPU Verification**: CPU runs the large target model to verify speculations
3. **Parallel Execution**: GPU speculates next batch while CPU verifies current batch
4. **Cache Optimization**: Reuses speculative results to reduce redundant computation
5. **Auto-tuning**: Dynamically adjusts speculation parameters for optimal performance

This approach enables efficient use of both CPU and GPU resources, making it possible to run larger models on devices with limited GPU memory.
