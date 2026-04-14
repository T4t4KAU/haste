# Baseline Benchmarks

This folder contains baseline autoregressive benchmarks for fair comparison with Haste.

## Baselines

1. **CPU AR**: pure CPU autoregressive inference (no draft model).
2. **CPU+GPU Offload AR**: device-map offload inference (no draft model).

## Run Baselines

CPU AR:

```bash
python -O baselines/bench_baselines.py \
  --mode cpu \
  --model-path /path/to/Qwen3-4B \
  --dataset-root ./datasets \
  --datasets alpaca,gsm8k \
  --prompt-limit 160 \
  --max-num-seqs 8 \
  --max-new-tokens 128 \
  --use-tqdm
```

CPU+GPU offload AR:

```bash
python -O baselines/bench_baselines.py \
  --mode offload \
  --model-path /path/to/Qwen3-4B \
  --dataset-root ./datasets \
  --datasets alpaca,gsm8k \
  --prompt-limit 160 \
  --max-num-seqs 8 \
  --max-new-tokens 128 \
  --use-tqdm
```

You can adjust GPU offload aggressiveness:

```bash
python -O baselines/bench_baselines.py \
  --mode offload \
  --model-path /path/to/Qwen3-4B \
  --dataset-root ./datasets \
  --datasets alpaca,gsm8k \
  --prompt-limit 160 \
  --max-num-seqs 8 \
  --max-new-tokens 128 \
  --offload-gpu-mem-frac 0.6 \
  --use-tqdm
```

## Fair Comparison Tips

- Use the same datasets, prompt limit, `max_num_seqs`, and `max_new_tokens`.
- Use the same model (`Qwen3-4B`) for baselines and Haste target.
- Keep temperature at 0 (greedy) for all runs.
- Run on the same machine and isolate background GPU/CPU workloads.
