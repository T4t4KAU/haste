"""Baseline benchmark script for autoregressive inference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bench


@dataclass(frozen=True)
class BatchResult:
    prefill_tokens: int
    decode_tokens: int
    prefill_time: float
    decode_time: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Baseline benchmarks for autoregressive inference (CPU and CPU+GPU offload).",
    )
    parser.add_argument(
        "--mode",
        choices=("cpu", "offload"),
        default="cpu",
        help="Baseline mode: cpu for pure CPU AR, offload for CPU+GPU offload AR.",
    )
    parser.add_argument("--model-path", required=True, help="Path to the model.")
    parser.add_argument("--dataset-root", required=True, help="Root directory for datasets.")
    parser.add_argument(
        "--datasets",
        default="alpaca,gsm8k,humaneval,mt_bench_1,qa,sum",
        help="Comma-separated dataset names under dataset-root, or `all`.",
    )
    parser.add_argument(
        "--dataset-file",
        default="",
        help="Optional explicit JSONL file. Overrides --datasets.",
    )
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=32,
        help="Maximum number of prompts to benchmark. 0 means all loaded prompts.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=8,
        help="Maximum number of concurrent sequences per batch.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use-tqdm",
        action="store_true",
        help="Show generation progress bars.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="auto",
        help="Model dtype. Use float32 for CPU baseline.",
    )
    parser.add_argument(
        "--offload-gpu-mem-frac",
        type=float,
        default=0.6,
        help="GPU memory fraction for device_map=auto in offload mode.",
    )
    return parser


def _resolve_dtype(mode: str, dtype: str) -> torch.dtype | None:
    if dtype == "auto":
        if mode == "cpu":
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    return None


def _input_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device") and str(model.device) != "meta":
        return model.device
    if hasattr(model, "hf_device_map"):
        devices = list({str(v) for v in model.hf_device_map.values()})
        if any("cuda" in device for device in devices):
            return torch.device("cuda")
        if devices:
            return torch.device(devices[0])
    return torch.device("cpu")


def _load_model(mode: str, model_path: str, dtype: torch.dtype | None, offload_frac: float) -> torch.nn.Module:
    if mode == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        return model.cpu()

    max_memory = None
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_gb = max(1, int(total_gb * offload_frac))
        max_memory = {
            0: f"{gpu_gb}GiB",
            "cpu": "256GiB",
        }

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        max_memory=max_memory,
        local_files_only=True,
    )


def _batch_iter(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


@torch.inference_mode()
def _run_batch(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    use_tqdm: bool,
) -> BatchResult:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    prompt_lengths = attention_mask.sum(dim=1).tolist()

    device = _input_device(model)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    prefill_tokens = int(sum(prompt_lengths))
    decode_tokens = int(len(prompts) * max_new_tokens)

    start = perf_counter()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    prefill_time = perf_counter() - start

    logits = outputs.logits
    past = outputs.past_key_values

    progress = None
    if use_tqdm:
        progress = tqdm(total=decode_tokens, desc="Generating", dynamic_ncols=True)

    decode_time = 0.0
    for _ in range(max_new_tokens):
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype)],
            dim=1,
        )
        start = perf_counter()
        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        decode_time += perf_counter() - start
        logits = outputs.logits
        past = outputs.past_key_values
        if progress is not None:
            progress.update(len(prompts))

    if progress is not None:
        progress.close()

    return BatchResult(
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
        prefill_time=prefill_time,
        decode_time=decode_time,
    )


def _print_throughput(prefill_tokens: int, decode_tokens: int, prefill_time: float, decode_time: float) -> None:
    total_tokens = prefill_tokens + decode_tokens
    total_time = prefill_time + decode_time
    print("\nProfiling Summary")
    print("-" * 60)
    if prefill_time > 0:
        print(f"Prefill throughput: {prefill_tokens / prefill_time:.2f} tok/s")
    if decode_time > 0:
        print(f"Decode throughput: {decode_tokens / decode_time:.2f} tok/s")
    if total_time > 0:
        print(f"Overall throughput: {total_tokens / total_time:.2f} tok/s")
        print(f"Generation throughput: {decode_tokens / total_time:.2f} tok/s")


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dataset_files = bench.resolve_dataset_files(dataset_root, args.datasets, args.dataset_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    records = bench.load_prompt_records(
        dataset_files,
        tokenizer,
        prompt_limit=args.prompt_limit,
        turn_index=0,
        join_turns=False,
        shuffle=False,
        seed=args.seed,
    )
    prompts = [record.prompt for record in records]

    dtype = _resolve_dtype(args.mode, args.dtype)
    model = _load_model(args.mode, args.model_path, dtype, args.offload_gpu_mem_frac)
    model.eval()

    batches = _batch_iter(prompts, args.max_num_seqs)
    prefill_tokens = 0
    decode_tokens = 0
    prefill_time = 0.0
    decode_time = 0.0

    for batch_prompts in batches:
        result = _run_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            max_new_tokens=args.max_new_tokens,
            use_tqdm=args.use_tqdm,
        )
        prefill_tokens += result.prefill_tokens
        decode_tokens += result.decode_tokens
        prefill_time += result.prefill_time
        decode_time += result.decode_time

    print("\nBenchmark Summary")
    print("-" * 60)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_path}")
    print(f"Loaded prompts: {len(prompts)}")

    prompt_min, prompt_max, prompt_avg = bench.summarize_prompt_lengths(tokenizer, prompts)
    print(f"Prompt token length min/avg/max: {prompt_min}/{prompt_avg:.1f}/{prompt_max}")
    print(f"Batch limit (max_num_seqs): {args.max_num_seqs}")
    print(f"Prefill tokens: {prefill_tokens}")
    print(f"Decode tokens: {decode_tokens}")
    print(f"Total processed tokens: {prefill_tokens + decode_tokens}")

    _print_throughput(prefill_tokens, decode_tokens, prefill_time, decode_time)


if __name__ == "__main__":
    main()
