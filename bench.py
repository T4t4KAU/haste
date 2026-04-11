"""Benchmark script for Haste LLM engine."""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from haste import LLM, SamplingParams
from haste.utils.profiling import build_profile_report, save_profile_report


@dataclass(frozen=True)
class PromptRecord:
    """Prompt record with dataset information."""
    dataset: str  # Dataset name
    prompt_id: str  # Prompt ID
    prompt: str  # Prompt text


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser for benchmarking.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Benchmark haste on real prompts from the HCSD/Dovetail datasets.",
    )
    parser.add_argument("--target-model-path", required=True, help="Path to the target model.")
    parser.add_argument("--draft-model-path", required=True, help="Path to the draft model.")
    parser.add_argument("--dataset-root", required=True,help="Root directory for datasets.")
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
        default=0,
        help="Maximum number of concurrent sequences. 0 uses the hardware-based default.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=4096,
        help="Maximum number of batched tokens for the scheduler.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--draft-temperature", type=float, default=0.0)
    parser.add_argument("--speculate-k", type=int, default=1)
    parser.add_argument("--async-fan-out", type=int, default=1)
    parser.add_argument(
        "--auto-tune-kf",
        action="store_true",
        help="Dynamically search and adjust speculative lookahead/fan-out at runtime.",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--turn-index",
        type=int,
        default=0,
        help="Which item from `turns` to use when a prompt file is multi-turn.",
    )
    parser.add_argument(
        "--join-turns",
        action="store_true",
        help="Join every element in `turns` into one prompt instead of using --turn-index.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle prompts before applying --prompt-limit.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph capture and force eager mode.",
    )
    parser.add_argument(
        "--use-tqdm",
        action="store_true",
        help="Show generation progress bars.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed runtime logs, including auto-tune K/F transitions.",
    )
    parser.add_argument(
        "--profile-output",
        default="",
        help="Optional JSON file for the full profiling report.",
    )
    parser.add_argument(
        "--include-raw-metrics",
        action="store_true",
        help="Include raw per-step metric series in the JSON profiling report.",
    )
    return parser


def pick_benchmark_config() -> dict[str, int]:
    """Pick benchmark configuration based on hardware.
    
    Returns:
        dict[str, int]: Benchmark configuration
    """
    if not torch.cuda.is_available():
        return {
            "max_num_seqs": 16,
            "max_num_batched_tokens": 4096,
        }

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gb < 14:
        return {
            "max_num_seqs": 32,
            "max_num_batched_tokens": 4096,
        }

    return {
        "max_num_seqs": 128,
        "max_num_batched_tokens": 4096,
    }


def resolve_dataset_files(dataset_root: Path, datasets: str, dataset_file: str) -> list[Path]:
    """Resolve dataset files based on command line arguments.
    
    Args:
        dataset_root (Path): Root directory for datasets
        datasets (str): Comma-separated dataset names
        dataset_file (str): Explicit dataset file path
        
    Returns:
        list[Path]: List of dataset files
    """
    if dataset_file:
        return [Path(dataset_file).expanduser().resolve()]

    if datasets.strip().lower() == "all":
        return sorted(dataset_root.glob("*/question.jsonl"))

    files = []
    for name in datasets.split(","):
        dataset = name.strip()
        if not dataset:
            continue
        candidate = dataset_root / dataset / "question.jsonl"
        if not candidate.exists():
            raise FileNotFoundError(f"Dataset file not found: {candidate}")
        files.append(candidate)
    return files


def render_messages(messages: list[dict[str, Any]], tokenizer) -> str:
    """Render chat messages into a prompt string.
    
    Args:
        messages (list[dict[str, Any]]): List of messages
        tokenizer: Tokenizer to use for rendering
        
    Returns:
        str: Rendered prompt string
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    formatted = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        formatted.append(f"{role}: {content}")
    formatted.append("assistant:")
    return "\n\n".join(formatted)


def extract_prompt_text(payload: dict[str, Any], tokenizer, turn_index: int, join_turns: bool) -> str:
    """Extract prompt text from a payload.
    
    Args:
        payload (dict[str, Any]): Payload containing prompt information
        tokenizer: Tokenizer to use for rendering
        turn_index (int): Index of the turn to use
        join_turns (bool): Whether to join all turns
        
    Returns:
        str: Extracted prompt text
    """
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        return prompt

    messages = payload.get("messages")
    if isinstance(messages, list):
        return render_messages(messages, tokenizer)

    turns = payload.get("turns")
    if isinstance(turns, list) and turns:
        if join_turns:
            return "\n\n".join(str(turn) for turn in turns)
        clamped_index = min(max(turn_index, 0), len(turns) - 1)
        return str(turns[clamped_index])

    raise ValueError("Unsupported prompt record: expected `prompt`, `messages`, or `turns`.")


def load_prompt_records(
    files: list[Path],
    tokenizer,
    *,
    prompt_limit: int,
    turn_index: int,
    join_turns: bool,
    shuffle: bool,
    seed: int,
) -> list[PromptRecord]:
    """Load prompt records from dataset files.
    
    Args:
        files (list[Path]): List of dataset files
        tokenizer: Tokenizer to use for rendering
        prompt_limit (int): Maximum number of prompts to load
        turn_index (int): Index of the turn to use
        join_turns (bool): Whether to join all turns
        shuffle (bool): Whether to shuffle prompts
        seed (int): Seed for shuffling
        
    Returns:
        list[PromptRecord]: List of prompt records
    """
    records_by_dataset = []
    for path in files:
        dataset = path.parent.name
        dataset_records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                prompt_id = str(payload.get("question_id", payload.get("id", f"{dataset}-{line_number}")))
                dataset_records.append(
                    PromptRecord(
                        dataset=dataset,
                        prompt_id=prompt_id,
                        prompt=extract_prompt_text(payload, tokenizer, turn_index, join_turns),
                    )
                )
        records_by_dataset.append(dataset_records)

    records = [record for dataset_records in records_by_dataset for record in dataset_records]

    if shuffle:
        import random

        rng = random.Random(seed)
        rng.shuffle(records)
    elif prompt_limit > 0 and len(records_by_dataset) > 1:
        records = []
        dataset_iters = [iter(dataset_records) for dataset_records in records_by_dataset if dataset_records]
        while dataset_iters and len(records) < prompt_limit:
            next_iters = []
            for dataset_iter in dataset_iters:
                try:
                    records.append(next(dataset_iter))
                except StopIteration:
                    continue
                if len(records) >= prompt_limit:
                    break
                next_iters.append(dataset_iter)
            dataset_iters = next_iters

    if prompt_limit > 0:
        records = records[:prompt_limit]

    if not records:
        raise ValueError("No prompts were loaded for benchmarking.")
    return records


def summarize_prompt_lengths(tokenizer, prompts: list[str]) -> tuple[int, int, float]:
    """Summarize prompt lengths.
    
    Args:
        tokenizer: Tokenizer to use for encoding
        prompts (list[str]): List of prompts
        
    Returns:
        tuple[int, int, float]: Min, max, and average prompt lengths
    """
    lengths = [len(tokenizer.encode(prompt)) for prompt in prompts]
    total = sum(lengths)
    return min(lengths), max(lengths), total / len(lengths)


def print_series_summary(label: str, summary: dict[str, Any], unit: str = "") -> None:
    """Print summary of a series of values.
    
    Args:
        label (str): Label for the series
        summary (dict[str, Any]): Summary statistics
        unit (str, optional): Unit of measurement. Defaults to "".
    """
    if summary.get("count", 0) == 0:
        return
    suffix = f" {unit}" if unit else ""
    print(
        f"{label}: mean {summary['mean']:.2f}{suffix}, "
        f"p50 {summary['p50']:.2f}{suffix}, p95 {summary['p95']:.2f}{suffix}, "
        f"max {summary['max']:.2f}{suffix}"
    )


def print_profile_summary(report: dict[str, Any]) -> None:
    """Print profiling summary from the report.
    
    Args:
        report (dict[str, Any]): Profiling report
    """
    throughput = report["throughput"]
    cache = report["cache"]
    acceptance = report["acceptance"]
    stages = report["stages"]
    runners = report["runners"]

    print("\nProfiling Summary")
    print("-" * 60)
    if throughput["prefill_tok_per_s"] is not None:
        print(f"Prefill throughput: {throughput['prefill_tok_per_s']:.2f} tok/s")
    if throughput["decode_tok_per_s"] is not None:
        print(f"Decode throughput: {throughput['decode_tok_per_s']:.2f} tok/s")
    if throughput["overall_tok_per_s"] is not None:
        print(f"Overall throughput: {throughput['overall_tok_per_s']:.2f} tok/s")
    if throughput["generation_tok_per_s"] is not None:
        print(f"Generation throughput: {throughput['generation_tok_per_s']:.2f} tok/s")
    if cache["avg_hit_rate"] is not None:
        print(f"Average cache hit rate: {cache['avg_hit_rate']:.2%}")
    if acceptance["avg_accepted_spec_fraction"] is not None:
        print(f"Accepted speculative fraction: {acceptance['avg_accepted_spec_fraction']:.2%}")

    print_series_summary("Engine step", stages["engine_step_ms"], "ms")
    print_series_summary("Scheduler", stages["scheduler_ms"], "ms")
    print_series_summary("Speculate", stages["speculate_ms"], "ms")
    print_series_summary("Verify", stages["verify_ms"], "ms")
    print_series_summary("Rollback", stages["rollback_ms"], "ms")
    print_series_summary("Postprocess", stages["postprocess_ms"], "ms")

    target = runners.get("target", {})
    target_verify = target.get("verify", {})
    if target_verify:
        print_series_summary("Target verify runner", target_verify.get("time_ms", {}), "ms")

    draft_worker = runners.get("draft_worker", {})
    if draft_worker:
        if draft_worker.get("auto_tune_enabled"):
            print(
                "Auto-tuned K/F: "
                f"static K={draft_worker.get('static_speculate_k')} "
                f"static F={draft_worker.get('static_async_fan_out')} "
                f"final K={draft_worker.get('final_effective_k')} "
                f"final F={draft_worker.get('final_effective_f')}"
            )
        print_series_summary("Draft request wait", draft_worker.get("request_wait_ms", {}), "ms")
        print_series_summary("Draft worker serve", draft_worker.get("worker_serve_ms", {}), "ms")
        print_series_summary("Draft cache populate", draft_worker.get("cache_populate_ms", {}), "ms")
        print_series_summary("Effective lookahead", draft_worker.get("effective_lookahead", {}))
        print_series_summary("Effective fan-out cap", draft_worker.get("effective_fan_out_cap", {}))
        if draft_worker.get("fast_populate_rate") is not None:
            print(f"Fast populate rate: {draft_worker['fast_populate_rate']:.2%}")


def main():
    """Main function to run benchmarking."""
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dataset_files = resolve_dataset_files(dataset_root, args.datasets, args.dataset_file)
    cfg = pick_benchmark_config()

    llm = LLM(
        model=args.target_model_path,
        draft_model=args.draft_model_path,
        speculate=True,
        speculate_k=args.speculate_k,
        draft_async=True,
        async_fan_out=args.async_fan_out,
        async_auto_tune=args.auto_tune_kf,
        verbose=args.verbose,
        enforce_eager=args.enforce_eager,
        max_num_seqs=args.max_num_seqs or cfg["max_num_seqs"],
        max_num_batched_tokens=args.max_num_batched_tokens or cfg["max_num_batched_tokens"],
        max_model_len=args.max_model_len,
    )

    try:
        records = load_prompt_records(
            dataset_files,
            llm.tokenizer,
            prompt_limit=args.prompt_limit,
            turn_index=args.turn_index,
            join_turns=args.join_turns,
            shuffle=args.shuffle,
            seed=args.seed,
        )
        prompts = [record.prompt for record in records]
        sampling_params = [
            SamplingParams(
                temperature=args.temperature,
                draft_temperature=args.draft_temperature,
                ignore_eos=True,
                max_new_tokens=args.max_new_tokens,
            )
            for _ in prompts
        ]

        warmup_prompt = prompts[0] if prompts else "Benchmark warmup"
        warmup_params = SamplingParams(
            temperature=args.temperature,
            draft_temperature=args.draft_temperature,
            ignore_eos=True,
            max_new_tokens=min(args.max_new_tokens, 8),
        )
        for _ in range(max(args.warmup_runs, 0)):
            llm.generate(
                [warmup_prompt],
                [warmup_params],
                use_tqdm=False,
                stream_callback=lambda *_: None,
            )

        prompt_min, prompt_max, prompt_avg = summarize_prompt_lengths(llm.tokenizer, prompts)

        start = perf_counter()
        outputs, metrics = llm.generate(
            prompts,
            sampling_params,
            use_tqdm=args.use_tqdm,
            stream_callback=lambda *_: None,
        )
        elapsed = perf_counter() - start
    finally:
        llm.shutdown()

    generated_new_tokens = sum(len(output["token_ids"]) for output in outputs)
    requested_new_tokens = sum(sp.max_new_tokens for sp in sampling_params)
    processed_tokens = metrics["prefill_total_tokens"] + metrics["decode_total_tokens"]
    dataset_counts = Counter(record.dataset for record in records)
    batch_limit = args.max_num_seqs or cfg["max_num_seqs"]
    scheduler_token_cap = args.max_num_batched_tokens or cfg["max_num_batched_tokens"]

    profile_report = build_profile_report(
        metrics,
        wall_time_sec=elapsed,
        generated_new_tokens=generated_new_tokens,
        requested_new_tokens=requested_new_tokens,
        speculate_k=args.speculate_k,
        metadata={
            "target_model_path": args.target_model_path,
            "draft_model_path": args.draft_model_path,
            "dataset_files": [str(path) for path in dataset_files],
            "dataset_mix": dict(dataset_counts),
            "prompt_count": len(records),
            "prompt_token_length": {
                "min": prompt_min,
                "avg": prompt_avg,
                "max": prompt_max,
            },
            "max_num_seqs": batch_limit,
            "max_num_batched_tokens": scheduler_token_cap,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "draft_temperature": args.draft_temperature,
            "speculate_k": args.speculate_k,
            "async_fan_out": args.async_fan_out,
            "auto_tune_kf": args.auto_tune_kf,
            "verbose": args.verbose,
            "warmup_runs": args.warmup_runs,
        },
        include_raw_metrics=args.include_raw_metrics,
    )

    print("\nBenchmark Summary")
    print("-" * 60)
    print(f"Dataset files: {', '.join(str(path) for path in dataset_files)}")
    print(f"Loaded prompts: {len(records)}")
    print(f"Dataset mix: {dict(dataset_counts)}")
    print(f"Prompt token length min/avg/max: {prompt_min}/{prompt_avg:.1f}/{prompt_max}")
    print(f"Batch limit (max_num_seqs): {batch_limit}")
    print(f"Scheduler token cap: {scheduler_token_cap}")
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Prefill tokens: {metrics['prefill_total_tokens']}")
    print(f"Decode tokens: {metrics['decode_total_tokens']}")
    print(f"Total processed tokens: {processed_tokens}")
    print(f"Requested new tokens: {requested_new_tokens}")
    print(f"Generated new tokens: {generated_new_tokens}")
    print_profile_summary(profile_report)

    if args.profile_output:
        output_path = save_profile_report(args.profile_output, profile_report)
        print(f"Profile report saved to: {output_path}")


if __name__ == "__main__":
    main()
