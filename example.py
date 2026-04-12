"""Example script for running inference with Haste LLM engine."""

import argparse
from time import perf_counter

from haste import LLM, SamplingParams
from haste.utils.profiling import build_profile_report, save_profile_report


def print_series_summary(label: str, summary: dict, unit: str = ""):
    """Print summary of a series of values.
    
    Args:
        label (str): Label for the series
        summary (dict): Summary statistics
        unit (str, optional): Unit of measurement. Defaults to "".
    """
    if summary.get("count", 0) == 0:
        return
    suffix = f" {unit}" if unit else ""
    print(
        f"{label}: mean {summary['mean']:.2f}{suffix}, "
        f"p50 {summary['p50']:.2f}{suffix}, p95 {summary['p95']:.2f}{suffix}"
    )


def print_profile_summary(report: dict):
    """Print profiling summary from the report.
    
    Args:
        report (dict): Profiling report
    """
    throughput = report["throughput"]
    cache = report["cache"]
    acceptance = report["acceptance"]
    stages = report["stages"]
    runners = report["runners"]
    metadata = report.get("metadata", {})
    mode = metadata.get("mode", "spec_async")

    print("\nProfiling Summary")
    print("-" * 50)
    if throughput["prefill_tok_per_s"] is not None:
        print(f"Prefill throughput: {throughput['prefill_tok_per_s']:.2f} tok/s")
    if throughput["decode_tok_per_s"] is not None:
        print(f"Decode throughput: {throughput['decode_tok_per_s']:.2f} tok/s")
    if throughput["overall_tok_per_s"] is not None:
        print(f"Overall throughput: {throughput['overall_tok_per_s']:.2f} tok/s")
    if throughput["generation_tok_per_s"] is not None:
        print(f"Generation throughput: {throughput['generation_tok_per_s']:.2f} tok/s")
    if mode == "spec_async" and cache["avg_hit_rate"] is not None:
        print(f"Average cache hit rate: {cache['avg_hit_rate']:.2%}")
    if mode in {"spec_sync", "spec_async"} and acceptance["avg_accepted_spec_fraction"] is not None:
        print(f"Accepted speculative fraction: {acceptance['avg_accepted_spec_fraction']:.2%}")

    print_series_summary("Engine step", stages["engine_step_ms"], "ms")
    print_series_summary("Speculate", stages["speculate_ms"], "ms")
    print_series_summary("Verify", stages["verify_ms"], "ms")
    print_series_summary("Postprocess", stages["postprocess_ms"], "ms")

    draft_worker = runners.get("draft_worker", {})
    if mode == "spec_async" and draft_worker:
        if draft_worker.get("auto_tune_enabled"):
            print(
                "Auto-tuned K/F: "
                f"static K={draft_worker.get('static_speculate_k')} "
                f"static F={draft_worker.get('static_async_fan_out')} "
                f"final K={draft_worker.get('final_effective_k')} "
                f"final F={draft_worker.get('final_effective_f')}"
            )
        print_series_summary("Effective lookahead", draft_worker.get("effective_lookahead", {}))
        print_series_summary("Effective fan-out cap", draft_worker.get("effective_fan_out_cap", {}))
        if draft_worker.get("fast_populate_rate") is not None:
            print(f"Fast populate rate: {draft_worker['fast_populate_rate']:.2%}")


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Run inference with Haste LLM engine.",
    )
    parser.add_argument(
        "--mode",
        choices=("ar", "spec_sync", "spec_async"),
        default="spec_async",
        help="Decoding mode: autoregressive (`ar`), synchronous speculative (`spec_sync`), or asynchronous speculative (`spec_async`).",
    )
    parser.add_argument(
        "--target-model-path",
        required=True,
        help="Path to the target model.",
    )
    parser.add_argument(
        "--draft-model-path",
        default="",
        help="Path to the draft model. Required for speculative modes.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for target model.",
    )
    parser.add_argument(
        "--draft-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for draft model.",
    )
    parser.add_argument(
        "--speculate-k",
        type=int,
        default=7,
        help="Number of tokens to speculate.",
    )
    parser.add_argument(
        "--async-fan-out",
        type=int,
        default=3,
        help="Number of async draft runners.",
    )
    parser.add_argument(
        "--auto-tune-kf",
        action="store_true",
        help="Dynamically search and adjust speculative lookahead/fan-out at runtime.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed runtime logs, including auto-tune K/F transitions.",
    )
    parser.add_argument(
        "--profile-output",
        default="",
        help="Optional JSON file for the profiling report.",
    )
    parser.add_argument(
        "--include-raw-metrics",
        action="store_true",
        help="Include raw per-step metric series in the profiling report.",
    )
    return parser


def validate_mode_args(args: argparse.Namespace) -> None:
    """Validate command line argument combinations."""
    if args.mode in {"spec_sync", "spec_async"} and not args.draft_model_path:
        raise ValueError("--draft-model-path is required when --mode is spec_sync or spec_async.")
    if args.auto_tune_kf and args.mode != "spec_async":
        raise ValueError("--auto-tune-kf is only supported when --mode is spec_async.")


def build_llm_kwargs(args: argparse.Namespace) -> dict:
    """Build LLM keyword arguments from command line arguments."""
    speculate = args.mode != "ar"
    draft_async = args.mode == "spec_async"
    kwargs = {
        "model": args.target_model_path,
        "speculate": speculate,
        "speculate_k": args.speculate_k,
        "draft_async": draft_async,
        "async_fan_out": args.async_fan_out,
        "async_auto_tune": args.auto_tune_kf,
        "verbose": args.verbose,
        "max_num_seqs": 32,
        "max_num_batched_tokens": 4096,
        "max_model_len": 4096,
        "enforce_eager": False,
    }
    if speculate:
        kwargs["draft_model"] = args.draft_model_path
    return kwargs

def main():
    """Main function to run inference with Haste LLM engine."""
    args = build_parser().parse_args()
    validate_mode_args(args)
    prompts = [
        "Peking University is",
        "The capital of France is",
        "Write a short poem about spring:",
    ]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        draft_temperature=args.draft_temperature,
        max_new_tokens=args.max_new_tokens,
    )

    print("Initializing LLM...")
    llm = LLM(**build_llm_kwargs(args))
    print("LLM initialized successfully!")

    try:
        print("\nGenerating responses...")
        start = perf_counter()
        outputs, metrics = llm.generate(prompts, sampling_params, use_tqdm=True)
        elapsed = perf_counter() - start

        for prompt, output in zip(prompts, outputs):
            print(f"\nPrompt: {prompt}")
            print("-" * 50)
            print(output["text"])

        generated_new_tokens = sum(len(output["token_ids"]) for output in outputs)
        requested_new_tokens = sum(sp.max_new_tokens for sp in [sampling_params] * len(prompts))
        profile_report = build_profile_report(
            metrics,
            wall_time_sec=elapsed,
            generated_new_tokens=generated_new_tokens,
            requested_new_tokens=requested_new_tokens,
            speculate_k=args.speculate_k,
            metadata={
                "mode": args.mode,
                "target_model_path": args.target_model_path,
                "draft_model_path": args.draft_model_path if args.mode != "ar" else None,
                "prompt_count": len(prompts),
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "draft_temperature": args.draft_temperature,
                "speculate_k": args.speculate_k,
                "async_fan_out": args.async_fan_out,
                "auto_tune_kf": args.auto_tune_kf,
                "verbose": args.verbose,
            },
            include_raw_metrics=args.include_raw_metrics,
        )
        print(f"Mode: {args.mode}")
        print_profile_summary(profile_report)

        if args.profile_output:
            output_path = save_profile_report(args.profile_output, profile_report)
            print(f"Profile report saved to: {output_path}")
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
