"""Profiling and metrics utilities."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_METRICS = {
    "cache_hits": [],
    "accepted_suffix_lens_with_recovery": [],
    "accepted_suffix_lens_on_hit": [],
    "accepted_suffix_lens_on_miss": [],
    "prefill_total_time": 0.0,
    "decode_total_time": 0.0,
    "prefill_total_tokens": 0,
    "decode_total_tokens": 0,
    "target_step_times": [],
    "target_verify_times": [],
    "scheduler_times": [],
    "prefill_step_times": [],
    "decode_step_times": [],
    "prefill_batch_sizes": [],
    "decode_batch_sizes": [],
    "prefill_step_tokens": [],
    "decode_step_tokens": [],
    "prefill_speculator_times": [],
    "prefill_verifier_times": [],
    "speculate_times": [],
    "verify_times": [],
    "rollback_times": [],
    "postprocess_times": [],
    "effective_lookaheads": [],
    "engine_wall_time": 0.0,
    "num_requests": 0,
    "completed_requests": 0,
    "num_engine_steps": 0,
    "runner_profiles": {},
}


def fresh_metrics() -> dict[str, Any]:
    """Create a fresh metrics dictionary with default values.
    
    Returns:
        dict[str, Any]: Fresh metrics dictionary
    """
    return copy.deepcopy(DEFAULT_METRICS)


def reset_metrics(metrics: dict[str, Any]) -> None:
    """Reset metrics to default values.
    
    Args:
        metrics (dict[str, Any]): Metrics dictionary to reset
    """
    metrics.clear()
    metrics.update(fresh_metrics())


def safe_divide(numerator: float | int, denominator: float | int) -> float | None:
    """Safely divide two numbers, returning None if denominator is zero.
    
    Args:
        numerator (float | int): Numerator
        denominator (float | int): Denominator
        
    Returns:
        float | None: Result of division or None if denominator is zero
    """
    if not denominator:
        return None
    return float(numerator) / float(denominator)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Calculate percentile of sorted values.
    
    Args:
        sorted_values (list[float]): Sorted list of values
        fraction (float): Fraction for percentile (0.0 to 1.0)
        
    Returns:
        float: Calculated percentile
    """
    if not sorted_values:
        raise ValueError("percentile requires a non-empty list")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_numeric_series(values: list[float] | list[int], scale: float = 1.0) -> dict[str, Any]:
    """Summarize a numeric series with various statistics.
    
    Args:
        values (list[float] | list[int]): List of values to summarize
        scale (float, optional): Scale factor for values. Defaults to 1.0.
        
    Returns:
        dict[str, Any]: Summary statistics
    """
    if not values:
        return {"count": 0}

    scaled = sorted(float(value) * scale for value in values)
    count = len(scaled)
    total = sum(scaled)
    return {
        "count": count,
        "sum": total,
        "mean": total / count,
        "min": scaled[0],
        "p50": _percentile(scaled, 0.50),
        "p90": _percentile(scaled, 0.90),
        "p95": _percentile(scaled, 0.95),
        "max": scaled[-1],
    }


def build_runner_mode_summary(snapshot: dict[str, list[float] | list[int]], mode: str) -> dict[str, Any]:
    """Build summary for a specific runner mode.
    
    Args:
        snapshot (dict[str, list[float] | list[int]]): Metrics snapshot
        mode (str): Mode to summarize (e.g., "prefill", "decode")
        
    Returns:
        dict[str, Any]: Mode summary
    """
    total_times = snapshot[f"{mode}_total_times"]
    input_tokens = snapshot[f"{mode}_input_tokens"]
    batch_sizes = snapshot[f"{mode}_batch_sizes"]
    prepare_times = snapshot[f"{mode}_prepare_times"]
    model_times = snapshot[f"{mode}_model_times"]
    sample_times = snapshot[f"{mode}_sample_times"]

    total_time = sum(total_times)
    total_tokens = sum(input_tokens)
    return {
        "calls": len(total_times),
        "throughput_tok_per_s": safe_divide(total_tokens, total_time),
        "input_tokens_total": total_tokens,
        "time_ms": summarize_numeric_series(total_times, scale=1000.0),
        "prepare_ms": summarize_numeric_series(prepare_times, scale=1000.0),
        "model_ms": summarize_numeric_series(model_times, scale=1000.0),
        "sample_ms": summarize_numeric_series(sample_times, scale=1000.0),
        "batch_size": summarize_numeric_series(batch_sizes),
        "input_tokens": summarize_numeric_series(input_tokens),
    }


def build_runner_profile_summary(
    snapshot: dict[str, Any],
    *,
    device: str,
    is_draft: bool,
) -> dict[str, Any]:
    """Build runner profile summary.
    
    Args:
        snapshot (dict[str, Any]): Metrics snapshot
        device (str): Device information
        is_draft (bool): Whether this is a draft runner
        
    Returns:
        dict[str, Any]: Runner profile summary
    """
    return {
        "device": device,
        "is_draft": is_draft,
        "prefill": build_runner_mode_summary(snapshot, "prefill"),
        "decode": build_runner_mode_summary(snapshot, "decode"),
        "verify": build_runner_mode_summary(snapshot, "verify"),
    }


def build_draft_worker_profile_summary(snapshot: dict[str, Any], *, device: str) -> dict[str, Any]:
    """Build draft worker profile summary.
    
    Args:
        snapshot (dict[str, Any]): Metrics snapshot
        device (str): Device information
        
    Returns:
        dict[str, Any]: Draft worker profile summary
    """
    return {
        "device": device,
        "request_wait_ms": summarize_numeric_series(snapshot["request_wait_times"], scale=1000.0),
        "exposed_wait_ms": summarize_numeric_series(snapshot.get("exposed_wait_times", []), scale=1000.0),
        "worker_total_ms": summarize_numeric_series(snapshot["worker_total_times"], scale=1000.0),
        "worker_serve_ms": summarize_numeric_series(snapshot["worker_serve_times"], scale=1000.0),
        "cache_populate_ms": summarize_numeric_series(snapshot["cache_populate_times"], scale=1000.0),
        "populate_branch_count": summarize_numeric_series(snapshot.get("populate_branch_counts", [])),
        "fast_populate_rate": safe_divide(
            sum(snapshot.get("fast_populate_flags", [])),
            len(snapshot.get("fast_populate_flags", [])),
        ),
        "request_batch_size": summarize_numeric_series(snapshot["request_batch_sizes"]),
        "worker_batch_size": summarize_numeric_series(snapshot["worker_batch_sizes"]),
        "cache_hit_rate": summarize_numeric_series(snapshot["cache_hit_rates"]),
        "tree_cache_size": summarize_numeric_series(snapshot["tree_cache_sizes"]),
        "effective_lookahead": summarize_numeric_series(snapshot.get("effective_lookaheads", [])),
        "effective_fan_out_cap": summarize_numeric_series(snapshot.get("effective_fan_out_caps", [])),
    }


def build_profile_report(
    metrics: dict[str, Any],
    *,
    wall_time_sec: float | None = None,
    generated_new_tokens: int | None = None,
    requested_new_tokens: int | None = None,
    speculate_k: int | None = None,
    metadata: dict[str, Any] | None = None,
    include_raw_metrics: bool = False,
) -> dict[str, Any]:
    """Build a comprehensive profile report.
    
    Args:
        metrics (dict[str, Any]): Metrics dictionary
        wall_time_sec (float | None, optional): Wall time in seconds. Defaults to None.
        generated_new_tokens (int | None, optional): Number of generated new tokens. Defaults to None.
        requested_new_tokens (int | None, optional): Number of requested new tokens. Defaults to None.
        speculate_k (int | None, optional): Speculation length. Defaults to None.
        metadata (dict[str, Any] | None, optional): Metadata. Defaults to None.
        include_raw_metrics (bool, optional): Whether to include raw metrics. Defaults to False.
        
    Returns:
        dict[str, Any]: Profile report
    """
    prefill_tokens = int(metrics.get("prefill_total_tokens", 0))
    decode_tokens = int(metrics.get("decode_total_tokens", 0))
    processed_tokens = prefill_tokens + decode_tokens
    prefill_time = float(metrics.get("prefill_total_time", 0.0))
    decode_time = float(metrics.get("decode_total_time", 0.0))
    total_wall_time = wall_time_sec if wall_time_sec is not None else float(metrics.get("engine_wall_time", 0.0))

    accepted_suffixes = metrics.get("accepted_suffix_lens_with_recovery", [])
    accepted_total = sum(accepted_suffixes)
    speculative_steps = len(accepted_suffixes)
    accepted_spec_tokens = max(0, accepted_total - speculative_steps)
    effective_lookaheads = metrics.get("effective_lookaheads", [])
    effective_spec_budget = sum(effective_lookaheads)
    if not effective_spec_budget and speculate_k and speculative_steps:
        effective_spec_budget = speculative_steps * speculate_k

    cache_hits = metrics.get("cache_hits", [])
    report = {
        "metadata": metadata or {},
        "totals": {
            "num_requests": int(metrics.get("num_requests", 0)),
            "completed_requests": int(metrics.get("completed_requests", 0)),
            "num_engine_steps": int(metrics.get("num_engine_steps", 0)),
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "processed_tokens": processed_tokens,
            "requested_new_tokens": requested_new_tokens,
            "generated_new_tokens": generated_new_tokens,
            "prefill_time_sec": prefill_time,
            "decode_time_sec": decode_time,
            "wall_time_sec": total_wall_time,
        },
        "throughput": {
            "prefill_tok_per_s": safe_divide(prefill_tokens, prefill_time),
            "decode_tok_per_s": safe_divide(decode_tokens, decode_time),
            "overall_tok_per_s": safe_divide(processed_tokens, total_wall_time),
            "generation_tok_per_s": safe_divide(generated_new_tokens or 0, total_wall_time),
        },
        "acceptance": {
            "speculative_steps": speculative_steps,
            "accepted_tokens_with_recovery_total": accepted_total,
            "avg_tokens_per_step_with_recovery": safe_divide(accepted_total, speculative_steps),
            "avg_accepted_spec_tokens": safe_divide(accepted_spec_tokens, speculative_steps),
            "avg_accepted_spec_fraction": (
                safe_divide(accepted_spec_tokens, effective_spec_budget)
                if effective_spec_budget
                else None
            ),
            "effective_lookahead": summarize_numeric_series(effective_lookaheads),
            "accepted_suffix_len_on_hit": summarize_numeric_series(metrics.get("accepted_suffix_lens_on_hit", [])),
            "accepted_suffix_len_on_miss": summarize_numeric_series(metrics.get("accepted_suffix_lens_on_miss", [])),
        },
        "cache": {
            "avg_hit_rate": safe_divide(sum(cache_hits), len(cache_hits)) if cache_hits else None,
            "per_step_hit_rate": summarize_numeric_series(cache_hits),
        },
        "stages": {
            "engine_step_ms": summarize_numeric_series(metrics.get("target_step_times", []), scale=1000.0),
            "scheduler_ms": summarize_numeric_series(metrics.get("scheduler_times", []), scale=1000.0),
            "prefill_step_ms": summarize_numeric_series(metrics.get("prefill_step_times", []), scale=1000.0),
            "decode_step_ms": summarize_numeric_series(metrics.get("decode_step_times", []), scale=1000.0),
            "prefill_batch_size": summarize_numeric_series(metrics.get("prefill_batch_sizes", [])),
            "decode_batch_size": summarize_numeric_series(metrics.get("decode_batch_sizes", [])),
            "prefill_step_tokens": summarize_numeric_series(metrics.get("prefill_step_tokens", [])),
            "decode_step_tokens": summarize_numeric_series(metrics.get("decode_step_tokens", [])),
            "prefill_speculator_ms": summarize_numeric_series(metrics.get("prefill_speculator_times", []), scale=1000.0),
            "prefill_verifier_ms": summarize_numeric_series(metrics.get("prefill_verifier_times", []), scale=1000.0),
            "speculate_ms": summarize_numeric_series(metrics.get("speculate_times", []), scale=1000.0),
            "verify_ms": summarize_numeric_series(metrics.get("verify_times", []), scale=1000.0),
            "rollback_ms": summarize_numeric_series(metrics.get("rollback_times", []), scale=1000.0),
            "postprocess_ms": summarize_numeric_series(metrics.get("postprocess_times", []), scale=1000.0),
            "target_verify_ms": summarize_numeric_series(metrics.get("target_verify_times", []), scale=1000.0),
        },
        "runners": metrics.get("runner_profiles", {}),
    }

    if include_raw_metrics:
        report["raw_metrics"] = metrics

    return report


def save_profile_report(path: str | Path, report: dict[str, Any]) -> Path:
    """Save profile report to a file.
    
    Args:
        path (str | Path): Path to save the report
        report (dict[str, Any]): Profile report
        
    Returns:
        Path: Path where the report was saved
    """
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
