"""Benchmark entrypoint for the vendored Dovetail baseline."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
import transformers
from tqdm import tqdm

from .decoding import greedy_generate, speculative_greedy_generate
from .modeling import describe_model, load_model, load_tokenizer
from .prompts import load_prompt_records, render_prompt


def _select_draft_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_output_path(output_arg: str) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path.cwd() / "outputs" / f"benchmark_{timestamp}.json").resolve()


def _method_list(methods_arg: str) -> list[str]:
    methods = [item.strip() for item in methods_arg.split(",") if item.strip()]
    unknown = sorted(set(methods) - {"cpu_only", "dovetail"})
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    if not methods:
        raise ValueError("At least one method must be selected.")
    return methods


def _eos_token_id(tokenizer, model) -> int | None:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        return int(eos_token_id)
    config_eos = getattr(model.config, "eos_token_id", None)
    if isinstance(config_eos, list):
        return int(config_eos[0]) if config_eos else None
    if config_eos is not None:
        return int(config_eos)
    return None


def _build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    methods = sorted({record["method"] for record in records})
    for method in methods:
        method_records = [record for record in records if record["method"] == method]
        total_prompt_tokens = sum(record["prompt_tokens"] for record in method_records)
        total_generated_tokens = sum(record["generated_tokens"] for record in method_records)
        total_prefill_sec = sum(record.get("prefill_sec", 0.0) for record in method_records)
        total_decode_sec = sum(record.get("decode_sec", 0.0) for record in method_records)
        total_elapsed_sec = sum(record["elapsed_sec"] for record in method_records)
        total_proposed_tokens = sum(record.get("proposed_tokens", 0) for record in method_records)
        total_accepted_tokens = sum(record.get("accepted_tokens", 0) for record in method_records)
        total_speculative_steps = sum(record.get("speculative_steps", 0) for record in method_records)
        summary[method] = {
            "samples": len(method_records),
            "total_prompt_tokens": total_prompt_tokens,
            "total_generated_tokens": total_generated_tokens,
            "total_prefill_sec": total_prefill_sec,
            "total_decode_sec": total_decode_sec,
            "total_elapsed_sec": total_elapsed_sec,
            "prefill_tokens_per_sec": (
                total_prompt_tokens / total_prefill_sec if total_prefill_sec > 0 else 0.0
            ),
            "decode_tokens_per_sec": (
                total_generated_tokens / total_decode_sec if total_decode_sec > 0 else 0.0
            ),
            "overall_tokens_per_sec": (
                (total_prompt_tokens + total_generated_tokens) / total_elapsed_sec
                if total_elapsed_sec > 0
                else 0.0
            ),
            "generation_tokens_per_sec": (
                total_generated_tokens / total_elapsed_sec if total_elapsed_sec > 0 else 0.0
            ),
            "avg_tokens_per_sec": mean(record["tokens_per_sec"] for record in method_records),
            "median_tokens_per_sec": median(record["tokens_per_sec"] for record in method_records),
            "avg_generated_tokens": mean(record["generated_tokens"] for record in method_records),
            "avg_elapsed_sec": mean(record["elapsed_sec"] for record in method_records),
        }
        acceptance_rates = [
            record["acceptance_rate"]
            for record in method_records
            if record["acceptance_rate"] is not None
        ]
        avg_accepted = [
            record["avg_accepted_tokens"]
            for record in method_records
            if record["avg_accepted_tokens"] is not None
        ]
        if acceptance_rates:
            summary[method]["avg_acceptance_rate"] = mean(acceptance_rates)
        if avg_accepted:
            summary[method]["avg_accepted_tokens"] = mean(avg_accepted)
        if total_proposed_tokens > 0:
            summary[method]["overall_acceptance_rate"] = total_accepted_tokens / total_proposed_tokens
            summary[method]["total_proposed_tokens"] = total_proposed_tokens
            summary[method]["total_accepted_tokens"] = total_accepted_tokens
        if total_speculative_steps > 0:
            summary[method]["overall_avg_accepted_tokens"] = (
                total_accepted_tokens / total_speculative_steps
            )

    if "cpu_only" in summary and "dovetail" in summary:
        baseline_tps = summary["cpu_only"]["overall_tokens_per_sec"]
        dovetail_tps = summary["dovetail"]["overall_tokens_per_sec"]
        baseline_avg_tps = summary["cpu_only"]["avg_tokens_per_sec"]
        dovetail_avg_tps = summary["dovetail"]["avg_tokens_per_sec"]
        if baseline_tps > 0:
            summary["dovetail"]["speedup_vs_cpu_only"] = dovetail_tps / baseline_tps
        if baseline_avg_tps > 0:
            summary["dovetail"]["avg_speedup_vs_cpu_only"] = dovetail_avg_tps / baseline_avg_tps
    return summary


def _parse_dataset_names(datasets_arg: str) -> list[str] | None:
    if not datasets_arg.strip():
        return None
    return [item.strip() for item in datasets_arg.split(",") if item.strip()]


def _count_total_runs(records, turn_mode: str) -> int:
    total = 0
    for record in records:
        if turn_mode == "sequential" and record.turns:
            total += len(record.turns)
        else:
            total += 1
    return total


def _run_generation(
    *,
    method: str,
    target_model,
    draft_model,
    input_ids,
    target_device: str,
    draft_device: str,
    max_new_tokens: int,
    num_draft_tokens: int,
    eos_token_id: int | None,
):
    if method == "cpu_only":
        return greedy_generate(
            target_model,
            input_ids=input_ids,
            device=target_device,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
    if draft_model is None:
        raise RuntimeError("Draft model was not loaded.")
    return speculative_greedy_generate(
        target_model,
        draft_model,
        input_ids=input_ids,
        target_device=target_device,
        draft_device=draft_device,
        max_new_tokens=max_new_tokens,
        num_draft_tokens=num_draft_tokens,
        eos_token_id=eos_token_id,
    )


def _run_record(
    *,
    record,
    method: str,
    tokenizer,
    target_model,
    draft_model,
    target_device: str,
    draft_device: str,
    max_new_tokens: int,
    num_draft_tokens: int,
    eos_token_id: int | None,
    system_prompt: str,
    turn_mode: str,
    progress,
) -> dict[str, Any]:
    if record.turns and turn_mode == "sequential":
        conversation_messages: list[dict[str, str]] = []
        turn_results: list[dict[str, Any]] = []
        generated_turn_texts: list[str] = []
        total_prompt_tokens = 0
        total_generated_tokens = 0
        total_prefill_sec = 0.0
        total_decode_sec = 0.0
        total_elapsed_sec = 0.0
        total_proposed_tokens = 0
        total_accepted_tokens = 0
        total_speculative_steps = 0

        for turn_index, user_turn in enumerate(record.turns):
            conversation_messages.append({"role": "user", "content": user_turn})
            prompt_text = render_prompt(
                record,
                tokenizer,
                system_prompt=system_prompt,
                messages_override=conversation_messages,
                turn_mode=turn_mode,
            )
            input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
            prompt_tokens = int(input_ids.shape[-1])
            result = _run_generation(
                method=method,
                target_model=target_model,
                draft_model=draft_model,
                input_ids=input_ids,
                target_device=target_device,
                draft_device=draft_device,
                max_new_tokens=max_new_tokens,
                num_draft_tokens=num_draft_tokens,
                eos_token_id=eos_token_id,
            )
            decoded_text = tokenizer.decode(
                result.generated_token_ids,
                skip_special_tokens=True,
            ).strip()

            conversation_messages.append({"role": "assistant", "content": decoded_text})
            generated_turn_texts.append(decoded_text)
            total_prompt_tokens += prompt_tokens
            total_generated_tokens += len(result.generated_token_ids)
            total_prefill_sec += result.prefill_sec
            total_decode_sec += result.decode_sec
            total_elapsed_sec += result.elapsed_sec
            total_proposed_tokens += result.proposed_tokens
            total_accepted_tokens += result.accepted_tokens
            total_speculative_steps += result.speculative_steps

            turn_results.append(
                {
                    "turn_index": turn_index,
                    "user_turn": user_turn,
                    "prompt_text": prompt_text,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": len(result.generated_token_ids),
                    "generated_token_ids": result.generated_token_ids,
                    "generated_text": decoded_text,
                    "prefill_sec": result.prefill_sec,
                    "decode_sec": result.decode_sec,
                    "elapsed_sec": result.elapsed_sec,
                    "tokens_per_sec": result.tokens_per_sec,
                    "proposed_tokens": result.proposed_tokens,
                    "accepted_tokens": result.accepted_tokens,
                    "acceptance_rate": result.acceptance_rate,
                    "avg_accepted_tokens": result.avg_accepted_tokens,
                    "speculative_steps": result.speculative_steps,
                }
            )
            progress.update(1)

        return {
            "prompt_id": record.prompt_id,
            "source": record.source,
            "category": record.category,
            "method": method,
            "turn_mode": turn_mode,
            "num_turns": len(record.turns),
            "prompt_tokens": total_prompt_tokens,
            "prompt_text": None,
            "generated_tokens": total_generated_tokens,
            "generated_token_ids": None,
            "generated_text": "\n\n".join(generated_turn_texts).strip(),
            "generated_turn_texts": generated_turn_texts,
            "prefill_sec": total_prefill_sec,
            "decode_sec": total_decode_sec,
            "elapsed_sec": total_elapsed_sec,
            "tokens_per_sec": (
                total_generated_tokens / total_elapsed_sec if total_elapsed_sec > 0 else 0.0
            ),
            "proposed_tokens": total_proposed_tokens,
            "accepted_tokens": total_accepted_tokens,
            "acceptance_rate": (
                total_accepted_tokens / total_proposed_tokens
                if total_proposed_tokens > 0
                else None
            ),
            "avg_accepted_tokens": (
                total_accepted_tokens / total_speculative_steps
                if total_speculative_steps > 0
                else None
            ),
            "speculative_steps": total_speculative_steps,
            "turn_results": turn_results,
        }

    prompt_text = render_prompt(
        record,
        tokenizer,
        system_prompt=system_prompt,
        turn_mode=turn_mode,
    )
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    prompt_tokens = int(input_ids.shape[-1])
    result = _run_generation(
        method=method,
        target_model=target_model,
        draft_model=draft_model,
        input_ids=input_ids,
        target_device=target_device,
        draft_device=draft_device,
        max_new_tokens=max_new_tokens,
        num_draft_tokens=num_draft_tokens,
        eos_token_id=eos_token_id,
    )
    decoded_text = tokenizer.decode(
        result.generated_token_ids,
        skip_special_tokens=True,
    ).strip()
    progress.update(1)
    return {
        "prompt_id": record.prompt_id,
        "source": record.source,
        "category": record.category,
        "method": method,
        "turn_mode": turn_mode,
        "num_turns": len(record.turns) if record.turns else 1,
        "prompt_text": prompt_text,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": len(result.generated_token_ids),
        "generated_token_ids": result.generated_token_ids,
        "generated_text": decoded_text,
        "prefill_sec": result.prefill_sec,
        "decode_sec": result.decode_sec,
        "elapsed_sec": result.elapsed_sec,
        "tokens_per_sec": result.tokens_per_sec,
        "proposed_tokens": result.proposed_tokens,
        "accepted_tokens": result.accepted_tokens,
        "acceptance_rate": result.acceptance_rate,
        "avg_accepted_tokens": result.avg_accepted_tokens,
        "speculative_steps": result.speculative_steps,
    }


def run_benchmark(args):
    methods = _method_list(args.methods)
    dataset_names = _parse_dataset_names(args.datasets)
    draft_device = _select_draft_device(args.draft_device)
    target_device = "cpu"

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    prompt_limit = args.prompt_limit if args.prompt_limit > 0 else None
    prompts = load_prompt_records(
        args.prompt_file,
        limit=prompt_limit,
        dataset_root=args.dataset_root,
        dataset_names=dataset_names,
    )

    tokenizer = load_tokenizer(
        args.target_model_path,
        trust_remote_code=args.trust_remote_code,
    )

    target_model = load_model(
        args.target_model_path,
        device=target_device,
        dtype_name=args.target_dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    draft_model = None
    if "dovetail" in methods:
        draft_model = load_model(
            args.draft_model_path,
            device=draft_device,
            dtype_name=args.draft_dtype,
            attn_implementation=args.attn_implementation,
            trust_remote_code=args.trust_remote_code,
        )
        if target_model.config.vocab_size != draft_model.config.vocab_size:
            raise ValueError(
                "Target and draft models must share the same vocabulary size for speculative decoding."
            )

    eos_token_id = _eos_token_id(tokenizer, target_model)
    warmup_prompt = render_prompt(prompts[0], tokenizer, system_prompt=args.system_prompt)
    warmup_ids = tokenizer(warmup_prompt, return_tensors="pt")["input_ids"]
    warmup_tokens = min(args.max_new_tokens, 16)
    for _ in range(max(0, args.warmup_runs)):
        if "cpu_only" in methods:
            greedy_generate(
                target_model,
                input_ids=warmup_ids,
                device=target_device,
                max_new_tokens=warmup_tokens,
                eos_token_id=eos_token_id,
            )
        if "dovetail" in methods and draft_model is not None:
            speculative_greedy_generate(
                target_model,
                draft_model,
                input_ids=warmup_ids,
                target_device=target_device,
                draft_device=draft_device,
                max_new_tokens=warmup_tokens,
                num_draft_tokens=args.num_draft_tokens,
                eos_token_id=eos_token_id,
            )

    records: list[dict[str, Any]] = []
    total_runs = _count_total_runs(prompts, args.turn_mode) * len(methods)
    progress = tqdm(total=total_runs, desc="Benchmark", unit="run")

    for prompt_record in prompts:
        for method in methods:
            records.append(
                _run_record(
                    record=prompt_record,
                    method=method,
                    tokenizer=tokenizer,
                    target_model=target_model,
                    draft_model=draft_model,
                    target_device=target_device,
                    draft_device=draft_device,
                    max_new_tokens=args.max_new_tokens,
                    num_draft_tokens=args.num_draft_tokens,
                    eos_token_id=eos_token_id,
                    system_prompt=args.system_prompt,
                    turn_mode=args.turn_mode,
                    progress=progress,
                )
            )

    progress.close()

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "methods": methods,
            "target_model_path": args.target_model_path,
            "draft_model_path": args.draft_model_path,
            "prompt_file": args.prompt_file,
            "dataset_root": args.dataset_root,
            "datasets": dataset_names,
            "turn_mode": args.turn_mode,
            "max_new_tokens": args.max_new_tokens,
            "num_draft_tokens": args.num_draft_tokens,
            "target_dtype": args.target_dtype,
            "draft_dtype": args.draft_dtype,
            "target_device": target_device,
            "draft_device": draft_device,
            "attn_implementation": args.attn_implementation,
            "cpu_threads": torch.get_num_threads(),
            "warmup_runs": args.warmup_runs,
            "system_prompt": args.system_prompt,
            "trust_remote_code": args.trust_remote_code,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "models": {
            "target": asdict(
                describe_model(
                    target_model,
                    requested_path=args.target_model_path,
                    device=target_device,
                    dtype_name=args.target_dtype,
                )
            ),
            "draft": (
                asdict(
                    describe_model(
                        draft_model,
                        requested_path=args.draft_model_path,
                        device=draft_device,
                        dtype_name=args.draft_dtype,
                    )
                )
                if draft_model is not None
                else None
            ),
        },
        "summary": _build_summary(records),
        "records": records,
    }

    output_path = _resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path, report
