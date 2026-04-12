"""LLM-judge evaluation script for local Haste-compatible model servers.

This script:
1. Loads prompts from local JSONL datasets.
2. Queries a local OpenAI-compatible model endpoint for candidate answers.
3. Uses a stronger remote judge model to score those answers with templates from
   ``datasets/judge_prompts.jsonl``.
4. Reports judge-based average score and pass-rate style accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_JUDGE_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_JUDGE_MODEL = "doubao-seed-2-0-pro-260215"
DEFAULT_CANDIDATE_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_CANDIDATE_MODEL = "haste-local"
DEFAULT_JUDGE_PROMPT_FILE = "datasets/judge_prompts.jsonl"
DEFAULT_OUTPUT_PATH = "outputs/eval_results.json"
CANDIDATE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Provide only the final answer to the user. "
    "Do not output hidden reasoning, chain-of-thought, or any `<think>` tags."
)

SCORE_PATTERN = re.compile(r"\[\[(\d+)\]\]")


def build_parser() -> argparse.ArgumentParser:
    """Build command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a local model with an external LLM judge.",
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets",
        help="Root directory that contains dataset subdirectories with question.jsonl files.",
    )
    parser.add_argument(
        "--datasets",
        default="alpaca,gsm8k",
        help="Comma-separated dataset names under dataset-root, or `all`.",
    )
    parser.add_argument(
        "--dataset-file",
        default="",
        help="Optional explicit JSONL dataset file. Overrides --datasets.",
    )
    parser.add_argument(
        "--judge-prompt-file",
        default=DEFAULT_JUDGE_PROMPT_FILE,
        help="Path to the judge prompt JSONL file.",
    )
    parser.add_argument(
        "--judge-prompt-name",
        default="",
        help="Optional explicit judge prompt name. If omitted, eval.py auto-selects a template.",
    )
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=20,
        help="Maximum number of samples to evaluate. 0 means all loaded prompts.",
    )
    parser.add_argument(
        "--turn-limit",
        type=int,
        default=2,
        help="Maximum number of turns to evaluate for multi-turn datasets.",
    )
    parser.add_argument(
        "--candidate-base-url",
        default=DEFAULT_CANDIDATE_BASE_URL,
        help="Base URL of the local OpenAI-compatible model server.",
    )
    parser.add_argument(
        "--candidate-api-key",
        default=os.environ.get("HASTE_EVAL_CANDIDATE_API_KEY", "EMPTY"),
        help="API key for the local candidate endpoint. A dummy value is fine for the bundled server.",
    )
    parser.add_argument(
        "--candidate-model",
        default=DEFAULT_CANDIDATE_MODEL,
        help="Model name sent to the candidate chat completion endpoint.",
    )
    parser.add_argument(
        "--judge-base-url",
        default=os.environ.get("HASTE_EVAL_JUDGE_BASE_URL", DEFAULT_JUDGE_BASE_URL),
        help="Base URL of the judge model endpoint.",
    )
    parser.add_argument(
        "--judge-api-key",
        default=os.environ.get("HASTE_EVAL_JUDGE_API_KEY", ""),
        help="API key for the judge model endpoint. Can also be provided via HASTE_EVAL_JUDGE_API_KEY.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("HASTE_EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        help="Judge model name.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to request from the local model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the local model.",
    )
    parser.add_argument(
        "--pass-score",
        type=int,
        default=7,
        help="Score threshold used to compute pass-rate style accuracy.",
    )
    parser.add_argument(
        "--judge-max-retries",
        type=int,
        default=3,
        help="Maximum retries when the judge API call fails or returns an unparsable score.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between judge retries.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the evaluation JSON report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample progress and judge details.",
    )
    return parser


def get_openai_client(*, base_url: str, api_key: str):
    """Create an OpenAI-compatible client lazily.

    Importing the package lazily keeps unit tests for pure eval helpers free
    from the optional dependency.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "The `openai` package is required for eval.py. Install it with `pip install openai`."
        ) from exc

    return OpenAI(base_url=base_url, api_key=api_key)


def resolve_dataset_files(dataset_root: Path, datasets: str, dataset_file: str) -> list[Path]:
    """Resolve dataset files from command line arguments."""
    if dataset_file:
        path = Path(dataset_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return [path]

    dataset_root = dataset_root.expanduser().resolve()
    if datasets.strip().lower() == "all":
        files = sorted(dataset_root.glob("*/question.jsonl"))
    else:
        files = []
        for name in datasets.split(","):
            dataset_name = name.strip()
            if not dataset_name:
                continue
            candidate = dataset_root / dataset_name / "question.jsonl"
            if not candidate.exists():
                raise FileNotFoundError(f"Dataset file not found: {candidate}")
            files.append(candidate)

    if not files:
        raise ValueError("No dataset files were resolved.")
    return files


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_judge_prompts(path: Path) -> dict[str, dict[str, Any]]:
    """Load judge prompt templates from JSONL."""
    prompts = {}
    for record in load_jsonl(path):
        prompts[record["name"]] = record
    if not prompts:
        raise ValueError(f"No judge prompts loaded from {path}")
    return prompts


def trim_sample_turns(sample: dict[str, Any], turn_limit: int) -> dict[str, Any]:
    """Trim sample turns and references to a configurable maximum."""
    turns = list(sample.get("turns", []))
    if not turns:
        raise ValueError(f"Sample {sample.get('question_id')} does not contain `turns`.")

    trimmed = dict(sample)
    trimmed["turns"] = turns[:turn_limit]
    if "reference" in sample and isinstance(sample["reference"], list):
        trimmed["reference"] = list(sample["reference"])[:turn_limit]
    return trimmed


def is_math_sample(sample: dict[str, Any], dataset_name: str) -> bool:
    """Heuristic for math-style judge templates."""
    category = str(sample.get("category", "")).lower()
    return "math" in category or dataset_name.lower() == "gsm8k"


def select_judge_prompt_name(
    sample: dict[str, Any],
    *,
    dataset_name: str,
    override: str = "",
) -> str:
    """Select the appropriate single-model judge prompt template."""
    if override:
        return override

    num_turns = len(sample["turns"])
    math_sample = is_math_sample(sample, dataset_name)
    has_reference = bool(sample.get("reference"))

    if num_turns > 1:
        if math_sample and has_reference:
            return "single-math-v1-multi-turn"
        return "single-v1-multi-turn"

    if math_sample and has_reference:
        return "single-math-v1"
    return "single-v1"


def build_judge_prompt(
    *,
    sample: dict[str, Any],
    answers: list[str],
    prompt_record: dict[str, Any],
) -> tuple[str, str]:
    """Render judge system prompt and user prompt text."""
    system_prompt = prompt_record["system_prompt"]
    template = prompt_record["prompt_template"]
    turns = sample["turns"]
    references = sample.get("reference", [])

    if prompt_record["name"] == "single-v1":
        prompt = template.format(question=turns[0], answer=answers[0])
    elif prompt_record["name"] == "single-math-v1":
        prompt = template.format(
            question=turns[0],
            ref_answer_1=references[0] if references else "",
            answer=answers[0],
        )
    elif prompt_record["name"] == "single-v1-multi-turn":
        if len(turns) < 2 or len(answers) < 2:
            raise ValueError("single-v1-multi-turn requires at least 2 turns and 2 answers.")
        prompt = template.format(
            question_1=turns[0],
            answer_1=answers[0],
            question_2=turns[1],
            answer_2=answers[1],
        )
    elif prompt_record["name"] == "single-math-v1-multi-turn":
        if len(turns) < 2 or len(answers) < 2 or len(references) < 2:
            raise ValueError("single-math-v1-multi-turn requires 2 turns, 2 answers, and 2 references.")
        prompt = template.format(
            question_1=turns[0],
            ref_answer_1=references[0],
            answer_1=answers[0],
            question_2=turns[1],
            ref_answer_2=references[1],
            answer_2=answers[1],
        )
    else:
        raise ValueError(f"Unsupported judge prompt for single-model eval: {prompt_record['name']}")

    return system_prompt, prompt


def parse_judge_score(text: str) -> int:
    """Parse judge score like ``[[7]]`` from output text."""
    match = SCORE_PATTERN.search(text)
    if match is None:
        raise ValueError(f"Judge output does not contain a score marker: {text}")
    score = int(match.group(1))
    if not 1 <= score <= 10:
        raise ValueError(f"Judge score must be between 1 and 10, got {score}")
    return score


def request_candidate_chat_completion(
    client,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_new_tokens: int,
) -> str:
    """Query the local candidate model."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    choice = response.choices[0]
    content = getattr(choice.message, "content", "")
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                chunks.append(item.get("text", ""))
            elif hasattr(item, "text"):
                chunks.append(getattr(item, "text", ""))
        return "".join(chunks).strip()
    return str(content).strip()


def extract_response_text(response: Any) -> str:
    """Extract text from a Responses API object."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    output = getattr(response, "output", None)
    if output:
        chunks: list[str] = []
        for item in output:
            content = getattr(item, "content", None) or item.get("content", []) if isinstance(item, dict) else []
            for block in content:
                text = getattr(block, "text", None)
                if text:
                    chunks.append(str(text))
                elif isinstance(block, dict) and "text" in block:
                    chunks.append(str(block["text"]))
        if chunks:
            return "".join(chunks).strip()

    raise ValueError(f"Unable to extract text from judge response: {response}")


def request_judge_score(
    client,
    *,
    model: str,
    system_prompt: str,
    prompt: str,
) -> str:
    """Query the judge model, preferring Responses API and falling back to chat completions."""
    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": system_prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                },
            ],
        )
        return extract_response_text(response)
    except Exception:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content
        return str(content).strip()


def generate_answers_for_sample(
    candidate_client,
    *,
    candidate_model: str,
    sample: dict[str, Any],
    temperature: float,
    max_new_tokens: int,
) -> list[str]:
    """Run the local model on all turns of a sample."""
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": CANDIDATE_SYSTEM_PROMPT,
        }
    ]
    answers: list[str] = []
    for question in sample["turns"]:
        messages.append({"role": "user", "content": question})
        answer = request_candidate_chat_completion(
            candidate_client,
            model=candidate_model,
            messages=messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        answers.append(answer)
        messages.append({"role": "assistant", "content": answer})
    return answers


def score_sample_with_retries(
    judge_client,
    *,
    judge_model: str,
    system_prompt: str,
    prompt: str,
    max_retries: int,
    retry_sleep: float,
) -> tuple[int, str]:
    """Judge one sample with retries."""
    last_error: Exception | None = None
    last_text = ""
    for attempt in range(1, max_retries + 1):
        try:
            last_text = request_judge_score(
                judge_client,
                model=judge_model,
                system_prompt=system_prompt,
                prompt=prompt,
            )
            return parse_judge_score(last_text), last_text
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_sleep)
    raise RuntimeError(f"Judge failed after {max_retries} attempts. Last output: {last_text}") from last_error


def summarize_results(results: list[dict[str, Any]], *, pass_score: int) -> dict[str, Any]:
    """Build aggregate summary."""
    if not results:
        return {
            "num_samples": 0,
            "average_score": None,
            "accuracy": None,
            "pass_score": pass_score,
            "score_distribution": {},
            "per_dataset": {},
        }

    scores = [result["judge_score"] for result in results]
    passed = sum(score >= pass_score for score in scores)
    score_distribution: dict[str, int] = {}
    per_dataset: dict[str, dict[str, Any]] = {}

    for score in scores:
        score_distribution[str(score)] = score_distribution.get(str(score), 0) + 1

    dataset_groups: dict[str, list[int]] = {}
    for result in results:
        dataset_groups.setdefault(result["dataset"], []).append(result["judge_score"])
    for dataset_name, dataset_scores in dataset_groups.items():
        dataset_passed = sum(score >= pass_score for score in dataset_scores)
        per_dataset[dataset_name] = {
            "num_samples": len(dataset_scores),
            "average_score": mean(dataset_scores),
            "accuracy": dataset_passed / len(dataset_scores),
        }

    return {
        "num_samples": len(scores),
        "average_score": mean(scores),
        "accuracy": passed / len(scores),
        "pass_score": pass_score,
        "min_score": min(scores),
        "max_score": max(scores),
        "score_distribution": score_distribution,
        "per_dataset": per_dataset,
    }


def save_report(path: Path, report: dict[str, Any]) -> Path:
    """Save evaluation report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return path


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    if not args.judge_api_key:
        raise ValueError(
            "--judge-api-key is required. You can also provide it via HASTE_EVAL_JUDGE_API_KEY."
        )

    dataset_root = Path(args.dataset_root)
    judge_prompt_file = Path(args.judge_prompt_file)
    dataset_files = resolve_dataset_files(dataset_root, args.datasets, args.dataset_file)
    judge_prompts = load_judge_prompts(judge_prompt_file)

    candidate_client = get_openai_client(
        base_url=args.candidate_base_url,
        api_key=args.candidate_api_key,
    )
    judge_client = get_openai_client(
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
    )

    samples: list[tuple[str, dict[str, Any]]] = []
    for dataset_path in dataset_files:
        dataset_name = dataset_path.parent.name
        for record in load_jsonl(dataset_path):
            samples.append((dataset_name, trim_sample_turns(record, args.turn_limit)))

    if args.prompt_limit > 0:
        samples = samples[: args.prompt_limit]

    if not samples:
        raise ValueError("No evaluation samples were loaded.")

    print("Starting evaluation...", flush=True)
    print(f"Candidate endpoint: {args.candidate_base_url}", flush=True)
    print(f"Judge endpoint: {args.judge_base_url}", flush=True)
    print(f"Loaded samples: {len(samples)}", flush=True)

    results: list[dict[str, Any]] = []
    for index, (dataset_name, sample) in enumerate(samples, start=1):
        judge_prompt_name = select_judge_prompt_name(
            sample,
            dataset_name=dataset_name,
            override=args.judge_prompt_name,
        )
        if judge_prompt_name not in judge_prompts:
            raise KeyError(f"Judge prompt `{judge_prompt_name}` not found in {judge_prompt_file}")

        if args.verbose:
            print(
                f"[{index}/{len(samples)}] dataset={dataset_name} "
                f"question_id={sample.get('question_id')} judge_prompt={judge_prompt_name}",
                flush=True,
            )

        answers = generate_answers_for_sample(
            candidate_client,
            candidate_model=args.candidate_model,
            sample=sample,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        system_prompt, judge_prompt = build_judge_prompt(
            sample=sample,
            answers=answers,
            prompt_record=judge_prompts[judge_prompt_name],
        )
        score, judge_output = score_sample_with_retries(
            judge_client,
            judge_model=args.judge_model,
            system_prompt=system_prompt,
            prompt=judge_prompt,
            max_retries=args.judge_max_retries,
            retry_sleep=args.retry_sleep,
        )

        result = {
            "dataset": dataset_name,
            "question_id": sample.get("question_id"),
            "category": sample.get("category"),
            "judge_prompt_name": judge_prompt_name,
            "turns": sample["turns"],
            "reference": sample.get("reference"),
            "answers": answers,
            "judge_score": score,
            "judge_output": judge_output,
        }
        results.append(result)

        if args.verbose:
            print(f"  -> score={score}", flush=True)

    summary = summarize_results(results, pass_score=args.pass_score)
    report = {
        "config": {
            "dataset_root": str(dataset_root.resolve()),
            "dataset_files": [str(path.resolve()) for path in dataset_files],
            "judge_prompt_file": str(judge_prompt_file.resolve()),
            "judge_prompt_name": args.judge_prompt_name or "auto",
            "candidate_base_url": args.candidate_base_url,
            "candidate_model": args.candidate_model,
            "judge_base_url": args.judge_base_url,
            "judge_model": args.judge_model,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "pass_score": args.pass_score,
            "turn_limit": args.turn_limit,
        },
        "summary": summary,
        "results": results,
    }
    output_path = save_report(Path(args.output), report)

    print("\nEvaluation Summary")
    print("-" * 60)
    print(f"Samples: {summary['num_samples']}")
    if summary["average_score"] is not None:
        print(f"Average judge score: {summary['average_score']:.2f}/10")
    if summary["accuracy"] is not None:
        print(f"Accuracy (score >= {args.pass_score}): {summary['accuracy']:.2%}")
    print(f"Saved report: {output_path.resolve()}")

    if summary["per_dataset"]:
        print("\nPer-dataset summary")
        for dataset_name, dataset_summary in summary["per_dataset"].items():
            print(
                f"- {dataset_name}: avg_score={dataset_summary['average_score']:.2f}, "
                f"accuracy={dataset_summary['accuracy']:.2%}, "
                f"samples={dataset_summary['num_samples']}"
            )


if __name__ == "__main__":
    main()
