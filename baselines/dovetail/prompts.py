"""Prompt utilities for the vendored Dovetail baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import DEFAULT_DOVETAIL_DATA_ROOT, DEFAULT_PROMPT_FILE


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    prompt: str | None
    messages: list[dict[str, str]] | None
    turns: list[str] | None
    category: str | None
    reference: list[str] | None
    source: str | None


def discover_dataset_files(dataset_root: str | Path | None = None) -> dict[str, Path]:
    root = Path(dataset_root) if dataset_root else DEFAULT_DOVETAIL_DATA_ROOT
    dataset_files: dict[str, Path] = {}
    if not root.exists():
        return dataset_files
    for child in sorted(root.iterdir()):
        question_file = child / "question.jsonl"
        if child.is_dir() and question_file.exists():
            dataset_files[child.name] = question_file
    return dataset_files


def _load_prompt_file(
    path: Path,
    *,
    limit: int | None = None,
    source: str | None = None,
) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload: dict[str, Any] = json.loads(line)
            prompt_id = str(
                payload.get(
                    "id",
                    payload.get(
                        "question_id",
                        f"{source or path.stem}-{line_number}",
                    ),
                )
            )
            prompt = payload.get("prompt")
            messages = payload.get("messages")
            turns = payload.get("turns")
            if turns is not None:
                turns = [str(turn) for turn in turns]
            if prompt is None and messages is None and turns is None:
                raise ValueError(
                    f"{path}:{line_number} must contain `prompt`, `messages`, or `turns`."
                )
            records.append(
                PromptRecord(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    messages=messages,
                    turns=turns,
                    category=payload.get("category"),
                    reference=payload.get("reference"),
                    source=source or path.parent.name,
                )
            )
            if limit and len(records) >= limit:
                break
    return records


def load_prompt_records(
    prompt_file: str | Path | None = None,
    *,
    limit: int | None = None,
    dataset_root: str | Path | None = None,
    dataset_names: list[str] | None = None,
) -> list[PromptRecord]:
    if dataset_names:
        dataset_files = discover_dataset_files(dataset_root)
        if not dataset_files:
            raise ValueError(f"No datasets found under {dataset_root or DEFAULT_DOVETAIL_DATA_ROOT}")
        selected_names = dataset_names
        if len(selected_names) == 1 and selected_names[0] == "all":
            selected_names = sorted(dataset_files)
        missing = [name for name in selected_names if name not in dataset_files]
        if missing:
            available = ", ".join(sorted(dataset_files))
            raise ValueError(
                f"Unknown datasets: {', '.join(missing)}. Available datasets: {available}"
            )

        records: list[PromptRecord] = []
        for dataset_name in selected_names:
            remaining = None if limit is None else max(0, limit - len(records))
            if remaining == 0:
                break
            records.extend(
                _load_prompt_file(
                    dataset_files[dataset_name],
                    limit=remaining,
                    source=dataset_name,
                )
            )
        if not records:
            raise ValueError("No prompts found in the selected datasets.")
        return records

    path = Path(prompt_file) if prompt_file else DEFAULT_PROMPT_FILE
    records = _load_prompt_file(path, limit=limit, source=path.stem)
    if not records:
        raise ValueError(f"No prompts found in {path}")
    return records


def render_messages(messages: list[dict[str, str]], tokenizer, *, system_prompt: str = "") -> str:
    formatted_messages = list(messages)
    if system_prompt and (
        not formatted_messages or formatted_messages[0].get("role") != "system"
    ):
        formatted_messages = [{"role": "system", "content": system_prompt}] + formatted_messages
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    formatted = []
    for message in formatted_messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        formatted.append(f"{role}: {content}")
    formatted.append("assistant:")
    return "\n\n".join(formatted)


def render_prompt(
    record: PromptRecord,
    tokenizer,
    *,
    system_prompt: str = "",
    messages_override: list[dict[str, str]] | None = None,
    turn_mode: str = "sequential",
) -> str:
    if messages_override is not None:
        return render_messages(messages_override, tokenizer, system_prompt=system_prompt)

    if record.messages:
        return render_messages(record.messages, tokenizer, system_prompt=system_prompt)

    if record.turns:
        if turn_mode == "sequential":
            return render_messages(
                [{"role": "user", "content": record.turns[0]}],
                tokenizer,
                system_prompt=system_prompt,
            )
        return render_messages(
            [{"role": "user", "content": record.turns[0]}],
            tokenizer,
            system_prompt=system_prompt,
        )

    prompt_text = record.prompt or ""
    if system_prompt:
        return f"{system_prompt}\n\n{prompt_text}"
    return prompt_text
