"""Constants for the vendored Dovetail baseline."""

from __future__ import annotations

from pathlib import Path


DEFAULT_TARGET_MODEL_PATH = "Qwen3-32B"
DEFAULT_DRAFT_MODEL_PATH = "Qwen3-0.6B"

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "datasets" / "alpaca" / "question.jsonl"
DEFAULT_DOVETAIL_DATA_ROOT = PROJECT_ROOT / "datasets"
