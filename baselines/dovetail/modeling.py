"""Model loading helpers for the vendored Dovetail baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ResolvedModelInfo:
    requested_path: str
    resolved_path: str
    device: str
    dtype: str
    architecture: str
    num_hidden_layers: int | None
    hidden_size: int | None
    vocab_size: int | None


def resolve_hf_model_path(model_path: str | Path) -> str:
    path = Path(model_path).expanduser().resolve()
    if (path / "config.json").exists():
        return str(path)

    snapshots_dir = path / "snapshots"
    refs_main = path / "refs" / "main"
    if snapshots_dir.exists():
        if refs_main.exists():
            snapshot_name = refs_main.read_text(encoding="utf-8").strip()
            snapshot_path = snapshots_dir / snapshot_name
            if (snapshot_path / "config.json").exists():
                return str(snapshot_path)

        snapshot_candidates = sorted(
            child for child in snapshots_dir.iterdir() if child.is_dir()
        )
        for candidate in reversed(snapshot_candidates):
            if (candidate / "config.json").exists():
                return str(candidate)

    raise FileNotFoundError(
        "Could not resolve a Hugging Face model snapshot from "
        f"`{model_path}`."
    )


def parse_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def load_tokenizer(model_path: str, trust_remote_code: bool = False):
    resolved = resolve_hf_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_path: str,
    *,
    device: str,
    dtype_name: str,
    attn_implementation: str | None,
    trust_remote_code: bool = False,
):
    resolved = resolve_hf_model_path(model_path)
    load_kwargs: dict[str, Any] = {
        "torch_dtype": parse_torch_dtype(dtype_name),
        "low_cpu_mem_usage": True,
        "local_files_only": True,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(resolved, **load_kwargs)
    except TypeError:
        load_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(resolved, **load_kwargs)

    model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return model


def describe_model(model, requested_path: str, device: str, dtype_name: str) -> ResolvedModelInfo:
    config = model.config
    architectures = getattr(config, "architectures", None) or ["unknown"]
    return ResolvedModelInfo(
        requested_path=requested_path,
        resolved_path=resolve_hf_model_path(requested_path),
        device=device,
        dtype=dtype_name,
        architecture=architectures[0],
        num_hidden_layers=getattr(config, "num_hidden_layers", None),
        hidden_size=getattr(config, "hidden_size", None),
        vocab_size=getattr(config, "vocab_size", None),
    )
