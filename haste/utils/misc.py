"""Miscellaneous utility functions."""

import os
from pathlib import Path

from transformers import AutoTokenizer


def _looks_like_local_model_dir(path: Path) -> bool:
    """Check if a path looks like a local model directory.
    
    This covers plain Hugging Face snapshot directories as well as local
    ModelScope exports that contain model configs, tokenizer files, and
    safetensor shards directly under the given directory.
    
    Args:
        path (Path): Path to check
        
    Returns:
        bool: True if the path looks like a local model directory
    """
    if not path.is_dir():
        return False

    has_config = (path / "config.json").is_file() or (path / "configuration.json").is_file()
    has_tokenizer = (path / "tokenizer.json").is_file() or (path / "tokenizer_config.json").is_file()
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    return has_config and (has_tokenizer or has_weights)


def resolve_pretrained_path(model_path: str) -> str:
    """Resolve a local pretrained model path.
    
    Supported layouts include:
    
    - Hugging Face cache roots such as
      ``.../hub/models--ORG--NAME``. In that layout, the actual files live under
      ``snapshots/<revision>`` and the root only contains metadata and symlinks.
    - Local unpacked model directories, including ModelScope downloads such as
      ``/path/to/Qwen3-1.7B`` that directly contain ``config.json`` and
      ``*.safetensors`` files.
    
    Args:
        model_path (str): Path to the model
        
    Returns:
        str: Resolved path to the model
    """
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if _looks_like_local_model_dir(path):
        return str(path)

    snapshots_dir = path / "snapshots"
    refs_main = path / "refs" / "main"
    if snapshots_dir.is_dir():
        if refs_main.is_file():
            revision = refs_main.read_text().strip()
            candidate = snapshots_dir / revision
            if _looks_like_local_model_dir(candidate):
                return str(candidate)

        snapshot_dirs = sorted(
            (p for p in snapshots_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if snapshot_dirs:
            for snapshot_dir in snapshot_dirs:
                if _looks_like_local_model_dir(snapshot_dir):
                    return str(snapshot_dir)

    direct_model_children = [child for child in path.iterdir() if _looks_like_local_model_dir(child)]
    if len(direct_model_children) == 1:
        return str(direct_model_children[0])

    return str(path)


def infer_model_family(model_path: str) -> str:
    """Infer model family based on model path name.
    
    Args:
        model_path (str): Path to the model
        
    Returns:
        str: Model family name
    """
    model_path_lower = model_path.lower()
    if "qwen" in model_path_lower:
        return "qwen"
    else:
        return "unknown"


def decode_tokens(token_ids: list[int], tokenizer: AutoTokenizer) -> list[str]:
    """Decode token IDs into text.
    
    Args:
        token_ids (list[int]): List of token IDs
        tokenizer (AutoTokenizer): Tokenizer to use for decoding
        
    Returns:
        list[str]: Decoded text for each token
    """
    decoded = []
    for token in token_ids:
        try:
            text = tokenizer.decode([token], skip_special_tokens=False)
            decoded.append(text)
        except Exception:
            decoded.append(f"<token_id:{token}>")
    return decoded
