"""Model weight loading utilities."""

import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

from tqdm import tqdm

from haste.utils.misc import resolve_pretrained_path


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """Default weight loader function.
    
    Args:
        param (nn.Parameter): Parameter to load weights into
        loaded_weight (torch.Tensor): Loaded weights
    """
    param.data.copy_(loaded_weight)


def load_safetensors_model(model: nn.Module, path: str, packed_modules_mapping: dict):
    """Load model weights from safetensors files.
    
    Args:
        model (nn.Module): Model to load weights into
        path (str): Path to the directory containing safetensors files
        packed_modules_mapping (dict): Mapping for packed modules
    """
    safetensor_files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found under model path: {path}")
    for file in tqdm(safetensor_files, desc="Loading model files"):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


def load_model(model: nn.Module, path: str):
    """Load model weights from a directory.
    
    Args:
        model (nn.Module): Model to load weights into
        path (str): Path to the directory containing model weights
    """
    resolved_path = resolve_pretrained_path(path)
    print(f"[load_model] loading model from {resolved_path}")
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    load_safetensors_model(model, resolved_path, packed_modules_mapping)
    
    print(f"[load_model] model loaded from {resolved_path}")
