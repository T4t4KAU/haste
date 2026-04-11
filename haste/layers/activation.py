"""Activation functions module."""

import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """SiLU activation with element-wise multiplication.
    
    This class implements a custom activation function that splits the input tensor
    into two parts along the last dimension, applies SiLU activation to the first part,
    and multiplies it with the second part.
    """

    def __init__(self):
        """Initialize the SiluAndMul module."""
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SiluAndMul module.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after applying SiLU and multiplication
        """
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
