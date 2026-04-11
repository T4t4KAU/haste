"""Layer normalization modules."""

import torch
from torch import nn


class RMSHeadNorm(nn.Module):
    """RMS (Root Mean Square) head normalization.
    
    This class implements RMS normalization specifically designed for model heads,
    with support for both standalone normalization and normalization with residual connection.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        """Initialize the RMSHeadNorm module.
        
        Args:
            hidden_size (int): Hidden size of the input
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for standalone RMS normalization.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Normalized tensor
        """
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for RMS normalization with residual connection.
        
        Args:
            x (torch.Tensor): Input tensor
            residual (torch.Tensor): Residual tensor to add
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Normalized tensor and updated residual
        """
        orig_dtype = x.dtype
        x = x.float().add(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x.mul(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the RMSHeadNorm module.
        
        Args:
            x (torch.Tensor): Input tensor
            residual (torch.Tensor | None, optional): Residual tensor to add. Defaults to None.
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Normalized tensor and updated residual
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)


class RMSDNorm(nn.Module):
    """RMS (Root Mean Square) deep normalization.
    
    This class implements RMS normalization for deep layers, with a different method
    signature to allow torch compile to specialize differently based on shape.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        """Initialize the RMSDNorm module.
        
        Args:
            hidden_size (int): Hidden size of the input
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def norm_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for standalone RMS normalization.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Normalized tensor
        """
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x.mul(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul(self.weight)
        return x

    def add_norm_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for RMS normalization with residual connection.
        
        Args:
            x (torch.Tensor): Input tensor
            residual (torch.Tensor): Residual tensor to add
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Normalized tensor and updated residual
        """
        orig_dtype = x.dtype
        x = x.float().add(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x.mul(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the RMSDNorm module.
        
        Args:
            x (torch.Tensor): Input tensor
            residual (torch.Tensor | None, optional): Residual tensor to add. Defaults to None.
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Normalized tensor and updated residual
        """
        if residual is None:
            return self.norm_forward(x)
        else:
            return self.add_norm_forward(x, residual)
