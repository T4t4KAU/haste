"""Linear layers module with support for parallelism."""

import torch
from torch import nn
import torch.nn.functional as F


def divide(numerator, denominator):
    """Divide two numbers with assertion for exact division.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        
    Returns:
        int: Result of exact division
    """
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """Base class for linear layers.
    
    This class provides a common interface for all linear layer implementations.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):
        """Initialize the LinearBase module.
        
        Args:
            input_size (int): Input size
            output_size (int): Output size
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_group = None  # Set to None in single-device mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the linear layer.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """Replicated linear layer.
    
    This class implements a linear layer where weights are replicated across devices.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """Initialize the ReplicatedLinear module.
        
        Args:
            input_size (int): Input size
            output_size (int): Output size
            bias (bool, optional): Whether to use bias. Defaults to False.
        """
        super().__init__(input_size, output_size)
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Load weights for the linear layer.
        
        Args:
            param (nn.Parameter): Parameter to load weights into
            loaded_weight (torch.Tensor): Loaded weights
        """
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ReplicatedLinear module.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return F.linear(x.to(self.weight.dtype), self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """Column parallel linear layer.
    
    This class implements a linear layer with column-wise parallelism.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """Initialize the ColumnParallelLinear module.
        
        Args:
            input_size (int): Input size
            output_size (int): Output size
            bias (bool, optional): Whether to use bias. Defaults to False.
        """
        super().__init__(input_size, output_size)
        self.input_size_per_partition = input_size
        self.output_size_per_partition = output_size  # Handle entire output dimension in single-device mode

        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Load weights for the column parallel linear layer.
        
        Args:
            param (nn.Parameter): Parameter to load weights into
            loaded_weight (torch.Tensor): Loaded weights
        """
        param_data = param.data
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ColumnParallelLinear module.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return F.linear(x.to(self.weight.dtype), self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Merged column parallel linear layer.
    
    This class implements a column parallel linear layer that merges multiple output sizes.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        """Initialize the MergedColumnParallelLinear module.
        
        Args:
            input_size (int): Input size
            output_sizes (list[int]): List of output sizes to merge
            bias (bool, optional): Whether to use bias. Defaults to False.
        """
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias=bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """Load weights for the merged column parallel linear layer.
        
        Args:
            param (nn.Parameter): Parameter to load weights into
            loaded_weight (torch.Tensor): Loaded weights
            loaded_shard_id (int): Shard ID to load
        """
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id])
        shard_size = self.output_sizes[loaded_shard_id]
        param_data = param_data.narrow(0, shard_offset, shard_size)
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """QKV parallel linear layer.
    
    This class implements a linear layer specifically for QKV projections in attention.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        """Initialize the QKVParallelLinear module.
        
        Args:
            hidden_size (int): Hidden size
            head_size (int): Head size
            total_num_heads (int): Total number of heads
            total_num_kv_heads (int | None, optional): Total number of KV heads. Defaults to None.
            bias (bool, optional): Whether to use bias. Defaults to False.
        """
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.num_heads = total_num_heads  # Handle all heads in single-device mode
        self.num_kv_heads = self.total_num_kv_heads  # Handle all KV heads in single-device mode
        input_size = hidden_size
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """Load weights for the QKV parallel linear layer.
        
        Args:
            param (nn.Parameter): Parameter to load weights into
            loaded_weight (torch.Tensor): Loaded weights
            loaded_shard_id (str): Shard ID to load ("q", "k", or "v")
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(0, shard_offset, shard_size)
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """Row parallel linear layer.
    
    This class implements a linear layer with row-wise parallelism.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """Initialize the RowParallelLinear module.
        
        Args:
            input_size (int): Input size
            output_size (int): Output size
            bias (bool, optional): Whether to use bias. Defaults to False.
        """
        super().__init__(input_size, output_size)
        self.input_size_per_partition = input_size  # Handle entire input dimension in single-device mode
        self.output_size_per_partition = output_size

        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Load weights for the row parallel linear layer.
        
        Args:
            param (nn.Parameter): Parameter to load weights into
            loaded_weight (torch.Tensor): Loaded weights
        """
        param_data = param.data
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RowParallelLinear module.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return F.linear(x.to(self.weight.dtype), self.weight, self.bias)
