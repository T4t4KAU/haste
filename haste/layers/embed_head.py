"""Embedding and language model head modules."""

import torch
from torch import nn
import torch.nn.functional as F

from haste.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """Vocabulary parallel embedding layer.
    
    This class implements a vocabulary parallel embedding layer that can handle
    distributed vocabulary across multiple devices. Currently, it's implemented
    for single-device mode.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        draft_async: bool = False,
    ):
        """Initialize the vocabulary parallel embedding layer.
        
        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embeddings
            draft_async (bool, optional): Whether to use async draft mode. Defaults to False.
        """
        super().__init__()

        self.draft_async = draft_async

        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = num_embeddings  # Handle entire vocabulary in single-device mode
        self.vocab_start_idx = 0  # Start from 0
        self.vocab_end_idx = num_embeddings  # End at vocabulary size
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Load weights for the embedding layer.
        
        Args:
            param (nn.Parameter): Parameter to load weights into
            loaded_weight (torch.Tensor): Loaded weights
        """
        # Directly copy entire weights in single-device mode
        param_data = param.data
        assert param.data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """Forward pass of the embedding layer.
        
        Args:
            x (torch.Tensor): Input tensor of token IDs
            
        Returns:
            torch.Tensor: Embedded tokens
        """
        # Direct embedding operation in single-device mode
        y = F.embedding(x, self.weight)
        # print(f'in vocab parallel embedding, shape of input: {x.shape}, shape of output: {y.shape}') # [nt] -> [nt, D]
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """Parallel language model head.
    
    This class implements a parallel language model head that can handle
    distributed vocabulary across multiple devices.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        draft_async: bool = False,
    ):
        """Initialize the parallel language model head.
        
        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embeddings
            bias (bool, optional): Whether to use bias. Defaults to False.
            draft_async (bool, optional): Whether to use async draft mode. Defaults to False.
        """
        assert not bias, "ERROR in ParallelLMHead: bias is not supported"
        super().__init__(num_embeddings, embedding_dim, draft_async)
        self.draft_async = draft_async

    def forward(self, x: torch.Tensor, last_only: bool = True):
        """Forward pass of the language model head.
        
        Args:
            x (torch.Tensor): Input tensor of shape [nt = B*S, D]
            last_only (bool, optional): Whether to return only the last token's logits. Defaults to True.
            
        Returns:
            torch.Tensor: Logits of shape [nt, V] or [b, V] or [b, L, V]
        """
        context = get_context()
        if context.cu_seqlens_q is not None:  # Multi-query decode (prefill, glue, verify, tree decode)
            if context.is_prefill:
                if last_only:
                    # [nt, D] -> [b, D] which later becomes [b, V]
                    last_indices = context.cu_seqlens_q[1:] - 1
                    x = x[last_indices].contiguous()
                else:
                    # Return logits for all tokens in prefill
                    flat_logits = F.linear(x, self.weight)
                    return flat_logits
            else:  # Multi-query decode path (glue, verify, tree)
                flat_logits = F.linear(x, self.weight)
                # Check if constant query len (verify/tree) or variable (glue)
                batch_size = context.cu_seqlens_q.size(0) - 1
                total_tokens = x.size(0)
                if total_tokens % batch_size == 0:
                    constant_query_len = total_tokens // batch_size
                    return flat_logits.view(batch_size, constant_query_len, flat_logits.size(-1))
                return flat_logits  # Variable-length: return flat [N, V]

        # Decode, get single token
        logits = F.linear(x, self.weight)
        return logits
