"""Qwen3 model implementation."""

import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from haste.layers.activation import SiluAndMul
from haste.layers.attention import Attention
from haste.layers.layernorm import RMSHeadNorm, RMSDNorm
from haste.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from haste.layers.rotary_embedding import get_rope
from haste.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """Qwen3 attention module.
    
    This class implements the attention mechanism for Qwen3 model.
    """

    def __init__( 
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        # speculation args 
        draft: bool = False,
        speculate: bool = False,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
    ) -> None:
        """Initialize the Qwen3Attention module.
        
        Args:
            hidden_size (int): Hidden size of the model
            num_heads (int): Number of attention heads
            num_kv_heads (int): Number of key/value heads
            max_position (int, optional): Maximum position embedding. Defaults to 4096 * 32.
            head_dim (int | None, optional): Head dimension. Defaults to None.
            rms_norm_eps (float, optional): Epsilon for RMS norm. Defaults to 1e-06.
            qkv_bias (bool, optional): Whether to use bias in QKV projection. Defaults to False.
            rope_theta (float, optional): Theta for RoPE. Defaults to 10000.
            rope_scaling (tuple | None, optional): RoPE scaling. Defaults to None.
            draft (bool, optional): Whether this is a draft model. Defaults to False.
            speculate (bool, optional): Whether to use speculative decoding. Defaults to False.
            spec_k (int, optional): Speculation length. Defaults to 1.
            async_fan_out (int, optional): Async fan-out. Defaults to 1.
            draft_async (bool, optional): Whether to use async draft mode. Defaults to False.
        """
        super().__init__()
        self.draft = draft
        self.draft_async = draft_async

        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = self.total_num_kv_heads
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            draft=draft,
            speculate=speculate,
            draft_async=draft_async,
            F=async_fan_out,
            K=spec_k,
        )
        self.q_norm = RMSHeadNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSHeadNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
        """Forward pass of the Qwen3Attention module.
        
        Args:
            positions (torch.Tensor): Position indices
            hidden_states (torch.Tensor): Input hidden states
            
        Returns:
            torch.Tensor: Output hidden states
        """
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.reshape(-1, self.head_dim) # [num_tokens, D] = [b*s, nh*hd] -> [b*s*nh, hd]
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.reshape(q.shape) # back to [b*s, nh*hd]

        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.reshape(k.shape)

        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output

        
class Qwen3MLP(nn.Module):
    """Qwen3 MLP module.
    
    This class implements the MLP for Qwen3 model.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        """Initialize the Qwen3MLP module.
        
        Args:
            hidden_size (int): Hidden size of the model
            intermediate_size (int): Intermediate size of the MLP
            hidden_act (str): Activation function
        """
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        """Forward pass of the Qwen3MLP module.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 decoder layer.
    
    This class implements a single decoder layer for Qwen3 model.
    """

    def __init__(
        self,
        config: Qwen3Config,
        draft: bool,
        speculate: bool,
        spec_k: int,
        async_fan_out: int,
        draft_async: bool,
    ) -> None:
        """Initialize the Qwen3DecoderLayer module.
        
        Args:
            config (Qwen3Config): Model configuration
            draft (bool): Whether this is a draft model
            speculate (bool): Whether to use speculative decoding
            spec_k (int): Speculation length
            async_fan_out (int): Async fan-out
            draft_async (bool): Whether to use async draft mode
        """
        super().__init__() 
        self.draft = draft
        self.speculate = speculate
        self.spec_k = spec_k
        self.async_fan_out = async_fan_out
        self.draft_async = draft_async
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            draft=self.draft,
            speculate=self.speculate,
            spec_k=self.spec_k,
            async_fan_out=self.async_fan_out,
            draft_async=self.draft_async,
        )

        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Qwen3DecoderLayer module.
        
        Args:
            positions (torch.Tensor): Position indices
            hidden_states (torch.Tensor): Input hidden states
            residual (torch.Tensor | None): Residual tensor
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output hidden states and residual
        """
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """Qwen3 model.
    
    This class implements the full Qwen3 model.
    """

    def __init__(
        self,
        config: Qwen3Config,
        draft: bool = False,
        speculate: bool = False,
        spec_k: int = 1,
        async_fan_out: int = 1, 
        draft_async: bool = False,
    ) -> None:
        """Initialize the Qwen3Model module.
        
        Args:
            config (Qwen3Config): Model configuration
            draft (bool, optional): Whether this is a draft model. Defaults to False.
            speculate (bool, optional): Whether to use speculative decoding. Defaults to False.
            spec_k (int, optional): Speculation length. Defaults to 1.
            async_fan_out (int, optional): Async fan-out. Defaults to 1.
            draft_async (bool, optional): Whether to use async draft mode. Defaults to False.
        """
        super().__init__()
        self.draft = draft
        self.speculate = speculate
        self.spec_k = spec_k
        self.async_fan_out = async_fan_out
        self.draft_async = draft_async
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            draft_async=self.draft_async,
        )
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                config,
                draft=self.draft,
                speculate=self.speculate,
                spec_k=self.spec_k,
                async_fan_out=self.async_fan_out,
                draft_async=self.draft_async,
            )
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the Qwen3Model module.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            positions (torch.Tensor): Position indices
            
        Returns:
            torch.Tensor: Output hidden states
        """
        hidden_states = self.embed_tokens(input_ids)  # torch.Size([4096, 2560]) always through residual stream 
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 model for causal language modeling.
    
    This class implements the Qwen3 model for causal language modeling tasks.
    """
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config, 
        draft: bool = False,
        speculate: bool = False,
        use_eagle: bool = False,
        spec_k: int = 1,
        async_fan_out: int = 1, 
        draft_async: bool = False,
        auto_tune_kf: bool = False,
    ) -> None:
        """Initialize the Qwen3ForCausalLM module.
        
        Args:
            config (Qwen3Config): Model configuration
            draft (bool, optional): Whether this is a draft model. Defaults to False.
            speculate (bool, optional): Whether to use speculative decoding. Defaults to False.
            use_eagle (bool, optional): Whether to use Eagle optimization. Defaults to False.
            spec_k (int, optional): Speculation length. Defaults to 1.
            async_fan_out (int, optional): Async fan-out. Defaults to 1.
            draft_async (bool, optional): Whether to use async draft mode. Defaults to False.
            auto_tune_kf (bool, optional): Whether to auto-tune K and F parameters. Defaults to False.
        """
        super().__init__()

        self.draft = draft
        self.draft_async = draft_async

        if auto_tune_kf:
            print(
                "Starting Qwen3ForCausalLM init, "
                f"draft={draft}, speculate={speculate}, "
                f"spec_k_max={spec_k}, async_fan_out_max={async_fan_out}, auto_tune_kf=True"
            )
        else:
            print(
                "Starting Qwen3ForCausalLM init, "
                f"draft={draft}, speculate={speculate}, "
                f"spec_k={spec_k}, async_fan_out={async_fan_out}"
            )
        self.model = Qwen3Model(config, draft, speculate, spec_k, async_fan_out, draft_async)
        self.async_fan_out = async_fan_out
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            draft_async=draft_async,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        if auto_tune_kf:
            print(
                "Finishing Qwen3ForCausalLM init, "
                f"draft={draft}, speculate={speculate}, "
                f"spec_k_max={spec_k}, async_fan_out_max={async_fan_out}, auto_tune_kf=True"
            )
        else:
            print(
                "Finishing Qwen3ForCausalLM init, "
                f"draft={draft}, speculate={speculate}, "
                f"spec_k={spec_k}, async_fan_out={async_fan_out}"
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the Qwen3ForCausalLM module.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            positions (torch.Tensor): Position indices
            
        Returns:
            torch.Tensor: Output hidden states
        """
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        last_only: bool = True, 
    ) -> torch.Tensor:
        """Compute logits from hidden states.
        
        Args:
            hidden_states (torch.Tensor): Hidden states
            last_only (bool, optional): Whether to return only the last token's logits. Defaults to True.
            
        Returns:
            torch.Tensor: Logits
        """
        logits = self.lm_head(hidden_states, last_only=last_only)
        return logits
