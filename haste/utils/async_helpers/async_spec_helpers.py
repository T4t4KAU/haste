"""Asynchronous speculative decoding helpers."""

import torch
from haste.config import Config
from transformers import AutoTokenizer


@torch.inference_mode()
def compute_megaspec_lookahead(MQ_LEN: int, K: int) -> int:
    """Compute lookahead for megaspec.
    
    Args:
        MQ_LEN (int): Multi-query length
        K (int): Speculation length
        
    Returns:
        int: Computed lookahead
    """
    return K + 1 + K * MQ_LEN 


@torch.inference_mode()
def make_glue_decode_input_ids(
    draft_tokens: torch.Tensor,  # [B, K]
    rec_tokens: torch.Tensor,   # [B]
) -> torch.Tensor:
    """Create glue token input IDs with recovery token first.
    
    Args:
        draft_tokens (torch.Tensor): Draft tokens of shape [B, K]
        rec_tokens (torch.Tensor): Recovery tokens of shape [B]
        
    Returns:
        torch.Tensor: Glue token input IDs of shape [B, K+1]
    """
    assert draft_tokens.shape[0] == rec_tokens.shape[0], f"Expected draft_tokens and rec_tokens to have the same number of rows, got {draft_tokens.shape[0]} and {rec_tokens.shape[0]}"
    
    # we need flat [num_ttl_q_tokens, num_q_heads, head_dim] for multi-query decode inputs 
    # all info about batch size etc goes through wrapper.plan()
    
    out = torch.cat([rec_tokens.unsqueeze(1), draft_tokens], dim=1).view(-1) # [B, K+1] -> B(K+1)
    
    return out 


def get_forked_recovery_tokens_from_logits(
    config: Config,
    logits: torch.Tensor,
    cache_hits: torch.Tensor,
    returned_tokens: torch.Tensor,
    tokenizer: AutoTokenizer,
    fan_out_list: list[int] | None = None,
    fan_out_list_miss: list[int] | None = None,
    lookahead: int | None = None,
):
    """Get forked recovery tokens from logits.
    
    Args:
        config (Config): Configuration object
        logits (torch.Tensor): Logits of shape [B, K+1, V]
        cache_hits (torch.Tensor): Cache hit indicators of shape [B]
        returned_tokens (torch.Tensor): Returned tokens of shape [B, K+1]
        tokenizer (AutoTokenizer): Tokenizer
        fan_out_list (list[int] | None, optional): Fan-out list. Defaults to None.
        fan_out_list_miss (list[int] | None, optional): Fan-out list for misses. Defaults to None.
        lookahead (int | None, optional): Lookahead. Defaults to None.
        
    Returns:
        torch.Tensor: Forked recovery tokens of shape [B, sum(fan_out_list)]
    """
    B, _, V_actual = logits.shape
    del tokenizer, V_actual

    K = config.speculate_k if lookahead is None else lookahead
    fan_out_list = config.fan_out_list if fan_out_list is None else fan_out_list
    fan_out_list_miss = config.fan_out_list_miss if fan_out_list_miss is None else fan_out_list_miss
    fan_out_list = fan_out_list[: K + 1]
    fan_out_list_miss = fan_out_list_miss[: K + 1]
    # lets scatter then repeat the temp matched expt -- we should do better
    assert cache_hits.shape == (B,), f"cache_hits must have shape (B,), got {cache_hits.shape}"
    assert logits.shape[0] == B and logits.shape[1] == K+1, f"logits must have shape (B, K+1, V), got {logits.shape}"
    assert len(fan_out_list) == K + 1, f"fan_out_list must have length K+1={K+1}, got {len(fan_out_list)}"
    assert returned_tokens.shape == (B, K+1), f"returned_tokens must have shape (B, K+1), got {returned_tokens.shape}"
    
    # Use scatter_ to set returned tokens to -inf so we don't include those in forked tokens 
    # Don't touch the last sequence position, only scatter the first K positions
    logits[:, :-1, :] = logits[:, :-1, :].scatter(
        dim=2,
        index=returned_tokens[:, 1:].unsqueeze(2),
        value=float('-inf'),
    )
    
    # Compute top-k once at max fanout, then mask per row/position
    k_max = max(max(fan_out_list), max(fan_out_list_miss))
    _, topk_idx = torch.topk(logits, k_max, dim=-1)  # [B, K+1, k_max]
    
    # Build per-b, per-(K+1) counts depending on cache_hits
    hit_counts = torch.as_tensor(
        fan_out_list, device=logits.device, dtype=torch.int64)           # [K+1]
    miss_counts = torch.as_tensor(
        fan_out_list_miss, device=logits.device, dtype=torch.int64)     # [K+1]
    ch_bool = cache_hits.to(torch.bool).view(
        B, 1)                                                # [B,1]
    counts_b = torch.where(ch_bool, hit_counts.view(1, -1).expand(B, -1),
                           miss_counts.view(1, -1).expand(B, -1))                                  # [B, K+1]

    # [k_max]
    ar = torch.arange(k_max, device=logits.device)
    # [B, K+1, k_max]
    mask = ar.view(1, 1, -1) < counts_b.view(B, K + 1, 1)

    idxs_flat = topk_idx.masked_select(mask).view(
        B, -1)
    expected_width = max(sum(fan_out_list), sum(fan_out_list_miss))
    assert idxs_flat.shape == (B, expected_width), f"idxs_flat should be (B, {expected_width}), got {idxs_flat.shape}"

    
    # return idxs_flat, topk_idx
    return idxs_flat


def apply_sampler_x_rescaling(probs: torch.Tensor, sampler_x: float, F: int) -> torch.Tensor:
    """Apply sampler_x rescaling to probabilities.
    
    Args:
        probs (torch.Tensor): Probability tensor of shape [B, S, V] where S can be =1
        sampler_x (float): Rescaling factor for top-F probabilities
        F (int): Number of top probabilities to rescale
        
    Returns:
        torch.Tensor: Rescaled and renormalized probabilities
    """
    # Find topF indices with highest probs
    _, topk_indices = torch.topk(probs, F+1, dim=-1)  # [B, S, F]

    # Create a mask for topF positions
    topf_mask = torch.zeros_like(probs, dtype=torch.bool)
    topf_mask.scatter_(dim=-1, index=topk_indices, value=True)

    # Rescale topF probs by sampler_x factor
    probs = torch.where(topf_mask, probs * sampler_x, probs)

    # Renormalize to get valid distribution
    probs = probs / probs.sum(dim=-1, keepdim=True)

    return probs
