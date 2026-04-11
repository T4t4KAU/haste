"""Mask helper functions for attention mechanisms."""

from haste.config import Config
import torch

_mask_cache = {
    'glue_and_rec_mask': None,
    'diag_components': None,
    'ones_tensor': None,
    'cached_params': None
}


@torch.inference_mode()
def get_mask_iter_i(i: int, prefix_len: int, K: int, F: int) -> torch.Tensor:
    """Get attention mask for iteration i.
    
    Args:
        i (int): Current iteration
        prefix_len (int): Length of the prefix
        K (int): Number of tokens to speculate
        F (int): Fan-out factor
        
    Returns:
        torch.Tensor: Attention mask
    """
    q_len = F * (K + 1)  # Calculate query length - each speculative position is expanded into F branches
    prefix_mask = torch.ones(q_len, prefix_len)  # Build prefix mask - allow all query positions to see the full prefix

    # Ensure each position can only see itself and previous speculative tokens
    glue_and_rec_mask = (torch.arange(K + 1) <= torch.arange(K + 1)
                         .unsqueeze(1)).repeat_interleave(F, dim=0).to(torch.int32)
    
    diags = [torch.diag(torch.ones(q_len)) for _ in range(i + 1)]  # Each diagonal matrix represents a verification branch
    mask = torch.cat([prefix_mask, glue_and_rec_mask, *diags], dim=1)  # Concatenate complete mask
    assert mask.size(0) == q_len, f"ERROR in get_mask_iter_i: mask should have length q_len, got {mask.size(0)}"
    assert mask.size(1) == prefix_len + (K + 1) + (i + 1) * q_len, f"ERROR in get_mask_iter_i: mask should have length q_len + (K+1) + (i+1) * q_len, got {mask.size(1)}"
    return mask.to(torch.bool)


@torch.inference_mode()
def _precompute_mask_components(K: int, F: int, max_step: int, max_context_len: int, 
                                device: torch.device, fan_out_list: torch.Tensor, fan_out_list_miss: torch.Tensor):
    """Precompute mask components to avoid repeated calculations.
    
    Args:
        K (int): Number of tokens to speculate
        F (int): Fan-out factor
        max_step (int): Maximum step
        max_context_len (int): Maximum context length
        device (torch.device): Device to use
        fan_out_list (torch.Tensor): Fan-out list for cache hits
        fan_out_list_miss (torch.Tensor): Fan-out list for cache misses
        
    Returns:
        tuple: Precomputed mask components
    """
    new_row_idx_hit = torch.arange(K + 1, device=device).repeat_interleave(fan_out_list)  # Expand rows based on branch factor
    glue_and_rec_mask_hit = torch.tril(torch.ones(K + 1, K + 1, device=device), diagonal=0)[new_row_idx_hit].to(torch.int32)
    new_row_idx_miss = torch.arange(K + 1, device=device).repeat_interleave(fan_out_list_miss)  # Expand rows based on branch factor
    glue_and_rec_mask_miss = torch.tril(torch.ones(K + 1, K + 1, device=device), diagonal=0)[new_row_idx_miss].to(torch.int32)
    
    MQ_LEN = glue_and_rec_mask_hit.shape[0]
    assert glue_and_rec_mask_miss.shape[0] == MQ_LEN, f"ERROR in _precompute_mask_components: glue_and_rec_mask_miss should have length MQ_LEN={MQ_LEN}, got {glue_and_rec_mask_miss.shape[0]}"
    
    diag_components = {}
    for step in range(max_step + 1):
        diags = [torch.diag(torch.ones(max_context_len, device=device)) for _ in range(step + 1)]
        if diags:
            diag_components[step] = torch.cat(diags, dim=1)
        else:
            diag_components[step] = torch.empty(MQ_LEN, 0, device=device)
        
    ones_tensor = torch.ones(MQ_LEN, max_context_len, device=device)
    
    return glue_and_rec_mask_hit, glue_and_rec_mask_miss, diag_components, ones_tensor


def _get_custom_mask_optimized(context_lens: int, step: int, K: int, F: int, B: int, device: torch.device,
                                glue_and_rec_mask_hit: torch.Tensor, glue_and_rec_mask_miss: torch.Tensor, 
                                diag_components: dict, ones_tensor: torch.Tensor, cache_hits: torch.Tensor):
    """Get optimized custom mask.
    
    Args:
        context_lens (int): Context lengths
        step (int): Current step
        K (int): Number of tokens to speculate
        F (int): Fan-out factor
        B (int): Batch size
        device (torch.device): Device to use
        glue_and_rec_mask_hit (torch.Tensor): Glue and recovery mask for cache hits
        glue_and_rec_mask_miss (torch.Tensor): Glue and recovery mask for cache misses
        diag_components (dict): Diagonal components
        ones_tensor (torch.Tensor): Ones tensor
        cache_hits (torch.Tensor): Cache hit indicators
        
    Returns:
        torch.Tensor: Optimized custom mask
    """
    MQ_LEN = glue_and_rec_mask_hit.shape[0]
    glue_added = K + 1
    tree_decode_added = (step + 1) * MQ_LEN
    ttl_added = tree_decode_added + glue_added
    
    masks = []
    for b in range(B):
        prefix_len = context_lens[b] - ttl_added
        assert prefix_len >= 0, f"ERROR in _get_custom_mask_optimized: prefix_len should be non-negative, got {prefix_len}"
        prefix_mask = ones_tensor[:, :prefix_len]
        
        if cache_hits[b] == 1:
            glue_and_rec_mask = glue_and_rec_mask_hit
        else:
            glue_and_rec_mask = glue_and_rec_mask_miss
        
        mask = torch.cat([prefix_mask, glue_and_rec_mask, diag_components[step]], dim=1)
        
        assert mask.shape == (MQ_LEN, context_lens[b]), f"ERROR in _get_custom_mask_optimized: mask should have shape {(MQ_LEN, context_lens[b])}, got {mask.shape}"
        masks.append(mask.view(-1))
        
    return torch.cat(masks, dim=0).to(torch.bool)


@torch.inference_mode()
def get_custom_mask_cached(config: Config, context_lens: int, step: int, K: int, F: int, B: int, 
                           device: torch.device, fan_out_list: list[int], fan_out_list_miss: list[int], cache_hits: torch.Tensor):
    """Get cached custom mask.
    
    Args:
        config (Config): Configuration object
        context_lens (int): Context lengths
        step (int): Current step
        K (int): Number of tokens to speculate
        F (int): Fan-out factor
        B (int): Batch size
        device (torch.device): Device to use
        fan_out_list (list[int]): Fan-out list for cache hits
        fan_out_list_miss (list[int]): Fan-out list for cache misses
        cache_hits (torch.Tensor): Cache hit indicators
        
    Returns:
        torch.Tensor: Cached custom mask
    """
    global _mask_cache
    
    max_step = K + 1  # Maximum steps
    
    fan_out_tensor = torch.tensor(fan_out_list, dtype=torch.int64, device=device)
    fan_out_tensor_miss = torch.tensor(fan_out_list_miss, dtype=torch.int64, device=device)
    current_params = (K, F, max_step, config.max_model_len, device, tuple(fan_out_list), tuple(fan_out_list_miss))
    
    if (_mask_cache['cached_params'] is None or _mask_cache['cached_params'] != current_params):
        glue_and_rec_mask_hit, glue_and_rec_mask_miss, diag_components, ones_tensor = \
            _precompute_mask_components(K, F, max_step, config.max_model_len, device, fan_out_tensor, fan_out_tensor_miss)
       
        # Update cache
        _mask_cache['glue_and_rec_mask_hit'] = glue_and_rec_mask_hit
        _mask_cache['glue_and_rec_mask_miss'] = glue_and_rec_mask_miss
        _mask_cache['diag_components'] = diag_components
        _mask_cache['ones_tensor'] = ones_tensor
        _mask_cache['cached_params'] = current_params
        
    mask = _get_custom_mask_optimized(context_lens, step, K, F, B, device,
                                _mask_cache['glue_and_rec_mask_hit'], 
                                _mask_cache['glue_and_rec_mask_miss'], 
                                _mask_cache['diag_components'], 
                                _mask_cache['ones_tensor'], 
                                cache_hits)
    
    return mask


@torch.inference_mode()
def flat_blocks_after_cat(L: torch.Tensor, M: torch.Tensor):
    """Build concatenation mask for variable-length blocks.
    
    Args:
        L (torch.Tensor): Lengths of variable-length blocks
        M (torch.Tensor): Base mask
        
    Returns:
        torch.Tensor: Flattened mask after concatenation
    """
    assert L.ndim == 1 and L.numel() > 0
    
    N, y = M.shape
    k = L.numel()
    device, dtype = M.device, M.dtype
    
    cols_per_block = L.to(torch.long) + y  # [k]
    total_cols = int(cols_per_block.sum().item())  # Calculate total columns
    
    T = torch.ones((N, total_cols), device=device, dtype=dtype)
    
    # Block start positions
    offs = torch.cat([torch.zeros(1, device=device, dtype=torch.long), cols_per_block.cumsum(0)[:-1]])
    pos = torch.arange(total_cols, device=device)  # [total_cols]
    blk = torch.repeat_interleave(torch.arange(k, device=device), cols_per_block)  # [total_cols]
    
    within = pos - offs[blk]
    mask = within.ge(L[blk])
    T[:, mask] = M.repeat(1, k)
    
    blocks = T.split(cols_per_block.tolist(), dim=1)
    out = torch.cat([b.reshape(-1) for b in blocks], dim=0)
    return out 


_vec_cache = {}


def get_custom_mask_vectorized(config: Config, context_lens, step: int, K: int, F: int, B: int, 
                           device: torch.device, cache_hits: torch.Tensor):
    """Get vectorized custom mask.
    
    Args:
        config (Config): Configuration object
        context_lens: Context lengths
        step (int): Current step
        K (int): Number of tokens to speculate
        F (int): Fan-out factor
        B (int): Batch size
        device (torch.device): Device to use
        cache_hits (torch.Tensor): Cache hit indicators
        
    Returns:
        torch.Tensor: Vectorized custom mask
    """
    global _vec_cache
    
    cache_key = (K, tuple(config.fan_out_list), tuple(config.fan_out_list_miss), device)
    if _vec_cache.get('key') != cache_key:
        fan_out_tensor = torch.tensor(config.fan_out_list, dtype=torch.long, device=device)
        fan_out_tensor_miss = torch.tensor(config.fan_out_list_miss, dtype=torch.long, device=device)
        
        MQ_LEN = int(fan_out_tensor.sum().item())
        
        tril = torch.tril(torch.ones(K + 1, K + 1, device=device)).to(torch.bool)      
        glue_hit = tril[torch.arange(K + 1, device=device).repeat_interleave(fan_out_tensor)]
        glue_miss = tril[torch.arange(K + 1, device=device).repeat_interleave(fan_out_tensor_miss)]
        glue_differs = not torch.equal(glue_hit, glue_miss)
        
        eye = torch.eye(MQ_LEN, device=device, dtype=torch.bool)
        max_step = K + 1
        M_hit_per_step = {}
        for s in range(max_step + 1):
            M_hit_per_step[s] = torch.cat([glue_hit, eye.repeat(1, s + 1)], dim=1)
        
        _vec_cache = {
            'key': cache_key,
            'glue_miss': glue_miss,
            'glue_differs': glue_differs,
            'MQ_LEN': MQ_LEN,
            'M_hit_per_step': M_hit_per_step,
        }
        
    c = _vec_cache
    MQ_LEN = c['MQ_LEN']
    M_hit = c['M_hit_per_step'][step]
    
    glue_added = K + 1
    tree_decode_added = (step + 1) * MQ_LEN
    ttl_added = glue_added + tree_decode_added
    
    prefix_lens = context_lens - ttl_added
    assert torch.all(prefix_lens >= 0), f"prefix_lens must be non-negative, but got {prefix_lens}"
    
    out = flat_blocks_after_cat(prefix_lens, M_hit)
    
    if c['glue_differs']:
        miss_mask = (cache_hits == 0)
        if miss_mask.any():
            y = M_hit.shape[1]
            cols_per_block = prefix_lens.to(torch.long) + y  # [k]
            elems_per_block = MQ_LEN * cols_per_block
            block_starts = torch.zeros(B + 1, device=device, dtype=torch.long)
            block_starts[1:] = elems_per_block.cumsum(0)[:-1]
            
            miss_idx = miss_mask.nonzero(as_tuple=True)[0]
            num_miss = miss_idx.numel()

            ms = block_starts[miss_idx]
            mw = cols_per_block[miss_idx]
            mp = prefix_lens[miss_idx]
            
            rows = torch.arange(MQ_LEN, device=device)
            gcols = torch.arange(K + 1, device=device)
            
            # flat_idx[m, r, g] = ms[m] + r * mw[m] + mp[m] + g
            flat_idx = (ms[:, None, None]
                        + rows[None, :, None] * mw[:, None, None]
                        + mp[:, None, None]
                        + gcols[None, None, :])
            
            glue_miss = c['glue_miss']
            vals = glue_miss.unsqueeze(0).expand(num_miss, -1, -1)
            out[flat_idx.reshape(-1)] = vals.reshape(-1)
    
    return out


@torch.inference_mode()
def get_custom_mask(config: Config, context_lens: int, step: int, K: int, F: int, B: int, device: torch.device, cache_hits: torch.Tensor):
    """Get custom mask based on batch size.
    
    Args:
        config (Config): Configuration object
        context_lens (int): Context lengths
        step (int): Current step
        K (int): Number of tokens to speculate
        F (int): Fan-out factor
        B (int): Batch size
        device (torch.device): Device to use
        cache_hits (torch.Tensor): Cache hit indicators
        
    Returns:
        torch.Tensor: Custom mask
    """
    if B <= 8:
        return get_custom_mask_cached(config, context_lens, step, K, F, B, device, fan_out_list=config.fan_out_list, fan_out_list_miss=config.fan_out_list_miss, cache_hits=cache_hits)
    else:
        return get_custom_mask_vectorized(config, context_lens, step, K, F, B, device, cache_hits)
