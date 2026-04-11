"""Runner helper functions for tensor preparation."""

import torch
from haste.engine.sequence import Sequence


def prepare_decode_tensors_from_seqs(
    seqs: list[Sequence],
    block_size: int,
    is_draft: bool,
    verify: bool = False,
    k: int = 1,
    device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare tensors for decoding operations.
    
    Args:
        seqs (list[Sequence]): List of sequences
        block_size (int): Block size
        is_draft (bool): Whether this is for draft model
        verify (bool): Whether this is for verification
        k (int): Number of tokens to verify
        device (torch.device): Device to use
        
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Input IDs, positions, slot mapping, and context lengths
    """
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []
    
    # Normal decoding mode
    if not verify:
       assert k == -1, "k should be -1 for normal decoding or draft fwd in speculation"
       
       # Iterate through each sequence
       for seq in seqs:
           block_table = seq.draft_block_table if is_draft else seq.block_table 
           assert len(seq) // block_size <= len(block_table), "in sync spec draft decode, not enough blocks allocated"
           num_cached_tokens = seq.num_cached_tokens if not is_draft else seq.num_draft_cached_tokens
           assert num_cached_tokens == len(seq) - 1, f"num_cached_tokens ({num_cached_tokens}) does not match seq length ({len(seq) - 1})"
           
           input_ids.append(seq.last_token)  # Add the last token
           positions.append(len(seq) - 1)  # Add the position of the last token
           context_lens.append(len(seq))  # Add sequence length
           
           pos = seq.num_tokens - 1
           block_idx = pos // block_size
           pos_in_block = pos % block_size
           slot_mapping.append(block_table[block_idx] * block_size + pos_in_block)
    
    # Verification decoding mode
    else:
        assert not is_draft, "verify path only supported for target model"  # We prep tensors to send to draft for glue on the target 
        assert k > 0, "k should be > 0 for target fwd in verify"
        
        for seq_idx, seq in enumerate(seqs):
           assert (seq.num_tokens - 1) // block_size <= len(seq.block_table), "in sync spec target verify, not enough blocks allocated"
           
           pos0 = seq.num_tokens - (k + 1)  # Verification start position
           input_ids.extend(seq[pos0:])  # Collect all tokens from pos0 to current
           positions.extend(list(range(pos0, pos0 + k + 1)))  # Collect corresponding position information
           assert seq.num_cached_tokens == pos0, f"num_cached_tokens={seq.num_cached_tokens} != pos0={pos0} (num_tokens={seq.num_tokens}, k={k})"
           context_lens.append(len(seq))  # Add context length
           
           # Iterate through each verification position
           for j in range(k + 1):
               pos = pos0 + j
               block_idx = pos // block_size  # Determine block index
               block_id = seq.block_table[block_idx]  # Determine block ID
               pos_in_block = pos % block_size  # Determine position within block
               slot_mapping.append(block_id * block_size + pos_in_block)  # Calculate final physical memory position
    
    pin_memory = device is not None and device.type == "cuda"
    
    input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=pin_memory).to(device, non_blocking=pin_memory)
    positions = torch.tensor(positions, dtype=torch.int64, pin_memory=pin_memory).to(device, non_blocking=pin_memory)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=pin_memory).to(device, non_blocking=pin_memory)
    context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=pin_memory).to(device, non_blocking=pin_memory)
    
    return input_ids, positions, slot_mapping, context_lens


def prepare_block_tables_from_seqs(seqs: list[Sequence], 
                                   is_draft: bool = False, 
                                   device: torch.device = None) -> torch.Tensor:
    """Convert sequence block tables to tensors.
    
    Args:
        seqs (list[Sequence]): List of sequences
        is_draft (bool): Whether this is for draft model
        device (torch.device): Device to use
        
    Returns:
        torch.Tensor: Block tables tensor
    """
    if is_draft:
        max_len = max(len(seq.draft_block_table) for seq in seqs)
        block_tables = [seq.draft_block_table + [-1] * (max_len - len(seq.draft_block_table)) for seq in seqs]  # Pad to maximum length
    
    else:
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
    
    # pin_memory is only effective for CUDA devices
    pin_memory = device is not None and device.type == "cuda"
    
    block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=pin_memory).to(device, non_blocking=pin_memory)
    
    return block_tables


def prepare_prefill_tensors_from_seqs(seqs: list[Sequence], block_size: int, is_draft: bool = False, 
                                      skip_first_token: bool = False, device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare tensors for prefill operations.
    
    Args:
        seqs (list[Sequence]): List of sequences
        block_size (int): Block size
        is_draft (bool): Whether this is for draft model
        skip_first_token (bool): Whether to skip the first token
        device (torch.device): Device to use
        
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
            Input IDs, positions, cumulative sequence lengths for queries, cumulative sequence lengths for keys,
            maximum query sequence length, maximum key sequence length, and slot mapping
    """
    input_ids = []
    positions = []
    cu_seqlens_q = [0]  # Cumulative lengths for query sequences
    cu_seqlens_k = [0]  # Cumulative lengths for key sequences
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    
    for seq in seqs:
        seqlen = len(seq)
        if is_draft:
            num_cached_tokens = seq.num_draft_cached_tokens
            block_table = seq.draft_block_table
        else:
            num_cached_tokens = seq.num_cached_tokens
            block_table = seq.block_table
        
        start = num_cached_tokens + (skip_first_token if is_draft else 0)  # Calculate start position
        input_ids.extend(seq[start:])  # Collect all tokens from start to current
        pos_offset = -skip_first_token if is_draft else 0  # Calculate position offset
        positions.extend(list(range(start + pos_offset, seqlen + pos_offset)))  # Collect corresponding position information
        seqlen_q = seqlen - start
        seqlen_k = seqlen + pos_offset
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(max_seqlen_q, seqlen_q)
        max_seqlen_k = max(max_seqlen_k, seqlen_k)
        
        # First prefill has no block table
        if not block_table:
            continue
        
        for pos in range(start + pos_offset, seq.num_tokens + pos_offset):
            block_idx = pos // block_size
            pos_in_block = pos % block_size
            slot_mapping.append(block_table[block_idx] * block_size + pos_in_block)
        
    pin_memory = device is not None and device.type == "cuda"
    
    # Convert to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=pin_memory).to(device, non_blocking=pin_memory)
    positions = torch.tensor(positions, dtype=torch.int64, pin_memory=pin_memory).to(device, non_blocking=pin_memory)
    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=pin_memory).to(device, non_blocking=pin_memory)
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=pin_memory).to(device, non_blocking=pin_memory)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=pin_memory).to(device, non_blocking=pin_memory)
   
    return input_ids, positions, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping
