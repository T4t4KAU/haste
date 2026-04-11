"""Block manager for KV cache memory management."""

from collections import deque
import xxhash
import numpy as np

from haste.engine.sequence import Sequence


class Block:
    """Block class for KV cache."""

    def __init__(self, block_id):
        """Initialize block.
        
        Args:
            block_id (int): Block ID
        """
        self.block_id = block_id
        self.ref_count = 0  # Reference count
        self.hash = -1  # Block hash value
        self.token_ids = []  # Token IDs stored in the block

    def update(self, hash: int, token_ids: list[int]):
        """Update block with hash and token IDs.
        
        Args:
            hash (int): Block hash value
            token_ids (list[int]): Token IDs to store
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """Reset block to initial state."""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """Block manager for KV cache memory management."""

    def __init__(
        self, 
        num_blocks: int, 
        block_size: int, 
        is_draft: bool = False,
        speculate_k: int = -1, 
        max_model_len: int = -1, 
        verbose: bool = False
    ):
        """Initialize block manager.
        
        Args:
            num_blocks (int): Number of blocks
            block_size (int): Block size
            is_draft (bool): Whether this is for draft model
            speculate_k (int): Number of tokens to speculate
            max_model_len (int): Maximum model sequence length
            verbose (bool): Whether to enable verbose logging
        """
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()  # Map from hash to block ID
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # Free block IDs
        self.used_block_ids: set[int] = set()  # Used block IDs
        self.is_draft: bool = is_draft  # Whether this is for draft model
        self.speculate_k: int = speculate_k  # Number of tokens to speculate
        self.verbose: bool = verbose  # Whether to enable verbose logging
        self.max_model_len: int = max_model_len  # Maximum model sequence length

    # Compute hash value
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """Compute hash value for token IDs.
        
        Args:
            token_ids (list[int]): Token IDs
            prefix (int): Prefix hash value
            
        Returns:
            int: Hash value
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # Allocate memory block
    def _allocate_block(self, block_id: int) -> Block:
        """Allocate memory block.
        
        Args:
            block_id (int): Block ID
            
        Returns:
            Block: Allocated block
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]
    
    # Allocate n memory blocks
    def _allocate_n_blocks(self, n: int) -> list[Block]:
        """Allocate n memory blocks.
        
        Args:
            n (int): Number of blocks to allocate
            
        Returns:
            list[Block]: List of allocated blocks
        """
        if len(self.free_block_ids) < n:
            raise RuntimeError(f"Insufficient free blocks: need {n}, have {len(self.free_block_ids)}")
        
        # Extract n block IDs in one operation
        block_ids = [self.free_block_ids.popleft() for _ in range(n)]

        # Reset all blocks and update tracking sets
        blocks = []
        for block_id in block_ids:
            block = self.blocks[block_id]
            assert block.ref_count == 0
            block.reset()
            self.used_block_ids.add(block_id)
            blocks.append(block)
        
        return blocks

    # Deallocate memory block
    def _deallocate_block(self, block_id: int) -> Block:
        """Deallocate memory block.
        
        Args:
            block_id (int): Block ID
            
        Returns:
            Block: Deallocated block
        """
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)  # Remove from used block set
        self.free_block_ids.append(block_id)  # Add to free block queue

    # Deallocate n memory blocks
    def _deallocate_n_blocks(self, block_ids: list[int]):  
        """Deallocate n memory blocks.
        
        Args:
            block_ids (list[int]): List of block IDs to deallocate
        """
        for block_id in block_ids:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0: 
                self._deallocate_block(block_id)

    # Check if blocks can be allocated
    def can_allocate(self, seq: Sequence) -> bool:
        """Check if blocks can be allocated for a sequence.
        
        Args:
            seq (Sequence): Sequence
            
        Returns:
            bool: Whether blocks can be allocated
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """Allocate blocks for a sequence.
        
        Args:
            seq (Sequence): Sequence
        """
        block_table = seq.draft_block_table if self.is_draft else seq.block_table
        assert not block_table
        h = -1
        cache_miss = False
        
        # Iterate through each block in the sequence
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)  # Get token_ids for current block
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            
            # Check if block already exists (prefix sharing)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:  # cache hit
                if self.is_draft:  # Cache draft sequence
                    seq.num_draft_cached_tokens += self.block_size
                else:  # Cache regular sequence
                    seq.num_cached_tokens += self.block_size
                    
                # Update block reference count for already allocated blocks
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
                
            block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """Deallocate blocks for a sequence.
        
        Args:
            seq (Sequence): Sequence
        """
        # Get block table for the sequence
        block_table = seq.draft_block_table if self.is_draft else seq.block_table
        
        # Iterate from the end to handle dependencies
        for block_id in reversed(block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        if self.is_draft:
            seq.num_draft_cached_tokens = 0
        else:
            seq.num_cached_tokens = 0
        block_table.clear()  # Clear data

    # Check if blocks can be appended
    def can_append(self, seq: Sequence, lookahead_num_tokens: int = 1) -> bool:
        """Check if blocks can be appended for a sequence.
        
        Args:
            seq (Sequence): Sequence
            lookahead_num_tokens (int): Number of lookahead tokens
            
        Returns:
            bool: Whether blocks can be appended
        """
        block_table = seq.draft_block_table if self.is_draft else seq.block_table

        # Check if maximum model length would be exceeded
        if seq.num_tokens + lookahead_num_tokens > self.max_model_len:
            print(f'[block_manager] WARNING: Sequence length + lookahead would exceed max model length', flush=True)
            return False

        # Calculate required number of blocks
        # ceil((current sequence length + lookahead length) / block size)
        target_blocks = (seq.num_tokens + lookahead_num_tokens +
                         self.block_size - 1) // self.block_size
        current_blocks = len(block_table)

        if target_blocks > current_blocks:
            needed = target_blocks - current_blocks
            return len(self.free_block_ids) >= needed
        else:
            return True  # Current blocks are sufficient

    # Append memory blocks
    # If current blocks are insufficient, allocate new blocks
    # If current blocks are sufficient, use existing blocks
    def may_append(self, seq: Sequence, lookahead_num_tokens: int = 1):
        """Append memory blocks for a sequence if needed.
        
        Args:
            seq (Sequence): Sequence
            lookahead_num_tokens (int): Number of lookahead tokens
        """
        block_table = seq.draft_block_table if self.is_draft else seq.block_table

        # Calculate required number of blocks
        # ceil((current sequence length + lookahead length) / block size)
        target_blocks = (seq.num_tokens + lookahead_num_tokens +
                         self.block_size - 1) // self.block_size
        current_blocks = len(block_table)

        if target_blocks > current_blocks:
            needed = target_blocks - current_blocks
            new_blocks = self._allocate_n_blocks(needed)
            for block in new_blocks:
                block_table.append(block.block_id)
