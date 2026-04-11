"""Scheduler module for managing sequence scheduling and resource allocation."""

from collections import deque
import torch
from transformers import AutoTokenizer

from haste.config import Config
from haste.engine.sequence import Sequence, SequenceStatus
from haste.engine.block_manager import BlockManager


class Scheduler:
    """Scheduler class for managing sequence scheduling and resource allocation.
    
    This class handles the scheduling of sequences for prefill and decode operations,
    manages KV cache block allocation, and handles sequence lifecycle events.
    """
    
    def __init__(self, target_config: Config, draft_config: Config | None = None):
        """Initialize the scheduler with the specified configurations.
        
        Args:
            target_config (Config): Configuration for the target model
            draft_config (Config | None, optional): Configuration for the draft model. Defaults to None.
        """
        self.max_num_seqs = target_config.max_num_seqs  # Maximum number of sequences
        self.fan_out_list = target_config.fan_out_list  # Fan-out list
        self.fan_out_list_miss = target_config.fan_out_list_miss  # Fan-out list after cache miss
        if target_config.draft_async:
            self.MQ_LEN = sum(self.fan_out_list)
        self.max_num_batched_tokens = target_config.max_num_batched_tokens
        self.max_model_len = target_config.max_model_len
        self.eos = target_config.eos
        self.speculate = target_config.speculate
        self.F = target_config.async_fan_out
        self.K = target_config.speculate_k  # Speculation length
        self.block_size = target_config.kvcache_block_size  # KV cache block size
        self.verbose = target_config.verbose  # Whether to print verbose information
        self.draft_async = target_config.draft_async  # Whether to use async draft
        
        # Initialize KV cache block manager
        self.block_manager = BlockManager(
            target_config.num_kvcache_blocks, target_config.kvcache_block_size, 
            is_draft=False, verbose=self.verbose, max_model_len=self.max_model_len)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            target_config.model,
            use_fast=True,
            local_files_only=True,
        )
        
        if self.speculate:
            # Initialize draft KV cache block manager
            self.draft_block_manager = BlockManager(
                draft_config.num_kvcache_blocks, draft_config.kvcache_block_size, is_draft=True, 
                speculate_k=self.K, verbose=self.verbose, max_model_len=self.max_model_len)
            
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """Check if all sequences have been processed.
        
        Returns:
            bool: True if there are no waiting or running sequences, False otherwise
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """Add a sequence to the waiting queue.
        
        Args:
            seq (Sequence): Sequence to add
        """
        self.waiting.append(seq)
    
    def bms_can_append(self, seq: Sequence, target_lookahead_len: int, draft_lookahead_len: int | None = None) -> bool:
        """Check if blocks can be appended for the sequence.
        
        Args:
            seq (Sequence): Sequence to check
            target_lookahead_len (int): Lookahead length for target model
            draft_lookahead_len (int | None, optional): Lookahead length for draft model. Defaults to None.
            
        Returns:
            bool: True if blocks can be appended, False otherwise
        """
        target_can_append = self.block_manager.can_append(seq, target_lookahead_len)
        if self.speculate:
            draft_can_append = self.draft_block_manager.can_append(seq, draft_lookahead_len)
        else:
            assert draft_lookahead_len is None, f"ERROR in bms_can_append: draft_lookahead_len should be None if not speculate, but got {draft_lookahead_len}"
            draft_can_append = True

        return target_can_append and draft_can_append
        
    def bms_can_allocate(self, seq: Sequence) -> bool:
        """Check if blocks can be allocated for the sequence.
        
        Args:
            seq (Sequence): Sequence to check
            
        Returns:
            bool: True if blocks can be allocated, False otherwise
        """
        return self.block_manager.can_allocate(seq) and \
            (not self.speculate or self.draft_block_manager.can_allocate(seq))


    def schedule(self) -> tuple[list[Sequence], bool]:
        """Schedule sequences for execution.
        
        Returns:
            tuple[list[Sequence], bool]: List of scheduled sequences and whether it's a prefill operation
        """
        # Prefill phase
        scheduled_seqs = []
        num_batched_tokens = 0

        while self.waiting:
            seq = self.waiting[0]  # Get the first sequence in the waiting queue
            
            # Calculate remaining tokens
            remain = len(seq) - seq.num_cached_tokens
            
            if num_batched_tokens + remain > self.max_num_batched_tokens or not self.bms_can_allocate(seq):
                break
        
            self.block_manager.allocate(seq)
            if self.speculate:
                self.draft_block_manager.allocate(seq)
            
            num_batched_tokens += remain
            
            seq.status = SequenceStatus.RUNNING  # Mark sequence as running
            self.waiting.popleft()  # Remove from waiting queue
            self.running.append(seq)  # Add to running queue
            scheduled_seqs.append(seq)  # Add to scheduled sequences
        
        if scheduled_seqs:
            if __debug__: print(f'[scheduler] returning {len(scheduled_seqs)} sequences for prefill', flush=True)
            return scheduled_seqs, True

        num_seqs_decoded = 0
        sync_spec = self.speculate and not self.draft_async
        async_spec = self.speculate and self.draft_async
        
        if async_spec:
            target_lookhead_len = self.K + 1
            draft_lookhead_len = self.K + 1
        
        elif sync_spec:
            target_lookhead_len = self.K + 1
            draft_lookhead_len = self.K + 1
        else:
            target_lookhead_len = 1
            draft_lookhead_len = None
        
        # Decode scheduling loop
        while self.running and num_seqs_decoded < self.max_num_seqs:
            seq = self.running.popleft()
            
            # Preempt other sequences if resources are insufficient
            while not self.bms_can_append(seq, target_lookhead_len, draft_lookhead_len):
                if self.running:
                    preempted_seq = self.running.pop()
                    self.preempt(preempted_seq)
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs_decoded += 1
                self.block_manager.may_append(seq, target_lookhead_len)
                if self.speculate:
                    self.draft_block_manager.may_append(seq, draft_lookhead_len)
                scheduled_seqs.append(seq)
        
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False
            
    def preempt(self, seq: Sequence):
        """Preempt a sequence and return it to the waiting queue.
        
        Args:
            seq (Sequence): Sequence to preempt
        """
        seq.status = SequenceStatus.WAITING  # Change status
        seq.recovery_token_id = None
        self.block_manager.deallocate(seq)  # Release memory
        if self.speculate:
            self.draft_block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        
        # Update sequence state
        seq.num_prompt_tokens = seq.num_tokens
        seq.last_spec_step_accepted_len = -1
        seq.extend_count = 0
        seq.extend_token_ids = None       

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """Postprocess sequences after non-speculative decoding.
        
        Args:
            seqs (list[Sequence]): List of sequences to postprocess
            token_ids (list[int]): Generated token IDs
            is_prefill (bool): Whether this is a prefill operation
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)  # Add generated token to sequence
            if is_prefill:
                seq.num_cached_tokens = seq.num_prompt_tokens
            else:
                seq.num_cached_tokens += 1
                
            # Check if sequence is finished
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_new_tokens:
                seq.status = SequenceStatus.FINISHED  # Mark as finished
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
            
            else:
                block_table = seq.block_table
                last_block = self.block_manager.blocks[block_table[-1]]
                
                if seq.last_block_num_tokens == self.block_size:
                    tokens_ids = seq.block(seq.num_blocks - 1)
                    prefix = self.block_manager.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                    h = self.block_manager.compute_hash(tokens_ids, prefix)
                    
                    # Update hash for the last block
                    last_block.update(h, tokens_ids)
                    self.block_manager.hash_to_block_id[h] = last_block.block_id
                
    def _handle_eos_and_max_new_tokens(self, seq: Sequence, new_suffix: list[int]) -> tuple[list[int], bool]:
        """Handle EOS and max new tokens constraints.
        
        Args:
            seq (Sequence): Sequence to process
            new_suffix (list[int]): New tokens to add
            
        Returns:
            tuple[list[int], bool]: Processed suffix and whether the sequence is finished
        """
        finished = False
        
        # Truncate at EOS if not ignoring EOS
        if not seq.ignore_eos and self.eos in new_suffix:
            new_suffix = new_suffix[:new_suffix.index(self.eos) + 1]
        
        # Limit max new tokens
        if seq.num_completion_tokens + len(new_suffix) >= seq.max_new_tokens:
            new_suffix = new_suffix[:seq.max_new_tokens - seq.num_completion_tokens]
        
        # Limit max model length
        if seq.num_tokens + len(new_suffix) >= self.max_model_len:
            max_allowed_suffix_len = self.max_model_len - seq.num_tokens
            new_suffix = new_suffix[:max(0, max_allowed_suffix_len)]
        
        new_suffix_len = len(new_suffix)
        
        # Check if sequence is finished
        if ((not seq.ignore_eos and self.eos in new_suffix) or
                seq.num_completion_tokens + new_suffix_len == seq.max_new_tokens or
                seq.num_tokens + new_suffix_len >= self.max_model_len):  
            finished = True
            
        
        assert seq.num_completion_tokens <= seq.max_new_tokens,\
                f"seq.num_completion_tokens = {seq.num_completion_tokens} and seq.max_new_tokens = {seq.max_new_tokens}"

        return new_suffix, finished
    
    def _update_kv_caches(self, seq: Sequence, new_suffix: list[int]):
        """Update KV caches for the sequence.
        
        Args:
            seq (Sequence): Sequence to update
            new_suffix (list[int]): New tokens added to the sequence
        """
        # Calculate required blocks for the new sequence length
        required_blocks = (seq.num_tokens + len(new_suffix) + self.block_size - 1) // self.block_size
        
        spec_blocks_target = len(seq.block_table)  # Target model block count
        spec_blocks_draft = len(seq.draft_block_table)  # Draft model block count
        
        # Check for over-allocation
        spec_crossed_target = spec_blocks_target > required_blocks
        spec_crossed_draft = spec_blocks_draft > required_blocks
        
        # Release over-allocated blocks for target model
        if spec_crossed_target:
            excess_blocks = spec_blocks_target - required_blocks
            blocks_to_deallocate = seq.block_table[-excess_blocks:]
            
            for block_id in blocks_to_deallocate:
                block = self.block_manager.blocks[block_id]
                block.ref_count -= 1
                if block.ref_count == 0:
                    self.block_manager._deallocate_block(block_id)
            
            seq.block_table = seq.block_table[:-excess_blocks]
        
        # Release over-allocated blocks for draft model
        if spec_crossed_draft:
            excess_blocks = spec_blocks_draft - required_blocks
            blocks_to_deallocate = seq.draft_block_table[-excess_blocks:]
            
            for block_id in blocks_to_deallocate:
                block = self.draft_block_manager.blocks[block_id]
                block.ref_count -= 1
                if block.ref_count == 0:
                    self.draft_block_manager._deallocate_block(block_id)
            
            seq.draft_block_table = seq.draft_block_table[:-excess_blocks]

    def _finalize_blocks(self, block_manager: BlockManager, seq: Sequence, block_table: list[int], block_index: int):
        """Finalize blocks by computing and updating hashes.
        
        Args:
            block_manager (BlockManager): Block manager to use
            seq (Sequence): Sequence containing the block
            block_table (list[int]): Block table for the sequence
            block_index (int): Index of the block to finalize
        """
        token_ids = seq.block(block_index)
        prefix = block_manager.blocks[block_table[-2]].hash if len(block_table) > 1 else -1  # Hash of previous block
        h = block_manager.compute_hash(token_ids, prefix)
        last_block = block_manager.blocks[block_table[-1]]
        last_block.update(h, token_ids)  # Update hash for the last block
        block_manager.hash_to_block_id[h] = last_block.block_id  # Update hash to block ID mapping
        
    def _update_sequence_metadata(self, seq: Sequence, new_suffix: list[int], recovery_token: int):
        """Update sequence metadata after speculative decoding.
        
        Args:
            seq (Sequence): Sequence to update
            new_suffix (list[int]): New tokens added to the sequence
            recovery_token (int): Recovery token for next speculation
        """
        new_suffix_len = len(new_suffix)
        assert new_suffix_len >= 1, "ERROR in _update_sequence_metadata: new_suffix_len = 0, should be non-empty"
    
        seq.token_ids.extend(new_suffix)  # Add new suffix
        seq.num_tokens += new_suffix_len  # Update count
        seq.last_token = new_suffix[-1]  # Update last token
        seq.num_cached_tokens += new_suffix_len  # Update target cache length
        seq.num_draft_cached_tokens += new_suffix_len  # Update draft cache length
        
        seq.last_spec_step_accepted_len = new_suffix_len  # Record accepted tokens for this speculation step
        seq.recovery_token_id = recovery_token  # Record recovery token for next speculation
        
        assert seq.last_block_num_tokens == seq.last_block_num_tokens_draft, f"ERROR in _update_sequence_metadata: seq.last_block_num_tokens = {seq.last_block_num_tokens} and seq.last_draft_block_num_tokens = {seq.last_block_num_tokens_draft}"
        assert seq.block_table, "ERROR in _update_sequence_metadata: seq.block_table is empty"
        assert seq.draft_block_table, "ERROR in _update_sequence_metadata: seq.draft_block_table is empty"
        
        new_total = seq.num_tokens  # New total token count
        for block_index in range(len(seq.block_table)):
            if (block_index + 1) * self.block_size > new_total:
                continue

            target_block = self.block_manager.blocks[seq.block_table[block_index]]
            if target_block.hash == -1:
                self._finalize_blocks(self.block_manager, seq, seq.block_table, block_index)

            draft_block = self.draft_block_manager.blocks[seq.draft_block_table[block_index]]
            if draft_block.hash == -1:
                self._finalize_blocks(self.draft_block_manager, seq, seq.draft_block_table, block_index)
        
    def postprocess_speculate(self, seqs: list[Sequence], new_suffixes: list[list[int]], next_recovery_tokens: list[int]):
        """Postprocess sequences after speculative decoding.
        
        Args:
            seqs (list[Sequence]): List of sequences to postprocess
            new_suffixes (list[list[int]]): New tokens for each sequence
            next_recovery_tokens (list[int]): Recovery tokens for next speculation
        """
        for i, (seq, new_suffix, next_recovery_token) in enumerate(zip(seqs, new_suffixes, next_recovery_tokens)):
            new_suffix, finished = self._handle_eos_and_max_new_tokens(seq, new_suffix)  # Handle EOS and length limits
            
            self._update_kv_caches(seq, new_suffix)  # Update KV caches
            self._update_sequence_metadata(seq, new_suffix, next_recovery_token)  # Update sequence metadata
            
            if finished:
                if __debug__: print(f'Sequence {seq.seq_id} finished, deallocating and marking as done + removing from running', flush=True)
                seq.status = SequenceStatus.FINISHED
                
                # Release cache blocks
                self.block_manager.deallocate(seq)
                self.draft_block_manager.deallocate(seq)
                self.running.remove(seq)  # Remove from running queue
        