"""Sequence module for managing sequence data and state."""

from copy import copy
from enum import Enum, auto
from itertools import count

from haste.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """Sequence status enum."""
    WAITING = auto()  # Sequence is waiting to be processed
    RUNNING = auto()  # Sequence is currently being processed
    FINISHED = auto()  # Sequence has completed


class Sequence:
    """Sequence class for managing sequence data and state during inference.
    
    This class tracks all relevant information about a sequence being processed,
    including token IDs, block tables, and various metadata.
    """
    
    counter = count()  # Class-level counter for generating unique sequence IDs
    
    # List of attributes to include in serialization
    _ATTRIBUTES = [
        'seq_id', 'status', 'token_ids', 'last_token', 'num_tokens',
        'num_prompt_tokens', 'num_cached_tokens', 'block_table',
        'last_spec_step_accepted_len', 'draft_block_table',
        'num_draft_cached_tokens', 'temperature', 'draft_temperature', 'max_new_tokens',
        'ignore_eos', 'recovery_token_id', 'last_target_hidden_state',
        'extend_token_ids', 'extend_count',
    ]
    
    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """Initialize a new sequence.
        
        Args:
            token_ids (list[int]): List of token IDs for the sequence
            sampling_params (SamplingParams, optional): Sampling parameters. Defaults to SamplingParams().
        """
        self.block_size = 256  # Block size for KV cache management
        
        self.seq_id = next(Sequence.counter)  # Unique sequence ID
        self.status = SequenceStatus.WAITING  # Initial status
        self.token_ids = copy(token_ids)  # Copy token IDs to avoid modification
        self.last_token = token_ids[-1]  # Last token in the sequence
        self.num_tokens = len(self.token_ids)  # Total number of tokens
        self.num_prompt_tokens = len(token_ids)  # Number of prompt tokens
        self.num_cached_tokens = 0  # Number of cached tokens in KV cache
        self.block_table = []  # Block table for KV cache management
        self.last_spec_step_accepted_len = -1  # Length of accepted tokens in last speculation step
        
        self.draft_block_table = []  # Block table for draft model KV cache
        self.num_draft_cached_tokens = 0  # Number of cached tokens in draft model KV cache
        
        # Sampling parameters
        self.temperature = sampling_params.temperature  # Temperature for sampling
        self.draft_temperature = sampling_params.draft_temperature  # Temperature for draft model sampling
        self.max_new_tokens = sampling_params.max_new_tokens  # Maximum number of new tokens to generate
        self.ignore_eos = sampling_params.ignore_eos  # Whether to ignore EOS token
        
        self.recovery_token_id = None  # Recovery token ID for speculative decoding
        self.last_target_hidden_state = None  # Last hidden state from target model

        self.extend_token_ids = None  # Extended token IDs for recovery generation
        self.extend_count = 0  # Number of extended tokens


    def __len__(self):
        """Return the total number of tokens in the sequence.
        
        Returns:
            int: Total number of tokens
        """
        return self.num_tokens

    def __getitem__(self, key):
        """Get token IDs at the specified index or slice.
        
        Args:
            key (int or slice): Index or slice to access
            
        Returns:
            int or list[int]: Token ID(s) at the specified position(s)
        """
        return self.token_ids[key]

    @property
    def is_finished(self):
        """Check if the sequence has finished processing.
        
        Returns:
            bool: True if the sequence is finished, False otherwise
        """
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """Get the number of completion tokens generated.
        
        Returns:
            int: Number of completion tokens
        """
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """Get the prompt token IDs.
        
        Returns:
            list[int]: Prompt token IDs
        """
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """Get the completion token IDs.
        
        Returns:
            list[int]: Completion token IDs
        """
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """Get the number of cached blocks for the target model.
        
        Returns:
            int: Number of cached blocks
        """
        return (self.num_cached_tokens + self.block_size - 1) // self.block_size

    @property
    def num_blocks(self):
        """Get the total number of blocks for the sequence.
        
        Returns:
            int: Total number of blocks
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size
    
    @property
    def num_draft_cached_blocks(self):
        """Get the number of cached blocks for the draft model.
        
        Returns:
            int: Number of draft cached blocks
        """
        return (self.num_draft_cached_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """Get the number of tokens in the last block for the target model.
        
        Returns:
            int: Number of tokens in the last block
        """
        return self.num_tokens - (self.num_cached_blocks - 1) * self.block_size
    
    @property
    def last_block_num_tokens_draft(self):
        """Get the number of tokens in the last block for the draft model.
        
        Returns:
            int: Number of tokens in the last draft block
        """
        return self.num_tokens - (self.num_draft_cached_blocks - 1) * self.block_size

    def block(self, i):
        """Get the tokens in the specified block.
        
        Args:
            i (int): Block index
            
        Returns:
            list[int]: Tokens in the specified block
        """
        assert 0 <= i < self.num_blocks, f"block index {i} out of range, num_blocks={self.num_blocks}"
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        """Append a token to the end of the sequence.
        
        Args:
            token_id (int): Token ID to append
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
    
    def clone_spec(self):
        """Create a clone of the sequence for speculative decoding.
        
        Returns:
            Sequence: Cloned sequence with identical state
        """
        cloned = Sequence.__new__(Sequence)
        cloned.block_size = self.block_size
        for attr in self._ATTRIBUTES:
            value = getattr(self, attr, None)
            if isinstance(value, list):
                value = value.copy()
            setattr(cloned, attr, value)
        return cloned
    
    def __getstate__(self):
        """Get the state for serialization.
        
        Returns:
            dict: State dictionary
        """
        state = {}
        for attr in self._ATTRIBUTES:
            state[attr] = getattr(self, attr)
        return state
        

    def __setstate__(self, state):
        """Set the state from deserialization.
        
        Args:
            state (dict): State dictionary
        """
        for attr in self._ATTRIBUTES:
            setattr(self, attr, state.get(attr))
