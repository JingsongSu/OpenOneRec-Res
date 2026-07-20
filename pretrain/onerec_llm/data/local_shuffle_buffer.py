"""
Local shuffle buffer for data randomization during iteration.

This module provides a fixed-size buffer that randomizes the order of data samples
using hash-based indexing with SortedDict for efficient random access.
"""

import hashlib
import logging
import threading
import traceback
from collections import defaultdict

from sortedcontainers import SortedDict

logger = logging.getLogger(__name__)


class LocalShuffleBuffer:
    """
    A buffer class to implement local data shuffling.
    
    Maintains a fixed-size buffer to randomize the order of data samples during iteration.
    Uses hash-based indexing with SortedDict for efficient random access.
    
    Attributes:
        buffer_size: Maximum capacity of the buffer
        random_fetch: Probability to randomly fetch a sample before buffer is full
        buffer: SortedDict storing samples (key: hash, value: sample)
        count: Statistics counter (adds, conflicts, buffer_epoch)
        buffer_multiply: Large multiplier to avoid hash collisions across epochs
        lock: Thread lock for thread-safe operations
    """
    
    def __init__(self, buffer_size: int = 2048, random_fetch: float = 0.01) -> None:
        """
        Initialize the LocalShuffleBuffer.
        
        Args:
            buffer_size: Maximum capacity of the buffer (default: 2048)
            random_fetch: Probability to randomly fetch a sample before buffer is full (0.0-1.0, default: 0.01)
        """
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")
        if not 0.0 <= random_fetch <= 1.0:
            raise ValueError(f"random_fetch must be between 0.0 and 1.0, got {random_fetch}")
        
        self.buffer_size = buffer_size
        self.random_fetch = random_fetch
        self.buffer = SortedDict()  # key: hash, value: sample
        self.count = defaultdict(int)
        self.count["buffer_epoch"] = 0
        # Large multiplier (0xffffffffffffffff) to avoid hash collisions across epochs
        self.buffer_multiply = int('f' * 16, 16)
        self.lock = threading.Lock()

    def _calc_sample_hash(self, obj: dict, buffer_epoch: int = None) -> int:
        """
        Calculate a unique hash for a sample to use as buffer key.
        
        Maps sample identifier to integer with random-like distribution using MD5 hash.
        Adds epoch-based offset to prevent cross-epoch hash collisions.
        
        Args:
            obj: Sample object containing "uuid" and "source" keys
            buffer_epoch: Optional epoch index. If None, uses current buffer_epoch
            
        Returns:
            Integer hash value
        """
        if buffer_epoch is None:
            buffer_epoch = self.count["buffer_epoch"]
        
        # Create unique string from sample identifiers
        unique_str = f"{obj['uuid']}{obj['source']}@ep{buffer_epoch}"
        
        # Generate MD5 hash and convert to integer (use first 16 hex chars = 64 bits)
        hash_obj = hashlib.md5(unique_str.encode('utf-8'))
        hex_str = hash_obj.hexdigest()[:16]
        base_hash = int(hex_str, 16)
        
        # Add epoch-based offset to prevent cross-epoch collisions
        return base_hash + self.buffer_multiply * buffer_epoch

    def add(self, obj: dict, fn: str = None, epoch: int = None) -> bool:
        """
        Add a sample to the buffer.
        
        Args:
            obj: Sample object to add to buffer (must contain "uuid" and "source" keys)
            fn: Optional filename/identifier for logging
            epoch: Optional epoch index
            
        Returns:
            True if sample was added and buffer isn't ready for extraction,
            False if extraction should occur (buffer full or random fetch triggered)
        """
        try:
            # Calculate hash for the sample
            obj_hash = self._calc_sample_hash(obj, buffer_epoch=epoch)
            self.count["add"] += 1
            
            # Update buffer epoch every buffer_size additions
            if self.count["add"] % self.buffer_size == 0:
                self.count["buffer_epoch"] += 1

            # Handle hash collisions (duplicate unique identifiers)
            if obj_hash in self.buffer:
                self.count["conflict"] += 1
                # Log warning periodically for collision rate
                if self.count["conflict"] % 100 == 0:
                    conflict_rate = self.count["conflict"] / self.count["add"]
                    logger.warning(
                        f"{'=' * 30}\n"
                        f"Potential duplicate samples with same uuid/source! "
                        f"uuid={obj['uuid']}, source={obj['source']}, fn={fn}, "
                        f"conflict_rate={conflict_rate:.4f}, add_count={self.count['add']}\n"
                        f"{'=' * 30}"
                    )
            
            with self.lock:
                self.buffer[obj_hash] = obj

            # Random fetch trigger: small probability to extract before buffer is full
            # This prevents downstream timeout errors
            if (obj_hash % 10000) < int(10000 * self.random_fetch):
                return False  # Trigger extraction
            
            # Check if buffer has reached capacity
            return len(self.buffer) < self.buffer_size
                
        except Exception as e:
            logger.error(f"Error in LocalShuffleBuffer.add(): {traceback.format_exc()}")
            raise

    def get(self) -> dict:
        """
        Extract a sample from the buffer.
        
        Returns:
            A sample object from the buffer
            
        Raises:
            ValueError: If buffer is empty
        """
        if len(self.buffer) == 0:
            raise ValueError("Cannot get sample from empty buffer")

        with self.lock:
            # Pop first item from SortedDict (provides random-like access due to hashing)
            # popitem(0) removes the first (smallest) key-value pair
            return self.buffer.popitem(0)[1]

    def __len__(self) -> int:
        """Return current number of samples in the buffer."""
        return len(self.buffer)
