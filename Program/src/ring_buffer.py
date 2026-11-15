"""
ring_buffer.py — High-Performance Zero-Allocation Circular Ring Buffer

Designed for real-time DSP pipelines where memory allocations and np.roll()
cause latency spikes, GC pauses, and cache invalidation.
"""

import threading
import numpy as np
from typing import Tuple, Optional


class CircularRingBuffer:
    """
    Thread-safe, pre-allocated circular ring buffer for real-time streaming data.

    Characteristics:
    - O(1) push operations (single sample or vectorized chunks)
    - Zero memory reallocations during streaming
    - Monotonically increasing sample counter for packet drop / overflow detection
    - Contiguous memory output on window reads
    """

    def __init__(self, capacity: int, dtype: np.dtype = np.float64):
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.dtype = dtype
        self._buffer = np.zeros(self.capacity, dtype=self.dtype)
        self._head = 0          # Index of next write
        self._total_written = 0  # Monotonic counter
        self._lock = threading.Lock()

    def push(self, value: float) -> None:
        """Push a single scalar sample into the ring buffer in O(1) time."""
        with self._lock:
            self._buffer[self._head] = value
            self._head = (self._head + 1) % self.capacity
            self._total_written += 1

    def push_chunk(self, chunk: np.ndarray) -> None:
        """
        Push multiple samples into the ring buffer efficiently.
        Handles wrap-around with at most 2 slice assignments.
        """
        arr = np.asarray(chunk, dtype=self.dtype)
        n = len(arr)
        if n == 0:
            return

        with self._lock:
            if n >= self.capacity:
                # If chunk is larger than capacity, only keep the most recent elements
                self._buffer[:] = arr[-self.capacity:]
                self._head = 0
                self._total_written += n
                return

            end = self._head + n
            if end <= self.capacity:
                self._buffer[self._head:end] = arr
            else:
                first_part = self.capacity - self._head
                self._buffer[self._head:] = arr[:first_part]
                self._buffer[:n - first_part] = arr[first_part:]

            self._head = (self._head + n) % self.capacity
            self._total_written += n

    def get_latest(self, n_samples: Optional[int] = None) -> np.ndarray:
        """
        Extract the most recent n_samples in chronological order.
        Returns a contiguous 1D numpy array.
        """
        with self._lock:
            available = min(self._total_written, self.capacity)
            if n_samples is None or n_samples > available:
                n = available
            else:
                n = max(0, int(n_samples))

            if n == 0:
                return np.empty(0, dtype=self.dtype)

            start = (self._head - n) % self.capacity
            if start + n <= self.capacity:
                return self._buffer[start:start + n].copy()
            else:
                part1 = self._buffer[start:]
                part2 = self._buffer[:(start + n) % self.capacity]
                return np.concatenate((part1, part2))

    def get_all(self) -> np.ndarray:
        """Return all valid buffered samples up to capacity in chronological order."""
        return self.get_latest(self.capacity)

    @property
    def total_written(self) -> int:
        """Monotonic counter of total samples received since inception."""
        return self._total_written

    @property
    def is_full(self) -> bool:
        """True if the buffer has accumulated at least `capacity` samples."""
        return self._total_written >= self.capacity

    @property
    def size(self) -> int:
        """Current number of active samples stored."""
        return min(self._total_written, self.capacity)

    def reset(self) -> None:
        """Clear the buffer state without reallocating underlying memory."""
        with self._lock:
            self._buffer.fill(0)
            self._head = 0
            self._total_written = 0
