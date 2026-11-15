"""
tests/test_ring_buffer.py — Unit Tests for Circular Ring Buffer
"""

import threading
import numpy as np
import pytest
from src.ring_buffer import CircularRingBuffer


def test_ring_buffer_basic_push_and_retrieval():
    buf = CircularRingBuffer(capacity=5)
    assert buf.size == 0
    assert not buf.is_full

    for i in range(3):
        buf.push(float(i))

    assert buf.size == 3
    assert not buf.is_full
    np.testing.assert_array_equal(buf.get_latest(3), np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(buf.get_latest(2), np.array([1.0, 2.0]))


def test_ring_buffer_wrap_around_ordering():
    buf = CircularRingBuffer(capacity=4)
    # Push 6 items: 0, 1, 2, 3, 4, 5
    for i in range(6):
        buf.push(float(i))

    assert buf.is_full
    assert buf.total_written == 6
    # Oldest 2 items dropped; should contain [2, 3, 4, 5]
    np.testing.assert_array_equal(buf.get_all(), np.array([2.0, 3.0, 4.0, 5.0]))
    np.testing.assert_array_equal(buf.get_latest(2), np.array([4.0, 5.0]))


def test_ring_buffer_push_chunk():
    buf = CircularRingBuffer(capacity=6)
    buf.push_chunk(np.array([10.0, 20.0, 30.0]))
    assert buf.size == 3
    np.testing.assert_array_equal(buf.get_latest(3), np.array([10.0, 20.0, 30.0]))

    # Push chunk causing wrap-around
    buf.push_chunk(np.array([40.0, 50.0, 60.0, 70.0]))
    assert buf.is_full
    assert buf.total_written == 7
    # Capacity is 6, should contain [20, 30, 40, 50, 60, 70]
    np.testing.assert_array_equal(buf.get_all(), np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0]))


def test_ring_buffer_thread_safety():
    buf = CircularRingBuffer(capacity=1000)
    n_threads = 4
    n_pushes_per_thread = 250

    def worker(worker_id):
        for i in range(n_pushes_per_thread):
            buf.push(float(worker_id * 1000 + i))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert buf.total_written == n_threads * n_pushes_per_thread
    assert buf.size == 1000
    res = buf.get_all()
    assert len(res) == 1000
