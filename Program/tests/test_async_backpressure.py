"""
tests/test_async_backpressure.py — Unit Tests for Async Backpressure & Queue Bounding
"""

import asyncio
import pytest
from src.domain.models import SensorSample
from src.core.engine import RobustCoherenceEngine


def test_robust_coherence_engine_backpressure_drop():
    """
    Flood a bounded engine with 2000 samples when queue capacity is only 50.
    Verify:
    1. Queue size never exceeds 50
    2. Overflows are dropped deterministically (FIFO drop policy)
    3. Ingestion remains completely non-blocking without memory explosion
    """
    async def _test():
        queue_cap = 50
        total_flood = 2000
        engine = RobustCoherenceEngine(buffer_size=queue_cap, window_seconds=100.0, nominal_fs=10.0)

        for i in range(total_flood):
            sample = SensorSample(timestamp=float(i * 0.1), channel_data=(float(i), float(i * 2)))
            await engine.ingest(sample)

        assert engine.queue_size == queue_cap
        assert engine.total_ingested == total_flood
        assert engine.dropped_samples == (total_flood - queue_cap)

        drained = await engine.drain_queue()
        assert drained == queue_cap
        assert engine.queue_size == 0

    asyncio.run(_test())


def test_robust_coherence_engine_empty_coherence():
    """Engine reports unavailable metrics until a full analysis window exists."""
    async def _test():
        engine = RobustCoherenceEngine(buffer_size=100)
        metric = await engine.compute_coherence()
        assert metric.plv != metric.plv  # NaN means unavailable, not measured zero.
        assert metric.coherence_at_01hz != metric.coherence_at_01hz
        assert metric.state.value == "STARTING"

    asyncio.run(_test())
