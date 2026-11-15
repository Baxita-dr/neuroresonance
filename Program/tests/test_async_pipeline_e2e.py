"""
tests/test_async_pipeline_e2e.py — End-to-End Async Pipeline Integration Test
"""

import asyncio
import numpy as np
import pytest
from src.domain.models import SensorSample, ResonanceStateEnum
from src.core.engine import RobustCoherenceEngine


def test_end_to_end_async_coherence_computation():
    """
    Feed a full asynchronous window of phase-locked 0.1 Hz samples and verify
    end-to-end descriptive PLV without a state category.
    """
    async def _test():
        fs = 10.0
        n_samples = 1100  # 110 seconds at 10 Hz
        t = np.arange(n_samples, dtype=float) / fs
        sig_resp = np.sin(2 * np.pi * 0.10 * t)
        sig_hbo2 = np.sin(2 * np.pi * 0.10 * (t - 3.8))

        engine = RobustCoherenceEngine(buffer_size=1200, window_seconds=100.0, nominal_fs=fs)

        for i in range(n_samples):
            sample = SensorSample(
                timestamp=float(t[i]),
                channel_data=(float(sig_resp[i]), float(sig_hbo2[i])),
            )
            await engine.ingest(sample)

        metric = await engine.compute_coherence()

        assert metric.plv > 0.65
        assert metric.state is ResonanceStateEnum.DESCRIPTIVE
        assert metric.plv_percent > 90.0

    asyncio.run(_test())
