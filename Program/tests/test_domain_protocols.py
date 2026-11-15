"""
tests/test_domain_protocols.py — Structural Typing & Domain Integrity Tests
"""

import pytest
from dataclasses import FrozenInstanceError
from src.domain.models import SensorSample, ResonanceStateEnum
from src.domain.protocols import StreamProcessor, CoherenceEstimatorProtocol
from src.infrastructure.async_lsl import AsyncLSLStreamProcessor
from src.core.engine import RobustCoherenceEngine


def test_sensor_sample_immutability():
    """Verify SensorSample cannot be mutated during concurrent streaming pipeline."""
    sample = SensorSample(timestamp=123.456, channel_data=(1.0, 2.0, -0.5))

    with pytest.raises(FrozenInstanceError):
        sample.timestamp = 999.999  # Attempting mutation must raise FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        sample.channel_data = (0.0,)


def test_stream_processor_protocol_conformance():
    """Verify AsyncLSLStreamProcessor strictly implements StreamProcessor Protocol."""
    processor = AsyncLSLStreamProcessor()
    assert isinstance(processor, StreamProcessor)


def test_coherence_engine_protocol_conformance():
    """Verify RobustCoherenceEngine strictly implements CoherenceEstimatorProtocol."""
    engine = RobustCoherenceEngine()
    assert isinstance(engine, CoherenceEstimatorProtocol)
