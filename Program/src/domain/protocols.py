"""
src/domain/protocols.py — Domain Protocols & Structural Interfaces

Defines the dependency inversion boundaries for biometric streaming,
phase-locking analysis, and hardware drivers.
"""

from typing import AsyncGenerator, Protocol, runtime_checkable
import numpy as np
from src.domain.models import SensorSample, CoherenceMetric, NVCValidationResult


@runtime_checkable
class StreamProcessor(Protocol):
    """
    Protocol for asynchronous hardware and biometric stream providers.
    Decouples LSL, OpenCV Webcams, and synthetic simulators.
    """
    async def process_stream(self) -> AsyncGenerator[SensorSample, None]:
        """Yield sensor samples asynchronously with backpressure."""
        ...


@runtime_checkable
class CoherenceEstimatorProtocol(Protocol):
    """
    Protocol for phase-locking and resonance computation engines.
    """
    async def ingest(self, sample: SensorSample) -> None:
        """Ingest a sensor sample with bounded backpressure queueing."""
        ...

    async def compute_coherence(self) -> CoherenceMetric:
        """Compute the current Phase-Locking Value (PLV) across buffered epochs."""
        ...


@runtime_checkable
class NVCValidatorProtocol(Protocol):
    """
    Deprecated compatibility protocol. HbO₂/HbR correlation alone cannot
    validate neurovascular coupling; this interface is not used by the app.
    """
    def validate(self, hbo2_signal: np.ndarray, hbr_signal: np.ndarray) -> NVCValidationResult:
        """Legacy signature; do not interpret correlation as NVC validation."""
        ...
