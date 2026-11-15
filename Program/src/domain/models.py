"""
src/domain/models.py — Pure Domain Value Objects & Data Contracts

Immutable domain models decoupled from hardware drivers, UI frameworks,
and storage.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class ResonanceStateEnum(str, Enum):
    """Descriptive/lifecycle status; deprecated category members are never emitted."""
    HIGH_PHASE_CONSISTENCY = "HIGH_PHASE_CONSISTENCY"  # Deprecated; do not interpret as a state.
    INTERMEDIATE_PHASE_CONSISTENCY = "INTERMEDIATE_PHASE_CONSISTENCY"  # Deprecated.
    LOW_PHASE_CONSISTENCY = "LOW_PHASE_CONSISTENCY"  # Deprecated.
    DESCRIPTIVE = "DESCRIPTIVE_PLV_NO_STATE_CLASSIFICATION"
    RESONANCE = "HIGH_PHASE_CONSISTENCY"  # Deprecated compatibility alias
    PARTIAL = "INTERMEDIATE_PHASE_CONSISTENCY"  # Deprecated compatibility alias
    DYSREGULATED = "LOW_PHASE_CONSISTENCY"  # Deprecated compatibility alias
    STARTING = "STARTING"
    IDLE = "IDLE"


@dataclass(frozen=True)
class SensorSample:
    """
    Immutable biometric sample contract.

    Attributes:
        timestamp: Monotonic or wall-clock epoch timestamp (seconds).
        channel_data: Tuple of float amplitudes (e.g. (respiration, HbO2, HbR)).
    """
    timestamp: float
    channel_data: Tuple[float, ...]

    def __post_init__(self):
        if not isinstance(self.channel_data, tuple):
            # Enforce immutability guarantee
            object.__setattr__(self, "channel_data", tuple(self.channel_data))


@dataclass(frozen=True)
class CoherenceMetric:
    """
    Immutable representation of descriptive phase metrics.
    """
    plv: float
    plv_percent: float
    phase_lag_ms: float
    coherence_at_01hz: float
    state: ResonanceStateEnum
    dominant_frequency_hz: float
    timestamp: float


@dataclass(frozen=True)
class NVCValidationResult:
    """
    Deprecated legacy type. Its ``is_valid_nvc`` field is not a scientifically
    valid NVC assessment; correlation alone cannot validate NVC.
    """
    is_valid_nvc: bool
    anticorrelation_r: float
    artifact_probability: float  # Legacy misnomer; value is an uncalibrated score.
    snr_db: float
