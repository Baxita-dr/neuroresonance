"""
NeuroResonance: experimental respiration and fNIRS signal visualization tools.

The package includes signal processing, synthetic examples, camera-derived
motion proxies, and experimental LSL input. Its metrics are not clinically or
physiologically validated.
"""

__version__ = "1.0.0"
__author__ = "Bastian Fernandez; Aleksandra Zwolińska (XIV LO Wrocław)"

# Safe imports for signal processing & coherence engine
try:
    from src.domain.biometrics import BiometricProfile
except ImportError:
    pass

try:
    from src.ring_buffer import CircularRingBuffer
except ImportError:
    pass

try:
    from src.signal_processing import (
        bandpass_filter,
        compute_instantaneous_phase,
        compute_instantaneous_frequency,
        compute_power_spectrum,
        NeurovascularCouplingCheck,
        ProcessedSignal,
        z_score_normalize,
        min_max_normalize,
    )
except ImportError:
    pass

try:
    from src.coherence_engine import (
        compute_plv,
        compute_coherence,
        compute_lissajous,
        ResonanceEstimator,
        ResonanceState,
    )
except ImportError:
    pass

try:
    from src.ai_diagnostics import (
        NeuroDiagnosticsEngine,
        FallbackDiagnosticsEngine,
        get_diagnostics_engine,
    )
except ImportError:
    pass

try:
    from src.lsl_streamer import fNIRSSimulator, LSLBridge
except ImportError:
    pass

try:
    from src.camera_tracker import CameraTracker, RespiratorySignal
except ImportError:
    pass
