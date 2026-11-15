"""
tests/test_coherence_engine.py — Phase-Locking Value & Resonance Engine Tests
"""

import numpy as np
import pytest
from src.coherence_engine import (
    compute_plv,
    compute_lissajous,
    ResonanceEstimator,
    ResonanceState,
)


def test_plv_ground_truth_identical_phases():
    """Identical phase series must yield PLV == 1.0000 exactly."""
    n = 500
    phases = np.linspace(-np.pi, np.pi, n)
    plv = compute_plv(phases, phases)
    assert np.isclose(plv, 1.0, atol=1e-5)


def test_plv_ground_truth_constant_phase_lag():
    """Constant phase difference (arbitrary delay) must yield PLV == 1.0000."""
    n = 500
    phase1 = np.linspace(0, 10 * np.pi, n)
    delta_phi = np.pi / 3.0  # 60 degrees constant lag
    phase2 = phase1 + delta_phi

    plv = compute_plv(phase1, phase2)
    assert np.isclose(plv, 1.0, atol=1e-5)


def test_plv_uncorrelated_noise():
    """Uncorrelated phase noise must yield near-zero PLV (bounded by 1/sqrt(N))."""
    np.random.seed(42)
    n = 2000
    phase1 = np.random.uniform(-np.pi, np.pi, n)
    phase2 = np.random.uniform(-np.pi, np.pi, n)

    plv = compute_plv(phase1, phase2)
    # Expected value for N=2000 is ~ 1/sqrt(2000) ~ 0.022
    assert plv < 0.10


def test_lissajous_orbit_computation():
    """Verify parametric Lissajous generation returns balanced 2D coordinates."""
    n = 200
    p1 = np.linspace(0, 2 * np.pi, n)
    p2 = p1 + np.pi / 2.0  # Circle in phase space

    x, y = compute_lissajous(p1, p2, amplitude1=1.0, amplitude2=1.0, n_points=100)
    assert len(x) == 100
    assert len(y) == 100
    # Coordinates must lie inside [-1.0, 1.0] unit bounds
    assert np.all(x >= -1.0) and np.all(x <= 1.0)
    assert np.all(y >= -1.0) and np.all(y <= 1.0)


def test_resonance_estimator_pipeline():
    """Estimator reports a descriptive PLV after a full timestamped window."""
    estimator = ResonanceEstimator(fs_resp=10.0, fs_fnirs=10.0, window_s=100.0)
    t = np.arange(0.0, 110.0, 0.1)

    # 0.1 Hz coherent signals
    resp = np.sin(2 * np.pi * 0.10 * t)
    # HbO2 delayed by 3.5 s (phase lag ~ 0.1 * 2*pi * 3.5 = 0.7*pi)
    hbo2 = np.sin(2 * np.pi * 0.10 * (t - 3.5))
    hbr = -0.7 * hbo2

    state = estimator.update(
        resp_signal=resp,
        hbo2_signal=hbo2,
        hbr_signal=hbr,
        timestamps_resp=t,
        timestamps_fnirs=t,
    )

    assert isinstance(state, ResonanceState)
    assert state.plv_percent > 90.0
    assert state.state_label == "DESCRIPTIVE PLV (NO STATE CLASSIFICATION)"
    assert state.state_color == "#4FC3F7"
    assert state.is_significant is None
    assert np.isnan(state.phase_lag_ms)
    assert np.isclose(state.dominant_freq_hz, 0.10, atol=0.03)


def test_unvalidated_surrogate_api_is_disabled():
    """Surrogate significance is disabled rather than returning a misleading p-value."""
    from src.coherence_engine import compute_plv_block_surrogate
    with pytest.raises(NotImplementedError, match="Surrogate p-values were removed"):
        compute_plv_block_surrogate(np.zeros(20), np.zeros(20))


def test_estimator_never_assigns_a_plv_category():
    """The compatibility classifier returns a descriptive status at every PLV."""
    label, color = ResonanceEstimator._classify(0.99)
    assert label == "DESCRIPTIVE PLV (NO STATE CLASSIFICATION)"
    assert color == "#4FC3F7"


def test_alignment_rejects_a_long_acquisition_gap():
    """A long missing span is rejected instead of being linearly invented."""
    ts = np.arange(0.0, 10.0, 0.1)
    values = np.sin(2.0 * np.pi * 0.1 * ts)
    keep = np.ones(ts.size, dtype=bool)
    keep[35:43] = False  # 0.8 s rendering/acquisition pause

    aligned = ResonanceEstimator._align_on_overlap(
        values[keep], ts[keep], values, ts, fs_out=2.0
    )

    assert aligned is None


def test_plv_nan_resilience():
    """Verify compute_plv and estimator handle NaN inputs gracefully (M2)."""
    p1 = np.array([0.1, 0.2, np.nan, 0.4, 0.5])
    p2 = np.array([0.1, 0.2, 0.3, np.nan, 0.5])
    plv = compute_plv(p1, p2)
    assert np.isfinite(plv)
    assert 0.0 <= plv <= 1.0

    # Estimator update with NaNs
    estimator = ResonanceEstimator(fs_resp=10.0, fs_fnirs=10.0, window_s=100.0)
    sig = np.array([1.0, np.nan, 2.0, 3.0] * 50)
    t = np.linspace(0, 20.0, len(sig))
    state = estimator.update(sig, sig, sig, t, t)
    assert isinstance(state, ResonanceState)
    assert np.isnan(state.plv_percent)
    assert state.state_label.startswith("INSUFFICIENT DATA")
