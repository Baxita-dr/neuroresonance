import numpy as np
import pytest

from src.dashboard_metrics import build_session_insight_state, estimate_sampling_rate_hz


def test_realtime_insight_uses_estimator_coherence_not_plv_fraction():
    state = build_session_insight_state(
        plv_pct=78.4,
        phase_lag_ms=125.0,
        estimator_coherence=0.31,
        state_label="DESCRIPTIVE PLV (NO STATE CLASSIFICATION)",
        pacing_reference_hz=0.1,
    )

    assert state["plv_pct"] == 78.4
    assert state["coherence"] == 0.31
    assert state["coherence"] != state["plv_pct"] / 100.0


def test_measured_sampling_rate_uses_timestamp_spacing_and_handles_missing_data():
    timestamps = np.array([0.0, 0.1, 0.2, 0.3, np.nan])
    assert estimate_sampling_rate_hz(timestamps) == pytest.approx(10.0)
    assert estimate_sampling_rate_hz([1.0]) is None
    assert estimate_sampling_rate_hz([1.0, 1.0, np.nan]) is None
