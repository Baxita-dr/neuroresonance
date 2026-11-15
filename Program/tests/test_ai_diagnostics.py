"""
tests/test_ai_diagnostics.py — Unit Tests for Experimental Signal Summaries
"""

import pytest
from src.ai_diagnostics import (
    FallbackDiagnosticsEngine,
    NeuroDiagnosticsEngine,
    get_diagnostics_engine,
)


def test_fallback_diagnostics_engine_report_generation():
    """Verify FallbackDiagnosticsEngine generates valid structured Markdown report."""
    engine = FallbackDiagnosticsEngine()
    res_state = {
        "plv_pct": 78.5,
        "phase_lag_ms": 320.0,
        "coherence": 0.81,
        "state_label": "DESCRIPTIVE PLV (NO STATE CLASSIFICATION)",
        "biometrics": {
            "age": 28,
            "height_cm": 178.0,
            "weight_kg": 72.0,
            "sex": "M",
            "calibrated_dpf": 5.65,
            "expected_neurovascular_lag_s": 3.6,
            "pacing_reference_hz": 0.1,
        },
    }
    nvc_data = {
        "anti_correlation_r": -0.68,
        "significance": "not assessed",
        "calibrated_dpf": 5.65,
    }
    psd_data = {
        "dominant_freq_hz": 0.092,
        "band_power": {"Mayer_0.1Hz": 0.75},
    }

    report = engine.generate_report(
        resonance_state=res_state,
        nvc_check=nvc_data,
        power_spectrum_data=psd_data,
        session_duration_s=120.0,
    )

    assert isinstance(report, str)
    assert "## Experimental signal summary" in report
    assert "78.5%" in report
    assert "HbO₂/HbR sample correlation" in report
    assert "320" not in report


def test_fallback_diagnostics_engine_realtime_insight():
    """Verify real-time descriptions do not depend on PLV categories."""
    engine = FallbackDiagnosticsEngine()

    state_high = {
        "plv_pct": 82.0,
        "phase_lag_ms": 250.0,
        "coherence": 0.85,
        "state_label": "DESCRIPTIVE PLV (NO STATE CLASSIFICATION)",
    }
    insight_high = engine.generate_realtime_insight(state_high)
    assert isinstance(insight_high, str)
    assert len(insight_high) > 10

    state_low = {
        "plv_pct": 25.0,
        "phase_lag_ms": 800.0,
        "coherence": 0.20,
        "state_label": "DESCRIPTIVE PLV (NO STATE CLASSIFICATION)",
    }
    insight_low = engine.generate_realtime_insight(state_low)
    assert isinstance(insight_low, str)
    assert len(insight_low) > 10


@pytest.mark.parametrize("coherence", [None, float("nan")])
def test_realtime_insight_reports_unavailable_coherence_without_raising(coherence):
    engine = FallbackDiagnosticsEngine()

    insight = engine.generate_realtime_insight({
        "plv_pct": 93.8,
        "coherence": coherence,
        "state_label": "DESCRIPTIVE PLV (NO STATE CLASSIFICATION)",
    })

    assert "PLV=93.8%" in insight
    assert "spectral coherence=not available" in insight


def test_remote_diagnostics_engine_is_disabled():
    with pytest.raises(RuntimeError, match="Remote LLM diagnostics are disabled"):
        NeuroDiagnosticsEngine(api_key="must-not-be-used")


def test_get_diagnostics_engine_without_api_key_returns_fallback(monkeypatch):
    """Verify get_diagnostics_engine safely returns fallback engine when no key is supplied."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    engine = get_diagnostics_engine(api_key="")
    assert isinstance(engine, FallbackDiagnosticsEngine)


def test_ai_diagnostics_nan_and_provenance_handling():
    """Verify unavailable PLV and provenance are reported without categories."""

    engine = FallbackDiagnosticsEngine()
    res_state = {
        "plv_pct": float("nan"),
        "phase_lag_ms": 0.0,
        "coherence": 0.0,
        "state_label": "INSUFFICIENT SIGNAL",
        "data_provenance": "SIMULATOR (PACED RESPIRATION + SYNTHETIC fNIRS-LIKE SIGNALS)",
    }
    nvc_data = {"anti_correlation_r": 0.0, "significance": "not assessed"}
    psd_data = {"dominant_freq_hz": 0.10}

    report = engine.generate_report(
        resonance_state=res_state,
        nvc_check=nvc_data,
        power_spectrum_data=psd_data,
        session_duration_s=60.0,
    )
    # Provenance notice must be present
    assert "SIMULATOR" in report
    assert "PLV: **not available**" in report
    assert "software category" not in report
