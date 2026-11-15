"""
Unit tests for signal-processing utilities
"""

import numpy as np
import pytest
from src.signal_processing import (
    bandpass_filter,
    compute_instantaneous_phase,
    compute_instantaneous_frequency,
    compute_power_spectrum,
    HemoglobinCorrelationSummary,
    z_score_normalize,
)


def test_butterworth_bandpass_attenuation():
    """Verify that 0.1 Hz signal passes and 2.0 Hz high-frequency noise is attenuated."""
    fs = 10.0
    duration_s = 60.0
    t = np.linspace(0, duration_s, int(fs * duration_s), endpoint=False)

    # Signal: 0.1 Hz target + 2.0 Hz noise
    sig_target = np.sin(2 * np.pi * 0.10 * t)
    sig_noise = np.sin(2 * np.pi * 2.00 * t)
    sig_composite = sig_target + sig_noise

    # Bandpass 0.07 - 0.15 Hz
    filtered = bandpass_filter(sig_composite, lowcut=0.07, highcut=0.15, fs=fs, order=4)

    # Power in target band vs noise band
    target_power_original = np.mean(sig_target ** 2)
    target_power_filtered = np.mean(filtered ** 2)

    # Attenuation check: output power should closely match target power, noise suppressed
    assert np.isclose(target_power_filtered, target_power_original, rtol=0.25)
    # Correlation with original 0.1 Hz target in steady-state (ignoring filter transient edges)
    corr = np.corrcoef(filtered[int(fs * 2):-int(fs * 2)], sig_target[int(fs * 2):-int(fs * 2)])[0, 1]
    assert corr > 0.92


def test_hilbert_instantaneous_phase_accuracy():
    """Verify Hilbert analytic transform accurately recovers known harmonic phase."""
    fs = 100.0
    t = np.linspace(0, 10.0, int(fs * 10.0), endpoint=False)
    freq = 1.0  # 1 Hz

    # Cosine signal has phase 2*pi*f*t
    sig = np.cos(2 * np.pi * freq * t)
    phase, amp = compute_instantaneous_phase(sig)

    # Check amplitude envelope ~ 1.0 (excluding boundary edge effects)
    mid_slice = slice(int(fs), -int(fs))
    assert np.allclose(amp[mid_slice], 1.0, atol=0.05)

    # Instantaneous frequency should be ~ 1.0 Hz
    inst_freq = compute_instantaneous_frequency(phase, fs=fs)
    assert np.allclose(inst_freq[mid_slice], 1.0, atol=0.08)


def test_hbo2_hbr_correlation_summary_is_descriptive():
    """Verify descriptive correlation thresholds, not NVC or artifact validity."""
    checker = HemoglobinCorrelationSummary(window_s=20.0, fs=10.0)
    n = 200

    # Negative correlation is a display-threshold result only.
    hbo2_clean = np.sin(np.linspace(0, 4 * np.pi, n))
    hbr_clean = -0.7 * hbo2_clean + np.random.normal(0, 0.05, n)

    result_negative = checker.update(hbo2_clean, hbr_clean)
    assert result_negative["below_anticorrelation_display_threshold"] is True
    assert result_negative["anticorrelation_r"] < -0.5
    assert 0.0 <= result_negative["artifact_score"] <= 1.0
    assert result_negative["artifact_probability"] == result_negative["artifact_score"]

    # Positive correlation does not identify its cause.
    hbo2_artifact = hbo2_clean.copy()
    hbr_artifact = 0.8 * hbo2_clean  # Positive correlation!
    result_positive = checker.update(hbo2_artifact, hbr_artifact)
    assert result_positive["below_anticorrelation_display_threshold"] is False
    assert result_positive["anticorrelation_r"] > 0.0


def test_artifact_score_does_not_count_channel_baseline_step():
    """Different HbO2/HbR baselines do not create a variance artifact at the join."""
    score = HemoglobinCorrelationSummary._compute_artifact_score(
        np.full(100, 12.0), np.full(100, -8.0)
    )
    assert score == 0.0


def test_power_spectrum_peak_detection():
    """Verify Welch PSD accurately locates dominant Mayer wave peak."""
    fs = 10.0
    t = np.linspace(0, 100.0, int(fs * 100.0), endpoint=False)
    sig = np.sin(2 * np.pi * 0.100 * t)  # 0.100 Hz Mayer frequency

    freqs, pxx, dom_freq, dom_power = compute_power_spectrum(sig, fs=fs, nperseg=512)
    assert np.isclose(dom_freq, 0.100, atol=0.02)
    assert dom_power > 0.1


def test_bandpass_filter_short_signal_graceful():
    """Verify bandpass_filter on signals of lengths 13-27 returns safely without raising ValueError (L1)."""
    fs = 10.0
    for length in (5, 13, 20, 24, 27):
        sig = np.sin(np.linspace(0, 2 * np.pi, length))
        filtered = bandpass_filter(sig, lowcut=0.07, highcut=0.15, fs=fs, order=4)
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == length
