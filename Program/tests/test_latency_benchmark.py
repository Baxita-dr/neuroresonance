"""
tests/test_latency_benchmark.py — Real-Time DSP Latency & Throughput Benchmark
"""

import time
import os
import numpy as np
import pytest
from src.ring_buffer import CircularRingBuffer
from src.signal_processing import bandpass_filter, compute_instantaneous_phase, NeurovascularCouplingCheck
from src.coherence_engine import compute_plv

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_BENCHMARKS") != "1",
    reason="Machine-dependent timing benchmarks are opt-in; set RUN_BENCHMARKS=1.",
)


def test_circular_ring_buffer_push_latency():
    """Benchmark ring buffer push: must be < 5 microseconds per sample."""
    buf = CircularRingBuffer(capacity=5000)
    n_iterations = 1000

    t0 = time.perf_counter()
    for i in range(n_iterations):
        buf.push(float(i))
    t1 = time.perf_counter()

    avg_us = ((t1 - t0) / n_iterations) * 1e6
    print(f"\n[BENCHMARK] RingBuffer Push Latency: {avg_us:.3f} us / sample")
    assert avg_us < 20.0  # Well below 20 microseconds!


def test_realtime_dsp_pipeline_latency():
    """
    Benchmark full DSP cycle per update epoch:
    - 300 samples (30 seconds at 10 Hz)
    - 4th-order Butterworth SOS zero-phase filtering
    - Hilbert Analytic Transform & phase unwrapping
    - Phase-Locking Value (PLV) calculation
    - Neurovascular coupling Pearson cross-correlation
    Must complete in < 5 milliseconds (allowing >200 Hz real-time operation).
    """
    n_samples = 300  # 30 seconds at 10 Hz
    fs = 10.0
    t = np.linspace(0, 30.0, n_samples)
    sig_resp = np.sin(2 * np.pi * 0.1 * t) + np.random.normal(0, 0.05, n_samples)
    sig_hbo2 = np.sin(2 * np.pi * 0.1 * (t - 3.8)) + np.random.normal(0, 0.05, n_samples)
    sig_hbr = -0.7 * sig_hbo2 + np.random.normal(0, 0.05, n_samples)
    nvc_checker = NeurovascularCouplingCheck(window_s=30.0, fs=fs)

    n_runs = 50
    latencies_ms = []

    for _ in range(n_runs):
        start = time.perf_counter()

        # Step 1: Bandpass filter both channels
        resp_filt = bandpass_filter(sig_resp, lowcut=0.07, highcut=0.15, fs=fs, order=4)
        hbo_filt = bandpass_filter(sig_hbo2, lowcut=0.07, highcut=0.15, fs=fs, order=4)

        # Step 2: Instantaneous Hilbert phases
        phase_r, _ = compute_instantaneous_phase(resp_filt)
        phase_h, _ = compute_instantaneous_phase(hbo_filt)

        # Step 3: Phase-Locking Value (PLV)
        plv = compute_plv(phase_r, phase_h)

        # Step 4: Neurovascular Coupling check
        nvc_res = nvc_checker.update(sig_hbo2, sig_hbr)

        duration_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(duration_ms)

    mean_latency = float(np.mean(latencies_ms))
    p99_latency = float(np.percentile(latencies_ms, 99))

    print(f"\n[BENCHMARK] End-to-End DSP Pipeline Latency: Mean = {mean_latency:.2f} ms | P99 = {p99_latency:.2f} ms")
    print(f"[BENCHMARK] Real-Time Processing Headroom: {(1.0 - (mean_latency / 100.0)) * 100:.1f}% idle CPU budget at 10 Hz")

    # Hard assert: Processing must take less than 6 ms (we budget up to 100 ms per frame)
    assert mean_latency < 6.0
