"""Measure the production 100-second estimator path on deterministic data."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coherence_engine import ResonanceEstimator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    fs = 10.0
    timestamps = np.arange(0.0, 100.1, 1.0 / fs, dtype=np.float64)
    rng = np.random.default_rng(20261115)
    phase = 2.0 * np.pi * 0.1 * timestamps
    respiration = np.sin(phase) + rng.normal(0.0, 0.02, timestamps.size)
    hbo2 = np.sin(phase - 0.8) + rng.normal(0.0, 0.025, timestamps.size)
    hbr = -0.55 * hbo2 + rng.normal(0.0, 0.025, timestamps.size)
    estimator = ResonanceEstimator(fs_resp=fs, fs_fnirs=fs, window_s=100.0)

    def update_once() -> float:
        start = time.perf_counter()
        state = estimator.update(
            resp_signal=respiration,
            hbo2_signal=hbo2,
            hbr_signal=hbr,
            timestamps_resp=timestamps,
            timestamps_fnirs=timestamps,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if not np.isfinite(state.plv_percent):
            raise RuntimeError(f"Estimator returned unavailable PLV: {state.state_label}")
        return elapsed_ms

    for _ in range(5):
        update_once()  # Warm up SciPy and native-library paths.
    measurements = np.asarray([update_once() for _ in range(args.iterations)], dtype=np.float64)
    p50, p95, p99 = np.percentile(measurements, [50, 95, 99])
    print(f"Production 100 s estimator benchmark ({args.iterations} iterations)")
    print(f"mean={measurements.mean():.3f} ms  p50={p50:.3f} ms  p95={p95:.3f} ms  p99={p99:.3f} ms")
    print("Synthetic deterministic input only; timings depend on this computer and do not validate hardware acquisition.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
