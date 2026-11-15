"""Reproducible toy-data check for descriptive PLV behavior.

Run from Program/: python tools/run_synthetic_phase_consistency.py --replicates 1000 --seed 20261115
The generated oscillators are deliberately simple and are not a biological
null model, a significance test, or validation against participant data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.coherence_engine import ResonanceEstimator


def summarize(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {"n": 0, "median_plv_percent": float("nan"), "q05": float("nan"), "q95": float("nan")}
    return {
        "n": int(finite.size),
        "median_plv_percent": float(np.median(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "q95": float(np.quantile(finite, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261115)
    args = parser.parse_args()
    if args.replicates <= 0:
        parser.error("--replicates must be positive")

    rng = np.random.default_rng(args.seed)
    sample_rate_hz = 10.0
    timestamps = np.arange(0.0, 110.0, 1.0 / sample_rate_hz)
    estimator = ResonanceEstimator(fs_resp=sample_rate_hz, fs_fnirs=sample_rate_hz, window_s=100.0)
    paired: list[float] = []
    independent: list[float] = []

    for _ in range(args.replicates):
        resp_frequency = 0.1
        resp_phase = rng.uniform(-np.pi, np.pi)
        resp = np.sin(2.0 * np.pi * resp_frequency * timestamps + resp_phase)
        resp += rng.normal(0.0, 0.20, size=timestamps.size)

        paired_hbo = np.sin(2.0 * np.pi * resp_frequency * timestamps + rng.uniform(-np.pi, np.pi))
        paired_hbo += rng.normal(0.0, 0.20, size=timestamps.size)
        estimate = estimator.update(resp, paired_hbo, np.zeros_like(paired_hbo), timestamps, timestamps)
        paired.append(float(estimate.plv_percent))

        independent_frequency = rng.uniform(0.07, 0.15)
        independent_hbo = np.sin(2.0 * np.pi * independent_frequency * timestamps + rng.uniform(-np.pi, np.pi))
        independent_hbo += rng.normal(0.0, 0.20, size=timestamps.size)
        estimate = estimator.update(resp, independent_hbo, np.zeros_like(independent_hbo), timestamps, timestamps)
        independent.append(float(estimate.plv_percent))

    print(json.dumps({
        "seed": args.seed,
        "replicates": args.replicates,
        "sample_rate_hz": sample_rate_hz,
        "analysis_window_s": 100.0,
        "toy_scenarios": {
            "same_frequency_phase_locked": summarize(paired),
            "independent_frequency_oscillators": summarize(independent),
        },
        "interpretation": (
            "Synthetic engineering check only. These oscillator distributions are not participant data, "
            "a validated physiological null, a significance test, or evidence for biological coupling."
        ),
    }, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
