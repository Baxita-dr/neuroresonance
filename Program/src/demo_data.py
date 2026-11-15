"""Deterministic synthetic data for short, repeatable jury demonstrations."""

from __future__ import annotations

import numpy as np


DEMO_SEED = 20261115
DEMO_SAMPLE_RATE_HZ = 10.0
DEMO_DURATION_S = 180.0


def build_deterministic_demo_recording() -> list[dict]:
    """Build one 180-second synthetic stream with known channel mapping.

    The constructed channels share a 0.1 Hz component by design. This fixture
    demonstrates import, playback and descriptive metrics; it is not a
    physiological recording or a model of biological coupling.
    """
    timestamps = np.arange(
        0.0,
        DEMO_DURATION_S,
        1.0 / DEMO_SAMPLE_RATE_HZ,
        dtype=np.float64,
    )
    rng = np.random.default_rng(DEMO_SEED)
    phase = 2.0 * np.pi * 0.1 * timestamps
    respiration = np.sin(phase) + 0.12 * np.sin(2.0 * phase + 0.2)
    respiration += rng.normal(0.0, 0.015, timestamps.size)

    hbo2 = 0.7 * np.sin(phase - 0.8) + 0.08 * np.sin(2.0 * phase - 0.3)
    hbo2 += rng.normal(0.0, 0.025, timestamps.size)
    hbr = -0.55 * hbo2 + rng.normal(0.0, 0.025, timestamps.size)
    values = np.column_stack((respiration, hbo2, hbr))

    return [{
        "index": 0,
        "name": "NeuroResonance synthetic demo (seed 20261115)",
        "type": "synthetic demonstration",
        "source_id": "neuroresonance-demo-v1",
        "channel_format": "double64",
        "channel_count": 3,
        "nominal_srate_hz": DEMO_SAMPLE_RATE_HZ,
        "sample_count": int(timestamps.size),
        "duration_s": float(timestamps[-1]) if timestamps.size else 0.0,
        "channels": [
            {"index": 0, "label": "Respiration (synthetic)", "unit": "a.u.", "type": "synthetic"},
            {"index": 1, "label": "HbO2-like (synthetic)", "unit": "a.u.", "type": "synthetic"},
            {"index": 2, "label": "HbR-like (synthetic)", "unit": "a.u.", "type": "synthetic"},
        ],
        "is_numeric": True,
        "time_series": values,
        "timestamps": timestamps,
    }]
