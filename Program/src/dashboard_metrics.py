"""Small pure helpers shared by the dashboard and its regression tests."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np


def estimate_sampling_rate_hz(timestamps: Any) -> Optional[float]:
    """Estimate rate from observed timestamp spacing, robust to jitter/outliers."""
    values = np.asarray(timestamps, dtype=np.float64).ravel()
    if values.size < 2:
        return None
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    intervals = np.diff(values)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if intervals.size == 0:
        return None
    median_interval = float(np.median(intervals))
    if not math.isfinite(median_interval) or median_interval <= 0:
        return None
    return 1.0 / median_interval


def summarize_signal_quality(timestamps: Any, **signals: Any) -> Dict[str, Any]:
    """Return descriptive sample coverage and timing checks for a signal window."""
    ts = np.asarray(timestamps, dtype=np.float64).ravel()
    arrays = {name: np.asarray(values, dtype=np.float64).ravel() for name, values in signals.items()}
    lengths = [ts.size, *(values.size for values in arrays.values())]
    n = min(lengths) if lengths else 0
    if n <= 0:
        return {
            "sample_count": 0,
            "timestamp_coverage_pct": 0.0,
            "joint_coverage_pct": 0.0,
            "channel_coverage_pct": {name: 0.0 for name in arrays},
            "estimated_rate_hz": None,
            "duration_s": 0.0,
            "median_interval_s": None,
            "long_gap_count": 0,
            "nonmonotonic_interval_count": 0,
            "length_mismatch": len(set(lengths)) > 1,
        }

    ts = ts[:n]
    timestamp_valid = np.isfinite(ts)
    channel_valid = {
        name: np.isfinite(values[:n])
        for name, values in arrays.items()
    }
    joint_valid = timestamp_valid.copy()
    for valid in channel_valid.values():
        joint_valid &= valid

    ordered_ts = ts[timestamp_valid]
    intervals = np.diff(ordered_ts)
    positive_intervals = intervals[np.isfinite(intervals) & (intervals > 0.0)]
    median_interval = float(np.median(positive_intervals)) if positive_intervals.size else None
    estimated_rate = 1.0 / median_interval if median_interval is not None else None
    if estimated_rate is not None and not math.isfinite(estimated_rate):
        estimated_rate = None
    long_gap_count = (
        int(np.count_nonzero(positive_intervals > 3.0 * median_interval))
        if median_interval is not None else 0
    )
    duration = float(ordered_ts[-1] - ordered_ts[0]) if ordered_ts.size > 1 else 0.0
    if not math.isfinite(duration):
        duration = 0.0

    return {
        "sample_count": int(n),
        "timestamp_coverage_pct": 100.0 * float(np.count_nonzero(timestamp_valid)) / n,
        "joint_coverage_pct": 100.0 * float(np.count_nonzero(joint_valid)) / n,
        "channel_coverage_pct": {
            name: 100.0 * float(np.count_nonzero(valid)) / n
            for name, valid in channel_valid.items()
        },
        "estimated_rate_hz": float(estimated_rate) if estimated_rate is not None else None,
        "duration_s": duration,
        "median_interval_s": median_interval,
        "long_gap_count": long_gap_count,
        "nonmonotonic_interval_count": int(np.count_nonzero(intervals <= 0.0)),
        "length_mismatch": len(set(lengths)) > 1,
    }


def build_realtime_insight_state(
    *,
    plv_pct: float,
    phase_lag_ms: Optional[float],
    coherence: Optional[float],
    state_label: str,
    pacing_reference_hz: Optional[float],
) -> Dict[str, Any]:
    """Build AI summary input while preserving distinct estimator measures."""
    del phase_lag_ms  # Legacy compatibility parameter; phase lag is not reported.
    return {
        "plv_pct": float(plv_pct),
        "coherence": float(coherence) if coherence is not None and math.isfinite(float(coherence)) else None,
        "state_label": state_label,
        "signal_configuration": {
            "pacing_reference_hz": round(float(pacing_reference_hz), 4) if pacing_reference_hz is not None else None,
        },
    }


def build_session_insight_state(
    *,
    plv_pct: float,
    phase_lag_ms: Optional[float],
    estimator_coherence: Optional[float],
    state_label: str,
    pacing_reference_hz: Optional[float],
) -> Dict[str, Any]:
    """Build the dashboard summary from estimator outputs without deriving coherence from PLV."""
    return build_realtime_insight_state(
        plv_pct=plv_pct,
        phase_lag_ms=phase_lag_ms,
        coherence=estimator_coherence,
        state_label=state_label,
        pacing_reference_hz=pacing_reference_hz,
    )
