"""Monotonic elapsed-time helpers for live acquisition sessions."""

from __future__ import annotations

import math
import time
from typing import Optional


def elapsed_seconds(started_at: Optional[float], now: Optional[float] = None) -> float:
    """Return non-negative elapsed time from a ``time.monotonic()`` origin."""
    if started_at is None:
        return 0.0
    current = time.monotonic() if now is None else float(now)
    started = float(started_at)
    if not math.isfinite(current) or not math.isfinite(started):
        raise ValueError("Session clock values must be finite.")
    return max(0.0, current - started)


def duration_reached(started_at: Optional[float], duration_s: float, now: Optional[float] = None) -> bool:
    """Whether a monotonic session clock has reached its configured duration."""
    duration = float(duration_s)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Session duration must be a finite non-negative number.")
    return elapsed_seconds(started_at, now) >= duration
