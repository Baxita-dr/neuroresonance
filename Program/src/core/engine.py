"""
src/core/engine.py — Production-Grade Asynchronous Coherence Engine

Implements explicit backpressure, bounded FIFO queues, and zero-leak memory
lifecycle as required by enterprise biometric processing standards.
"""

import asyncio
import logging
import math
import time
import numpy as np
from typing import Optional, List, Tuple

from src.domain.models import SensorSample, CoherenceMetric, ResonanceStateEnum
from src.domain.protocols import CoherenceEstimatorProtocol
from src.coherence_engine import ResonanceEstimator

logger = logging.getLogger("NeuroResonanceCore")


class RobustCoherenceEngine(CoherenceEstimatorProtocol):
    """
    Asynchronous, backpressure-managed Phase-Locking Engine.

    Features:
    - Bounded FIFO queue (prevents memory leaks when incoming stream > processing rate)
    - Non-blocking ingestion: drops oldest sample upon queue saturation
    - Windowed vectorized PLV calculation using Hilbert Analytic Transform
    """

    def __init__(
        self,
        buffer_size: int = 1024,
        window_seconds: float = 100.0,
        nominal_fs: float = 10.0,
    ) -> None:
        self._buffer_size = int(buffer_size)
        self._window_seconds = float(window_seconds)
        self._nominal_fs = float(nominal_fs)
        if self._buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        if not np.isfinite(self._nominal_fs) or self._nominal_fs <= 0:
            raise ValueError("nominal_fs must be positive and finite.")
        if not np.isfinite(self._window_seconds) or self._window_seconds < 100.0:
            raise ValueError("window_seconds must be at least 100 seconds.")
        self._window_samples = int(math.ceil(self._window_seconds * self._nominal_fs)) + 2
        # Keep the asynchronous ingestion/backpressure API while sharing the
        # same DSP implementation used by the dashboard.
        self._estimator = ResonanceEstimator(
            fs_resp=self._nominal_fs,
            fs_fnirs=self._nominal_fs,
            window_s=self._window_seconds,
        )

        self._queue: asyncio.Queue[SensorSample] = asyncio.Queue(maxsize=self._buffer_size)
        self._history: List[SensorSample] = []
        self._dropped_samples: int = 0
        self._total_ingested: int = 0

    @property
    def queue_size(self) -> int:
        """Current number of pending samples in the async FIFO queue."""
        return self._queue.qsize()

    @property
    def dropped_samples(self) -> int:
        """Total count of dropped samples due to backpressure saturation."""
        return self._dropped_samples

    @property
    def total_ingested(self) -> int:
        """Monotonic counter of total samples processed."""
        return self._total_ingested

    async def ingest(self, sample: SensorSample) -> None:
        """
        Ingest a SensorSample with non-blocking backpressure protection.
        If queue is saturated, eject the oldest sample (FIFO drop policy).
        """
        if len(sample.channel_data) < 2:
            logger.warning("Rejected SensorSample with fewer than 2 channels: %s", sample.channel_data)
            return

        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._dropped_samples += 1
            except asyncio.QueueEmpty:
                pass

        try:
            self._queue.put_nowait(sample)
        except asyncio.QueueFull:
            self._dropped_samples += 1
            return
        self._total_ingested += 1

    async def drain_queue(self) -> int:
        """
        Drain all pending samples from the async queue into the calculation epoch history.
        Maintains history size bounded to window_samples to avoid memory leaks.
        """
        drained = 0
        while not self._queue.empty():
            try:
                sample = self._queue.get_nowait()
                self._history.append(sample)
                drained += 1
            except asyncio.QueueEmpty:
                break

        # Bounded retention: keep at most window_samples
        if len(self._history) > self._window_samples:
            self._history = self._history[-self._window_samples:]

        return drained

    async def compute_coherence(self) -> CoherenceMetric:
        """
        Compute instantaneous phase and Phase-Locking Value (PLV) across the active window.
        Assumes channel 0 = respiration, channel 1 = HbO2 (or hemodynamic proxy).
        """
        await self.drain_queue()

        if len(self._history) < self._window_samples:
            # Do not produce a plausible-looking statistic before the full window exists.
            return CoherenceMetric(
                plv=float("nan"),
                plv_percent=float("nan"),
                phase_lag_ms=float("nan"),
                coherence_at_01hz=float("nan"),
                state=ResonanceStateEnum.STARTING,
                dominant_frequency_hz=float("nan"),
                timestamp=time.time(),
            )

        # Unpack channels into numpy arrays, validating channel count
        valid_samples = [s for s in self._history if len(s.channel_data) >= 2]
        if len(valid_samples) < self._window_samples:
            return CoherenceMetric(
                plv=float("nan"),
                plv_percent=float("nan"),
                phase_lag_ms=float("nan"),
                coherence_at_01hz=float("nan"),
                state=ResonanceStateEnum.STARTING,
                dominant_frequency_hz=float("nan"),
                timestamp=time.time(),
            )

        t_arr = np.array([s.timestamp for s in valid_samples], dtype=np.float64)
        c0_arr = np.array([s.channel_data[0] for s in valid_samples], dtype=np.float64)
        c1_arr = np.array([s.channel_data[1] for s in valid_samples], dtype=np.float64)

        try:
            # The async protocol provides respiration and one hemodynamic
            # channel; HbR is not required for the shared PLV estimator.
            estimate = self._estimator.update(
                resp_signal=c0_arr,
                hbo2_signal=c1_arr,
                hbr_signal=np.zeros_like(c1_arr),
                timestamps_resp=t_arr,
                timestamps_fnirs=t_arr,
            )
        except Exception as exc:
            logger.warning("DSP calculation error in RobustCoherenceEngine: %s", exc)
            raise RuntimeError("DSP calculation failed; no metrics were produced.") from exc

        plv_pct = float(estimate.plv_percent)
        plv = plv_pct / 100.0 if np.isfinite(plv_pct) else float("nan")
        dom_f = float(estimate.dominant_freq_hz)
        coh_val = float(estimate.coherence_at_01hz)
        state = ResonanceStateEnum.STARTING if not np.isfinite(plv_pct) else ResonanceStateEnum.DESCRIPTIVE

        return CoherenceMetric(
            plv=plv,
            plv_percent=plv_pct,
            phase_lag_ms=float("nan"),
            coherence_at_01hz=float(coh_val),
            state=state,
            dominant_frequency_hz=dom_f,
            timestamp=t_arr[-1] if len(t_arr) > 0 else time.time(),
        )
