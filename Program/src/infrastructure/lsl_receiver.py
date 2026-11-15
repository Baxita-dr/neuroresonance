"""Background polling and bounded handoff for a selected LSL session.

``SelectedLSLSession`` owns stream selection, metadata checks, clock correction,
timestamp alignment, and recording. This adapter moves its polling work off the
Streamlit script thread and hands timestamped samples to the UI through a
bounded queue. If the UI falls behind, the receiver reports the loss so the
analysis window can be reset; the session recorder still sees raw samples
before they enter this queue.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("NeuroResonanceLSLReceiver")


class LSLBackgroundReceiver:
    """Poll one connected selected LSL session on a daemon worker thread."""

    def __init__(
        self,
        session: Any,
        *,
        queue_capacity: int = 60000,
        max_samples_per_pull: int = 6000,
        poll_interval_s: float = 0.01,
    ) -> None:
        if queue_capacity <= 0 or max_samples_per_pull <= 0:
            raise ValueError("Queue capacity and pull size must be positive.")
        if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
            raise ValueError("Poll interval must be a finite positive number.")
        self.session = session
        self.max_samples_per_pull = int(max_samples_per_pull)
        self.poll_interval_s = float(poll_interval_s)
        self._samples: queue.Queue[tuple[float, float, float, float, float]] = queue.Queue(
            maxsize=int(queue_capacity)
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self._warnings: List[str] = []
        self._error: Optional[str] = None
        self._dropped_samples = 0
        self._overflow_since_read = False
        self._last_correction = float("nan")

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def dropped_samples(self) -> int:
        with self._state_lock:
            return self._dropped_samples

    @property
    def error(self) -> Optional[str]:
        with self._state_lock:
            return self._error

    def quality_summary(self) -> Dict[str, Any]:
        """Snapshot receiver-side issues for the local session metadata."""
        with self._state_lock:
            return {
                "dropped_analysis_samples": self._dropped_samples,
                "receiver_error": self._error,
            }

    def start(self) -> None:
        if self.is_running:
            return
        if not self.session.is_connected:
            raise RuntimeError("Cannot start LSL receiver: selected session is disconnected.")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="NeuroResonance-LSL-Receiver",
            daemon=True,
        )
        self._thread.start()

    def _add_warning(self, message: str) -> None:
        with self._state_lock:
            if message not in self._warnings:
                self._warnings.append(message)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if not self.session.is_connected:
                self._add_warning("Selected LSL stream disconnected; restart acquisition after checking the device.")
                return
            try:
                chunk = self.session.pull_chunk(max_samples=self.max_samples_per_pull)
                warnings = list(chunk.get("quality_warnings", []))
                for warning in warnings:
                    self._add_warning(str(warning))
                timestamps = np.asarray(chunk.get("timestamps", []), dtype=np.float64).ravel()
                hbo = np.asarray(chunk.get("HbO2", []), dtype=np.float64).ravel()
                hbr = np.asarray(chunk.get("HbR", []), dtype=np.float64).ravel()
                resp = np.asarray(chunk.get("Respiration", []), dtype=np.float64).ravel()
                raw_timestamps = np.asarray(chunk.get("timestamps_raw", []), dtype=np.float64).ravel()
                correction = float(chunk.get("time_correction_s", float("nan")))
                self._last_correction = correction
                if not (timestamps.size == hbo.size == hbr.size):
                    raise ValueError("LSL sample values and corrected timestamps have different lengths.")
                if resp.size != timestamps.size:
                    resp = np.full(timestamps.size, np.nan, dtype=np.float64)
                if raw_timestamps.size != timestamps.size:
                    raw_timestamps = np.full(timestamps.size, np.nan, dtype=np.float64)

                for index, timestamp in enumerate(timestamps):
                    sample = (
                        float(timestamp),
                        float(raw_timestamps[index]),
                        float(hbo[index]),
                        float(hbr[index]),
                        float(resp[index]),
                    )
                    try:
                        self._samples.put_nowait(sample)
                    except queue.Full:
                        try:
                            self._samples.get_nowait()
                        except queue.Empty:
                            pass
                        self._samples.put_nowait(sample)
                        with self._state_lock:
                            self._dropped_samples += 1
                            self._overflow_since_read = True
            except Exception as exc:
                message = f"LSL receiver stopped after {type(exc).__name__}: {exc}"
                with self._state_lock:
                    self._error = message
                    self._warnings.append(message)
                logger.exception("Background LSL acquisition failed")
                return

            if timestamps.size == 0:
                self._stop_event.wait(self.poll_interval_s)

    def get_chunk(self, max_samples: int = 6000) -> Dict[str, Any]:
        """Return queued timestamped samples without waiting for the device."""
        if max_samples <= 0:
            raise ValueError("max_samples must be positive.")
        rows = []
        for _ in range(int(max_samples)):
            try:
                rows.append(self._samples.get_nowait())
            except queue.Empty:
                break
        with self._state_lock:
            warnings = list(self._warnings)
            self._warnings.clear()
            overflow = self._overflow_since_read
            self._overflow_since_read = False
            dropped_samples = self._dropped_samples
            error = self._error
        if overflow:
            warnings.append(
                f"LSL UI queue overflow dropped samples (total: {dropped_samples}); "
                "the analysis window must be reset. Raw samples remain in the local recording."
            )
        columns = list(zip(*rows)) if rows else [(), (), (), (), ()]
        return {
            "timestamps": np.asarray(columns[0], dtype=np.float64),
            "timestamps_raw": np.asarray(columns[1], dtype=np.float64),
            "HbO2": np.asarray(columns[2], dtype=np.float64),
            "HbR": np.asarray(columns[3], dtype=np.float64),
            "Respiration": np.asarray(columns[4], dtype=np.float64),
            "time_correction_s": self._last_correction,
            "quality_warnings": warnings,
            "receiver_error": error,
            "dropped_samples": dropped_samples,
        }

    def stop(self, timeout_s: float = 5.0) -> bool:
        """Request shutdown and wait for the polling thread to leave the inlet."""
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=max(0.0, float(timeout_s)))
        stopped = not thread.is_alive()
        if not stopped:
            self._add_warning("LSL receiver did not stop before the shutdown timeout; inlet cleanup may be delayed.")
        return stopped
