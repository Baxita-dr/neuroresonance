"""
camera_tracker.py
=================
Experimental webcam motion proxy based on mean luminance in a selected chest ROI.
The signal is not calibrated to physical displacement, is not a reference
measurement, and can be affected by lighting, framing, camera motion, and other
changes in the image. The tracker only removes a linear trend; downstream
analysis owns bandpass filtering and does not establish that the signal is
respiration.
"""

from __future__ import annotations

import threading
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "OpenCV is required: pip install opencv-python"
    ) from exc

try:
    from scipy.signal import detrend
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "SciPy is required: pip install scipy"
    ) from exc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class RespiratorySignal:
    """Container for a snapshot of the tracked respiratory waveform.

    Attributes
    ----------
    raw : np.ndarray
        Unfiltered mean-luminance timeseries extracted from the chest ROI.
        Units: mean 8-bit pixel value (0–255).
    filtered : np.ndarray
        Legacy field name for the linearly detrended luminance trace. No
        bandpass filtering is applied here; the estimator applies its bandpass
        once to the aligned respiration and fNIRS signals. NaN-padded until a
        detrended snapshot is available.
    timestamps : np.ndarray
        Unix timestamps (float seconds) corresponding to each sample.
    sampling_rate : float
        Effective capture rate in Hz (frames per second) at which samples were
        collected.  May differ slightly from the requested ``fps_target`` due to
        hardware constraints.
    """

    raw: np.ndarray
    filtered: np.ndarray
    timestamps: np.ndarray
    sampling_rate: float = 30.0
    # Derived convenience properties computed post-init
    duration_s: float = field(init=False)
    n_samples: int = field(init=False)

    @property
    def detrended(self) -> np.ndarray:
        """Return detrended luminance (alias for the legacy ``filtered`` field)."""
        return self.filtered

    def __post_init__(self) -> None:
        self.n_samples = len(self.raw)
        self.duration_s = self.n_samples / max(self.sampling_rate, 1e-6)

    # ------------------------------------------------------------------
    # Quick statistics helpers
    # ------------------------------------------------------------------

    def dominant_frequency_hz(self) -> float:
        """Return the peak frequency of the detrended signal via FFT."""
        if self.n_samples < 4:
            return float("nan")
        sig = self.filtered if not np.all(np.isnan(self.filtered)) else self.raw
        # Strip any NaN / non-finite padding before mean removal and FFT
        valid = np.isfinite(sig)
        if not np.any(valid):
            valid_raw = np.isfinite(self.raw)
            if not np.any(valid_raw):
                return float("nan")
            sig = self.raw[valid_raw]
        else:
            sig = sig[valid]

        if len(sig) < 4:
            return float("nan")

        freqs = np.fft.rfftfreq(len(sig), d=1.0 / self.sampling_rate)
        power = np.abs(np.fft.rfft(sig - sig.mean())) ** 2
        # Restrict to respiratory band
        mask = (freqs >= 0.1) & (freqs <= 0.5)
        if not mask.any():
            return float("nan")
        peak_idx = np.argmax(power[mask])
        return float(freqs[mask][peak_idx])

    def breaths_per_minute(self) -> float:
        """Convert dominant frequency to breaths per minute."""
        return self.dominant_frequency_hz() * 60.0

    def summary(self) -> str:
        """Return a human-readable stats string."""
        bpm = self.breaths_per_minute()
        return (
            f"RespiratorySignal | "
            f"n={self.n_samples} samples | "
            f"duration={self.duration_s:.1f}s | "
            f"fs={self.sampling_rate:.1f} Hz | "
            f"dominant={self.dominant_frequency_hz():.3f} Hz "
            f"({bpm:.1f} breaths/min)"
        )


# ---------------------------------------------------------------------------
# Main tracker class
# ---------------------------------------------------------------------------


class CameraTracker:
    """Real-time respiratory tracker using webcam luminance changes.

    The tracker runs a background thread that:
    1. Captures frames at *fps_target* from the default webcam.
    2. Converts each frame to grayscale and extracts a chest-region-of-interest
       (ROI) defined by fractional coordinates of the frame dimensions.
    3. Computes the mean pixel value of the ROI — a proxy for chest displacement.
    4. Appends the value to a circular deque of length ``buffer_seconds * fps_target``.
    5. Periodically detrends the rolling luminance trace. Bandpass filtering is
       applied once later by ``ResonanceEstimator`` after timestamp alignment.

    Parameters
    ----------
    roi_y_frac : tuple[float, float]
        Vertical fraction (top, bottom) of the frame defining the ROI.
        Default ``(0.3, 0.7)`` typically covers the chest for a user sitting
        upright in front of a laptop camera.
    roi_x_frac : tuple[float, float]
        Horizontal fraction (left, right) of the frame defining the ROI.
        Default ``(0.2, 0.8)`` excludes the arms/background edges.
    fps_target : int
        Desired capture rate in frames per second.  The actual rate is capped
        by the hardware and OS scheduler.
    buffer_seconds : int
        Duration of the rolling circular buffer in seconds.  Longer buffers
        yield more stable frequency estimates but increase memory use slightly.

    Thread Safety
    -------------
    ``get_signal()`` and ``get_frame_annotated()`` acquire a lightweight lock
    before reading shared state, so they are safe to call from any thread.

    Example
    -------
    >>> tracker = CameraTracker()
    >>> tracker.start()
    >>> time.sleep(10)
    >>> sig = tracker.get_signal()
    >>> print(sig.summary())
    >>> tracker.stop()
    """

    def __init__(
        self,
        camera_source: Union[int, str] = 0,
        roi_y_frac: Tuple[float, float] = (0.40, 0.85),
        roi_x_frac: Tuple[float, float] = (0.25, 0.75),
        fps_target: int = 30,
        buffer_seconds: int = 30,
    ) -> None:
        self.camera_source: Union[int, str] = camera_source
        # --- ROI parameters ---------------------------------------------------
        self._roi_y_frac = roi_y_frac
        self._roi_x_frac = roi_x_frac

        # --- Timing -----------------------------------------------------------
        self._fps_target: int = fps_target
        self._frame_interval: float = 1.0 / fps_target

        # --- Circular buffers -------------------------------------------------
        maxlen = buffer_seconds * fps_target
        self._raw_buffer: Deque[float] = deque(maxlen=maxlen)
        self._ts_buffer: Deque[float] = deque(maxlen=maxlen)
        self._filtered_buffer: Deque[float] = deque(maxlen=maxlen)

        # Minimum samples before detrending the rolling trace (0.5 s × fps)
        self._min_filter_samples: int = max(10, int(0.5 * fps_target))

        # --- Shared state protected by a lock ---------------------------------
        self._lock = threading.Lock()
        self._current_frame: Optional[np.ndarray] = None  # latest BGR frame
        self._last_frame_monotonic: Optional[float] = None
        self._effective_fps: float = float(fps_target)
        self._capture_target: Optional[Union[int, str]] = None

        # --- Thread control ---------------------------------------------------
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None  # type: ignore[name-defined]

        logger.debug(
            "CameraTracker initialised | ROI y=%s x=%s fps=%d buffer=%ds",
            roi_y_frac,
            roi_x_frac,
            fps_target,
            buffer_seconds,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True if the capture thread is alive and the stop event is not set."""
        return (
            self._thread is not None
            and self._thread.is_alive()
            and not self._stop_event.is_set()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, camera_source: Optional[Union[int, str]] = None) -> None:
        """Open the webcam or network video stream and start the background capture thread.

        Parameters
        ----------
        camera_source : int or str, optional
            Device index (e.g. 0, 1) or stream URL (e.g. 'http://192.168.1.X:8080/video', 'rtsp://...').
            If not provided, uses the camera_source specified at initialisation.

        Raises
        ------
        RuntimeError
            If the video source cannot be opened or is already running.
        """
        if self.is_running:
            raise RuntimeError("CameraTracker is already running.")

        if camera_source is not None:
            self.camera_source = camera_source

        # Parse camera source: integer index vs remote network stream / URL
        target_src: Union[int, str]
        if isinstance(self.camera_source, str):
            clean_src = self.camera_source.strip()
            if clean_src.isdigit():
                target_src = int(clean_src)
            else:
                target_src = clean_src
        else:
            target_src = int(self.camera_source)

        self._cap = cv2.VideoCapture(target_src)
        if not self._cap.isOpened():
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
            src_desc = f"device index {target_src}" if isinstance(target_src, int) else f"network stream '{target_src}'"
            raise RuntimeError(
                f"Cannot open video source ({src_desc}). "
                "Ensure the camera is connected or network stream is online."
            )

        # Hint the driver to deliver frames at the target rate.
        # Note: not all cameras/drivers honour CAP_PROP_FPS.
        self._cap.set(cv2.CAP_PROP_FPS, self._fps_target)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Retrieve actual camera fps for the timestamped signal metadata
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if actual_fps and actual_fps > 0:
            self._effective_fps = actual_fps
        else:
            self._effective_fps = float(self._fps_target)
        self._capture_target = target_src

        # Clear any stale data
        with self._lock:
            self._raw_buffer.clear()
            self._ts_buffer.clear()
            self._filtered_buffer.clear()
            self._current_frame = None
            self._last_frame_monotonic = None
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CameraTracker-capture",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "CameraTracker started | effective fps=%.1f", self._effective_fps
        )

    def stop(self) -> None:
        """Signal the background thread to stop and release the camera.

        Blocks until the thread exits (up to 3 seconds).
        """
        if not self._thread:
            return

        self._stop_event.set()
        self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            logger.warning("Camera read did not stop promptly; releasing capture to unblock it.")
            with self._lock:
                capture = self._cap
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
            self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            logger.error("Capture thread remains active after camera release; refusing unsafe restart.")
            return

        with self._lock:
            if self._cap:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
        self._thread = None

        logger.info("CameraTracker stopped.")

    def get_signal(self, n_samples: Optional[int] = None) -> RespiratorySignal:
        """Return a snapshot of the recent respiratory signal.

        Parameters
        ----------
        n_samples : int, optional
            If given, only the last *n_samples* samples are returned.
            Defaults to the full buffer contents.

        Returns
        -------
        RespiratorySignal
            Dataclass containing the raw and detrended signals, timestamps,
            and effective sampling rate.
        """
        with self._lock:
            raw = np.array(self._raw_buffer, dtype=np.float64)
            ts = np.array(self._ts_buffer, dtype=np.float64)
            filt = np.array(self._filtered_buffer, dtype=np.float64)
            fs = self._effective_fps

        # Pad the detrended array with NaN if shorter than raw
        if len(filt) < len(raw):
            padding = np.full(len(raw) - len(filt), np.nan)
            filt = np.concatenate([padding, filt])

        if n_samples is not None:
            raw = raw[-n_samples:]
            ts = ts[-n_samples:]
            filt = filt[-n_samples:]

        return RespiratorySignal(
            raw=raw,
            filtered=filt,
            timestamps=ts,
            sampling_rate=fs,
        )

    @property
    def last_frame_age_s(self) -> float:
        """Seconds since the latest decoded frame, or infinity before the first frame."""
        with self._lock:
            last_frame = self._last_frame_monotonic
        if last_frame is None:
            return float("inf")
        return max(0.0, time.monotonic() - last_frame)

    def get_frame_annotated(self, max_age_s: Optional[float] = None) -> Optional[np.ndarray]:
        """Return the latest captured BGR frame with annotations overlaid.

        Annotations include:
        - A green rectangle showing the chest ROI.
        - Overlay text "NeuroResonance | Respiratory Tracking".
        - Current buffer fill level (number of samples collected).

        Returns
        -------
        np.ndarray or None
            Annotated BGR frame, or None if no frame has been captured yet.
        """
        with self._lock:
            if self._current_frame is None:
                return None
            if (
                max_age_s is not None
                and self._last_frame_monotonic is not None
                and time.monotonic() - self._last_frame_monotonic > max_age_s
            ):
                return None
            frame = self._current_frame.copy()
            n_samples = len(self._raw_buffer)

        h, w = frame.shape[:2]

        # --- Draw ROI rectangle & targeting aids ---
        y0 = int(self._roi_y_frac[0] * h)
        y1 = int(self._roi_y_frac[1] * h)
        x0 = int(self._roi_x_frac[0] * w)
        x1 = int(self._roi_x_frac[1] * w)

        cv2.rectangle(frame, (x0, y0), (x1, y1), color=(0, 255, 136), thickness=2)
        cv2.putText(
            frame,
            "CHEST ROI | WYCENTRUJ KLATKE PIERSIOWA",
            (x0 + 8, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.42,
            color=(0, 255, 136),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        # --- Title overlay ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame,
            "NeuroResonance | Respiratory Motion Tracker",
            (10, 25),
            font,
            fontScale=0.6,
            color=(0, 255, 136),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        # --- User guidance & buffer fill overlay ---
        cv2.putText(
            frame,
            "Odleglosc: 50-70 cm | Oprzyj plecy i oddychaj torem klatkowo-brzusznym",
            (10, h - 28),
            font,
            fontScale=0.42,
            color=(180, 220, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        src_info = f"Source: {self.camera_source}" if self.camera_source != 0 else "Source: Webcam (Device 0)"
        cv2.putText(
            frame,
            f"Buffer: {n_samples} frames | {src_info}",
            (10, h - 10),
            font,
            fontScale=0.42,
            color=(160, 160, 160),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        return frame

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_bandpass(self, signal: np.ndarray) -> np.ndarray:
        """Return a linearly detrended luminance trace without bandpass filtering.

        Parameters
        ----------
        signal : np.ndarray
            1-D array of raw luminance values.

        Returns
        -------
        np.ndarray
            Detrended signal of the same length. The legacy method name is
            retained for compatibility; filtering is performed once by the
            downstream ``ResonanceEstimator``.
        """
        values = np.asarray(signal, dtype=float).ravel()
        if values.size < 2:
            return values.copy()
        return detrend(values, type="linear")

    def _extract_roi_mean(self, gray: np.ndarray) -> float:
        """Extract the mean pixel value from the chest ROI of a grayscale frame.

        The mean is an uncalibrated image-intensity motion proxy. Its variation
        can reflect breathing, but can also reflect lighting, framing, camera
        motion, or unrelated movement.

        Parameters
        ----------
        gray : np.ndarray
            Grayscale (H × W) frame.

        Returns
        -------
        float
            Mean pixel intensity in the ROI (range 0–255).
        """
        h, w = gray.shape
        y0 = int(self._roi_y_frac[0] * h)
        y1 = int(self._roi_y_frac[1] * h)
        x0 = int(self._roi_x_frac[0] * w)
        x1 = int(self._roi_x_frac[1] * w)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            return 0.0
        return float(roi.mean())

    def _reopen_capture(self) -> bool:
        """Replace a failed video capture handle while preserving the tracker thread."""
        target = self._capture_target
        if target is None or self._stop_event.is_set():
            return False

        current = self._cap
        if current is not None:
            try:
                current.release()
            except Exception:
                pass

        try:
            replacement = cv2.VideoCapture(target)
            if not replacement.isOpened():
                replacement.release()
                self._cap = None
                return False
            replacement.set(cv2.CAP_PROP_FPS, self._fps_target)
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                replacement.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            actual_fps = replacement.get(cv2.CAP_PROP_FPS)
            with self._lock:
                if self._stop_event.is_set():
                    replacement.release()
                    return False
                self._cap = replacement
                if actual_fps and actual_fps > 0:
                    self._effective_fps = float(actual_fps)
            logger.info("Camera capture recovered for source %r", target)
            return True
        except Exception as exc:
            self._cap = None
            logger.warning("Could not reopen camera source %r: %s", target, exc)
            return False

    def _capture_loop(self) -> None:
        """Main background thread: capture frames and update buffers.

        The loop:
        1. Reads a frame from the OpenCV VideoCapture.
        2. Converts to grayscale and computes ROI mean luminance.
        3. Appends to the raw circular buffer with a timestamp.
        4. Every 0.05 s, removes a linear trend from the rolling luminance
           buffer and refreshes ``_filtered_buffer``. Bandpass filtering is
           performed once downstream by ``ResonanceEstimator``.
        5. Stores the BGR frame for ``get_frame_annotated()``.
        6. Sleeps to maintain the target frame rate.

        The loop exits when ``_stop_event`` is set.
        """
        frame_count = 0
        consecutive_failures = 0
        reconnect_delay_s = 1.0
        next_reconnect_at = 0.0
        t_last_filter = time.perf_counter()
        # Refresh the detrended rolling trace without introducing filter delay.
        filter_interval_s = 0.05

        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            capture = self._cap
            if capture is None:
                ret, bgr_frame = False, None
            else:
                try:
                    ret, bgr_frame = capture.read()
                except Exception as exc:
                    logger.debug("Camera read failed: %s", exc)
                    ret, bgr_frame = False, None
            if not ret or bgr_frame is None:
                consecutive_failures += 1
                now = time.perf_counter()
                if consecutive_failures >= 15 and now >= next_reconnect_at:
                    logger.warning(
                        "Camera frame stream stalled (%d failed reads); reconnecting.",
                        consecutive_failures,
                    )
                    recovered = self._reopen_capture()
                    next_reconnect_at = now + reconnect_delay_s
                    if recovered:
                        consecutive_failures = 0
                        reconnect_delay_s = 1.0
                    else:
                        reconnect_delay_s = min(8.0, reconnect_delay_s * 2.0)
                time.sleep(self._frame_interval)
                continue

            consecutive_failures = 0
            reconnect_delay_s = 1.0

            ts = time.time()

            # Convert to grayscale and extract ROI mean
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            roi_mean = self._extract_roi_mean(gray)

            # --- Update raw buffer (thread-safe) ------------------------------
            with self._lock:
                self._raw_buffer.append(roi_mean)
                self._ts_buffer.append(ts)
                self._current_frame = bgr_frame
                self._last_frame_monotonic = time.monotonic()
                frame_count += 1

            # --- Refresh detrended signal periodically -----------------------
            now = time.perf_counter()
            if (now - t_last_filter) >= filter_interval_s:
                with self._lock:
                    raw_snap = np.array(self._raw_buffer, dtype=np.float64)

                if len(raw_snap) >= self._min_filter_samples:
                    filtered = self._apply_bandpass(raw_snap)
                    with self._lock:
                        self._filtered_buffer.clear()
                        self._filtered_buffer.extend(filtered.tolist())
                    t_last_filter = now

            # --- Throttle to target FPS ---------------------------------------
            elapsed = time.perf_counter() - loop_start
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.debug("Capture loop exited after %d frames.", frame_count)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    COLLECT_SECONDS = 10

    print("=" * 60)
    print("NeuroResonance — Camera Respiratory Tracker  (smoke test)")
    print("=" * 60)
    print(f"Opening webcam and collecting {COLLECT_SECONDS} seconds of data …")
    print("Please sit still and breathe normally.")
    print()

    tracker = CameraTracker(fps_target=30, buffer_seconds=30)

    try:
        tracker.start()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    # Countdown with live frame preview (press 'q' to quit early)
    for remaining in range(COLLECT_SECONDS, 0, -1):
        print(f"  Collecting … {remaining:2d}s remaining", end="\r", flush=True)

        frame = tracker.get_frame_annotated()
        if frame is not None:
            cv2.imshow("NeuroResonance | Respiratory Tracking", frame)
            # 1 ms wait — allows OpenCV to refresh the window
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[INFO] Interrupted by user.")
                break

        time.sleep(1.0)

    print()

    # --- Retrieve and display signal stats -----------------------------------
    signal = tracker.get_signal()
    print("\n--- Signal Statistics ---")
    print(signal.summary())
    print(f"  Raw signal range : {signal.raw.min():.2f} – {signal.raw.max():.2f}")
    print(f"  Raw signal std   : {signal.raw.std():.4f}")

    if not np.all(np.isnan(signal.filtered)):
        print(
            f"  Detrended range  : "
            f"{np.nanmin(signal.filtered):.4f} – {np.nanmax(signal.filtered):.4f}"
        )
        print(f"  Detrended std    : {np.nanstd(signal.filtered):.6f}")
    else:
        print("  (Detrended signal not yet available — buffer too short.)")

    print(f"\n  Estimated breathing rate: {signal.breaths_per_minute():.1f} breaths/min")

    tracker.stop()
    cv2.destroyAllWindows()

    print("\nDone.")
