"""
lsl_streamer.py — fNIRS Lab Streaming Layer simulator and hardware bridge.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BACKGROUND: fNIRS Physiology
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Functional near-infrared spectroscopy (fNIRS) estimates hemodynamic changes
from light absorption measured at selected wavelengths. This module generates
synthetic example signals; it does not model a device's optical measurement path.

    ΔOD_λ = ε_HbO2(λ) · Δ[HbO2] · L + ε_HbR(λ) · Δ[HbR] · L

The simulator does not calculate these quantities from raw optical data.

NEUROVASCULAR COUPLING (NVC)
─────────────────────────────
Task-evoked hemodynamic responses can involve changes in HbO₂ and HbR, but
their timing and magnitude vary with task, participant, and measurement setup.
An HbO₂/HbR correlation alone is not evidence of NVC.

MAYER WAVES
───────────
Low-frequency cardiovascular oscillations are often discussed near 0.1 Hz
(about a 10-second period). Similar-frequency components in fNIRS can have
multiple sources; their presence does not identify a cerebral origin.

SYNTHETIC RESPONSE KERNEL
─────────────────────────
This module uses a simple gamma-shaped kernel for software demonstrations.
It is not a validated physiological hemodynamic response function.

PINK NOISE (1/f)
─────────────────
The simulator creates shaped noise in the frequency domain for demonstration.
This does not establish that the resulting noise matches biological data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
import logging
from collections import deque
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import fftconvolve

logger = logging.getLogger("NeuroResonanceSimulator")

from src.lsl_runtime import (
    LSL_AVAILABLE as _LSL_AVAILABLE,
    StreamInfo,
    StreamInlet,
    StreamOutlet,
    local_clock,
    lsl_unavailable_message,
    resolve_byprop,
    resolve_stream,
)

if not _LSL_AVAILABLE:  # pragma: no cover - platform dependent
    print(
        f"[lsl_streamer] WARNING: {lsl_unavailable_message()} "
        "LSL push/pull disabled; the simulator will still generate synthetic signals internally."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _gamma_hrf(duration_s: float, sampling_rate: float,
               alpha: float = 6.0, beta: float = 1.0) -> np.ndarray:
    """
    Generate a normalized gamma-shaped kernel for synthetic examples.

    This kernel is not a validated or participant-specific fNIRS response model.

    Parameters
    ----------
    duration_s : float
        Length of the HRF kernel in seconds.
    sampling_rate : float
        Sampling rate in Hz.
    alpha : float
        Shape parameter α of the gamma distribution. Higher α shifts the peak
        later and broadens the response (default 6.0 → peak ≈ 6 s).
    beta : float
        Scale parameter β (inverse rate) in seconds (default 1.0 s).

    Returns
    -------
    hrf : np.ndarray
        Normalised HRF kernel, shape (n_samples,), peak amplitude = 1.
    """
    t = np.arange(0, duration_s, 1.0 / sampling_rate)
    # Gamma-variate: avoid t=0 divide by zero
    t_safe = np.where(t > 0, t, 1e-10)
    hrf = (t_safe ** alpha) * np.exp(-t_safe / beta)
    hrf /= hrf.max() + 1e-12  # normalise to unit peak
    return hrf


def _pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate pink (1/f) noise via the spectral (Voss-McCartney) method.

    Pink noise has a power spectral density P(f) ∝ 1/f, meaning lower
    frequencies carry more energy — a hallmark of biological time series
    including EEG, heart rate variability, and fNIRS.

    Implementation
    ──────────────
    1. Draw i.i.d. white noise samples in the time domain.
    2. FFT → complex frequency-domain representation X(f).
    3. Multiply |X(f)| by 1/√f (shaping the amplitude spectrum).
    4. IFFT → pink noise time series.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    signal : np.ndarray
        Zero-mean pink noise, unit standard deviation, shape (n_samples,).
    """
    white = rng.standard_normal(n_samples)
    fft_white = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    # Avoid divide-by-zero at DC (f=0); set DC to 0 (zero-mean signal)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaling = np.where(freqs > 0, 1.0 / np.sqrt(freqs), 0.0)
    fft_pink = fft_white * scaling
    pink = np.fft.irfft(fft_pink, n=n_samples)
    # Normalise to unit std
    std = pink.std()
    if std > 1e-12:
        pink /= std
    return pink


# ──────────────────────────────────────────────────────────────────────────────
# fNIRS Simulator
# ──────────────────────────────────────────────────────────────────────────────

class fNIRSSimulator:
    """
    Synthetic fNIRS-like signal generator with optional Lab Streaming Layer output.

    Generates example HbO₂/HbR time series from fixed mathematical
    components. These values are not physiological measurements or a validated
    model of resting-state signals, task responses, or NVC:

      • A low-frequency sine at the configured demonstration frequency.
      • A gamma-shaped kernel applied to synthetic random events.
      • Shaped noise and a constructed negative HbO₂/HbR relationship.

    Synthetic signal model (arbitrary units)
    ────────────────────────────────────────
        mayer(t)   = A_mayer · sin(2π · f_mayer · t + φ_mayer)
        hrf_comp   = A_hrf · convolve(neural_events, HRF)
        noise(t)   = σ_noise · pink_noise(t)
        HbO2(t)    = mayer(t) + hrf_comp(t) + noise(t)
        HbR(t)     = −0.6 · HbO2(t) + ε(t)   where ε is synthetic noise

    LSL stream
    ──────────
    Stream name : "NeuroResonance_fNIRS"
    Stream type : "fNIRS-SIM"
    Channels    : ["HbO2", "HbR"]  (or as configured)
    Units       : arbitrary synthetic units

    Parameters
    ----------
    sampling_rate : float
        Sampling rate in Hz (default 10.0 Hz for this software demonstration).
    channels : list[str]
        Channel names (default ["HbO2", "HbR"]).
    mayer_wave_freq : float
        Frequency of Mayer waves in Hz (default 0.1 Hz).
    neural_delay_s : float
        Deprecated compatibility parameter. The synthetic generator does not
        apply a physiological neural-to-hemodynamic delay.

    Examples
    --------
    >>> sim = fNIRSSimulator(sampling_rate=10.0)
    >>> sim.start()
    >>> time.sleep(5)
    >>> data = sim.get_latest(50)
    >>> print(data['HbO2'].mean())
    >>> sim.stop()
    """

    STREAM_NAME = "fNIRSSimulator_Mayer"
    STREAM_TYPE = "fNIRS-SIM"

    # Arbitrary synthetic amplitude scales; not calibrated concentrations.
    A_MAYER: float = 2.0        # Mayer wave amplitude
    A_HRF: float = 3.5          # Peak HRF amplitude for HbO2
    NOISE_SIGMA: float = 0.3    # Pink noise scaling factor
    HBR_ANTICORR: float = -0.6  # Chosen synthetic scaling, not an NVC parameter
    HBR_NOISE_SD: float = 0.3   # Additional synthetic HbR noise

    def __init__(
        self,
        sampling_rate: float = 10.0,
        channels: List[str] | None = None,
        mayer_wave_freq: float = 0.1,
        neural_delay_s: float = 4.0,
        seed: Optional[int] = None,
    ) -> None:
        self.sampling_rate = float(sampling_rate)
        self.channels = channels if channels is not None else ["HbO2", "HbR"]
        self.mayer_wave_freq = float(mayer_wave_freq)
        self.neural_delay_s = float(neural_delay_s)

        self._rng = np.random.default_rng(seed=seed)
        self._lock = threading.Lock()

        # Ring buffers: store last 60 s worth of samples
        self._buffer_size = int(self.sampling_rate * 60)
        self._hbo2_buf: deque[float] = deque(maxlen=self._buffer_size)
        self._hbr_buf: deque[float] = deque(maxlen=self._buffer_size)
        self._ts_buf: deque[float] = deque(maxlen=self._buffer_size)
        self._total_generated = 0
        self._new_sample_cursor = 0

        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._worker_error: Optional[str] = None
        self._worker_traceback: Optional[str] = None

        # Pre-compute HRF kernel (30 s window)
        self._hrf_kernel = _gamma_hrf(
            duration_s=30.0,
            sampling_rate=self.sampling_rate,
            alpha=6.0,
            beta=1.0,
        )

        # History buffer for continuous block-to-block HRF convolution
        self._event_history: np.ndarray = np.zeros(len(self._hrf_kernel))

        # Random Mayer-wave phase offset (each run is unique)
        self._mayer_phase = self._rng.uniform(0, 2 * np.pi)

        # LSL outlet (created in start())
        self._outlet: Optional["StreamOutlet"] = None

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    @property
    def worker_error(self) -> Optional[str]:
        """Return a background simulation failure, if one occurred."""
        with self._lock:
            return self._worker_error

    @property
    def worker_traceback(self) -> Optional[str]:
        """Return the original background failure stack for diagnostics."""
        with self._lock:
            return self._worker_traceback

    @property
    def is_running(self) -> bool:
        """Whether the sample-generation worker is currently alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        """
        Start the background simulation thread and LSL outlet.

        The thread generates samples at ``sampling_rate`` Hz and both
        buffers them internally (accessible via :meth:`get_latest`) and
        pushes them to the LSL stream.

        Calling ``start()`` on an already-running simulator is a no-op.
        """
        if self._thread is not None and self._thread.is_alive():
            return  # Already running

        self._stop_event.clear()
        with self._lock:
            self._worker_error = None
            self._worker_traceback = None
        self._outlet = self._create_outlet()

        self._thread = threading.Thread(
            target=self._run_loop,
            name="fNIRS-Sim",
            daemon=True,
        )
        self._thread.start()
        print(f"[fNIRSSimulator] Started — LSL stream '{self.STREAM_NAME}' "
              f"@ {self.sampling_rate} Hz")

    def stop(self) -> None:
        """
        Stop the simulation thread and close the LSL outlet gracefully.

        Blocks until the background thread terminates (up to 3 s).
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._outlet = None
        print("[fNIRSSimulator] Stopped.")

    def get_latest(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        Return the most recently generated samples from the internal buffer.

        Parameters
        ----------
        n_samples : int
            Number of samples to retrieve. If fewer samples are available
            (e.g., shortly after :meth:`start`), all available samples are
            returned.

        Returns
        -------
        dict with keys:
            ``'HbO2'``       — np.ndarray of synthetic HbO2-like values
            ``'HbR'``        — np.ndarray of synthetic HbR-like values
            ``'timestamps'`` — np.ndarray of LSL timestamps (seconds)
        """
        with self._lock:
            hbo2 = np.array(list(self._hbo2_buf)[-n_samples:], dtype=np.float64)
            hbr  = np.array(list(self._hbr_buf)[-n_samples:],  dtype=np.float64)
            ts   = np.array(list(self._ts_buf)[-n_samples:],   dtype=np.float64)
        return {"HbO2": hbo2, "HbR": hbr, "timestamps": ts}

    def get_new_samples(self) -> Dict[str, np.ndarray]:
        """Return every not-yet-consumed sample, including any detected loss."""
        with self._lock:
            first_retained = max(0, self._total_generated - len(self._ts_buf))
            start = max(self._new_sample_cursor, first_retained)
            dropped = max(0, first_retained - self._new_sample_cursor)
            offset = start - first_retained
            hbo2 = np.asarray(list(self._hbo2_buf)[offset:], dtype=np.float64)
            hbr = np.asarray(list(self._hbr_buf)[offset:], dtype=np.float64)
            timestamps = np.asarray(list(self._ts_buf)[offset:], dtype=np.float64)
            sample_indices = np.arange(start, self._total_generated, dtype=np.int64)
            self._new_sample_cursor = self._total_generated
        return {
            "HbO2": hbo2,
            "HbR": hbr,
            "timestamps": timestamps,
            "sample_indices": sample_indices,
            "dropped_samples": dropped,
        }

    # ------------------------------------------------------------------ #
    # Internal machinery                                                   #
    # ------------------------------------------------------------------ #

    def _create_outlet(self) -> Optional["StreamOutlet"]:
        """Create and return an LSL StreamOutlet, or None if pylsl unavailable."""
        if not _LSL_AVAILABLE:
            return None
        n_ch = len(self.channels)
        info = StreamInfo(
            name=self.STREAM_NAME,
            type=self.STREAM_TYPE,
            channel_count=n_ch,
            nominal_srate=self.sampling_rate,
            channel_format="float32",
            source_id=str(uuid.uuid4()),
        )
        # Annotate channel metadata
        chns = info.desc().append_child("channels")
        for ch_name in self.channels:
            ch = chns.append_child("channel")
            ch.append_child_value("label", ch_name)
            ch.append_child_value("unit", "arbitrary")
            ch.append_child_value("type", "synthetic fNIRS-like signal")
        return StreamOutlet(info)

    def _generate_block(self, t_start: float, n: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a block of *n* fNIRS samples starting at time *t_start*.

        Synthetic construction
        ──────────────────────
        1. Sine:        low-frequency component at f_mayer
        2. Kernel:      gamma-shaped response to synthetic random events
        3. Pink noise:  1/f-shaped synthetic noise
        4. HbR:         negative-scaled HbO2 plus synthetic noise

        Returns
        -------
        (hbo2, hbr) : tuple of np.ndarray, shape (n,)
        """
        dt = 1.0 / self.sampling_rate
        t = t_start + np.arange(n) * dt

        # ── 1. Mayer wave ─────────────────────────────────────────────────
        mayer = self.A_MAYER * np.sin(
            2 * np.pi * self.mayer_wave_freq * t + self._mayer_phase
        )

        # ── 2. Synthetic response-kernel component ─────────────────────────
        # Sparse synthetic event process: each sample has a 2% chance of event
        synthetic_events = (self._rng.uniform(size=n) < 0.02).astype(float)

        # Stateful convolution: prepend history to ensure continuous response across blocks
        k_len = len(self._hrf_kernel)
        combined_events = np.concatenate([self._event_history, synthetic_events])
        self._event_history = combined_events[-k_len:]

        # Convolve with HRF kernel
        hrf_conv = fftconvolve(combined_events, self._hrf_kernel, mode="full")
        hrf_slice = hrf_conv[k_len - 1 : k_len - 1 + n]
        if len(hrf_slice) < n:
            hrf_slice = np.pad(hrf_slice, (0, n - len(hrf_slice)))
        hrf_comp = hrf_slice[:n] * self.A_HRF

        # ── 3. Pink (1/f) noise ────────────────────────────────────────────
        noise = _pink_noise(n, self._rng) * self.NOISE_SIGMA

        # ── 4. Combine into HbO2 ──────────────────────────────────────────
        hbo2 = mayer + hrf_comp + noise  # arbitrary synthetic units

        # ── 5. Constructed HbR relation (not an NVC signature) ───────────
        hbr_noise = self._rng.normal(0.0, self.HBR_NOISE_SD, size=n)
        hbr = self.HBR_ANTICORR * hbo2 + hbr_noise  # arbitrary synthetic units

        return hbo2, hbr

    def _run_loop(self) -> None:
        """
        Background thread: continuously generate and push fNIRS samples.

        Runs at the nominal ``sampling_rate``; each iteration generates one
        sample, appends it to the internal ring buffer, and pushes it to the
        LSL outlet (if available).

        The thread exits cleanly when ``_stop_event`` is set.
        """
        try:
            self._run_loop_body()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._worker_error = message
                self._worker_traceback = traceback.format_exc()
            logger.exception("Synthetic signal generator stopped unexpectedly")

    def _run_loop_body(self) -> None:
        """Generate samples until stopped; exceptions are captured by _run_loop."""
        dt = 1.0 / self.sampling_rate
        t_start = time.perf_counter()
        sample_count = 0

        BLOCK = max(int(self.sampling_rate * 5), 50)  # 5 s chunk for HRF & pink noise continuity
        pending_hbo2: deque[float] = deque()
        pending_hbr: deque[float] = deque()

        while not self._stop_event.is_set():
            # Time-align: sleep until next sample is due
            next_tick = t_start + (sample_count + 1) * dt
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

            t_elapsed = sample_count * dt

            # Refill chunk buffer to preserve HRF convolution context and 1/f pink noise spectrum
            if not pending_hbo2:
                hbo2_arr, hbr_arr = self._generate_block(t_elapsed, BLOCK)
                pending_hbo2.extend(hbo2_arr)
                pending_hbr.extend(hbr_arr)

            hbo2_val = float(pending_hbo2.popleft())
            hbr_val  = float(pending_hbr.popleft())

            ts = local_clock() if _LSL_AVAILABLE else time.perf_counter()

            # Push to LSL
            if self._outlet is not None:
                self._outlet.push_sample([hbo2_val, hbr_val], timestamp=ts)

            # Buffer
            with self._lock:
                self._hbo2_buf.append(hbo2_val)
                self._hbr_buf.append(hbr_val)
                self._ts_buf.append(ts)
                self._total_generated += 1

            sample_count += 1


# ──────────────────────────────────────────────────────────────────────────────
# LSL Bridge (real hardware or simulator)
# ──────────────────────────────────────────────────────────────────────────────

class LSLBridge:
    """
    LSL inlet bridge that connects to either a real fNIRS device or the
    :class:`fNIRSSimulator` stream transparently.

    Discovery logic
    ───────────────
    If ``stream_name`` is provided, the bridge resolves a stream with that
    exact name via ``pylsl.resolve_byprop``.  If ``stream_name`` is ``None``,
    it performs a broad search for any stream of type ``"fNIRS"`` via
    ``pylsl.resolve_stream`` and connects to the first match.

    Parameters
    ----------
    stream_name : str or None
        LSL stream name to connect to.  ``None`` discovers any fNIRS stream.
    timeout : float
        Maximum seconds to wait for stream resolution (default 5.0 s).

    Examples
    --------
    >>> bridge = LSLBridge(stream_name="NeuroResonance_fNIRS")
    >>> if bridge.connect():
    ...     chunk = bridge.pull_chunk(max_samples=50)
    ...     print(chunk['HbO2'])
    """

    def __init__(
        self,
        stream_name: Optional[str] = None,
        timeout: float = 3.0,
        auto_reconnect: bool = True,
        reconnect_interval_s: float = 2.0,
        fnirs_stream_name: Optional[str] = None,
        respiration_stream_name: Optional[str] = None,
        respiration_stream_type: Optional[str] = None,
    ) -> None:
        self.fnirs_stream_name = fnirs_stream_name or stream_name
        self.stream_name = self.fnirs_stream_name  # Legacy public attribute.
        self.respiration_stream_name = respiration_stream_name
        self.respiration_stream_type = respiration_stream_type
        self._dual_stream = bool(respiration_stream_name or respiration_stream_type)
        self.timeout = float(timeout)
        self.auto_reconnect = auto_reconnect
        self.reconnect_interval_s = float(reconnect_interval_s)

        self._inlet: Optional["StreamInlet"] = None
        self._respiration_inlet: Optional["StreamInlet"] = None
        self._channel_names: List[str] = []
        self._respiration_channel_names: List[str] = []
        self._connected = False
        self._lock = threading.Lock()
        self._pending_fnirs_values = np.empty((0, 2), dtype=np.float64)
        self._pending_fnirs_timestamps = np.empty(0, dtype=np.float64)
        self._pending_resp_values = np.empty(0, dtype=np.float64)
        self._pending_resp_timestamps = np.empty(0, dtype=np.float64)

        # Telemetry & Resilience Stats
        self.stats = {
            "total_samples": 0,
            "disconnect_count": 0,
            "reconnect_count": 0,
            "dropped_samples": 0,
            "last_sample_time": 0.0,
            "jitter_ms": 0.0,
        }

        # Auto-reconnection watchdog
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_stop = threading.Event()
        if self.auto_reconnect and _LSL_AVAILABLE:
            self._start_watchdog()

    def _start_watchdog(self) -> None:
        """Background thread monitoring hardware connectivity and auto-reconnecting."""
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="LSL-Watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.is_set():
            if not self.is_connected:
                reconnected = self.connect(silent=True)
                if reconnected:
                    with self._lock:
                        self.stats["reconnect_count"] += 1
            self._watchdog_stop.wait(self.reconnect_interval_s)

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    @property
    def is_connected(self) -> bool:
        """``True`` if an LSL inlet is currently open and ready."""
        with self._lock:
            return (
                self._connected
                and self._inlet is not None
                and (not self._dual_stream or self._respiration_inlet is not None)
            )

    def connect(self, stream_type: str = "fNIRS", silent: bool = False) -> bool:
        """
        Resolve the LSL stream and open an inlet.
        """
        if not _LSL_AVAILABLE:
            if not silent:
                print(f"[LSLBridge] Cannot connect: {lsl_unavailable_message()}")
            return False
        if silent and self._watchdog_stop.is_set():
            return False

        if not silent:
            print(f"[LSLBridge] Searching for stream "
                  f"{'name=' + repr(self.stream_name) if self.stream_name else 'type=' + repr(stream_type)} "
                  f"(timeout={self.timeout} s)…")
        try:
            fnirs_streams = (
                resolve_byprop("name", self.fnirs_stream_name, timeout=self.timeout)
                if self.fnirs_stream_name
                else resolve_stream("type", stream_type, timeout=self.timeout)
            )
            if not fnirs_streams:
                if not silent:
                    print("[LSLBridge] No fNIRS stream found.")
                with self._lock:
                    self._connected = False
                return False

            respiration_streams = None
            if self._dual_stream:
                respiration_streams = (
                    resolve_byprop("name", self.respiration_stream_name, timeout=self.timeout)
                    if self.respiration_stream_name
                    else resolve_stream("type", self.respiration_stream_type or "Respiration", timeout=self.timeout)
                )
                if not respiration_streams:
                    if not silent:
                        print("[LSLBridge] No respiration stream found.")
                    with self._lock:
                        self._connected = False
                    return False

            info = fnirs_streams[0]
            with self._lock:
                if silent and self._watchdog_stop.is_set():
                    return False
                if self._inlet is not None:
                    try:
                        self._inlet.close_stream()
                    except Exception:
                        pass
                if self._respiration_inlet is not None:
                    try:
                        self._respiration_inlet.close_stream()
                    except Exception:
                        pass
                self._inlet = StreamInlet(info, max_buflen=360)
                self._channel_names = self._parse_channel_names(info)
                self._respiration_inlet = None
                self._respiration_channel_names = []
                if respiration_streams:
                    resp_info = respiration_streams[0]
                    self._respiration_inlet = StreamInlet(resp_info, max_buflen=360)
                    self._respiration_channel_names = self._parse_channel_names(resp_info)
                self._pending_fnirs_values = np.empty((0, 2), dtype=np.float64)
                self._pending_fnirs_timestamps = np.empty(0, dtype=np.float64)
                self._pending_resp_values = np.empty(0, dtype=np.float64)
                self._pending_resp_timestamps = np.empty(0, dtype=np.float64)
                self._connected = True

            if not silent:
                print(
                    f"[LSLBridge] Connected to '{info.name()}' "
                    f"({info.channel_count()} ch @ {info.nominal_srate()} Hz). "
                    f"Channels: {self._channel_names}"
                )
                if respiration_streams and not silent:
                    resp_info = respiration_streams[0]
                    print(
                        f"[LSLBridge] Respiration stream: '{resp_info.name()}' "
                        f"({resp_info.channel_count()} ch @ {resp_info.nominal_srate()} Hz). "
                        f"Channels: {self._respiration_channel_names}"
                    )
            return True

        except Exception as exc:  # noqa: BLE001
            if not silent:
                print(f"[LSLBridge] Connection error: {exc}")
            with self._lock:
                self._connected = False
                self.stats["disconnect_count"] += 1
            return False

    def pull_chunk(self, max_samples: int = 100) -> Dict[str, np.ndarray]:
        """Pull a chunk of samples with telemetry tracking and error isolation."""
        if self._dual_stream:
            return self._pull_dual_chunk(max_samples)

        empty: Dict[str, np.ndarray] = {
            "HbO2": np.empty(0),
            "HbR": np.empty(0),
            "Respiration": np.empty(0),
            "timestamps": np.empty(0),
        }

        with self._lock:
            if not self._connected or self._inlet is None:
                return empty
            inlet = self._inlet

        try:
            samples, timestamps = inlet.pull_chunk(
                timeout=0.0, max_samples=max_samples
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[LSLBridge] Stream drop detected: {exc}")
            with self._lock:
                self._connected = False
                self.stats["disconnect_count"] += 1
            return empty

        if not samples:
            return empty

        data = np.array(samples, dtype=np.float64)
        ts   = np.array(timestamps, dtype=np.float64)

        now = time.time()
        with self._lock:
            n_samples = len(ts)
            self.stats["total_samples"] += n_samples
            if self.stats["last_sample_time"] > 0:
                expected_dt = 1.0 / 10.0
                actual_dt = now - self.stats["last_sample_time"]
                self.stats["jitter_ms"] = abs(actual_dt - expected_dt) * 1000.0
            self.stats["last_sample_time"] = now

        # Identify HbO2 and HbR by parsed channel label metadata if available
        hbo_idx = 0
        hbr_idx = 1 if data.shape[1] > 1 else 0
        resp_idx = None
        with self._lock:
            names = ["".join(ch.lower().split()).replace("_", "") for ch in self._channel_names]
            # Match deoxy first and exclude it from oxy matching ("deoxy" contains "oxy").
            deoxy_tokens = ("hbr", "hhb", "deoxy")
            oxy_tokens = ("hbo2", "hbo", "o2hb", "oxy")
            hbr_candidates = [i for i, name in enumerate(names) if any(t in name for t in deoxy_tokens)]
            hbo_candidates = [i for i, name in enumerate(names) if any(t in name for t in oxy_tokens) and not any(t in name for t in deoxy_tokens)]
            if hbo_candidates:
                hbo_idx = hbo_candidates[0]
            if hbr_candidates:
                hbr_idx = hbr_candidates[0]
            resp_tokens = ("respiration", "resp", "breathing", "breath", "thorax", "belt")
            resp_candidates = [
                i for i, name in enumerate(names)
                if "rate" not in name and any(t in name for t in resp_tokens)
            ]
            if resp_candidates:
                resp_idx = resp_candidates[0]

        hbo2 = data[:, hbo_idx] if data.shape[1] > hbo_idx else np.empty(0)
        hbr  = data[:, hbr_idx] if data.shape[1] > hbr_idx else np.empty(0)
        respiration = data[:, resp_idx] if resp_idx is not None and data.shape[1] > resp_idx else np.empty(0)

        return {"HbO2": hbo2, "HbR": hbr, "Respiration": respiration, "timestamps": ts}

    def _pull_dual_chunk(self, max_samples: int) -> Dict[str, np.ndarray]:
        """Pull independent fNIRS/respiration streams and align by LSL time."""
        empty = {
            "HbO2": np.empty(0),
            "HbR": np.empty(0),
            "Respiration": np.empty(0),
            "timestamps": np.empty(0),
        }
        with self._lock:
            fnirs_inlet = self._inlet
            resp_inlet = self._respiration_inlet
            fnirs_names = list(self._channel_names)
            resp_names = list(self._respiration_channel_names)
            if not self._connected or fnirs_inlet is None or resp_inlet is None:
                return empty

        try:
            fnirs_samples, fnirs_timestamps = fnirs_inlet.pull_chunk(
                timeout=0.0, max_samples=max_samples
            )
            resp_samples, resp_timestamps = resp_inlet.pull_chunk(
                timeout=0.0, max_samples=max_samples
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[LSLBridge] Dual-stream drop detected: {exc}")
            with self._lock:
                self._connected = False
                self.stats["disconnect_count"] += 1
            return empty

        fnirs_ts = np.asarray(fnirs_timestamps, dtype=np.float64).ravel()
        resp_ts = np.asarray(resp_timestamps, dtype=np.float64).ravel()
        fnirs_data = np.asarray(fnirs_samples, dtype=np.float64)
        resp_data = np.asarray(resp_samples, dtype=np.float64)
        if fnirs_data.size:
            fnirs_data = np.atleast_2d(fnirs_data)
            if fnirs_data.shape[0] != fnirs_ts.size:
                fnirs_data = fnirs_data.reshape(fnirs_ts.size, -1)
        else:
            fnirs_data = np.empty((0, 0), dtype=np.float64)
        if resp_data.size:
            resp_data = np.atleast_2d(resp_data)
            if resp_data.shape[0] != resp_ts.size:
                resp_data = resp_data.reshape(resp_ts.size, -1)
        else:
            resp_data = np.empty((0, 0), dtype=np.float64)

        if fnirs_data.size:
            normalized = ["".join(ch.lower().split()).replace("_", "") for ch in fnirs_names]
            deoxy_tokens = ("hbr", "hhb", "deoxy")
            oxy_tokens = ("hbo2", "hbo", "o2hb", "oxy")
            hbo_candidates = [i for i, name in enumerate(normalized)
                              if any(t in name for t in oxy_tokens)
                              and not any(t in name for t in deoxy_tokens)]
            hbr_candidates = [i for i, name in enumerate(normalized)
                              if any(t in name for t in deoxy_tokens)]
            hbo_idx = hbo_candidates[0] if hbo_candidates else 0
            hbr_idx = hbr_candidates[0] if hbr_candidates else min(1, fnirs_data.shape[1] - 1)
            hbo_idx = min(hbo_idx, fnirs_data.shape[1] - 1)
            hbr_idx = min(hbr_idx, fnirs_data.shape[1] - 1)
            fnirs_values = fnirs_data[:, [hbo_idx, hbr_idx]]
        else:
            fnirs_values = np.empty((0, 2), dtype=np.float64)

        if resp_data.size:
            normalized = ["".join(ch.lower().split()).replace("_", "") for ch in resp_names]
            resp_tokens = ("respiration", "resp", "breathing", "breath", "thorax", "belt")
            candidates = [i for i, name in enumerate(normalized)
                          if "rate" not in name and any(t in name for t in resp_tokens)]
            resp_idx = min(candidates[0] if candidates else 0, resp_data.shape[1] - 1)
            resp_values = resp_data[:, resp_idx]
        else:
            resp_values = np.empty(0, dtype=np.float64)

        self._pending_fnirs_values, self._pending_fnirs_timestamps = self._merge_timestamped(
            self._pending_fnirs_values, self._pending_fnirs_timestamps,
            fnirs_values, fnirs_ts,
        )
        self._pending_resp_values, self._pending_resp_timestamps = self._merge_timestamped(
            self._pending_resp_values, self._pending_resp_timestamps,
            resp_values, resp_ts,
        )

        f_ts = self._pending_fnirs_timestamps
        r_ts = self._pending_resp_timestamps
        if f_ts.size < 1 or r_ts.size < 2:
            return empty
        overlap_start = max(float(f_ts[0]), float(r_ts[0]))
        overlap_end = min(float(f_ts[-1]), float(r_ts[-1]))
        selected = (f_ts >= overlap_start) & (f_ts <= overlap_end)
        if not np.any(selected):
            # Discard fNIRS points older than the respiration stream's current
            # range; they can no longer be synchronized in a future chunk.
            obsolete = f_ts < r_ts[0]
            self._pending_fnirs_values = self._pending_fnirs_values[~obsolete]
            self._pending_fnirs_timestamps = f_ts[~obsolete]
            return empty

        out_ts = f_ts[selected]
        out_fnirs = self._pending_fnirs_values[selected]
        out_resp = np.interp(out_ts, r_ts, self._pending_resp_values)

        # Keep only un-emitted fNIRS points and the respiration sample just
        # before the emitted edge, so interpolation remains continuous.
        last_emitted = float(out_ts[-1])
        pending_fnirs = f_ts > last_emitted
        self._pending_fnirs_values = self._pending_fnirs_values[pending_fnirs]
        self._pending_fnirs_timestamps = f_ts[pending_fnirs]
        anchor = max(0, int(np.searchsorted(r_ts, last_emitted, side="right")) - 1)
        self._pending_resp_values = self._pending_resp_values[anchor:]
        self._pending_resp_timestamps = r_ts[anchor:]

        with self._lock:
            self.stats["total_samples"] += int(out_ts.size)
            self.stats["last_sample_time"] = time.time()
            if out_ts.size > 1:
                median_dt = float(np.median(np.diff(out_ts)))
                if median_dt > 0:
                    self.stats["jitter_ms"] = float(np.std(np.diff(out_ts) - median_dt) * 1000.0)

        return {
            "HbO2": out_fnirs[:, 0],
            "HbR": out_fnirs[:, 1],
            "Respiration": out_resp,
            "timestamps": out_ts,
        }

    @staticmethod
    def _merge_timestamped(
        old_values: np.ndarray,
        old_timestamps: np.ndarray,
        new_values: np.ndarray,
        new_timestamps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Append, sort, and remove duplicate timestamps from a pending buffer."""
        if new_timestamps.size == 0:
            return old_values, old_timestamps
        if new_values.shape[0] != new_timestamps.size:
            return old_values, old_timestamps
        timestamps = np.concatenate((old_timestamps, new_timestamps))
        values = np.concatenate((old_values, new_values), axis=0)
        valid = np.isfinite(timestamps)
        if values.ndim == 1:
            valid &= np.isfinite(values)
        else:
            valid &= np.all(np.isfinite(values), axis=1)
        timestamps, values = timestamps[valid], values[valid]
        if timestamps.size == 0:
            return values, timestamps
        order = np.argsort(timestamps)
        timestamps, values = timestamps[order], values[order]
        keep = np.concatenate(([True], np.diff(timestamps) > 1e-9))
        return values[keep], timestamps[keep]

    def disconnect(self) -> None:
        """Close the LSL inlet and stop reconnection watchdog cleanly."""
        self._watchdog_stop.set()
        watchdog = self._watchdog_thread
        if watchdog is not None and watchdog is not threading.current_thread():
            watchdog.join(timeout=max(1.0, self.timeout + 1.0))
        with self._lock:
            if self._inlet is not None:
                try:
                    self._inlet.close_stream()
                except Exception:
                    pass
                self._inlet = None
            if self._respiration_inlet is not None:
                try:
                    self._respiration_inlet.close_stream()
                except Exception:
                    pass
                self._respiration_inlet = None
            self._connected = False
            self._pending_fnirs_values = np.empty((0, 2), dtype=np.float64)
            self._pending_fnirs_timestamps = np.empty(0, dtype=np.float64)
            self._pending_resp_values = np.empty(0, dtype=np.float64)
            self._pending_resp_timestamps = np.empty(0, dtype=np.float64)
        print("[LSLBridge] Disconnected.")

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_channel_names(info) -> List[str]:
        """
        Extract channel label strings from LSL StreamInfo XML metadata.

        Returns a list of channel name strings, or positional defaults
        (``"ch_0"``, ``"ch_1"``, …) if the metadata is absent/malformed.
        """
        names: List[str] = []
        try:
            ch_node = info.desc().child("channels").child("channel")
            while ch_node.name() == "channel":
                label = ch_node.child_value("label")
                names.append(label if label else f"ch_{len(names)}")
                ch_node = ch_node.next_sibling()
        except Exception:  # noqa: BLE001
            pass
        if not names:
            names = [f"ch_{i}" for i in range(info.channel_count())]
        return names


# ──────────────────────────────────────────────────────────────────────────────
# Demo / self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    DEMO_DURATION_S = 15
    POLL_INTERVAL_S = 3
    WINDOW_SAMPLES  = 30   # samples to request per stat snapshot

    print("=" * 60)
    print(" NeuroResonance fNIRS LSL Streamer — Demo")
    print("=" * 60)
    print(f"Running simulator for {DEMO_DURATION_S} s …\n")

    # 1. Start simulator
    sim = fNIRSSimulator(
        sampling_rate=10.0,
        channels=["HbO2", "HbR"],
        mayer_wave_freq=0.1,
    )
    sim.start()
    time.sleep(1.0)  # brief settle

    # 2. Connect bridge to the simulator stream
    bridge = LSLBridge(stream_name=fNIRSSimulator.STREAM_NAME, timeout=5.0)
    bridge_ok = bridge.connect()

    # 3. Poll loop
    t0 = time.perf_counter()
    tick = 0
    try:
        while (elapsed := time.perf_counter() - t0) < DEMO_DURATION_S:
            time.sleep(POLL_INTERVAL_S)
            tick += 1
            elapsed = time.perf_counter() - t0

            # Stats from internal buffer
            buf_data = sim.get_latest(WINDOW_SAMPLES)
            hbo2 = buf_data["HbO2"]
            hbr  = buf_data["HbR"]

            if hbo2.size == 0:
                print(f"[{elapsed:5.1f}s] No data yet…")
                continue

            corr = (
                float(np.corrcoef(hbo2, hbr)[0, 1])
                if hbo2.size > 2
                else float("nan")
            )

            print(
                f"[{elapsed:5.1f}s] "
                f"HbO2 μ={hbo2.mean():+.3f} σ={hbo2.std():.3f} a.u. | "
                f"HbR  μ={hbr.mean():+.3f} σ={hbr.std():.3f} a.u. | "
                f"corr(HbO2,HbR)={corr:+.3f}"
            )

            # Pull via LSL bridge too
            if bridge_ok:
                chunk = bridge.pull_chunk(max_samples=100)
                n_pulled = chunk["HbO2"].size
                print(f"         [LSLBridge] pulled {n_pulled} samples via LSL inlet")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if bridge_ok:
            bridge.disconnect()
        sim.stop()

    print("\nDone. Synthetic signal summary:")
    print("  • HbO2/HbR correlation is imposed by this synthetic model; it does not validate NVC.")
    print("  • A configurable low-frequency sine is included for demonstration.")
    print("  • Shaped noise is synthetic and does not establish biological realism.")
    sys.exit(0)
