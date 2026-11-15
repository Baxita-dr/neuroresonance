"""
coherence_engine.py
===================
Phase-Locking Value (PLV) and coherence analysis for NeuroResonance.

Interpretation
--------------
This module filters two input signals and calculates descriptive phase
consistency. PLV does not establish causality, cerebral origin,
neurovascular coupling, baroreflex resonance, or clinical state. fNIRS can
contain systemic, extracerebral, and motion-related components.

PLV Formula
-----------
Given two instantaneous phase series φ₁(t) and φ₂(t) of length N:

    PLV = | 1/N · Σₜ exp( i · (φ₁(t) − φ₂(t)) ) |

The complex exponential maps each phase difference onto the unit circle.
Perfect phase-locking (constant difference) yields |mean| = 1; uniformly
distributed differences yield |mean| → 0.

References
----------
  Lachaux J-P et al. (1999). Measuring phase synchrony in brain signals.
      Human Brain Mapping, 8(4), 194–208.
  Lehrer P, Gevirtz R. (2014). Heart rate variability biofeedback: how and
      why does it work? Frontiers in Psychology, 5, 756.
  Shaffer F, Ginsberg JP. (2017). An overview of heart rate variability
      metrics and norms. Frontiers in Public Health, 5, 258.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy import signal as sp_signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    data : np.ndarray
        1-D input signal.
    lowcut : float
        Lower −3 dB frequency (Hz).
    highcut : float
        Upper −3 dB frequency (Hz).
    fs : float
        Sampling frequency (Hz).
    order : int
        Filter order (default 4; doubled by zero-phase ``filtfilt``).

    Returns
    -------
    np.ndarray
        Filtered signal, same length as *data*.
    """
    data = np.asarray(data, dtype=float).ravel()
    if fs <= 0 or data.size < (3 * order + 1):
        return data.copy()
    valid = np.isfinite(data)
    if not np.all(valid):
        if np.sum(valid) < 4:
            return np.zeros_like(data)
        indices = np.arange(len(data))
        data = np.interp(indices, indices[valid], data[valid])
    nyq = 0.5 * fs
    low = float(np.clip(lowcut / nyq, 1e-6, 1.0 - 1e-4))
    high = float(np.clip(highcut / nyq, 1e-6, 1.0 - 1e-4))
    if not (0.0 < low < high < 1.0):
        return data.copy()
    # SOS is stable at the narrow Mayer-wave band; ba-form filtfilt is not.
    sos = sp_signal.butter(order, [low, high], btype="band", output="sos")
    padlen = 3 * int(sos.shape[0])
    if data.size <= padlen:
        return data.copy()
    return sp_signal.sosfiltfilt(sos, data, padlen=padlen)


def _hilbert_phase(data: np.ndarray) -> np.ndarray:
    """
    Return the instantaneous phase (radians) via the analytic signal.

    Parameters
    ----------
    data : np.ndarray
        1-D real-valued signal (should be narrow-band for meaningful phase).

    Returns
    -------
    np.ndarray
        Unwrapped instantaneous phase in radians.
    """
    analytic = sp_signal.hilbert(data)
    return np.unwrap(np.angle(analytic))


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def compute_plv(
    phase1: np.ndarray,
    phase2: np.ndarray,
    window_samples: Optional[int] = None,
) -> float | np.ndarray:
    """
    Compute the Phase-Locking Value (PLV) between two instantaneous phase series.

    Mathematical Definition
    -----------------------
    For two phase series φ₁(t) and φ₂(t) sampled at N time points:

        PLV = | 1/N · Σₜ₌₁ᴺ exp( i · (φ₁(t) − φ₂(t)) ) |

    The complex mean of exp( i · Δφ ) lives inside the unit circle.
    Its modulus equals 1 when Δφ is constant (perfect locking) and
    approaches 0 when Δφ is uniformly distributed (no coupling).

    Parameters
    ----------
    phase1 : np.ndarray
        Instantaneous phase of signal 1 (radians), shape (N,).
    phase2 : np.ndarray
        Instantaneous phase of signal 2 (radians), shape (N,).
    window_samples : int or None
        If given, split the series into contiguous, non-overlapping blocks of
        this length and return one PLV per block. The last partial block is
        dropped. This is not a centred sliding window.
        If None (default), a single scalar PLV over the whole series is
        returned.

    Returns
    -------
    float or np.ndarray
        * Scalar float in [0, 1] when ``window_samples`` is None.
        * 1-D numpy array of floats in [0, 1] when ``window_samples`` is set.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 1000)
    >>> p1 = 2 * np.pi * 0.1 * t            # 0.1 Hz oscillator
    >>> p2 = 2 * np.pi * 0.1 * t + 0.3     # same freq, constant offset
    >>> compute_plv(p1, p2)                 # → ~1.0
    >>> compute_plv(p1, p2, window_samples=100)  # array of ~1.0 values
    """
    phase1 = np.asarray(phase1, dtype=float)
    phase2 = np.asarray(phase2, dtype=float)
    if window_samples is not None and window_samples <= 0:
        raise ValueError(f"window_samples must be positive, got {window_samples}.")
    phase1 = phase1.ravel()
    phase2 = phase2.ravel()

    if phase1.shape != phase2.shape:
        raise ValueError(
            f"phase1 and phase2 must have the same shape; "
            f"got {phase1.shape} and {phase2.shape}."
        )

    delta = phase1 - phase2
    valid = np.isfinite(delta)
    if not np.any(valid):
        return float("nan") if window_samples is None else np.empty(0, dtype=float)
    if window_samples is None:
        delta = delta[valid]
        complex_phase = np.exp(1j * delta)
        return float(np.abs(np.mean(complex_phase)))

    # Contiguous non-overlapping blocks. A partial tail block is dropped.
    n = len(delta)
    n_windows = n // window_samples
    if n_windows == 0:
        raise ValueError(
            f"window_samples ({window_samples}) exceeds signal length ({n})."
        )
    trimmed_delta = delta[: n_windows * window_samples]
    trimmed_valid = valid[: n_windows * window_samples]
    windows = trimmed_delta.reshape(n_windows, window_samples)
    valid_windows = trimmed_valid.reshape(n_windows, window_samples)
    phasors = np.where(valid_windows, np.exp(1j * np.where(valid_windows, windows, 0.0)), 0.0j)
    counts = valid_windows.sum(axis=1)
    result = np.full(n_windows, np.nan, dtype=float)
    nonempty = counts > 0
    result[nonempty] = np.abs(phasors[nonempty].sum(axis=1) / counts[nonempty])
    return result


def compute_plv_block_surrogate(
    phase1: np.ndarray,
    phase2: np.ndarray,
    bandwidth_hz: float = 0.08,
    fs: float = 2.0,
) -> Tuple[float, float, bool]:
    """Disabled legacy API: circular shifts are not a validated null model here."""
    raise NotImplementedError(
        "Surrogate p-values were removed: circular shifts of short, edge-trimmed "
        "autocorrelated signals do not provide a validated significance test."
    )


def compute_rayleigh_test(
    phase1: np.ndarray,
    phase2: np.ndarray,
    bandwidth_hz: float = 0.08,
    fs: float = 2.0,
) -> Tuple[float, float, bool]:
    """Deprecated compatibility wrapper for the circular-shift comparison."""
    raise NotImplementedError(
        "Rayleigh significance is not implemented for this pairwise PLV estimator."
    )


def compute_coherence(
    sig1: np.ndarray,
    sig2: np.ndarray,
    fs: float,
    nperseg: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute magnitude-squared coherence between two signals.

    Uses Welch's averaged periodogram method (via ``scipy.signal.coherence``).

    Parameters
    ----------
    sig1 : np.ndarray
        First signal, shape (N,).
    sig2 : np.ndarray
        Second signal, shape (N,).
    fs : float
        Sampling frequency of both signals (Hz).
    nperseg : int
        Length of each FFT segment (default 256).  Longer segments give finer
        frequency resolution ( Δf = fs / nperseg ) at the cost of fewer
        averages.

    Returns
    -------
    frequencies : np.ndarray
        Frequency axis in Hz, shape (nperseg//2 + 1,).
    coherence_values : np.ndarray
        Magnitude-squared coherence in [0, 1], same shape as *frequencies*.

    Notes
    -----
    **Respiratory / Mayer-wave band** extraction
    The returned tuple also exposes coherence specifically in 0.07–0.15 Hz.
    Call ``_coherence_at_band`` or use ``ResonanceEstimator`` for a single
    band-averaged value.

    At fs = 10 Hz and nperseg = 256 the frequency resolution is 0.039 Hz,
    which is coarse relative to the 0.1 Hz band.  Consider zero-padding or
    using a longer segment if the signal is long enough.
    """
    sig1 = np.asarray(sig1, dtype=float)
    sig2 = np.asarray(sig2, dtype=float)
    freqs, coh = sp_signal.coherence(sig1, sig2, fs=fs, nperseg=nperseg)
    return freqs, coh


def circular_mean(angles: np.ndarray) -> float:
    """Circular mean of angles in radians, in (-π, π].

    A linear mean of unwrapped phase runs away when the two oscillators are
    not at exactly the same frequency. The mean of the unit phasors stays
    on the correct lag.
    """
    angles = np.asarray(angles, dtype=float).ravel()
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return float("nan")
    return float(np.angle(np.mean(np.exp(1j * angles))))


def phase_lag_ms(delta_rad: float, freq_hz: float) -> float:
    """Deprecated compatibility helper; phase difference is not a physiological delay."""
    del delta_rad, freq_hz
    return float("nan")


def coherence_in_band(
    sig1: np.ndarray,
    sig2: np.ndarray,
    fs: float,
    low_hz: float = 0.07,
    high_hz: float = 0.15,
) -> float:
    """Welch coherence with 0.02 Hz resolution, available after 100 s only."""
    sig1 = np.asarray(sig1, dtype=float).ravel()
    sig2 = np.asarray(sig2, dtype=float).ravel()
    n = min(sig1.size, sig2.size)
    if n < 1 or not math.isfinite(float(fs)) or fs <= 0:
        return float("nan")
    nperseg = int(round(fs * 50.0))
    if n < int(round(fs * 100.0)) or nperseg < 16:
        return float("nan")
    paired = np.isfinite(sig1[:n]) & np.isfinite(sig2[:n])
    if not np.all(paired):
        return float("nan")
    if np.std(sig1[:n]) <= 1e-12 or np.std(sig2[:n]) <= 1e-12:
        return float("nan")
    freqs, coh = sp_signal.coherence(sig1[:n], sig2[:n], fs=fs, nperseg=nperseg)
    return _coherence_at_band(freqs, coh, low_hz, high_hz)


def _coherence_at_band(
    freqs: np.ndarray,
    coh: np.ndarray,
    low_hz: float = 0.07,
    high_hz: float = 0.15,
) -> float:
    """
    Return mean coherence in the specified frequency band.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency axis (Hz).
    coh : np.ndarray
        Coherence values (same length as *freqs*).
    low_hz, high_hz : float
        Band edges (Hz).

    Returns
    -------
    float
        Mean coherence in the band, or NaN if no bins fall in the band.
    """
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not mask.any():
        return float("nan")
    return float(np.mean(coh[mask]))


def compute_lissajous(
    phase1: np.ndarray,
    phase2: np.ndarray,
    amplitude1: np.ndarray | float = 1.0,
    amplitude2: np.ndarray | float = 1.0,
    n_points: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a parametric Lissajous / phase-orbit curve for visualisation.

    The curve visualizes paired phase values from the selected signals.
    Its shape is not evidence of a physiological coupling mechanism.

    Definition
    ----------
    Given instantaneous phases φ₁(t) and φ₂(t) and instantaneous amplitudes
    A₁(t) and A₂(t):

        x(t) = A₁(t) · cos(φ₁(t))
        y(t) = A₂(t) · sin(φ₂(t))

    The curve is a visualization of paired phases only. Its shape depends on
    scaling and signal construction and is not evidence of biological coupling.

    Parameters
    ----------
    phase1 : np.ndarray
        Instantaneous phase of oscillator 1 (radians), shape (M,).
    phase2 : np.ndarray
        Instantaneous phase of oscillator 2 (radians), shape (M,).
    amplitude1 : np.ndarray or float
        Instantaneous (or constant) amplitude of oscillator 1.
        Broadcast-compatible with *phase1*.
    amplitude2 : np.ndarray or float
        Instantaneous (or constant) amplitude of oscillator 2.
        Broadcast-compatible with *phase2*.
    n_points : int
        Number of points to subsample from the orbit for rendering.
        If the input is longer, uniform sub-sampling is applied; if shorter,
        all points are used.

    Returns
    -------
    x : np.ndarray
        Horizontal coordinate, shape (n_points,) or (len(phase1),).
    y : np.ndarray
        Vertical coordinate, same shape as *x*.
    """
    phase1 = np.asarray(phase1, dtype=float)
    phase2 = np.asarray(phase2, dtype=float)
    amplitude1 = np.asarray(amplitude1, dtype=float)
    amplitude2 = np.asarray(amplitude2, dtype=float)
    if int(n_points) <= 0:
        raise ValueError("n_points must be positive.")
    n_points = int(n_points)

    x = amplitude1 * np.cos(phase1)
    y = amplitude2 * np.sin(phase2)

    n = len(x)
    if n > n_points:
        idx = np.round(np.linspace(0, n - 1, n_points)).astype(int)
        x = x[idx]
        y = y[idx]

    return x, y


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ResonanceState:
    """
    Snapshot of the current resonance estimation.

    Attributes
    ----------
    plv_percent : float
        Phase-Locking Value expressed as a percentage (0–100).
    phase_lag_ms : float
        Deprecated compatibility field; unavailable (NaN) because phase lag
        is not interpretable without a validated physiological model.
    coherence_at_01hz : float
        Magnitude-squared coherence averaged over 0.07–0.15 Hz.
    state_label : str
        Descriptive status only; no physiological category is assigned.
    state_color : str
        Hex colour string suitable for UI rendering.
        Blue for descriptive PLV, grey/blue for unavailable data.
    dominant_freq_hz : float
        Peak power frequency in the respiration channel in 0.07–0.15 Hz.
    timestamp : float
        UNIX timestamp (seconds since epoch) when the estimate was produced.
    """

    plv_percent: float
    phase_lag_ms: float
    coherence_at_01hz: float
    state_label: str
    state_color: str
    dominant_freq_hz: float
    timestamp: float = field(default_factory=time.time)
    is_significant: Optional[bool] = None
    p_value: float = float("nan")
    rayleigh_z: float = float("nan")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def plv(self) -> float:
        """PLV as a fraction in [0, 1]."""
        return self.plv_percent / 100.0

    @property
    def is_resonant(self) -> bool:
        """PLV is descriptive; this legacy property never infers resonance."""
        return False

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"ResonanceState("
            f"PLV={self.plv_percent:.1f}%, "
            f"coh={self.coherence_at_01hz:.3f}, "
            f"f_dom={self.dominant_freq_hz:.3f} Hz, "
            f"state={self.state_label})"
        )


# ---------------------------------------------------------------------------
# ResonanceEstimator
# ---------------------------------------------------------------------------

# Respiratory / Mayer-wave band edges (Hz)
_RESP_LOW_HZ: float = 0.07
_RESP_HIGH_HZ: float = 0.15

# Fixed descriptive display color; there are no PLV categories or thresholds.
_COLOR_DESCRIPTIVE: str = "#4FC3F7"


class ResonanceEstimator:
    """
    Online estimator of descriptive phase consistency for fNIRS and respiration.

    The estimator ingests raw respiratory and fNIRS (HbO₂ / HbR) signals,
    resamples them to a common grid, bandpass-filters to the respiratory /
    Mayer-wave band (0.07–0.15 Hz), and reports the Phase-Locking Value and
    related metrics via :class:`ResonanceState`.

    Interpretation
    --------------
    Returned PLV is a descriptive statistic for the selected window. fNIRS
    PLV alone cannot separate cerebral from systemic,
    extracerebral, or motion-related signal components.

    Parameters
    ----------
    fs_resp : float
        Native sampling frequency of the respiratory signal (Hz).
        Default 30 Hz (typical for a respiratory belt).
    fs_fnirs : float
        Native sampling frequency of the fNIRS signal (Hz).
        Default 10 Hz (typical for portable fNIRS devices).
    window_s : float
        Duration of the analysis window (seconds).  Longer windows improve
        PLV stability at the cost of latency. Must be at least 100 s.
    overlap_s : float
        Overlap between consecutive windows (seconds).  Used for display
        continuity; actual update rate is determined by the caller.
        Default 5 s.
    """

    # Common-grid rate (Hz). 2 Hz puts the 0.07–0.15 Hz band at a
    # comfortable fraction of Nyquist. 1 Hz left almost no samples per
    # cycle and made a Welch coherence estimate a single segment (MSC ≡ 1).
    _FS_COMMON: float = 2.0

    def __init__(
        self,
        fs_resp: float = 30.0,
        fs_fnirs: float = 10.0,
        window_s: float = 100.0,
        overlap_s: float = 5.0,
    ) -> None:
        self.fs_resp = float(fs_resp)
        self.fs_fnirs = float(fs_fnirs)
        self.window_s = float(window_s)
        self.overlap_s = float(overlap_s)
        if not math.isfinite(self.fs_resp) or self.fs_resp <= 0:
            raise ValueError("fs_resp must be positive and finite.")
        if not math.isfinite(self.fs_fnirs) or self.fs_fnirs <= 0:
            raise ValueError("fs_fnirs must be positive and finite.")
        if not math.isfinite(self.window_s) or self.window_s < 100.0:
            raise ValueError("window_s must be at least 100 seconds for this analysis.")

        # Cache for Lissajous orbit from the most recent update
        self._last_phase_resp: Optional[np.ndarray] = None
        self._last_phase_hbo2: Optional[np.ndarray] = None
        self._last_amp_resp: Optional[np.ndarray] = None
        self._last_amp_hbo2: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resample_to_common(
        sig: np.ndarray,
        fs_in: float,
        fs_out: float,
    ) -> np.ndarray:
        """
        Resample *sig* from *fs_in* Hz to *fs_out* Hz using polyphase filtering.

        ``scipy.signal.resample_poly`` is preferred over FFT resampling because
        it handles arbitrary (non-power-of-two) ratios and is more robust to
        edge artefacts.

        The up/down integers are computed from the ratio ``fs_out / fs_in``
        expressed as a reduced fraction.
        """
        if fs_in <= 0 or fs_out <= 0:
            return np.asarray(sig, dtype=float).ravel().copy()
        if abs(fs_in - fs_out) / fs_in < 1e-6:
            return np.asarray(sig, dtype=float).ravel().copy()

        from math import gcd

        fs_in_i = round(fs_in * 1000)   # scale to avoid float rounding
        fs_out_i = round(fs_out * 1000)
        g = gcd(fs_in_i, fs_out_i)
        up = fs_out_i // g
        down = fs_in_i // g
        return sp_signal.resample_poly(sig, up, down)

    @staticmethod
    def _on_uniform_grid(
        sig: np.ndarray,
        timestamps: np.ndarray,
        fs_out: float,
    ) -> Optional[np.ndarray]:
        """Interpolate *sig* onto a uniform grid using its own timestamps.

        Declared sample rates are wrong when the UI loop, a webcam, or a
        dropped LSL packet delivers irregular intervals. The timestamps are
        the time base the phase estimate has to follow.
        """
        sig = np.asarray(sig, dtype=float).ravel()
        ts = np.asarray(timestamps, dtype=float).ravel()
        if sig.size != ts.size or sig.size < 4 or fs_out <= 0:
            return None
        valid = np.isfinite(sig) & np.isfinite(ts)
        if int(np.sum(valid)) < 4:
            return None
        sig = sig[valid]
        ts = ts[valid]
        order = np.argsort(ts)
        ts = ts[order]
        sig = sig[order]
        keep = np.concatenate(([True], np.diff(ts) > 1e-9))
        ts = ts[keep]
        sig = sig[keep]
        if ts.size < 4 or ts[-1] <= ts[0]:
            return None
        t_out = np.arange(ts[0], ts[-1], 1.0 / fs_out)
        if t_out.size < 4:
            return None
        return np.interp(t_out, ts, sig)

    @staticmethod
    def _align_on_overlap(sig_a: np.ndarray, ts_a: np.ndarray, sig_b: np.ndarray,
                          ts_b: np.ndarray, fs_out: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Low-pass each native stream before 2 Hz interpolation and reject outages."""
        def clean(sig: np.ndarray, ts: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            sig, ts = np.asarray(sig, float).ravel(), np.asarray(ts, float).ravel()
            if sig.size != ts.size:
                return None
            ok = np.isfinite(sig) & np.isfinite(ts)
            sig, ts = sig[ok], ts[ok]
            order = np.argsort(ts)
            sig, ts = sig[order], ts[order]
            keep = np.concatenate(([True], np.diff(ts) > 1e-9)) if ts.size else np.array([], bool)
            sig, ts = sig[keep], ts[keep]
            if ts.size < 4:
                return None
            typical = float(np.median(np.diff(ts)))
            intervals = np.diff(ts)
            if typical <= 0 or np.max(intervals) > 3.0 * typical:
                return None
            native_fs = 1.0 / typical
            if native_fs <= 0.60:
                return None
            native_grid = np.arange(ts[0], ts[-1] + 0.25 * typical, typical)
            if native_grid.size < 8:
                return None
            uniform = np.interp(native_grid, ts, sig)
            cutoff_hz = min(0.30, 0.40 * native_fs)
            if cutoff_hz >= native_fs / 2.0 or cutoff_hz <= _RESP_HIGH_HZ:
                return None
            sos = sp_signal.butter(6, cutoff_hz, btype="lowpass", fs=native_fs, output="sos")
            padlen = 3 * int(sos.shape[0])
            if uniform.size <= padlen:
                return None
            filtered = sp_signal.sosfiltfilt(sos, uniform, padlen=padlen)
            return filtered, native_grid
        a, b = clean(sig_a, ts_a), clean(sig_b, ts_b)
        if a is None or b is None or fs_out <= 0:
            return None
        start, end = max(a[1][0], b[1][0]), min(a[1][-1], b[1][-1])
        if end <= start:
            return None
        grid = np.arange(start, end, 1.0 / fs_out)
        if grid.size < 8:
            return None
        return np.interp(grid, a[1], a[0]), np.interp(grid, b[1], b[0])

    @staticmethod
    def _dominant_freq(sig: np.ndarray, fs: float) -> float:
        """
        Return the frequency (Hz) of peak power in the Mayer/respiratory band.
        """
        n = len(sig)
        if n < 4:
            return float("nan")
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        psd = np.abs(np.fft.rfft(sig)) ** 2
        mask = (freqs >= _RESP_LOW_HZ) & (freqs <= _RESP_HIGH_HZ)
        if not mask.any():
            return float("nan")
        return float(freqs[mask][np.argmax(psd[mask])])

    @staticmethod
    def _classify(plv: float, is_significant: bool = False) -> Tuple[str, str]:
        """Retained method name; all valid outputs are descriptive only."""
        return "DESCRIPTIVE PLV (NO STATE CLASSIFICATION)", _COLOR_DESCRIPTIVE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        resp_signal: np.ndarray,
        hbo2_signal: np.ndarray,
        hbr_signal: np.ndarray,
        timestamps_resp: np.ndarray,
        timestamps_fnirs: np.ndarray,
    ) -> ResonanceState:
        """
        Update the resonance estimate with new signal epochs.

        Pipeline
        --------
        1. **Validate and align** – timestamps define the native rate; duplicate
           samples are removed and long acquisition gaps are rejected.
        2. **Anti-alias and resample** – each stream is low-pass filtered on its
           timestamp-derived native grid before interpolation onto the common
           2 Hz grid.
        3. **Bandpass** – 4th-order zero-phase Butterworth filter, 0.07–0.15 Hz.
        4. **Hilbert** – instantaneous phase via analytic signal.
        5. **PLV** – descriptive Phase-Locking Value over the full window.
        6. **Coherence** – Welch magnitude-squared coherence in 0.07–0.15 Hz,
           available only when the 100-second minimum and input quality checks pass.

        Parameters
        ----------
        resp_signal : np.ndarray
            Raw respiratory signal sampled at ``fs_resp`` Hz.
        hbo2_signal : np.ndarray
            Oxygenated haemoglobin (HbO₂) signal sampled at ``fs_fnirs`` Hz.
        hbr_signal : np.ndarray
            Deoxygenated haemoglobin (HbR) signal sampled at ``fs_fnirs`` Hz.
            Retained for API compatibility; not currently used by this estimator.
        timestamps_resp : np.ndarray
            UNIX timestamps for each sample in *resp_signal*.
        timestamps_fnirs : np.ndarray
            UNIX timestamps for each sample in *hbo2_signal* / *hbr_signal*.

        Returns
        -------
        ResonanceState
            Current resonance metrics snapshot.
        """
        resp_signal = np.asarray(resp_signal, dtype=float)
        hbo2_signal = np.asarray(hbo2_signal, dtype=float)
        hbr_signal = np.asarray(hbr_signal, dtype=float)
        timestamps_resp = np.asarray(timestamps_resp, dtype=float)
        timestamps_fnirs = np.asarray(timestamps_fnirs, dtype=float)

        # Use only the most recent configured duration from both timestamped streams.
        if resp_signal.size != timestamps_resp.size or hbo2_signal.size != timestamps_fnirs.size:
            return ResonanceState(float("nan"), float("nan"), float("nan"), "INSUFFICIENT DATA: TIMESTAMP MISMATCH", "#4FC3F7", float("nan"))
        finite_resp_t = timestamps_resp[np.isfinite(timestamps_resp)]
        finite_fnirs_t = timestamps_fnirs[np.isfinite(timestamps_fnirs)]
        if finite_resp_t.size < 4 or finite_fnirs_t.size < 4:
            return ResonanceState(float("nan"), float("nan"), float("nan"), "INSUFFICIENT DATA: TIMESTAMPS", "#4FC3F7", float("nan"))
        common_end = min(float(np.max(finite_resp_t)), float(np.max(finite_fnirs_t)))
        common_start = max(float(np.min(finite_resp_t)), float(np.min(finite_fnirs_t)))
        available_s = common_end - common_start
        if available_s < self.window_s:
            return ResonanceState(float("nan"), float("nan"), float("nan"), f"INSUFFICIENT DATA: BUFFERING {max(0.0, available_s):.1f}/{self.window_s:.0f} s", "#4FC3F7", float("nan"))
        cutoff = common_end - self.window_s
        resp_mask = np.isfinite(timestamps_resp) & (timestamps_resp >= cutoff) & (timestamps_resp <= common_end)
        fnirs_mask = np.isfinite(timestamps_fnirs) & (timestamps_fnirs >= cutoff) & (timestamps_fnirs <= common_end)
        resp_signal, timestamps_resp = resp_signal[resp_mask], timestamps_resp[resp_mask]
        hbo2_signal, timestamps_fnirs = hbo2_signal[fnirs_mask], timestamps_fnirs[fnirs_mask]

        # Anti-alias filtering is performed at each native rate before 2 Hz interpolation.
        fs_c = self._FS_COMMON
        aligned = self._align_on_overlap(resp_signal, timestamps_resp, hbo2_signal, timestamps_fnirs, fs_c)
        if aligned is None:
            return ResonanceState(float("nan"), float("nan"), float("nan"), "INSUFFICIENT DATA: GAP OR SAMPLE RATE", "#4FC3F7", float("nan"))
        resp_r, hbo2_r = aligned
        min_len = min(len(resp_r), len(hbo2_r))
        if min_len < int(self.window_s * fs_c * 0.95):
            return ResonanceState(float("nan"), float("nan"), float("nan"), "INSUFFICIENT DATA: SHORT OVERLAP", "#4FC3F7", float("nan"))
        resp_r, hbo2_r = resp_r[:min_len], hbo2_r[:min_len]

        # ------------------------------------------------------------------ #
        # 2. Bandpass filter to respiratory / Mayer-wave band
        # ------------------------------------------------------------------ #
        resp_bp = _bandpass_filter(resp_r, _RESP_LOW_HZ, _RESP_HIGH_HZ, fs_c)
        hbo2_bp = _bandpass_filter(hbo2_r, _RESP_LOW_HZ, _RESP_HIGH_HZ, fs_c)

        # Phase is undefined for a flat/no-signal channel. Without this guard,
        # numerical Hilbert-transform residue can produce an arbitrary but
        # apparently perfect PLV (including PLV=1 for two constant inputs).
        resp_scale = max(float(np.std(resp_r)), 1e-12)
        hbo2_scale = max(float(np.std(hbo2_r)), 1e-12)
        if (
            float(np.std(resp_bp)) <= 1e-8 * resp_scale
            or float(np.std(hbo2_bp)) <= 1e-8 * hbo2_scale
        ):
            return ResonanceState(
                plv_percent=float("nan"),
                phase_lag_ms=float("nan"),
                coherence_at_01hz=float("nan"),
                state_label="STARTING (INSUFFICIENT SIGNAL)",
                state_color="#4FC3F7",
                dominant_freq_hz=float("nan"),
            )

        # ------------------------------------------------------------------ #
        # 3. Hilbert transform → instantaneous phase and amplitude
        # ------------------------------------------------------------------ #
        analytic_resp = sp_signal.hilbert(resp_bp)
        analytic_hbo2 = sp_signal.hilbert(hbo2_bp)

        phase_resp = np.unwrap(np.angle(analytic_resp))
        phase_hbo2 = np.unwrap(np.angle(analytic_hbo2))
        amp_resp = np.abs(analytic_resp)
        amp_hbo2 = np.abs(analytic_hbo2)

        # Cache for Lissajous orbit
        self._last_phase_resp = phase_resp
        self._last_phase_hbo2 = phase_hbo2
        self._last_amp_resp = amp_resp
        self._last_amp_hbo2 = amp_hbo2

        # ------------------------------------------------------------------ #
        # 4. Descriptive PLV; no surrogate p-value or software category is inferred.
        # ------------------------------------------------------------------ #
        # Hilbert phase is least reliable at the edges of a finite window.
        # Exclude up to 3 s at each edge, while retaining at least 8 samples
        # for short windows.
        edge_margin = min(int(round(3.0 * fs_c)), max(0, (min_len - 8) // 2))
        stop_idx = min_len - edge_margin if edge_margin else min_len
        phase_resp_plv = phase_resp[edge_margin:stop_idx]
        phase_hbo2_plv = phase_hbo2[edge_margin:stop_idx]

        plv_scalar = compute_plv(phase_resp_plv, phase_hbo2_plv)
        plv_percent = plv_scalar * 100.0

        # Phase lag is intentionally unavailable: it is not a physiological delay.
        f_dom = self._dominant_freq(resp_bp, fs_c)

        # ------------------------------------------------------------------ #
        # 5. Welch coherence is withheld until a 100 s record supports its resolution.
        # ------------------------------------------------------------------ #
        coh_band = coherence_in_band(
            resp_r, hbo2_r, fs=fs_c, low_hz=_RESP_LOW_HZ, high_hz=_RESP_HIGH_HZ
        )
        if not math.isfinite(coh_band):
            coh_band = float("nan")

        # ------------------------------------------------------------------ #
        state_label, state_color = self._classify(plv_scalar)
        return ResonanceState(
            plv_percent=plv_percent if math.isfinite(plv_percent) else float("nan"),
            phase_lag_ms=float("nan"),
            coherence_at_01hz=float(coh_band),
            state_label=state_label,
            state_color=state_color,
            dominant_freq_hz=float(f_dom),
        )

    def get_lissajous_orbit(self, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the Lissajous / phase-orbit from the most recent ``update`` call.

        The orbit maps the two instantaneous phases onto the unit circle:

            x(t) = A₁(t) · cos(φ_resp(t))
            y(t) = A₂(t) · sin(φ_HbO₂(t))

        A visual summary only; orbit geometry is not evidence of physiological
        coupling.

        Parameters
        ----------
        n_points : int
            Number of orbit points to return (sub-sampled uniformly).

        Returns
        -------
        x, y : np.ndarray
            Lissajous orbit coordinates, each of shape (n_points,) or shorter
            if fewer data are available.

        Raises
        ------
        RuntimeError
            If called before any successful :meth:`update`.
        """
        if self._last_phase_resp is None or self._last_phase_hbo2 is None:
            raise RuntimeError(
                "No data available.  Call update() at least once before "
                "get_lissajous_orbit()."
            )
        amp1 = self._last_amp_resp if self._last_amp_resp is not None else 1.0
        amp2 = self._last_amp_hbo2 if self._last_amp_hbo2 is not None else 1.0
        return compute_lissajous(
            self._last_phase_resp,
            self._last_phase_hbo2,
            amplitude1=amp1,
            amplitude2=amp2,
            n_points=n_points,
        )
