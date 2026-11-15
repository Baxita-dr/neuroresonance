"""
signal_processing.py
====================
Digital signal processing module for NeuroResonance.

Implements bandpass filtering, Hilbert-transform-based instantaneous phase/amplitude,
instantaneous frequency, power spectral density estimation (Welch method),
neurovascular coupling validation, and signal normalization utilities.

Theoretical Background
----------------------
Hilbert Transform & Analytic Signal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For a real-valued signal x(t), the analytic signal z(t) is defined as:

    z(t) = x(t) + j * H{x(t)}

where H{·} denotes the Hilbert transform. This yields:

    - Instantaneous amplitude (envelope): A(t) = |z(t)|
    - Instantaneous phase:               φ(t) = arg(z(t))
    - Instantaneous frequency:           f(t) = (1/2π) · dφ/dt

Reference: Gabor, D. (1946). "Theory of communication." Journal of the
Institution of Electrical Engineers, 93(26), 429–457.

Mayer Waves (0.07–0.15 Hz)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Mayer waves are low-frequency cardiovascular oscillations often discussed
near 0.1 Hz. Signals in this band can contain systemic and other components;
frequency alone does not identify their source or establish NVC.

Reference: Mayer, S. (1876). Studien zur Physiologie des Herzens und der
Blutgefässe. Sitzungsberichte der Akademie der Wissenschaften in Wien, 74,
281–307.

Respiratory Band (0.1–0.5 Hz)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Respiratory oscillations (typical resting rate 0.15–0.4 Hz) introduce
hemodynamic artifacts in fNIRS. Bandpass filtering in this range isolates
respiratory-coupled components. Care must be taken when separating Mayer waves
from respiratory artifacts given their overlapping frequency ranges.

Neurovascular Coupling (NVC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
HbO₂/HbR correlation is descriptive and can vary with task, physiology,
measurement geometry, and artifacts. It cannot validate NVC or identify an
artifact or physiological state by itself.

Reference: Scholkmann, F. et al. (2014). "A review on continuous wave functional
near-infrared spectroscopy and imaging instrumentation and methodology."
NeuroImage, 85, 6–27.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAYER_WAVE_LOW_HZ: float = 0.07   # Lower bound of Mayer wave band
MAYER_WAVE_HIGH_HZ: float = 0.15  # Upper bound of Mayer wave band
RESPIRATORY_LOW_HZ: float = 0.10  # Lower bound of respiratory band
RESPIRATORY_HIGH_HZ: float = 0.50 # Upper bound of respiratory band


# ---------------------------------------------------------------------------
# 1. Bandpass Filter
# ---------------------------------------------------------------------------

def bandpass_filter(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter to a 1-D signal.

    Uses second-order sections (SOS) representation for improved numerical
    stability over the transfer-function (ba) form, particularly important at
    low cut-off frequencies relative to the sampling rate.

    The minimum signal length required is ``3 * order + 1`` samples; shorter
    signals are returned unchanged with a warning.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal (real-valued).
    lowcut : float
        Lower cut-off frequency in Hz.  Must satisfy ``0 < lowcut < highcut < fs/2``.
    highcut : float
        Upper cut-off frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Filter order (default 4).  Higher orders sharpen the roll-off but may
        introduce ringing artefacts on short signals.

    Returns
    -------
    np.ndarray
        Filtered signal with the same shape as the input, dtype ``float64``.

    Raises
    ------
    ValueError
        If ``lowcut`` or ``highcut`` are outside the valid Nyquist range, or if
        ``lowcut >= highcut``.

    Notes
    -----
    ``scipy.signal.sosfiltfilt`` applies the filter forward *and* backward,
    yielding zero phase shift and an effective order of ``2 * order``.

    Mayer wave isolation example::

        filtered = bandpass_filter(raw, 0.07, 0.15, fs=10.0)

    Respiratory isolation example::

        filtered = bandpass_filter(raw, 0.10, 0.50, fs=10.0)
    """
    signal = np.asarray(signal, dtype=np.float64)

    nyq = 0.5 * fs
    if not (0 < lowcut < highcut):
        raise ValueError(
            f"Require 0 < lowcut < highcut, got lowcut={lowcut}, highcut={highcut}."
        )
    if highcut >= nyq:
        raise ValueError(
            f"highcut ({highcut} Hz) must be less than the Nyquist frequency "
            f"({nyq} Hz) for fs={fs} Hz."
        )

    low = lowcut / nyq
    high = highcut / nyq
    sos = sp_signal.butter(order, [low, high], btype="band", output="sos")
    padlen = 3 * int(sos.shape[0])
    if signal.size <= padlen:
        logger.warning(
            "Signal length %d is too short for bandpass_filter (min %d). "
            "Returning raw signal unchanged.",
            signal.size,
            padlen + 1,
        )
        return signal.copy()

    return sp_signal.sosfiltfilt(sos, signal, padlen=padlen)


# ---------------------------------------------------------------------------
# 2. Instantaneous Phase & Amplitude via Hilbert Transform
# ---------------------------------------------------------------------------

def compute_instantaneous_phase(
    signal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute instantaneous phase and amplitude via the Hilbert transform.

    The analytic signal z(t) is obtained as:

        z(t) = x(t) + j·H{x(t)}

    where H{·} is the Hilbert transform implemented via FFT.

    Instantaneous phase is extracted as φ(t) = arg(z(t)) ∈ (−π, π].
    Instantaneous amplitude (envelope) is A(t) = |z(t)|.

    Parameters
    ----------
    signal : np.ndarray
        1-D real-valued signal (preferably bandpass-filtered before analysis).

    Returns
    -------
    phase : np.ndarray
        Instantaneous phase in radians, shape ``(N,)``, range (−π, π].
    amplitude : np.ndarray
        Instantaneous amplitude (analytic envelope), shape ``(N,)``, non-negative.

    Notes
    -----
    For meaningful results the signal should be narrow-band (single dominant
    frequency component).  Apply :func:`bandpass_filter` before calling this
    function.

    The Hilbert transform is computed via ``scipy.signal.hilbert``, which pads
    to the next fast FFT length internally.

    References
    ----------
    Gabor, D. (1946). Theory of communication. *Journal of the Institution of
    Electrical Engineers*, 93(26), 429–457.
    """
    signal = np.asarray(signal, dtype=np.float64)
    analytic = sp_signal.hilbert(signal)
    phase = np.angle(analytic)
    amplitude = np.abs(analytic)
    return phase, amplitude


# ---------------------------------------------------------------------------
# 3. Instantaneous Frequency
# ---------------------------------------------------------------------------

def compute_instantaneous_frequency(
    phase: np.ndarray,
    fs: float,
) -> np.ndarray:
    """Compute instantaneous frequency from an instantaneous phase array.

    The instantaneous frequency is the time-derivative of the unwrapped phase,
    converted to Hz:

        f(t) = (1 / 2π) · dφ_unwrapped / dt

    Parameters
    ----------
    phase : np.ndarray
        Instantaneous phase in radians as returned by
        :func:`compute_instantaneous_phase`.
    fs : float
        Sampling frequency in Hz used to scale the derivative.

    Returns
    -------
    np.ndarray
        Instantaneous frequency in Hz, shape ``(N,)``.  The first sample is
        duplicated from the second to maintain array length.

    Notes
    -----
    Phase unwrapping (``np.unwrap``) removes discontinuities at ±π before
    differentiation, which prevents spurious large-frequency estimates at
    phase-wrapping boundaries.

    Numerical differentiation is performed with ``np.diff``, which reduces the
    array length by one; the missing leading sample is filled by repeating the
    first computed value.
    """
    phase = np.asarray(phase, dtype=np.float64)
    unwrapped = np.unwrap(phase)
    # Derivative: Δφ / Δt, then divide by 2π to convert to Hz
    diff_phase = np.diff(unwrapped) * fs / (2.0 * np.pi)
    # Prepend first value so output length matches input
    inst_freq = np.concatenate([[diff_phase[0]], diff_phase])
    return inst_freq


# ---------------------------------------------------------------------------
# 4. Neurovascular Coupling Check
# ---------------------------------------------------------------------------

class HemoglobinCorrelationSummary:
    """Summarize HbO₂/HbR correlation and simple signal heuristics.

    The correlation threshold, variance score, and SNR estimate returned by
    this class are descriptive software outputs. They do not validate NVC or
    determine signal artifact, health, or physiological state.

    Parameters
    ----------
    window_s : float
        Duration of the analysis window in seconds (default 30 s).
    fs : float
        Expected sampling frequency in Hz (default 10.0 Hz).

    Attributes
    ----------
    window_samples : int
        Number of samples corresponding to ``window_s`` at ``fs``.

    Examples
    --------
    >>> checker = HemoglobinCorrelationSummary(window_s=30, fs=10.0)
    >>> result = checker.update(hbo2_array, hbr_array)
    >>> print(result["below_anticorrelation_display_threshold"], result["anticorrelation_r"])

    References
    ----------
    Scholkmann, F. et al. (2014). A review on continuous wave functional
    near-infrared spectroscopy and imaging instrumentation and methodology.
    *NeuroImage*, 85, 6–27.
    """

    DISPLAY_THRESHOLD: float = -0.3
    NVC_STRICT_THRESHOLD: float = -0.5   # Deprecated compatibility constant; not a validity criterion
    NVC_LOOSE_THRESHOLD: float = DISPLAY_THRESHOLD  # Deprecated compatibility alias

    def __init__(self, window_s: float = 30.0, fs: float = 10.0) -> None:
        self.window_s = window_s
        self.fs = fs
        self.window_samples: int = max(2, int(window_s * fs))

    # ------------------------------------------------------------------
    def update(
        self,
        hbo2_signal: np.ndarray,
        hbr_signal: np.ndarray,
    ) -> dict:
        """Compute descriptive HbO₂/HbR statistics for the supplied windows.

        If the signals are longer than ``window_samples``, only the most recent
        ``window_samples`` samples are used.

        Parameters
        ----------
        hbo2_signal : np.ndarray
            Oxygenated haemoglobin concentration signal (a.u. or µmol/L).
        hbr_signal : np.ndarray
            Deoxygenated haemoglobin concentration signal (same units).

        Returns
        -------
        dict
            Dictionary containing:

            ``anticorrelation_r`` : float
                Pearson correlation coefficient between HbO₂ and HbR. The name
                is retained for compatibility; the value may be positive.
            ``below_anticorrelation_display_threshold`` : bool
                Whether the correlation is below a display threshold. This is
                not a validity or physiological classification.
            ``artifact_score`` : float
                Uncalibrated variance-irregularity score in [0, 1]. It does
                not identify motion artifacts or their cause.
            ``artifact_probability`` : float
                Deprecated compatibility alias for ``artifact_score``; it is
                not a probability.
            ``snr_db`` : float
                Signal-to-noise ratio in dB, estimated as
                10·log₁₀(signal power / noise power), where noise is
                approximated by the high-frequency residual after a 5-point
                moving-average smoothing.
        """
        hbo2 = np.asarray(hbo2_signal, dtype=np.float64).ravel()
        hbr = np.asarray(hbr_signal, dtype=np.float64).ravel()

        # Trim to window
        if hbo2.size > self.window_samples:
            hbo2 = hbo2[-self.window_samples :]
        if hbr.size > self.window_samples:
            hbr = hbr[-self.window_samples :]

        # --- Pearson r ---------------------------------------------------
        min_len = min(hbo2.size, hbr.size)
        if min_len < 3:
            r_value = float("nan")
        else:
            paired = np.isfinite(hbo2[:min_len]) & np.isfinite(hbr[:min_len])
            paired_hbo2, paired_hbr = hbo2[:min_len][paired], hbr[:min_len][paired]
            if paired_hbo2.size < 3 or np.std(paired_hbo2) <= 1e-12 or np.std(paired_hbr) <= 1e-12:
                r_value = float("nan")
            else:
                r_value, _ = stats.pearsonr(paired_hbo2, paired_hbr)

        below_display_threshold = bool(r_value < self.DISPLAY_THRESHOLD)

        # --- Uncalibrated variance-irregularity score --------------------
        variance_score = self._compute_artifact_score(hbo2, hbr)

        # --- SNR ---------------------------------------------------------
        snr_db = self._compute_snr_db(hbo2)

        return {
            "anticorrelation_r": float(r_value),
            "below_anticorrelation_display_threshold": below_display_threshold,
            "artifact_score": float(np.clip(variance_score, 0.0, 1.0)),
            # Deprecated compatibility alias; this value is not a probability.
            "artifact_probability": float(np.clip(variance_score, 0.0, 1.0)),
            "snr_db": float(snr_db),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_artifact_score(
        hbo2: np.ndarray,
        hbr: np.ndarray,
    ) -> float:
        """Compute an uncalibrated variance-irregularity score.

        Splits each channel independently into roughly ten blocks, with a
        minimum block size of five samples, computes the coefficient of
        variation of its block variances, then averages the channel scores. A
        higher score indicates less uniform variance within this window; it
        does not identify motion or any cause.

        Returns a normalized score in [0, 1] clipped at CV = 2.0.
        """
        def channel_score(channel: np.ndarray) -> float:
            values = np.asarray(channel, dtype=float).ravel()
            values = values[np.isfinite(values)]
            sub_win = max(5, values.size // 10)
            n_wins = values.size // sub_win
            if n_wins < 2:
                return 0.0

            variances = np.array([
                np.var(values[i * sub_win : (i + 1) * sub_win])
                for i in range(n_wins)
            ])
            mean_var = float(np.mean(variances))
            if mean_var < 1e-12:
                return 0.0
            cv = float(np.std(variances) / mean_var)
            return float(np.clip(cv / 2.0, 0.0, 1.0))

        # HbO2 and HbR can have different baselines and scales. Scoring each
        # channel separately avoids introducing an artificial discontinuity
        # where concatenated channels meet.
        return float(np.mean([channel_score(hbo2), channel_score(hbr)]))

    # Backward-compatible alias. The returned value is a score, not a
    # probability; new code should use ``_compute_artifact_score``.
    _compute_artifact_probability = _compute_artifact_score

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_snr_db(sig: np.ndarray) -> float:
        """Estimate SNR in dB using a moving-average residual noise model.

        Signal power is taken as the variance of the original signal.
        Noise power is approximated as the variance of the residual between
        the original and a 5-point (or length-of-signal) moving-average
        smoothed version.

        Returns NaN when the estimate is undefined (too few or flat samples).
        """
        sig = np.asarray(sig, dtype=np.float64).ravel()
        sig = sig[np.isfinite(sig)]
        if sig.size < 3:
            return float("nan")
        signal_power = float(np.var(sig))
        if not np.isfinite(signal_power) or signal_power <= 1e-20:
            return float("nan")

        kernel_len = min(5, sig.size if sig.size % 2 else sig.size - 1)
        kernel = np.ones(kernel_len) / kernel_len
        pad = kernel_len // 2
        padded = np.pad(sig, pad, mode="reflect")
        smoothed = np.convolve(padded, kernel, mode="valid")
        noise = sig - smoothed
        noise_power = float(np.var(noise))

        if noise_power < 1e-20:
            return float("nan")

        return float(10.0 * np.log10(signal_power / noise_power))


# Backward-compatible import alias. This class summarizes correlation; it does
# not validate neurovascular coupling.
NeurovascularCouplingCheck = HemoglobinCorrelationSummary


# ---------------------------------------------------------------------------
# 5. Power Spectrum (Welch Method)
# ---------------------------------------------------------------------------

def compute_power_spectrum(
    signal: np.ndarray,
    fs: float,
    nperseg: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Estimate the power spectral density using Welch's method.

    Welch's method reduces spectral variance by averaging periodograms
    computed on overlapping, windowed segments of the signal.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Length of each Welch segment.  Defaults to ``min(len(signal), 256)``
        if the signal has at least 256 samples, otherwise the full signal
        length.  Longer segments improve frequency resolution; shorter
        segments improve variance reduction.

    Returns
    -------
    frequencies : np.ndarray
        Frequency axis in Hz, shape ``(M,)``.
    power : np.ndarray
        Power spectral density (V²/Hz or unit²/Hz), shape ``(M,)``.
    dominant_frequency : float
        Frequency in Hz with the highest power.
    dominant_power : float
        Power at the dominant frequency.

    Notes
    -----
    Uses a Hann window (default in ``scipy.signal.welch``) with 50 % overlap.

    Mayer wave dominance check example::

        freqs, psd, f_dom, p_dom = compute_power_spectrum(sig, fs=10.0)
        in_mayer = MAYER_WAVE_LOW_HZ <= f_dom <= MAYER_WAVE_HIGH_HZ

    References
    ----------
    Welch, P. D. (1967). The use of fast Fourier transform for the estimation
    of power spectra. *IEEE Transactions on Audio and Electroacoustics*, 15(2),
    70–73.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = signal.size

    if nperseg is None:
        nperseg = min(n, 256)

    # Ensure nperseg does not exceed signal length
    nperseg = min(nperseg, n)

    frequencies, power = sp_signal.welch(signal, fs=fs, nperseg=nperseg)
    dominant_idx = int(np.argmax(power))
    dominant_frequency = float(frequencies[dominant_idx])
    dominant_power = float(power[dominant_idx])

    return frequencies, power, dominant_frequency, dominant_power


# ---------------------------------------------------------------------------
# 6. ProcessedSignal Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProcessedSignal:
    """Container for a fully processed fNIRS/haemodynamic signal.

    Stores the raw input alongside all derived representations produced by the
    signal-processing pipeline so they can be passed between pipeline stages
    or serialised for downstream analysis.

    Attributes
    ----------
    raw : np.ndarray
        Original, unfiltered signal samples.
    filtered : np.ndarray
        Bandpass-filtered signal (output of :func:`bandpass_filter`).
    phase : np.ndarray
        Instantaneous phase in radians (output of
        :func:`compute_instantaneous_phase`).
    amplitude : np.ndarray
        Instantaneous amplitude / envelope in signal units.
    instantaneous_freq : np.ndarray
        Instantaneous frequency in Hz (output of
        :func:`compute_instantaneous_frequency`).
    timestamps : np.ndarray
        UTC or relative timestamps (seconds) corresponding to each sample.
        If not provided at construction, defaults to an empty array.
    fs : float
        Sampling frequency in Hz.
    """

    raw: np.ndarray
    filtered: np.ndarray
    phase: np.ndarray
    amplitude: np.ndarray
    instantaneous_freq: np.ndarray
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    fs: float = 10.0

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialise all fields to a plain Python dictionary.

        NumPy arrays are converted to lists so the result is JSON-serialisable
        with the standard library's ``json`` module.

        Returns
        -------
        dict
            Keys match the dataclass field names; array values are Python lists.
        """
        d = asdict(self)
        # Convert numpy arrays to lists for JSON compatibility
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                d[key] = value.tolist()
        return d


# ---------------------------------------------------------------------------
# 7. Normalisation Utilities
# ---------------------------------------------------------------------------

def z_score_normalize(signal: np.ndarray) -> np.ndarray:
    """Standardise a signal to zero mean and unit variance (z-score).

    Transforms the signal as:

        z(t) = (x(t) − μ) / σ

    where μ and σ are the sample mean and standard deviation of the input.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal.

    Returns
    -------
    np.ndarray
        Z-scored signal with mean ≈ 0 and standard deviation ≈ 1.
        If the standard deviation is zero (constant signal), the zero-mean
        signal is returned unchanged (i.e. all zeros).

    Notes
    -----
    Z-score normalisation is useful when comparing signals with different
    physical units or amplitudes, e.g., across fNIRS channels or subjects.
    """
    signal = np.asarray(signal, dtype=np.float64)
    mu = np.mean(signal)
    sigma = np.std(signal)
    if sigma < 1e-12:
        logger.warning(
            "z_score_normalize: standard deviation is effectively zero "
            "(sigma=%.3e). Returning zero-mean signal.",
            sigma,
        )
        return signal - mu
    return (signal - mu) / sigma


def min_max_normalize(
    signal: np.ndarray,
    out_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Scale a signal to a specified output range using min–max normalisation.

    Transforms the signal as:

        x_norm(t) = out_min + (x(t) − x_min) · (out_max − out_min)
                                                / (x_max − x_min)

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal.
    out_range : tuple of float, optional
        Desired ``(min, max)`` output range (default ``(0, 1)``).

    Returns
    -------
    np.ndarray
        Signal scaled to ``out_range``.  If the input signal is constant
        (zero range), the lower bound of ``out_range`` is returned for all
        samples.

    Raises
    ------
    ValueError
        If ``out_range[0] >= out_range[1]``.

    Notes
    -----
    Min–max normalisation is sensitive to outliers; prefer :func:`z_score_normalize`
    when extreme values are expected.
    """
    signal = np.asarray(signal, dtype=np.float64)
    out_min, out_max = out_range

    if out_min >= out_max:
        raise ValueError(
            f"out_range must satisfy out_min < out_max, got {out_range}."
        )

    x_min = signal.min()
    x_max = signal.max()
    signal_range = x_max - x_min

    if signal_range < 1e-12:
        logger.warning(
            "min_max_normalize: signal range is effectively zero. "
            "Returning array filled with out_range lower bound (%.4f).",
            out_min,
        )
        return np.full_like(signal, out_min)

    normalised = (signal - x_min) / signal_range
    return normalised * (out_max - out_min) + out_min
