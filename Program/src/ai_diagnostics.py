"""
ai_diagnostics.py
=================
Local signal-summary engine for NeuroResonance.

Provides deterministic, local-only descriptions of experimental respiration
and fNIRS signal metrics. No remote model receives session data.
"""

from __future__ import annotations

import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """Convert a duration in seconds to a human-readable string (mm:ss).

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string such as ``'4 min 32 s'``.
    """
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes} min {secs:02d} s"


def _format_optional_metric(value: object, precision: int, suffix: str = "") -> str:
    """Format a finite metric while keeping missing estimates explicitly unavailable."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "not available"
    if not math.isfinite(numeric):
        return "not available"
    return f"{numeric:.{precision}f}{suffix}"


# ---------------------------------------------------------------------------
# NeuroDiagnosticsEngine compatibility stub (remote execution disabled)
# ---------------------------------------------------------------------------

class NeuroDiagnosticsEngine:
    """Deprecated compatibility stub; this project does not send metrics remotely."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        del api_key
        raise RuntimeError(
            "Remote LLM diagnostics are disabled. Use FallbackDiagnosticsEngine for local summaries."
        )


# ---------------------------------------------------------------------------
# FallbackDiagnosticsEngine  (rule-based, no API key required)
# ---------------------------------------------------------------------------

class FallbackDiagnosticsEngine:
    """Rule-based signal-summary engine that requires no external API.

    Generates deterministic Markdown reports and real-time insights locally.
    PLV thresholds, significance tests, clinical categories and phase lag are
    deliberately omitted because they are not validated by this project.
    """

    def generate_report(
        self,
        resonance_state: dict,
        nvc_check: dict,
        power_spectrum_data: dict,
        session_duration_s: float,
    ) -> str:
        """Generate a rule-based educational signal summary.

        Args:
            resonance_state: Dictionary with ``plv_pct``, ``coherence``,
                and optional data provenance.
            nvc_check: Dictionary with ``anti_correlation_r`` and
                optional legacy significance text (ignored).
            power_spectrum_data: Dictionary with ``dominant_freq_hz`` and
                optional ``band_power``.
            session_duration_s: Session length in seconds.

        Returns:
            A Markdown-formatted signal summary string.
        """
        plv = _format_optional_metric(resonance_state.get("plv_pct"), 1, "%")
        coherence = _format_optional_metric(resonance_state.get("coherence"), 3)
        nvc_r = _format_optional_metric(nvc_check.get("anti_correlation_r"), 3)
        dom_freq = _format_optional_metric(power_spectrum_data.get("dominant_freq_hz"), 3, " Hz")
        provenance = resonance_state.get("data_provenance") or "not supplied"
        duration = _format_duration(session_duration_s)
        return (
            "## Experimental signal summary\n\n"
            f"Session duration: **{duration}**. Data source: **{provenance}**.\n\n"
            f"Descriptive PLV: **{plv}**. Welch coherence in 0.07–0.15 Hz: **{coherence}**. "
            f"Respiration dominant frequency in that band: **{dom_freq}**. HbO₂/HbR sample correlation: **{nvc_r}**.\n\n"
            "These window-level statistics do not establish statistical significance, physiological coupling, "
            "causality, cerebral origin, neurovascular integrity, diagnosis, or a clinical state. "
            "Inspect acquisition metadata, channel mapping, motion, systemic physiology, and signal quality. "
            "No phase lag is reported because this implementation does not validate it as a physiological delay."
        )

    def generate_realtime_insight(self, resonance_state: dict) -> str:
        """Generate a concise real-time dashboard insight (rule-based).

        Args:
            resonance_state: Dictionary with PLV and coherence values.

        Returns:
            A 1–2 sentence plain-language description of the signal metrics.
        """
        plv = resonance_state.get("plv_pct", 0.0)
        coherence = resonance_state.get("coherence")
        return (
            "Experimental signal summary: "
            f"PLV={_format_optional_metric(plv, 1, '%')}, "
            f"spectral coherence={_format_optional_metric(coherence, 3)}. These values describe the "
            "selected signals and window only; they do not establish a clinical "
            "or autonomic state. Check signal quality and acquisition mode."
        )

# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_diagnostics_engine(
    api_key: Optional[str] = None,
) -> FallbackDiagnosticsEngine:
    """Return the local deterministic engine; ignore legacy remote API keys."""
    del api_key
    return FallbackDiagnosticsEngine()
