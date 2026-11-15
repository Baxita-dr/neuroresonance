"""
dashboard.py — NeuroResonance Laboratory Research Dashboard

Main interactive Streamlit application for respiration and fNIRS signal
visualization and descriptive phase metrics.

Usage:
    streamlit run src/dashboard.py
"""

import os
import sys
import time
import math
import logging
import traceback
import ipaddress
from datetime import datetime, timezone
from urllib.parse import urlsplit
from typing import Optional, Dict, Any, List, Tuple, Union

logger = logging.getLogger("NeuroResonanceDashboard")


def _validate_camera_url(value: str) -> str:
    """Allow only literal private-network camera endpoints on common media ports."""
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https", "rtsp", "rtsps"} or not parsed.hostname:
        raise ValueError("Podaj adres kamery z protokołem http(s)/rtsp(s).")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError as exc:
        raise ValueError("Adres kamery musi być prywatnym adresem IP, nie nazwą hosta.") from exc
    if not address.is_private or address.is_loopback or address.is_link_local or address.is_multicast:
        raise ValueError("Dozwolony jest wyłącznie adres IP urządzenia w prywatnej sieci lokalnej.")
    allowed_ports = {80, 443, 554, 8080, 4747, 8554}
    try:
        port = parsed.port or (443 if parsed.scheme in {"https", "rtsps"} else 80)
    except ValueError as exc:
        raise ValueError("Nieprawidłowy port kamery.") from exc
    if port not in allowed_ports:
        raise ValueError("Port kamery nie znajduje się na liście dozwolonych portów.")
    return value.strip()

# Path setup: allow importing from src when executed from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import streamlit as st
import plotly.graph_objects as go

from src.camera_tracker import CameraTracker, RespiratorySignal
from src.lsl_streamer import fNIRSSimulator
from src.lsl_integration import SelectedLSLSession, discover_lsl_streams, inspect_lsl_stream
from src.xdf_playback import XDFPlaybackSource, load_xdf_recording
from src.infrastructure.lsl_receiver import LSLBackgroundReceiver
from src.session_clock import elapsed_seconds, duration_reached
from src.dashboard_metrics import (
    build_session_insight_state,
    estimate_sampling_rate_hz,
    summarize_signal_quality,
)
from src.ring_buffer import CircularRingBuffer
from src.signal_processing import HemoglobinCorrelationSummary
from src.coherence_engine import (
    ResonanceEstimator,
    ResonanceState,
    compute_lissajous,
)
from src.ai_diagnostics import FallbackDiagnosticsEngine
from src import __version__ as APP_VERSION
from src.demo_data import DEMO_SEED, build_deterministic_demo_recording

# ===========================================================================
# Configuration & Constants
# ===========================================================================
PAGE_TITLE = "NeuroResonance"
PAGE_ICON = None

FS_DISPLAY = 10                  # Simulator target/sample-rate baseline (Hz), not displayed as an observation
ANALYSIS_WINDOW_SECONDS = 100    # Minimum analysis window; captures ~10 cycles at 0.1 Hz
BUFFER_SECONDS = ANALYSIS_WINDOW_SECONDS
BUFFER_LEN = FS_DISPLAY * BUFFER_SECONDS + 2  # preserve both boundaries of the full time window
DEFAULT_RESP_HZ = 0.1            # Fixed 6 breaths/min demonstration pacing reference
SLEEP_INTERVAL = 1.0             # Full-page charts and metrics refresh cadence (s)
CAMERA_PREVIEW_FPS = 15          # Camera preview redraw rate between chart refreshes

# ===========================================================================
# Streamlit Page Config & Custom Styling
# ===========================================================================
st.set_page_config(
    page_title=PAGE_TITLE,
    layout="wide",
    page_icon=PAGE_ICON,
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --nr-black: #05070c;
        --nr-panel: #0b111c;
        --nr-panel-raised: #101a29;
        --nr-line: rgba(93, 151, 214, .18);
        --nr-blue: #4d8dff;
        --nr-cyan: #54e5f5;
        --nr-muted: #8d9bb0;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--nr-black);
        color: #edf5ff;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 78% 4%, rgba(26, 88, 180, .17), transparent 30%),
            radial-gradient(ellipse at 8% 54%, rgba(17, 90, 119, .09), transparent 30%),
            #05070c;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1720px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        container-type: inline-size;
        container-name: nr-main;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1421 0%, #080d15 42%, #06090f 100%);
        border-right: 1px solid rgba(88, 145, 210, .18);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label { color: #c4d0e0; }
    [data-testid="stSidebar"] hr { border-color: rgba(111, 153, 204, .16); }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .75rem; }
    h1, h2, h3, h4, h5 { color: #f3f8ff; letter-spacing: -.025em; }
    h5 {
        color: #91a8c4;
        font-size: .7rem;
        font-weight: 700;
        letter-spacing: .09em;
        text-transform: uppercase;
        margin: .35rem 0 .65rem;
    }
    [data-testid="stCaptionContainer"] { color: var(--nr-muted); }
    [data-testid="stPlotlyChart"] {
        background: linear-gradient(150deg, rgba(16, 27, 43, .82), rgba(8, 13, 21, .94));
        border: 1px solid var(--nr-line);
        border-radius: 14px;
        padding: 8px 8px 2px;
        box-shadow: 0 12px 34px rgba(0, 0, 0, .22), inset 0 1px rgba(255,255,255,.025);
        overflow: hidden;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(17, 29, 47, .92), rgba(9, 14, 23, .96));
        border: 1px solid var(--nr-line);
        border-radius: 12px;
        padding: 12px 14px;
    }
    [data-testid="stMetricLabel"] { color: #9aacc2; font-size: .74rem; }
    [data-testid="stMetricValue"] { color: #eff8ff; font-size: 1.4rem; }
    [data-testid="stAlert"] {
        border: 1px solid rgba(68, 143, 226, .24);
        border-radius: 12px;
        background: linear-gradient(110deg, rgba(14, 45, 78, .5), rgba(11, 19, 31, .86));
    }
    [data-testid="stExpander"] {
        border: 1px solid var(--nr-line);
        border-radius: 12px;
        background: rgba(10, 16, 26, .7);
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {
        background-color: #0b1421;
        border-color: rgba(104, 157, 219, .25);
        border-radius: 9px;
        color: #edf5ff;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: var(--nr-cyan);
        box-shadow: 0 0 0 1px rgba(84, 229, 245, .24);
    }
    .stButton > button {
        border: 1px solid rgba(106, 187, 255, .45);
        border-radius: 10px;
        background: linear-gradient(112deg, #1767d4 0%, #187fe8 48%, #14a8cb 100%);
        color: #fff;
        font-weight: 650;
        box-shadow: 0 5px 20px rgba(19, 113, 219, .2), inset 0 1px rgba(255,255,255,.18);
        transition: transform .16s ease, filter .16s ease, box-shadow .16s ease;
    }
    .stButton > button:hover {
        border-color: rgba(130, 220, 255, .75);
        filter: brightness(1.12);
        box-shadow: 0 7px 24px rgba(28, 141, 245, .3), 0 0 20px rgba(33, 190, 231, .12);
        transform: translateY(-1px);
        color: #fff;
    }
    .stButton > button:focus { box-shadow: 0 0 0 2px #05070c, 0 0 0 4px rgba(84, 229, 245, .7); }
    .nr-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 7px 2px 15px;
        border-bottom: 1px solid rgba(111, 153, 204, .16);
        margin-bottom: 6px;
    }
    .nr-brand-mark {
        display: grid;
        place-items: center;
        width: 42px;
        height: 42px;
        border: 1px solid rgba(84, 229, 245, .42);
        border-radius: 13px;
        background: linear-gradient(145deg, rgba(27, 106, 167, .42), rgba(22, 47, 84, .3));
        color: var(--nr-cyan);
        font-size: 29px;
        line-height: 1;
        box-shadow: 0 0 25px rgba(49, 159, 224, .12), inset 0 0 15px rgba(84,229,245,.08);
    }
    .nr-brand-copy strong { display: block; color: #f3f8ff; font-size: 1.03rem; letter-spacing: -.02em; }
    .nr-brand-copy span { display: block; margin-top: 3px; color: #8da2ba; font-size: .62rem; letter-spacing: .16em; }
    .nr-hero {
        position: relative;
        overflow: hidden;
        display: flex;
        justify-content: space-between;
        align-items: center;
        min-height: 132px;
        margin: 0 0 1rem;
        padding: 22px 28px;
        border: 1px solid rgba(84, 154, 226, .25);
        border-radius: 18px;
        background:
            radial-gradient(ellipse at 86% 4%, rgba(36, 121, 230, .28), transparent 38%),
            radial-gradient(ellipse at 62% 120%, rgba(18, 165, 204, .13), transparent 36%),
            linear-gradient(110deg, #0a111b 0%, #0b1421 53%, #0a1728 100%);
        box-shadow: 0 18px 50px rgba(0, 0, 0, .28), inset 0 1px rgba(255,255,255,.035);
    }
    .nr-hero:after {
        content: "";
        position: absolute;
        right: 10%;
        top: -100px;
        width: 360px;
        height: 240px;
        border: 1px solid rgba(82, 185, 250, .13);
        border-radius: 50%;
        transform: rotate(-20deg);
        box-shadow: 0 0 55px rgba(37, 131, 235, .08), inset 0 0 35px rgba(20, 160, 214, .04);
        pointer-events: none;
    }
    .nr-hero-copy, .nr-hero-meta { position: relative; z-index: 1; }
    .nr-eyebrow { color: var(--nr-cyan); font-size: .67rem; font-weight: 750; letter-spacing: .17em; text-transform: uppercase; }
    .nr-hero h1 { margin: 5px 0 2px; font-size: clamp(1.8rem, 3vw, 2.35rem); letter-spacing: -.04em; }
    .nr-hero p { margin: 0; color: #aab9cc; font-size: .9rem; }
    .nr-hero-meta { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; max-width: 44%; }
    .nr-chip {
        display: inline-flex;
        align-items: center;
        padding: 7px 11px;
        border: 1px solid rgba(100, 163, 224, .25);
        border-radius: 999px;
        background: rgba(9, 17, 29, .72);
        color: #b9cce2;
        font-size: .68rem;
        font-weight: 650;
        letter-spacing: .06em;
        white-space: nowrap;
    }
    .nr-chip-accent { color: #7cecff; border-color: rgba(84, 229, 245, .32); background: rgba(23, 104, 157, .17); }
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem .8rem 2rem; }
        .nr-hero { align-items: flex-start; flex-direction: column; gap: 14px; padding: 20px; }
        .nr-hero-meta { justify-content: flex-start; max-width: 100%; }
    }
    @container nr-main (max-width: 1100px) {
        [data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
            gap: .9rem;
        }
        [data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
        }
    }
    @keyframes breathPulse {
        0%   { transform: scale(0.9); opacity: 0.6; box-shadow: 0 0 10px #4d8dff; }
        50%  { transform: scale(1.3); opacity: 1.0; box-shadow: 0 0 25px #54e5f5; }
        100% { transform: scale(0.9); opacity: 0.6; box-shadow: 0 0 10px #4d8dff; }
    }
    .resonance-pacer {
        display: inline-block;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #54e5f5;
        animation: breathPulse 10s ease-in-out infinite;
        vertical-align: middle;
        margin-right: 8px;
    }
    .metric-badge {
        padding: 6px 12px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# Session State Initialization
# ===========================================================================
def _init_session_state() -> None:
    defaults = {
        "session_active": False,
        "session_start_monotonic": None,
        # Hardware & processing handles
        "simulator": None,
        "bridge": None,
        "lsl_session": None,
        "lsl_receiver": None,
        "lsl_candidates": [],
        "xdf_recording": None,
        "xdf_file_name": None,
        "xdf_loaded_upload_id": None,
        "xdf_current_upload_id": None,
        "xdf_playback": None,
        "xdf_fnirs_stream_index": None,
        "xdf_hbo2_channel_index": None,
        "xdf_hbr_channel_index": None,
        "xdf_respiration_mode": "No measured respiration",
        "xdf_respiration_stream_index": None,
        "xdf_respiration_channel_index": None,
        "xdf_playback_speed": 1.0,
        "xdf_quality_warnings": [],
        "xdf_builtin_demo": False,
        "session_is_synthetic_demo": False,
        "lsl_fnirs_metadata": None,
        "lsl_fnirs_metadata_selector": None,
        "lsl_resp_metadata": None,
        "lsl_resp_metadata_selector": None,
        "lsl_recording_path": None,
        "lsl_recording_enabled": False,
        "lsl_origin": None,
        "lsl_last_quality_warnings": [],
        "lsl_last_receive_monotonic": None,
        "session_data_source": None,
        "sim_resp_rng": None,
        "camera": None,
        "estimator": None,
        "nvc_checker": None,
        # Signal buffers & zero-allocation ring buffers (empty at initialization)
        "ring_time": CircularRingBuffer(capacity=BUFFER_LEN),
        "ring_resp": CircularRingBuffer(capacity=BUFFER_LEN),
        "ring_hbo": CircularRingBuffer(capacity=BUFFER_LEN),
        "ring_hbr": CircularRingBuffer(capacity=BUFFER_LEN),
        "buf_time": np.empty(0, dtype=np.float64),
        "buf_resp": np.empty(0, dtype=np.float64),
        "buf_hbo": np.empty(0, dtype=np.float64),
        "buf_hbr": np.empty(0, dtype=np.float64),
        "valid_samples_count": 0,
        "session_active_duration": 0.0,
        "cached_report": None,
        "cached_session_bundle": None,
        "current_plv": float("nan"),
        "current_state": "IDLE",
        "analysis_error": None,
        "diagnostic_events": [],
        "diagnostic_last_ui_refresh": None,
        "state_color": "#888888",
        "nvc_valid": False,
        "nvc_r": None,
        "last_ai_insight": "System idle. Configure source and press Start.",
        "last_insight_time": 0.0,
        "annotated_frame": None,
        "camera_source": 0,
        "custom_cam_url": "http://192.168.1.100:8080/video",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session_state()


def _record_diagnostic(level: str, message: str, exc: Optional[BaseException] = None) -> None:
    """Keep a bounded, downloadable event log without storing signal samples."""
    severity = level.upper()
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    trace = ""
    if exc is not None:
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    events = st.session_state.setdefault("diagnostic_events", [])
    active = bool(st.session_state.get("session_active"))
    source = (
        st.session_state.get("session_data_source")
        if active
        else globals().get("DATA_SOURCE", st.session_state.get("session_data_source") or "UNKNOWN")
    )
    session_started = st.session_state.get("session_start_monotonic")
    elapsed = (
        elapsed_seconds(session_started)
        if active and session_started is not None
        else st.session_state.get("session_active_duration") if session_started is not None else None
    )
    signature = (severity, str(message), trace)
    if events and events[-1].get("_signature") == signature:
        events[-1]["last_timestamp"] = timestamp
        events[-1]["occurrences"] = int(events[-1].get("occurrences", 1)) + 1
    else:
        events.append({
            "timestamp": timestamp,
            "last_timestamp": timestamp,
            "severity": severity,
            "message": str(message),
            "source": str(source),
            "elapsed_s": elapsed,
            "traceback": trace,
            "occurrences": 1,
            "_signature": signature,
        })
        del events[:-100]
    st.session_state["diagnostic_events"] = events

    log_level = getattr(logging, severity, logging.INFO)
    logger.log(
        log_level,
        "%s",
        message,
        exc_info=(type(exc), exc, exc.__traceback__) if exc is not None else None,
    )


def _format_diagnostic_report() -> str:
    events = st.session_state.get("diagnostic_events", [])
    active = bool(st.session_state.get("session_active"))
    source = (
        st.session_state.get("session_data_source")
        if active
        else globals().get("DATA_SOURCE", st.session_state.get("session_data_source") or "UNKNOWN")
    )
    started = st.session_state.get("session_start_monotonic")
    elapsed = elapsed_seconds(started) if active and started is not None else st.session_state.get("session_active_duration", 0.0)
    playback = st.session_state.get("xdf_playback")
    target_duration = playback.duration_s if active and playback is not None else globals().get("session_duration")
    samples = len(st.session_state.get("buf_time", []))
    rate = st.session_state.get("measured_sample_rate_hz")
    last_refresh = st.session_state.get("diagnostic_last_ui_refresh") or "brak aktywnej sesji"
    errors = sum(item.get("severity") == "ERROR" for item in events)
    warnings = sum(item.get("severity") == "WARNING" for item in events)

    lines = [
        "NeuroResonance — lokalny raport diagnostyczny",
        f"Wygenerowano: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Źródło danych: {source}",
        f"Sesja aktywna: {'tak' if active else 'nie'}",
        f"Czas sesji: {float(elapsed or 0.0):.1f} s" + (f" / {float(target_duration):.1f} s" if target_duration is not None else ""),
        f"Próbki w buforze analizy: {samples}",
        f"Oszacowana częstotliwość próbkowania: {float(rate):.3f} Hz" if rate is not None and math.isfinite(float(rate)) and float(rate) > 0 else "Oszacowana częstotliwość próbkowania: brak",
        f"Ostatnie odświeżenie obliczeń: {last_refresh}",
        f"Zdarzenia: {len(events)}; błędy: {errors}; ostrzeżenia: {warnings}",
        "",
        "Historia zdarzeń (bez próbek sygnału):",
    ]
    for item in events:
        elapsed_text = f"; czas sesji: {item['elapsed_s']:.1f} s" if item.get("elapsed_s") is not None else ""
        lines.extend([
            "",
            f"[{item['severity']}] {item['timestamp']} — {item['source']}{elapsed_text}",
            f"{item['message']} (wystąpienia: {item.get('occurrences', 1)})",
        ])
        if item.get("traceback"):
            lines.extend(["", item["traceback"].rstrip()])
    return "\n".join(lines) + "\n"


def _render_diagnostics_panel() -> None:
    events = st.session_state.get("diagnostic_events", [])
    error_count = sum(item.get("severity") == "ERROR" for item in events)
    warning_count = sum(item.get("severity") == "WARNING" for item in events)
    title = f"Diagnostyka sesji — błędy: {error_count}, ostrzeżenia: {warning_count}"
    with st.expander(title, expanded=bool(error_count or events or st.session_state.get("session_active"))):
        active = bool(st.session_state.get("session_active"))
        source = st.session_state.get("session_data_source") if active else DATA_SOURCE
        started = st.session_state.get("session_start_monotonic")
        elapsed = elapsed_seconds(started) if active and started is not None else st.session_state.get("session_active_duration", 0.0)
        playback = st.session_state.get("xdf_playback")
        target_duration = playback.duration_s if active and playback is not None else (session_duration if active else None)
        samples = len(st.session_state.get("buf_time", []))
        rate = st.session_state.get("measured_sample_rate_hz")
        last_refresh = st.session_state.get("diagnostic_last_ui_refresh")
        status = "aktywna" if active else "zatrzymana / gotowa"
        rate_text = f" · fs: {float(rate):.2f} Hz" if rate is not None and math.isfinite(float(rate)) and float(rate) > 0 else ""
        duration_text = f" / {float(target_duration):.1f} s" if target_duration is not None else ""
        st.caption(
            f"Status: {status} · źródło: {source} · czas: {float(elapsed or 0.0):.1f} s{duration_text} · "
            f"próbki w buforze: {samples}{rate_text}"
        )
        if last_refresh:
            st.caption(f"Ostatni przebieg aktualizacji: {last_refresh}")
        if error_count:
            latest_error = next(item for item in reversed(events) if item.get("severity") == "ERROR")
            st.error(f"Ostatni błąd: {latest_error['message']}")
        elif not events:
            st.info("Nie zarejestrowano błędów. Uruchom symulację, aby sprawdzić działanie sesji.")
        else:
            st.success("Nie zarejestrowano błędów; szczegóły sesji są poniżej.")

        for item in reversed(events[-20:]):
            label = f"{item['severity']} · {item['timestamp']} · {item['source']} · {item['message'][:90]}"
            with st.expander(label, expanded=False):
                st.caption(
                    f"Ostatnie wystąpienie: {item['last_timestamp']} · "
                    f"wystąpienia: {item.get('occurrences', 1)}"
                )
                if item.get("traceback"):
                    st.code(item["traceback"], language="text")

        download_col, clear_col = st.columns(2)
        with download_col:
            st.download_button(
                "Pobierz raport diagnostyczny",
                data=_format_diagnostic_report(),
                file_name="neuroresonance_diagnostics.txt",
                mime="text/plain",
                key="download_diagnostics_report",
            )
        with clear_col:
            if st.button("Wyczyść historię", key="clear_diagnostics_history"):
                st.session_state["diagnostic_events"] = []
                st.rerun()
        st.caption("Raport zawiera czas, tryb pracy i błędy ze stosem wywołań; nie zapisuje próbek sygnału.")


def _clear_fnirs_mapping() -> None:
    st.session_state["lsl_fnirs_metadata"] = None
    st.session_state["lsl_fnirs_metadata_selector"] = None
    st.session_state["lsl_confirm_derived_hb"] = False
    st.session_state["lsl_confirm_measured_resp"] = False
    for key in ("lsl_hbo_index", "lsl_hbr_index", "lsl_resp_same_index"):
        st.session_state[key] = None


def _clear_resp_mapping() -> None:
    st.session_state["lsl_resp_metadata"] = None
    st.session_state["lsl_resp_metadata_selector"] = None
    st.session_state["lsl_resp_index"] = None
    st.session_state["lsl_confirm_measured_resp"] = False


def _clear_hb_confirmation() -> None:
    st.session_state["lsl_confirm_derived_hb"] = False


def _clear_resp_confirmation() -> None:
    st.session_state["lsl_confirm_measured_resp"] = False


def _clear_xdf_fnirs_mapping() -> None:
    st.session_state["xdf_hbo2_channel_index"] = None
    st.session_state["xdf_hbr_channel_index"] = None
    st.session_state["xdf_confirm_derived_hb"] = False
    _clear_xdf_resp_mapping()


def _clear_xdf_resp_mapping() -> None:
    st.session_state["xdf_respiration_channel_index"] = None
    st.session_state["xdf_confirm_measured_respiration"] = False

# ===========================================================================
# Sidebar Controls
# ===========================================================================
with st.sidebar:
    st.markdown(
        '<div class="nr-brand"><div class="nr-brand-mark">∿</div>'
        '<div class="nr-brand-copy"><strong>NeuroResonance</strong>'
        '<span>EXPERIMENTAL SIGNALS</span></div></div>',
        unsafe_allow_html=True,
    )

    DATA_SOURCE = st.selectbox(
        "Data Stream Source",
        options=["SIMULATOR", "WEBCAM", "LIVE_HARDWARE", "XDF_FILE"],
        index=0,
        disabled=st.session_state["session_active"],
        help="Choose synthetic, webcam, live LSL, or a recorded LabRecorder/XDF file. File replay uses the file timestamps."
    )
    if DATA_SOURCE == "SIMULATOR":
        st.caption(f"Repeatable synthetic mode · seed {DEMO_SEED} · generated signals are not physiological measurements.")

    if DATA_SOURCE == "LIVE_HARDWARE":
        disabled = st.session_state["session_active"]
        with st.expander("CortiView outlet guide (check your installed version)"):
            st.caption(
                "The CortiView 1.11.5 manual describes a numeric 'stats' outlet with alternating HbO/HbR values, "
                "a 'raw' outlet with two-wavelength optical data plus IMU channels, and a 'mirror' marker outlet. "
                "For this dashboard, use a derived-Hb stats stream only after checking its actual metadata and channel order. "
                "Raw wavelength data are not converted here; marker streams are excluded from signal choices."
            )
        if st.button("Discover LSL streams", disabled=disabled):
            try:
                st.session_state["lsl_candidates"] = discover_lsl_streams()
            except Exception as exc:
                _record_diagnostic("ERROR", "Nie udało się wyszukać strumieni LSL.", exc)
                st.error(f"LSL discovery failed: {exc}")

        candidates = st.session_state["lsl_candidates"]
        candidate_by_key = {item["selector"]: item for item in candidates}
        stream_keys = [
            item["selector"] for item in candidates
            if item.get("numeric_samples", False)
        ]
        non_numeric_candidates = [
            item for item in candidates if not item.get("numeric_samples", False)
        ]
        st.caption(
            f"Found {len(candidates)} LSL outlet(s); {len(stream_keys)} numeric outlet(s) can be used for signal plots."
        )
        if non_numeric_candidates:
            with st.expander("Non-numeric or unsupported outlets (not selectable as signals)"):
                for item in non_numeric_candidates:
                    st.write(
                        f"{item.get('name') or '(unnamed)'} | {item.get('type') or '(no type)'} | "
                        f"format={item.get('channel_format') or 'unknown'} | marker/text or unsupported samples"
                    )
        if candidates and not stream_keys:
            st.warning("No numeric LSL outlet was found. String marker outlets cannot be used as signal waveforms.")
        if st.session_state.get("lsl_fnirs_selector") not in stream_keys:
            st.session_state["lsl_fnirs_selector"] = None
            _clear_fnirs_mapping()
        if st.session_state.get("lsl_resp_selector") not in stream_keys:
            st.session_state["lsl_resp_selector"] = None
            _clear_resp_mapping()

        def _stream_label(key: Optional[str]) -> str:
            if key is None:
                return "Choose an outlet…"
            item = candidate_by_key[key]
            rate = item.get("nominal_srate_hz", 0.0)
            rate_text = f"{rate:g} Hz" if rate else "irregular rate"
            return (
                f"{item.get('name') or '(unnamed)'} | {item.get('type') or '(no type)'} | "
                f"{item.get('channel_count', 0)} ch | {rate_text} | "
                f"{item.get('hostname') or '(host unknown)'} | {key[:12]}"
            )

        st.selectbox(
            "fNIRS outlet",
            options=[None, *stream_keys],
            format_func=_stream_label,
            key="lsl_fnirs_selector",
            on_change=_clear_fnirs_mapping,
            disabled=disabled or not stream_keys,
        )
        fnirs_selector = st.session_state.get("lsl_fnirs_selector")
        if fnirs_selector in candidate_by_key:
            item = candidate_by_key[fnirs_selector]
            st.caption(
                f"Outlet: name={item['name']!r}; type={item['type']!r}; "
                f"source_id={item['source_id']!r}; hostname={item['hostname']!r}; "
                f"channels={item['channel_count']}; format={item['channel_format']!r}; "
                f"nominal rate={item['nominal_srate_hz']} Hz"
            )
            if st.button("Inspect selected fNIRS channels", disabled=disabled):
                try:
                    _clear_fnirs_mapping()
                    st.session_state["lsl_fnirs_metadata"] = inspect_lsl_stream(fnirs_selector)
                    st.session_state["lsl_fnirs_metadata_selector"] = fnirs_selector
                except Exception as exc:
                    _record_diagnostic("ERROR", "Nie udało się odczytać metadanych kanałów fNIRS z LSL.", exc)
                    st.error(f"Could not inspect fNIRS channels: {exc}")

        fnirs_meta = st.session_state.get("lsl_fnirs_metadata")
        fnirs_ready = (
            fnirs_meta is not None
            and st.session_state.get("lsl_fnirs_metadata_selector") == fnirs_selector
        )
        if fnirs_ready:
            fnirs_rows = fnirs_meta.get("channels", [])
            if not fnirs_meta.get("metadata_complete") or not fnirs_meta.get("channel_count_matches_metadata"):
                st.warning("The outlet has partial channel metadata. Confirm each selected index in the acquisition software before mapping.")
            st.caption("fNIRS channels: index | label | unit | type")
            for channel in fnirs_rows:
                st.write(
                    f"{channel['index']} | {channel['label'] or 'label not supplied'} | "
                    f"{channel['unit'] or 'unit unknown'} | {channel['type'] or 'type unknown'}"
                )

            def _fnirs_channel_label(index: Optional[int]) -> str:
                if index is None:
                    return "Choose a channel…"
                channel = fnirs_rows[index]
                return (
                    f"{index}: {channel['label'] or 'label not supplied'} [{channel['unit'] or 'unit unknown'}] "
                    f"({channel['type'] or 'type unknown'})"
                )

            fnirs_indices: List[Optional[int]] = [None, *[row["index"] for row in fnirs_rows]]
            st.selectbox("HbO₂ channel index", fnirs_indices, format_func=_fnirs_channel_label, key="lsl_hbo_index", on_change=_clear_hb_confirmation, disabled=disabled)
            st.selectbox("HbR channel index", fnirs_indices, format_func=_fnirs_channel_label, key="lsl_hbr_index", on_change=_clear_hb_confirmation, disabled=disabled)
            st.checkbox(
                "I verified these selected indices carry derived HbO₂/HbR values, not raw optical intensity or optical density.",
                key="lsl_confirm_derived_hb",
                disabled=disabled,
            )
        else:
            st.info("Select an outlet and inspect its full channel metadata before mapping HbO₂/HbR.")

        respiration_mode = st.radio(
            "Measured respiration outlet",
            ["Same as fNIRS outlet", "Separate outlet", "No measured respiration"],
            key="lsl_respiration_mode",
            on_change=_clear_resp_confirmation,
            disabled=disabled,
        )
        resp_selector = None
        if respiration_mode == "Same as fNIRS outlet":
            if fnirs_ready:
                resp_rows = fnirs_meta.get("channels", [])
                st.selectbox(
                    "Measured respiration channel index",
                    [None, *[row["index"] for row in resp_rows]],
                    format_func=lambda index: "Choose a channel…" if index is None else f"{index}: {resp_rows[index]['label'] or 'label not supplied'} [{resp_rows[index]['unit'] or 'unit unknown'}] ({resp_rows[index]['type'] or 'type unknown'})",
                    key="lsl_resp_same_index",
                    on_change=_clear_resp_confirmation,
                    disabled=disabled,
                )
        elif respiration_mode == "Separate outlet":
            st.selectbox(
                "Respiration outlet",
                options=[None, *stream_keys],
                format_func=_stream_label,
                key="lsl_resp_selector",
                on_change=_clear_resp_mapping,
                disabled=disabled or not stream_keys,
            )
            resp_selector = st.session_state.get("lsl_resp_selector")
            if resp_selector in candidate_by_key and st.button("Inspect selected respiration channels", disabled=disabled):
                try:
                    _clear_resp_mapping()
                    st.session_state["lsl_resp_metadata"] = inspect_lsl_stream(resp_selector)
                    st.session_state["lsl_resp_metadata_selector"] = resp_selector
                except Exception as exc:
                    _record_diagnostic("ERROR", "Nie udało się odczytać metadanych kanału oddechowego LSL.", exc)
                    st.error(f"Could not inspect respiration channels: {exc}")
            resp_meta = st.session_state.get("lsl_resp_metadata")
            resp_ready = resp_meta is not None and st.session_state.get("lsl_resp_metadata_selector") == resp_selector
            if resp_ready:
                resp_rows = resp_meta.get("channels", [])
                if not resp_meta.get("metadata_complete") or not resp_meta.get("channel_count_matches_metadata"):
                    st.warning("The outlet has partial channel metadata. Confirm the selected index in the acquisition software before mapping.")
                for channel in resp_rows:
                    st.write(
                        f"{channel['index']} | {channel['label'] or 'label not supplied'} | "
                        f"{channel['unit'] or 'unit unknown'} | {channel['type'] or 'type unknown'}"
                    )
                st.selectbox(
                    "Measured respiration channel index",
                    [None, *[row["index"] for row in resp_rows]],
                    format_func=lambda index: "Choose a channel…" if index is None else f"{index}: {resp_rows[index]['label'] or 'label not supplied'} [{resp_rows[index]['unit'] or 'unit unknown'}] ({resp_rows[index]['type'] or 'type unknown'})",
                    key="lsl_resp_index",
                    on_change=_clear_resp_confirmation,
                    disabled=disabled,
                )
        if respiration_mode != "No measured respiration":
            st.checkbox(
                "I verified this selected index is measured respiration, not a pacer or pacing reference.",
                key="lsl_confirm_measured_resp",
                disabled=disabled,
            )
        st.checkbox(
            "Save raw LSL samples to this computer (optional)",
            key="lsl_recording_enabled",
            disabled=disabled,
            help="Off by default. This writes raw numeric samples and timestamps as local CSV plus metadata JSON. Files are not encrypted; enable only when your data handling and consent requirements are satisfied.",
        )
        st.caption("Local capture is plaintext and is not encrypted. On POSIX systems, owner-only file permissions are applied where supported.")
        if st.session_state.get("lsl_recording_path"):
            st.caption(f"Local session files: {st.session_state['lsl_recording_path']}")

    if DATA_SOURCE == "XDF_FILE":
        disabled = st.session_state["session_active"]
        st.caption("XDF is the recording container commonly written by LabRecorder; LSL itself is a streaming protocol.")
        st.caption("XDF files are loaded into memory before playback; the compressed upload limit is 100 MiB. Large files may still use substantial memory while parsing.")
        if st.button("Load deterministic 3-minute synthetic demo", disabled=disabled, key="load_synthetic_xdf_demo"):
            demo_id = "neuroresonance-synthetic-demo-v1"
            st.session_state["xdf_recording"] = build_deterministic_demo_recording()
            st.session_state["xdf_file_name"] = "Built-in synthetic demo (180 s)"
            st.session_state["xdf_loaded_upload_id"] = demo_id
            st.session_state["xdf_current_upload_id"] = demo_id
            st.session_state["xdf_builtin_demo"] = True
            st.session_state["xdf_fnirs_stream_index"] = 0
            st.session_state["xdf_hbo2_channel_index"] = 1
            st.session_state["xdf_hbr_channel_index"] = 2
            st.session_state["xdf_respiration_mode"] = "Same as fNIRS stream"
            st.session_state["xdf_respiration_channel_index"] = 0
            st.session_state["xdf_confirm_derived_hb"] = True
            st.session_state["xdf_confirm_measured_respiration"] = True
            st.session_state["xdf_playback_speed"] = 4.0
            st.session_state["xdf_quality_warnings"] = []
            for key in ("ring_time", "ring_resp", "ring_hbo", "ring_hbr"):
                st.session_state[key].reset()
            for key in ("buf_time", "buf_resp", "buf_hbo", "buf_hbr"):
                st.session_state[key] = np.empty(0, dtype=np.float64)
            st.session_state["cached_report"] = None
            st.session_state["cached_session_bundle"] = None
            st.session_state["current_plv"] = float("nan")
            st.session_state["current_coherence"] = None
            st.session_state["current_dom_freq"] = float("nan")
            st.session_state["session_data_source"] = None
        if st.session_state.get("xdf_builtin_demo"):
            st.info("Built-in demo: deterministic synthetic channels with a constructed shared 0.1 Hz component. This is not a physiological recording.")
        st.markdown("[Download the official minimal XDF example](https://raw.githubusercontent.com/xdf-modules/example-files/master/minimal.xdf)")
        st.caption("The example contains a short numeric EEG test stream and text markers. It is only for checking import and playback, not a physiological or fNIRS recording.")
        uploaded_xdf = st.file_uploader("Choose a LabRecorder .xdf file", type=["xdf"], disabled=disabled, key="xdf_upload")
        if uploaded_xdf is not None:
            current_upload_id = getattr(uploaded_xdf, "file_id", None) or f"{uploaded_xdf.name}:{uploaded_xdf.size}"
            st.session_state["xdf_current_upload_id"] = current_upload_id
            st.session_state["xdf_builtin_demo"] = False
            if st.button("Load XDF recording", disabled=disabled):
                try:
                    with st.spinner("Reading XDF streams and timestamps..."):
                        streams, _header = load_xdf_recording(uploaded_xdf.getvalue())
                    st.session_state["xdf_recording"] = streams
                    st.session_state["xdf_file_name"] = uploaded_xdf.name
                    st.session_state["xdf_loaded_upload_id"] = current_upload_id
                    st.session_state["xdf_builtin_demo"] = False
                    st.session_state["xdf_quality_warnings"] = []
                    for key in ("ring_time", "ring_resp", "ring_hbo", "ring_hbr"):
                        st.session_state[key].reset()
                    for key in ("buf_time", "buf_resp", "buf_hbo", "buf_hbr"):
                        st.session_state[key] = np.empty(0, dtype=np.float64)
                    st.session_state["session_data_source"] = None
                    st.session_state["cached_report"] = None
                    st.session_state["cached_session_bundle"] = None
                    st.session_state["current_plv"] = float("nan")
                    st.session_state["current_coherence"] = None
                    st.session_state["current_dom_freq"] = float("nan")
                    _clear_xdf_fnirs_mapping()
                except Exception as exc:
                    st.session_state["xdf_recording"] = None
                    st.session_state["xdf_builtin_demo"] = False
                    _record_diagnostic("ERROR", f"Nie udało się zaimportować pliku XDF: {uploaded_xdf.name}.", exc)
                    st.error(f"XDF import failed: {exc}")
        else:
            if st.session_state.get("xdf_builtin_demo"):
                st.session_state["xdf_current_upload_id"] = "neuroresonance-synthetic-demo-v1"
            else:
                st.session_state["xdf_current_upload_id"] = None

        streams = st.session_state.get("xdf_recording") or []
        loaded_id = st.session_state.get("xdf_loaded_upload_id")
        current_id = st.session_state.get("xdf_current_upload_id")
        if streams:
            st.caption(f"Loaded: {st.session_state.get('xdf_file_name') or 'XDF recording'} ({len(streams)} stream(s))")
            if not st.session_state.get("xdf_builtin_demo") and loaded_id != current_id:
                st.warning("The loaded file is no longer selected. Select and load the intended XDF file before playback.")
            numeric_indices = [item["index"] for item in streams if item.get("is_numeric") and item.get("sample_count", 0) > 1]

            def _xdf_stream_label(index: Optional[int]) -> str:
                if index is None:
                    return "Choose a numeric stream…"
                item = streams[int(index)]
                rate = item.get("nominal_srate_hz", 0.0)
                rate_text = f"{rate:g} Hz nominal" if rate > 0 else "irregular rate"
                return f"{item['name']} | {item['type'] or 'no type'} | {item['channel_count']} ch | {rate_text} | {item['sample_count']} samples"

            if st.session_state.get("xdf_fnirs_stream_index") not in numeric_indices:
                st.session_state["xdf_fnirs_stream_index"] = None
                _clear_xdf_fnirs_mapping()
            st.selectbox(
                "Numeric signal stream",
                options=[None, *numeric_indices],
                format_func=_xdf_stream_label,
                key="xdf_fnirs_stream_index",
                on_change=_clear_xdf_fnirs_mapping,
                disabled=disabled or not numeric_indices,
            )
            selected_fnirs_index = st.session_state.get("xdf_fnirs_stream_index")
            fnirs_ready = selected_fnirs_index in numeric_indices
            if fnirs_ready:
                fnirs_meta = streams[int(selected_fnirs_index)]
                st.caption("Selected stream channel metadata (labels are descriptive only; verify their meaning in the acquisition setup):")
                for channel in fnirs_meta["channels"]:
                    st.write(f"{channel['index']} | {channel['label'] or 'label not supplied'} | {channel['unit'] or 'unit unknown'} | {channel['type'] or 'type unknown'}")

                def _xdf_channel_label(index: Optional[int]) -> str:
                    if index is None:
                        return "Choose a channel…"
                    item = fnirs_meta["channels"][int(index)]
                    return f"{index}: {item['label'] or 'unlabelled'} [{item['unit'] or 'unit unknown'}]"

                channel_indices: List[Optional[int]] = [None, *range(fnirs_meta["channel_count"])]
                st.selectbox("HbO₂ channel index", channel_indices, format_func=_xdf_channel_label, key="xdf_hbo2_channel_index", on_change=lambda: st.session_state.__setitem__("xdf_confirm_derived_hb", False), disabled=disabled)
                st.selectbox("HbR channel index", channel_indices, format_func=_xdf_channel_label, key="xdf_hbr_channel_index", on_change=lambda: st.session_state.__setitem__("xdf_confirm_derived_hb", False), disabled=disabled)
                if st.session_state.get("xdf_builtin_demo"):
                    st.caption("The built-in demo mapping is fixed synthetic data, not measured HbO₂/HbR.")
                else:
                    st.checkbox(
                        "I verified these selected indices contain already-derived HbO₂/HbR values.",
                        key="xdf_confirm_derived_hb",
                        disabled=disabled,
                    )
            else:
                st.info("Load a file and select a numeric stream before mapping HbO₂/HbR.")

            respiration_mode = st.radio(
                "Measured respiration in XDF",
                ["Same as fNIRS stream", "Separate stream", "No measured respiration"],
                key="xdf_respiration_mode",
                on_change=_clear_xdf_resp_mapping,
                disabled=disabled,
            )
            respiration_stream_index = None
            if respiration_mode == "Same as fNIRS stream" and fnirs_ready:
                respiration_stream_index = int(selected_fnirs_index)
                channel_indices = [None, *range(streams[respiration_stream_index]["channel_count"])]
                resp_meta = streams[respiration_stream_index]
                st.selectbox(
                    "Measured respiration channel index",
                    channel_indices,
                    format_func=lambda index: "Choose a channel…" if index is None else f"{index}: {resp_meta['channels'][int(index)]['label'] or 'unlabelled'} [{resp_meta['channels'][int(index)]['unit'] or 'unit unknown'}]",
                    key="xdf_respiration_channel_index",
                    on_change=lambda: st.session_state.__setitem__("xdf_confirm_measured_respiration", False),
                    disabled=disabled,
                )
            elif respiration_mode == "Separate stream":
                if st.session_state.get("xdf_respiration_stream_index") not in numeric_indices:
                    st.session_state["xdf_respiration_stream_index"] = None
                    _clear_xdf_resp_mapping()
                st.selectbox(
                    "Respiration stream",
                    options=[None, *numeric_indices],
                    format_func=_xdf_stream_label,
                    key="xdf_respiration_stream_index",
                    on_change=_clear_xdf_resp_mapping,
                    disabled=disabled or not numeric_indices,
                )
                respiration_stream_index = st.session_state.get("xdf_respiration_stream_index")
                if respiration_stream_index is not None:
                    resp_meta = streams[int(respiration_stream_index)]
                    st.selectbox(
                        "Measured respiration channel index",
                        [None, *range(resp_meta["channel_count"])],
                        format_func=lambda index: "Choose a channel…" if index is None else f"{index}: {resp_meta['channels'][int(index)]['label'] or 'unlabelled'} [{resp_meta['channels'][int(index)]['unit'] or 'unit unknown'}]",
                        key="xdf_respiration_channel_index",
                        on_change=lambda: st.session_state.__setitem__("xdf_confirm_measured_respiration", False),
                        disabled=disabled,
                    )
            if respiration_mode != "No measured respiration" and not st.session_state.get("xdf_builtin_demo"):
                st.checkbox(
                    "I verified this selected channel is measured respiration, not a pacing reference.",
                    key="xdf_confirm_measured_respiration",
                    disabled=disabled,
                )
            st.selectbox("Playback speed", options=[0.25, 0.5, 1.0, 2.0, 4.0], index=2, key="xdf_playback_speed", format_func=lambda speed: f"{speed:g}×", disabled=disabled)
        else:
            st.info("Upload an XDF recording and choose Load XDF recording to inspect its streams.")

    if DATA_SOURCE == "WEBCAM":
        cam_source_choice = st.selectbox(
            "Wybór sensora optycznego (Camera Input)",
            options=[
                "Kamera domyślna PC/Laptop (Index 0)",
                "Kamera zewnętrzna / wirtualna (Index 1)",
                "Kamera wirtualna (Index 2)",
                "Smartfon: Bezprzewodowy strumień Wi-Fi (IP Webcam / DroidCam / RTSP)",
            ],
            index=0,
            help="Wybierz fizyczny indeks kamery lub strumień sieciowy ze smartfona."
        )

        if cam_source_choice == "Smartfon: Bezprzewodowy strumień Wi-Fi (IP Webcam / DroidCam / RTSP)":
            custom_url = st.text_input(
                "Adres URL strumienia ze smartfona",
                value=st.session_state.get("custom_cam_url", "http://192.168.1.100:8080/video"),
                help="Wpisz adres IP i port podany w aplikacji w telefonie (np. http://192.168.1.X:8080/video)."
            )
            try:
                safe_url = _validate_camera_url(custom_url)
                st.session_state["camera_source"] = safe_url
                cam_src_tag = "Smartfon Wi-Fi (prywatny adres IP)"
            except ValueError as exc:
                _record_diagnostic("ERROR", "Nieprawidłowy adres kamery.", exc)
                st.warning(str(exc))
                st.session_state["camera_source"] = None
                cam_src_tag = "nieprawidłowy adres"
            st.session_state["custom_cam_url"] = custom_url
        elif "Index 1" in cam_source_choice:
            st.session_state["camera_source"] = 1
            cam_src_tag = "Kamera zewn./wirtualna (Index 1)"
        elif "Index 2" in cam_source_choice:
            st.session_state["camera_source"] = 2
            cam_src_tag = "Kamera wirtualna (Index 2)"
        else:
            st.session_state["camera_source"] = 0
            cam_src_tag = "Kamera domyślna (Index 0)"

        st.caption(f"[OPTICAL SENSOR: {cam_src_tag}]")

        with st.expander("Instrukcja: Kamera ze smartfona (Mobile Video Sensor)", expanded=False):
            st.markdown(
                """
                **Wariant A: Bezprzewodowo przez Wi-Fi (IP Webcam / DroidCam):**
                1. Zainstaluj darmową aplikację na telefon:
                   - **Android:** *IP Webcam* (Pavel Khlebovich) lub *DroidCam*.
                   - **iOS:** *IP Camera Lite* lub *DroidCam Wireless*.
                2. Podłącz smartfon do tej samej sieci Wi-Fi co ten komputer.
                3. W aplikacji w telefonie kliknij *Start server* i odczytaj adres IP (np. `http://192.168.1.45:8080/video`).
                4. Wybierz opcję *Smartfon: Bezprzewodowy strumień Wi-Fi* i wklej ten URL powyżej.

                **Wariant B: Przez kabel USB / sterownik wirtualny (Camo / Continuity):**
                1. Zainstaluj Camo, DroidCam Client lub Iriun Webcam na PC i telefonie.
                2. Wybierz *Kamera zewnętrzna / wirtualna (Index 1)*.
                """
            )

        with st.expander("Protokol pozycjonowania i pomiaru", expanded=False):
            st.markdown(
                """
                - **Postawa:** Wyprostowane plecy, cialo oparte stabilnie o fotel. Rece na biurku lub kolanach (brak ruchow rak).
                - **Dystans:** Zachowaj 50-70 cm od obiektywu kamery.
                - **Klatka piersiowa:** Wycentruj mostek i klatke wewnatrz zielonej ramki celowniczej.
                - **Oswietlenie:** Rownomierne swiatlo z przodu. Unikaj silnego zrodla swiatla z tylu (okno za plecami).
                - **Oddech:** Spokojny, plynny oddech torem klatkowo-brzusznym w zadanym rytmie.
                """
            )

    session_duration = st.slider(
        "Session Duration (seconds)",
        min_value=120,
        max_value=300,
        value=300,
        step=20,
        disabled=st.session_state["session_active"],
    )

    st.markdown("---")
    TARGET_RESP_HZ = DEFAULT_RESP_HZ
    TARGET_RESP_BPM = TARGET_RESP_HZ * 60.0

    st.caption(
        f"**Fixed pacing reference:** `{TARGET_RESP_BPM:.1f}` breaths/min ({TARGET_RESP_HZ:.3f} Hz)"
    )

    st.markdown("---")
    col_start, col_stop = st.columns(2)
    start_btn = col_start.button("START SESSION", width="stretch")
    stop_btn = col_stop.button("STOP SESSION", width="stretch")

    st.markdown("---")
    if DATA_SOURCE == "XDF_FILE":
        st.caption("File playback follows its recorded timestamps. No pacing signal is added to imported data.")
    else:
        st.markdown("### Fixed Pacing Reference")
        st.markdown(
            f'<div style="line-height: 24px;"><span class="resonance-pacer"></span>'
            f'<b>Target: {TARGET_RESP_BPM:.1f} breaths/min ({TARGET_RESP_HZ:.3f} Hz)</b></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Inhale {30.0 / TARGET_RESP_BPM:.1f}s → Exhale {30.0 / TARGET_RESP_BPM:.1f}s. This is a fixed demonstration target.")

    st.markdown("---")
    with st.expander("Dostęp z innych urządzeń", expanded=False):
        st.info("Serwer działa lokalnie na tym komputerze. Dostęp przez sieć Wi-Fi jest wyłączony.")

    st.markdown("---")
    st.caption("Cortivision Challenge: Breath / Brain & Body")
    st.caption("Author: Bastian Fernandez (XIV LO Wrocław)")

# ===========================================================================
# Backend Session Management & Process Lifecycle
# ===========================================================================
def _cleanup_session_resources(recording_status: str = "completed") -> None:
    """Stop this browser session's resources; handles survive Streamlit reruns."""
    lsl_session: Optional[SelectedLSLSession] = st.session_state.get("lsl_session")
    lsl_receiver: Optional[LSLBackgroundReceiver] = st.session_state.get("lsl_receiver")
    if lsl_receiver is not None:
        try:
            if not lsl_receiver.stop(timeout_s=5.0):
                recording_status = "interrupted"
                logger.error("LSL receiver did not exit before session cleanup timeout.")
                _record_diagnostic("ERROR", "Odbiornik LSL nie zatrzymał się przed upływem limitu czasu.")
                # Keep the receiver, inlet and recorder alive until the worker has
                # actually exited; closing them here would race with pull_chunk().
                return
            receiver_quality = lsl_receiver.quality_summary()
            if receiver_quality["receiver_error"]:
                recording_status = "interrupted"
                _record_diagnostic("ERROR", f"Odbiornik LSL zgłosił błąd: {receiver_quality['receiver_error']}")
            if lsl_session is not None:
                lsl_session.stats["analysis_receiver"] = receiver_quality
        except Exception as exc:
            recording_status = "interrupted"
            logger.exception("Could not stop background LSL receiver cleanly: %s", exc)
            _record_diagnostic("ERROR", "Nie udało się poprawnie zatrzymać odbiornika LSL.", exc)
            return
        st.session_state["lsl_receiver"] = None

    if lsl_session is not None:
        try:
            if lsl_session.stats.get("disconnect_count", 0):
                recording_status = "interrupted"
            lsl_session.close_recording(recording_status)
        except Exception as exc:
            logger.warning("Could not close LSL session cleanly: %s", exc)
            _record_diagnostic("ERROR", "Nie udało się zamknąć lokalnego zapisu LSL.", exc)
        try:
            lsl_session.disconnect()
        except Exception as exc:
            logger.warning("Could not disconnect LSL inlet cleanly: %s", exc)
            _record_diagnostic("ERROR", "Nie udało się rozłączyć inletu LSL.", exc)
        st.session_state["lsl_session"] = None

    for key, method_name in (("simulator", "stop"), ("camera", "stop"), ("bridge", "disconnect"), ("xdf_playback", "stop")):
        resource = st.session_state.get(key)
        if resource is not None:
            try:
                getattr(resource, method_name)()
            except Exception as exc:
                logger.warning("Could not stop session resource %s: %s", key, exc)
                _record_diagnostic("ERROR", f"Nie udało się zatrzymać zasobu sesji: {key}.", exc)
        st.session_state[key] = None


def _start_session() -> None:
    prepared_xdf_source: Optional[XDFPlaybackSource] = None
    if DATA_SOURCE == "XDF_FILE":
        streams = st.session_state.get("xdf_recording") or []
        if not streams:
            st.error("Upload and load an XDF file before starting playback.")
            return
        current_upload_id = st.session_state.get("xdf_current_upload_id")
        loaded_upload_id = st.session_state.get("xdf_loaded_upload_id")
        builtin_demo = bool(st.session_state.get("xdf_builtin_demo"))
        if not builtin_demo and (loaded_upload_id is None or loaded_upload_id != current_upload_id):
            st.error("The loaded XDF file is no longer selected. Select and load it again before playback.")
            return
        fnirs_index = st.session_state.get("xdf_fnirs_stream_index")
        hbo_index = st.session_state.get("xdf_hbo2_channel_index")
        hbr_index = st.session_state.get("xdf_hbr_channel_index")
        if fnirs_index is None or hbo_index is None or hbr_index is None:
            st.error("Select a numeric stream and explicit HbO₂/HbR channel indices first.")
            return
        if not builtin_demo and not st.session_state.get("xdf_confirm_derived_hb", False):
            st.error("Confirm that the selected XDF channels are already-derived HbO₂/HbR values.")
            return
        respiration_mode = st.session_state.get("xdf_respiration_mode", "No measured respiration")
        respiration_stream_index = None
        respiration_channel_index = None
        if respiration_mode == "Same as fNIRS stream":
            respiration_stream_index = int(fnirs_index)
            respiration_channel_index = st.session_state.get("xdf_respiration_channel_index")
        elif respiration_mode == "Separate stream":
            respiration_stream_index = st.session_state.get("xdf_respiration_stream_index")
            respiration_channel_index = st.session_state.get("xdf_respiration_channel_index")
        if respiration_mode != "No measured respiration":
            if respiration_stream_index is None or respiration_channel_index is None:
                st.error("Select an XDF respiration stream and channel index, or choose No measured respiration.")
                return
            if not builtin_demo and not st.session_state.get("xdf_confirm_measured_respiration", False):
                st.error("Confirm that the selected XDF channel contains measured respiration, not a pacing reference.")
                return
        try:
            prepared_xdf_source = XDFPlaybackSource(
                streams,
                fnirs_stream_index=int(fnirs_index),
                hbo2_channel_index=int(hbo_index),
                hbr_channel_index=int(hbr_index),
                respiration_stream_index=(int(respiration_stream_index) if respiration_stream_index is not None else None),
                respiration_channel_index=(int(respiration_channel_index) if respiration_channel_index is not None else None),
                speed=float(st.session_state.get("xdf_playback_speed", 1.0)),
            )
        except Exception as exc:
            _record_diagnostic("ERROR", "Nie udało się przygotować odtwarzania XDF.", exc)
            st.error(f"XDF playback could not start: {exc}")
            return

    _cleanup_session_resources()
    lingering = [
        key for key in ("lsl_receiver", "lsl_session", "simulator", "camera", "bridge", "xdf_playback")
        if st.session_state.get(key) is not None
    ]
    if lingering:
        st.error("Poprzednia sesja nadal zwalnia zasoby. Poczekaj, aż odbiornik zakończy pracę, i spróbuj ponownie.")
        return

    # Reset zero-allocation ring buffers without zero-padding (Finding H1 fix)
    st.session_state["ring_time"].reset()
    st.session_state["ring_resp"].reset()
    st.session_state["ring_hbo"].reset()
    st.session_state["ring_hbr"].reset()

    st.session_state["buf_time"] = np.empty(0, dtype=np.float64)
    st.session_state["buf_resp"] = np.empty(0, dtype=np.float64)
    st.session_state["buf_hbo"] = np.empty(0, dtype=np.float64)
    st.session_state["buf_hbr"] = np.empty(0, dtype=np.float64)
    st.session_state["valid_samples_count"] = 0
    st.session_state["session_active_duration"] = 0.0
    st.session_state["session_start_monotonic"] = None
    st.session_state["sim_resp_rng"] = (
        np.random.default_rng(DEMO_SEED) if DATA_SOURCE == "SIMULATOR" else None
    )
    st.session_state["cached_report"] = None
    st.session_state["cached_session_bundle"] = None
    st.session_state["last_session_state"] = None
    st.session_state["current_plv"] = float("nan")
    st.session_state["current_coherence"] = None
    st.session_state["current_dom_freq"] = float("nan")
    st.session_state["current_state"] = "STARTING (ACQUIRING SIGNAL)"
    st.session_state["analysis_error"] = None
    st.session_state["state_color"] = "#4FC3F7"
    st.session_state["last_ai_insight"] = "Initializing signal acquisition and hemodynamic pipeline..."
    st.session_state["session_data_source"] = DATA_SOURCE
    st.session_state["session_is_synthetic_demo"] = bool(
        DATA_SOURCE == "XDF_FILE" and st.session_state.get("xdf_builtin_demo")
    )
    st.session_state["lsl_origin"] = None
    st.session_state["lsl_last_receive_monotonic"] = None
    st.session_state["lsl_last_quality_warnings"] = []
    st.session_state["lsl_recording_path"] = None
    st.session_state["sim_clock_offset"] = None
    st.session_state["sim_last_receive_monotonic"] = None
    st.session_state["camera_wall_start"] = None
    st.session_state["sim_last_quality_warning"] = None
    st.session_state["measured_sample_rate_hz"] = None
    st.session_state["xdf_playback"] = None
    st.session_state["xdf_quality_warnings"] = []

    # 1. fNIRS data source
    if DATA_SOURCE in ("SIMULATOR", "WEBCAM"):
        try:
            sim = fNIRSSimulator(
                sampling_rate=FS_DISPLAY,
                mayer_wave_freq=0.1,
                seed=DEMO_SEED if DATA_SOURCE == "SIMULATOR" else None,
            )
            sim.start()
            st.session_state["simulator"] = sim
        except Exception as exc:
            _record_diagnostic("ERROR", "Nie udało się uruchomić generatora sygnału symulowanego.", exc)
            st.error(f"[SIMULATOR INITIALIZATION ERROR]: {type(exc).__name__}: {exc}")
            _cleanup_session_resources(recording_status="failed")
            return
    else:
        st.session_state["simulator"] = None

    # 2. Camera tracker
    if DATA_SOURCE == "WEBCAM":
        try:
            cam_src = st.session_state.get("camera_source", 0)
            if cam_src is None:
                st.error("Popraw adres kamery przed uruchomieniem sesji.")
                _cleanup_session_resources(recording_status="failed")
                return
            cam = CameraTracker(camera_source=cam_src, fps_target=30, buffer_seconds=ANALYSIS_WINDOW_SECONDS)
            cam.start()
            st.session_state["camera"] = cam
        except Exception as exc:
            _record_diagnostic("ERROR", "Nie udało się uruchomić kamery.", exc)
            st.error(f"[CAMERA INITIALIZATION ERROR]: {exc}")
            _stop_session()
            return
    else:
        st.session_state["camera"] = None

    # 3. Explicitly selected LSL outlets and operator-confirmed channel mapping
    if DATA_SOURCE == "LIVE_HARDWARE":
        try:
            fnirs_selector = st.session_state.get("lsl_fnirs_selector")
            fnirs_meta = st.session_state.get("lsl_fnirs_metadata")
            if not fnirs_selector or not fnirs_meta or st.session_state.get("lsl_fnirs_metadata_selector") != fnirs_selector:
                raise ValueError("Discover an outlet, inspect its channels, and select HbO₂/HbR indices first.")
            hbo_index = st.session_state.get("lsl_hbo_index")
            hbr_index = st.session_state.get("lsl_hbr_index")
            if hbo_index is None or hbr_index is None:
                raise ValueError("Select explicit channel indices for HbO₂ and HbR.")

            respiration_mode = st.session_state.get("lsl_respiration_mode", "No measured respiration")
            respiration_selector = None
            respiration_index = None
            mapping = {"HbO2": hbo_index, "HbR": hbr_index}
            if respiration_mode == "Same as fNIRS outlet":
                respiration_index = st.session_state.get("lsl_resp_same_index")
                if respiration_index is None:
                    raise ValueError("Select an explicit measured respiration channel, or choose 'No measured respiration'.")
                mapping["Respiration"] = respiration_index
            elif respiration_mode == "Separate outlet":
                respiration_selector = st.session_state.get("lsl_resp_selector")
                resp_meta = st.session_state.get("lsl_resp_metadata")
                if not respiration_selector or respiration_selector == fnirs_selector:
                    raise ValueError("Choose a distinct respiration outlet, then inspect its channels.")
                if not resp_meta or st.session_state.get("lsl_resp_metadata_selector") != respiration_selector:
                    raise ValueError("Inspect the selected respiration outlet and choose a channel index first.")
                respiration_index = st.session_state.get("lsl_resp_index")
                if respiration_index is None:
                    raise ValueError("Select an explicit respiration channel index.")

            measured_respiration = respiration_mode != "No measured respiration"
            if measured_respiration and not st.session_state.get("lsl_confirm_measured_resp", False):
                raise ValueError("Confirm that the selected respiration channel is measured, not paced.")
            if not st.session_state.get("lsl_confirm_derived_hb", False):
                raise ValueError("Confirm that the selected channels are already-derived HbO₂/HbR.")

            lsl_session = SelectedLSLSession(
                fnirs_selector=fnirs_selector,
                channel_mapping=mapping,
                respiration_selector=respiration_selector,
                respiration_channel_index=respiration_index if respiration_selector else None,
                confirmed_derived_hb_channels=True,
                confirmed_measured_respiration=measured_respiration,
                recording_enabled=bool(st.session_state.get("lsl_recording_enabled", False)),
            )
            lsl_session.connect()
            st.session_state["lsl_session"] = lsl_session
            st.session_state["lsl_recording_path"] = lsl_session.recording_path
            lsl_receiver = LSLBackgroundReceiver(lsl_session)
            lsl_receiver.start()
            st.session_state["lsl_receiver"] = lsl_receiver
            rate = lsl_session.fnirs_sample_rate_hz
            buffer_capacity = min(60000, max(BUFFER_LEN, int(math.ceil(rate * ANALYSIS_WINDOW_SECONDS)) + 2 if rate > 0 else BUFFER_LEN))
            st.session_state["ring_time"] = CircularRingBuffer(capacity=buffer_capacity)
            st.session_state["ring_resp"] = CircularRingBuffer(capacity=buffer_capacity)
            st.session_state["ring_hbo"] = CircularRingBuffer(capacity=buffer_capacity)
            st.session_state["ring_hbr"] = CircularRingBuffer(capacity=buffer_capacity)
            st.session_state["lsl_last_receive_monotonic"] = time.monotonic()
        except Exception as exc:
            _record_diagnostic("ERROR", "Nie udało się rozpocząć sesji LSL.", exc)
            st.error(f"LSL session could not start: {exc}")
            _cleanup_session_resources(recording_status="failed")
            return
    elif DATA_SOURCE == "XDF_FILE":
        st.session_state["bridge"] = None
        st.session_state["lsl_session"] = None
        st.session_state["lsl_receiver"] = None
        if prepared_xdf_source is None:
            st.error("No validated XDF playback source is available.")
            return
        st.session_state["xdf_playback"] = prepared_xdf_source
        st.session_state["xdf_quality_warnings"] = list(prepared_xdf_source.quality_warnings)
        rate = prepared_xdf_source.sample_rate_hz
        buffer_capacity = min(60000, max(BUFFER_LEN, int(math.ceil(rate * ANALYSIS_WINDOW_SECONDS)) + 2 if rate > 0 else BUFFER_LEN))
        st.session_state["ring_time"] = CircularRingBuffer(capacity=buffer_capacity)
        st.session_state["ring_resp"] = CircularRingBuffer(capacity=buffer_capacity)
        st.session_state["ring_hbo"] = CircularRingBuffer(capacity=buffer_capacity)
        st.session_state["ring_hbr"] = CircularRingBuffer(capacity=buffer_capacity)
    else:
        st.session_state["bridge"] = None
        st.session_state["lsl_session"] = None

    # 4. DSP & Coherence Engines
    st.session_state["estimator"] = ResonanceEstimator(
        fs_resp=FS_DISPLAY,
        fs_fnirs=FS_DISPLAY,
        window_s=ANALYSIS_WINDOW_SECONDS,
    )
    st.session_state["nvc_checker"] = HemoglobinCorrelationSummary(window_s=20.0, fs=FS_DISPLAY)

    # Store the fixed demonstration reference for the active session.
    st.session_state["target_resp_hz"] = TARGET_RESP_HZ
    st.session_state["target_resp_bpm"] = TARGET_RESP_BPM

    st.session_state["session_active"] = True
    st.session_state["session_start_monotonic"] = time.monotonic()
    st.session_state["camera_wall_start"] = time.time()
    st.session_state["diagnostic_last_ui_refresh"] = None
    if prepared_xdf_source is not None:
        prepared_xdf_source.start(now_monotonic=st.session_state["session_start_monotonic"])
    _record_diagnostic("INFO", f"Rozpoczęto sesję w trybie {DATA_SOURCE}.")


def _stop_session(reason: Optional[str] = None) -> None:
    last_state = st.session_state.get("current_state")
    if last_state and last_state != "IDLE":
        st.session_state["last_session_state"] = last_state
    xdf_playback: Optional[XDFPlaybackSource] = st.session_state.get("xdf_playback")
    if xdf_playback is not None:
        st.session_state["session_active_duration"] = xdf_playback.position_s
        for state_key, buffer_key in (("buf_time", "ring_time"), ("buf_resp", "ring_resp"), ("buf_hbo", "ring_hbo"), ("buf_hbr", "ring_hbr")):
            st.session_state[state_key] = st.session_state[buffer_key].get_all()
    _cleanup_session_resources()
    if st.session_state.get("lsl_receiver") is None and st.session_state.get("lsl_session") is None:
        # Capture consent applies to one fully stopped session only.
        st.session_state["lsl_recording_enabled"] = False
    session_started = st.session_state.get("session_start_monotonic")
    if session_started is not None and xdf_playback is None:
        st.session_state["session_active_duration"] = elapsed_seconds(session_started)

    st.session_state["session_active"] = False
    st.session_state["current_state"] = "IDLE"
    st.session_state["state_color"] = "#888888"
    if reason:
        _record_diagnostic("INFO", reason)


if start_btn and not st.session_state["session_active"]:
    try:
        _start_session()
    except Exception as exc:
        _record_diagnostic("ERROR", "Nieoczekiwany błąd podczas uruchamiania sesji.", exc)
        st.error(f"[SESSION START ERROR]: {type(exc).__name__}: {exc}")
        try:
            _cleanup_session_resources(recording_status="failed")
        except Exception as cleanup_exc:
            _record_diagnostic("ERROR", "Nie udało się posprzątać zasobów po błędzie startu.", cleanup_exc)
        st.session_state["session_active"] = False

if stop_btn and st.session_state["session_active"]:
    _stop_session("Sesja została zatrzymana przyciskiem STOP.")

# ===========================================================================
# Plotting Helpers (Plotly Dark Theme)
# ===========================================================================
def _plot_breathing_guide(elapsed_s: float) -> go.Figure:
    t = np.linspace(0, 11, 200)
    phase_offset = (elapsed_s * TARGET_RESP_HZ * 2 * np.pi) % (2 * np.pi)
    y = np.sin(2 * np.pi * TARGET_RESP_HZ * t + phase_offset)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=y,
        mode="lines",
        line=dict(color="#54E5F5", width=2.5),
        name="Target Wave"
    ))
    curr_y = np.sin(phase_offset)
    fig.add_trace(go.Scatter(
        x=[0], y=[curr_y],
        mode="markers",
        marker=dict(size=14, color="#E9FAFF", symbol="circle", line=dict(color="#54E5F5", width=2)),
        name="Current Phase"
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c4d2e3", family="Inter, Segoe UI, sans-serif"),
        height=180,
        margin=dict(l=20, r=20, t=25, b=20),
        title=dict(text=f"PACE · {TARGET_RESP_BPM:.0f} BPM", font=dict(size=13, color="#54E5F5")),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(93,151,214,.14)", range=[-1.3, 1.3], showticklabels=False),
        showlegend=False,
    )
    return fig


def _plot_respiratory_waveform(t_buf: np.ndarray, resp_buf: np.ndarray, title: str = "Respiration Signal / Motion Proxy") -> go.Figure:
    fig = go.Figure()
    if len(t_buf) == 0:
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(color="#4D8DFF", size=1), name="Awaiting Signal"))
    else:
        fig.add_trace(go.Scatter(
            x=t_buf, y=resp_buf,
            mode="lines",
            line=dict(color="#64A9FF", width=2.0),
            fill="tozeroy",
            fillcolor="rgba(55, 132, 255, 0.13)",
            name="Respiration"
        ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c4d2e3", family="Inter, Segoe UI, sans-serif"),
        height=200,
        margin=dict(l=35, r=15, t=25, b=30),
        title=dict(text=title, font=dict(size=13, color="#64A9FF")),
        xaxis=dict(title="Time (s)", showgrid=False),
        yaxis=dict(title="Amplitude (a.u.)", showgrid=True, gridcolor="rgba(93,151,214,.14)"),
        showlegend=False,
    )
    return fig


def _plot_fnirs_hemodynamics(t_buf: np.ndarray, hbo_buf: np.ndarray, hbr_buf: np.ndarray, title: str = "fNIRS HbO₂/HbR Signals") -> go.Figure:
    fig = go.Figure()
    if len(t_buf) == 0:
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(color="#54E5F5", size=1), name="Awaiting Signal"))
    else:
        fig.add_trace(go.Scatter(
            x=t_buf, y=hbo_buf,
            mode="lines",
            line=dict(color="#54E5F5", width=2.2),
            name="ΔHbO₂ (Oxy)"
        ))
        fig.add_trace(go.Scatter(
            x=t_buf, y=hbr_buf,
            mode="lines",
            line=dict(color="#9B8BFF", width=1.9),
            name="ΔHbR (Deoxy)"
        ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c4d2e3", family="Inter, Segoe UI, sans-serif"),
        height=200,
        margin=dict(l=35, r=15, t=25, b=30),
        title=dict(text=title, font=dict(size=13, color="#54E5F5")),
        xaxis=dict(title="Time (s)", showgrid=False),
        yaxis=dict(title="Signal (device units)", showgrid=True, gridcolor="rgba(93,151,214,.14)"),
        legend=dict(x=0.01, y=0.98, orientation="h", bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def _plot_lissajous_orbit(hbo: np.ndarray, resp: np.ndarray, title: str = "Paired Signal Phase Orbit") -> go.Figure:
    paired_len = min(len(hbo), len(resp))
    hbo = np.asarray(hbo[:paired_len], dtype=float)
    resp = np.asarray(resp[:paired_len], dtype=float)
    valid = np.isfinite(hbo) & np.isfinite(resp)
    hbo = hbo[valid]
    resp = resp[valid]
    if len(hbo) < 10:
        x, y = np.zeros(10), np.zeros(10)
    else:
        std_hbo = float(np.std(hbo))
        std_resp = float(np.std(resp))
        # When breathing stops (apnea), normalized respiration collapses to 0 line
        hbo_n = (hbo - np.mean(hbo)) / (std_hbo + 1e-6) if std_hbo > 0.04 else np.zeros_like(hbo)
        resp_n = (resp - np.mean(resp)) / (std_resp + 1e-6) if std_resp > 0.04 else np.zeros_like(resp)
        x = hbo_n
        y = resp_n

    n = len(x)
    colors = np.linspace(0, 1, n)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines+markers",
        marker=dict(size=3, color=colors, colorscale=[[0, "#3478ff"], [0.55, "#36b5ff"], [1, "#69f0ec"]], showscale=False),
        line=dict(color="rgba(70, 167, 255, 0.48)", width=1.7),
        name="Phase Orbit"
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c4d2e3", family="Inter, Segoe UI, sans-serif"),
        height=200,
        margin=dict(l=35, r=15, t=25, b=30),
        title=dict(text=title, font=dict(size=13, color="#E0E0E0")),
        xaxis=dict(title="Normalized ΔHbO₂", showgrid=True, gridcolor="rgba(93,151,214,.14)", range=[-3, 3]),
        yaxis=dict(title="Normalized Respiration", showgrid=True, gridcolor="rgba(93,151,214,.14)", range=[-3, 3]),
        showlegend=False,
    )
    if len(hbo) < 10:
        fig.add_annotation(text="Measured paired signals unavailable", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(color="#AAAAAA"))
    return fig


def _plot_plv_gauge(plv_pct: float) -> go.Figure:
    has_value = math.isfinite(plv_pct)
    fig = go.Figure(go.Indicator(
        mode="gauge+number" if has_value else "gauge",
        value=plv_pct if has_value else 0,
        domain=dict(x=[0.05, 0.95], y=[0.0, 0.80]),
        number=dict(suffix="%", font=dict(size=28, color="#FFFFFF")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#FFFFFF"),
            bar=dict(color="#54E5F5" if has_value else "#77869b"),
            bgcolor="#101a29",
            borderwidth=1,
            bordercolor="#243a58",
        ),
        title=dict(text=("DESCRIPTIVE PLV · NO CUTOFF" if has_value else f"PLV REQUIRES {ANALYSIS_WINDOW_SECONDS} s + MEASURED RESPIRATION"), font=dict(size=13, color="#AAAAAA"), align="center"),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c4d2e3", family="Inter, Segoe UI, sans-serif"),
        height=235,
        margin=dict(l=25, r=25, t=48, b=12),
    )
    return fig

# ===========================================================================
# Header & UI Layout Definition
# ===========================================================================
source_label = {
    "SIMULATOR": "SIMULATOR",
    "WEBCAM": "WEBCAM",
    "LIVE_HARDWARE": "LIVE LSL",
    "XDF_FILE": "XDF PLAYBACK",
}.get(DATA_SOURCE, DATA_SOURCE)
session_label = "SESSION ACTIVE" if st.session_state.get("session_active") else "AWAITING SIGNAL"

st.markdown(
    f"""
    <div class="nr-hero">
        <div class="nr-hero-copy">
            <div class="nr-eyebrow">NeuroResonance &nbsp;·&nbsp; experimental signal dashboard</div>
            <h1>Live session</h1>
            <p>Respiration and fNIRS signals, displayed as experimental window statistics.</p>
        </div>
        <div class="nr-hero-meta">
            <span class="nr-chip nr-chip-accent">{source_label}</span>
            <span class="nr-chip">{session_label}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main 70% / 30% Split Layout
col_screen, col_metrics = st.columns([70, 30])

with col_screen:
    st.markdown("##### Primary Biosignal Streams & Kinematics")
    if DATA_SOURCE == "WEBCAM" and not st.session_state["session_active"]:
        with st.expander("Protokol pozycji i polaczenia (Webcam Acquisition Protocol)", expanded=False):
            st.caption(
                "1. Odleglosc: Zachowaj dystans 50-70 cm od obiektywu kamery.\n\n"
                "2. Klatka piersiowa: Skieruj obiektyw tak, aby mostek i klatka piersiowa znajdowaly sie w obrebie zielonej ramki celowniczej.\n\n"
                "3. Stabilnosc ciala: Oprzyj plecy o fotel; unikaj gestykulacji rekami i gwaltownych ruchow glowy podczas akwizycji.\n\n"
                "4. Oswietlenie: Zapewnij rownomierne swiatlo z przodu. Unikaj silnego oswietlenia z tylu (okno za plecami)."
            )

    # Sub-row 1: Camera Preview + Respiratory Motion Proxy
    sub_cam, sub_resp = st.columns([44, 56])
    with sub_cam:
        cam_placeholder = st.empty()
    with sub_resp:
        resp_placeholder = st.empty()

    # Sub-row 2: fNIRS Signals + Lissajous Phase Orbit
    sub_fnirs, sub_orbit = st.columns([56, 44])
    with sub_fnirs:
        fnirs_placeholder = st.empty()
    with sub_orbit:
        orbit_placeholder = st.empty()

with col_metrics:
    st.markdown("##### Phase-Locking & Coherence Indicators")
    gauge_placeholder = st.empty()
    state_placeholder = st.empty()
    nvc_placeholder = st.empty()

    st.markdown("---")
    st.markdown("##### Quantitative Parameters & Lag")
    metrics_container = st.empty()
    insight_placeholder = st.empty()

    st.markdown("---")
    st.markdown("##### Experimental Signal Summary")
    btn_report = st.button("Generate Signal Summary", width="stretch")
    report_container = st.empty()

# ===========================================================================
# Experimental Signal Summary Generation Function
# ===========================================================================
def _generate_signal_summary() -> str:
    report_source = st.session_state.get("session_data_source") or DATA_SOURCE
    current_plv = st.session_state.get("current_plv", float("nan"))
    try:
        current_plv = float(current_plv)
    except (TypeError, ValueError):
        current_plv = float("nan")
    start_t = st.session_state.get("session_start_monotonic")
    active_dur = st.session_state.get("session_active_duration", 0.0)
    if active_dur and active_dur > 0:
        elapsed_s = active_dur
    elif start_t is not None:
        elapsed_s = elapsed_seconds(start_t)
    else:
        elapsed_s = 0.0
    if report_source == "XDF_FILE":
        playback = st.session_state.get("xdf_playback")
        if playback is not None:
            # XDF playback can run faster than wall time; base report eligibility
            # on consumed signal time, not time spent watching the demo.
            elapsed_s = max(elapsed_s, float(playback.position_s))

    if elapsed_s >= ANALYSIS_WINDOW_SECONDS and not math.isfinite(current_plv):
        return (
            "### [STATUS: NO DESCRIPTIVE PLV RESULT]\n\n"
            "Nie uzyskano poprawnego PLV po zebraniu pełnego okna. Sprawdź zmierzony kanał oddechowy, "
            "ciągłość timestampów, mapowanie kanałów i jakość sygnałów. Pacer nie zastępuje pomiaru oddechu."
        )
    if elapsed_s < ANALYSIS_WINDOW_SECONDS or not math.isfinite(current_plv):
        return (
            "### [STATUS: INSUFFICIENT DATA]\n\n"
            f"Zbierz ciągłe okno danych o długości co najmniej {ANALYSIS_WINDOW_SECONDS} s przed wygenerowaniem podsumowania."
        )

    engine = FallbackDiagnosticsEngine()

    session_plv = current_plv
    data_prov = (
        "DETERMINISTIC SYNTHETIC DEMO; CONSTRUCTED SHARED OSCILLATION"
        if st.session_state.get("session_is_synthetic_demo")
        else (
            "WEBCAM MOTION PROXY + SYNTHETIC fNIRS" if report_source == "WEBCAM"
            else ("SYNTHETIC RESPIRATION + fNIRS" if report_source == "SIMULATOR"
                  else ("IMPORTED XDF CHANNELS; USER-MAPPED" if report_source == "XDF_FILE"
                        else "LIVE LSL fNIRS + MEASURED LSL RESPIRATION"))
        )
    )

    active_resp_hz = None if report_source == "XDF_FILE" or st.session_state.get("session_is_synthetic_demo") else st.session_state.get("target_resp_hz", TARGET_RESP_HZ)
    curr_coh = st.session_state.get("current_coherence")
    measured_coh = float(curr_coh) if curr_coh is not None and math.isfinite(float(curr_coh)) else float("nan")

    res_state = {
        "data_provenance": data_prov,
        "plv_pct": round(session_plv, 1),
        "coherence": round(float(np.clip(measured_coh, 0.0, 1.0)), 3) if math.isfinite(measured_coh) else float("nan"),
        "state_label": "DESCRIPTIVE PLV; NO STATE CLASSIFICATION",
        "signal_configuration": {
            "pacing_reference_hz": round(active_resp_hz, 4) if active_resp_hz is not None else None,
            "analysis_band_hz": [0.07, 0.15],
        },
    }

    nvc_value = st.session_state.get("nvc_r")
    nvc_r = float(nvc_value) if nvc_value is not None and math.isfinite(float(nvc_value)) else float("nan")
    nvc_data = {
        "anti_correlation_r": round(nvc_r, 4),
        "interpretation": "Descriptive HbO2/HbR channel relationship; review with acquisition context",
    }

    dominant_resp_hz = st.session_state.get("current_dom_freq")
    psd_data = {
        "dominant_freq_hz": float(dominant_resp_hz) if dominant_resp_hz is not None else float("nan"),
        "band_power": {},
    }

    return engine.generate_report(
        resonance_state=res_state,
        nvc_check=nvc_data,
        power_spectrum_data=psd_data,
        session_duration_s=elapsed_s,
    )


def _build_session_bundle(summary_text: str) -> bytes:
    """Create a user-triggered, portable export with summary and signal provenance."""
    import csv
    import io
    import json
    import zipfile

    timestamps = np.asarray(st.session_state.get("buf_time", []), dtype=np.float64).ravel()
    respiration = np.asarray(st.session_state.get("buf_resp", []), dtype=np.float64).ravel()
    hbo2 = np.asarray(st.session_state.get("buf_hbo", []), dtype=np.float64).ravel()
    hbr = np.asarray(st.session_state.get("buf_hbr", []), dtype=np.float64).ravel()
    quality = summarize_signal_quality(
        timestamps,
        respiration=respiration,
        hbo2=hbo2,
        hbr=hbr,
    )
    sample_count = min(timestamps.size, respiration.size, hbo2.size, hbr.size)

    def finite_or_none(value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    source = (
        "SYNTHETIC_DEMO" if st.session_state.get("session_is_synthetic_demo")
        else (st.session_state.get("session_data_source") or DATA_SOURCE)
    )
    if st.session_state.get("session_data_source") == "XDF_FILE":
        channel_mapping = {
            "fnirs_stream_index": st.session_state.get("xdf_fnirs_stream_index"),
            "hbo2_channel_index": st.session_state.get("xdf_hbo2_channel_index"),
            "hbr_channel_index": st.session_state.get("xdf_hbr_channel_index"),
            "respiration_mode": st.session_state.get("xdf_respiration_mode"),
            "respiration_stream_index": (
                st.session_state.get("xdf_fnirs_stream_index")
                if st.session_state.get("xdf_respiration_mode") == "Same as fNIRS stream"
                else st.session_state.get("xdf_respiration_stream_index")
            ),
            "respiration_channel_index": st.session_state.get("xdf_respiration_channel_index"),
        }
    elif source == "LIVE_HARDWARE":
        channel_mapping = {
            "hbo2_channel_index": st.session_state.get("lsl_hbo_index"),
            "hbr_channel_index": st.session_state.get("lsl_hbr_index"),
            "respiration_mode": st.session_state.get("lsl_respiration_mode"),
            "respiration_channel_index": (
                st.session_state.get("lsl_resp_same_index")
                if st.session_state.get("lsl_respiration_mode") == "Same as fNIRS outlet"
                else st.session_state.get("lsl_resp_index")
            ),
        }
    else:
        channel_mapping = None
    manifest = {
        "schema_version": 1,
        "application": "NeuroResonance",
        "application_version": APP_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": source,
        "synthetic_data": bool(source in {"SYNTHETIC_DEMO", "SIMULATOR", "WEBCAM"}),
        "synthetic_seed": DEMO_SEED if source in {"SYNTHETIC_DEMO", "SIMULATOR"} else None,
        "channel_mapping": channel_mapping,
        "analysis": {
            "window_seconds": ANALYSIS_WINDOW_SECONDS,
            "phase_band_hz": [0.07, 0.15],
            "common_grid_hz": 2.0,
            "welch_segment_seconds": 50.0,
            "estimated_sample_rate_hz": quality["estimated_rate_hz"],
            "metrics": {
                "plv_percent": finite_or_none(st.session_state.get("current_plv")),
                "coherence_0_07_0_15_hz": finite_or_none(st.session_state.get("current_coherence")),
                "respiration_dominant_frequency_hz": finite_or_none(st.session_state.get("current_dom_freq")),
            },
            "interpretation": "Descriptive signal statistics; not a clinical or physiological conclusion.",
        },
        "quality": quality,
    }

    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow(["timestamp_s", "respiration", "hbo2", "hbr"])
    for index in range(sample_count):
        values = (timestamps[index], respiration[index], hbo2[index], hbr[index])
        writer.writerow([
            format(float(value), ".12g") if math.isfinite(float(value)) else ""
            for value in values
        ])

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("summary.md", summary_text)
        archive.writestr("signals.csv", csv_buffer.getvalue())
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        )
    return zip_buffer.getvalue()

if btn_report:
    with st.spinner("Preparing an experimental signal summary..."):
        summary = _generate_signal_summary()
        st.session_state["cached_report"] = summary
        current_plv = st.session_state.get("current_plv", float("nan"))
        st.session_state["cached_session_bundle"] = (
            _build_session_bundle(summary)
            if current_plv is not None and math.isfinite(float(current_plv))
            else None
        )

if st.session_state.get("cached_report"):
    with report_container.container():
        with st.expander("NeuroResonance Signal Summary", expanded=True):
            st.markdown(st.session_state["cached_report"])
        if st.session_state.get("cached_session_bundle") is not None:
            st.download_button(
                "Download session bundle (.zip)",
                data=st.session_state["cached_session_bundle"],
                file_name="neuroresonance_session_bundle.zip",
                mime="application/zip",
                key="download_session_bundle",
            )
            st.caption("Bundle includes the current signal window. Review it for sensitive data before sharing.")

# ===========================================================================
# Idle UI View
# ===========================================================================
def _render_idle_state() -> None:
    builtin_demo = bool(
        DATA_SOURCE == "XDF_FILE"
        and st.session_state.get("session_is_synthetic_demo")
        and st.session_state.get("session_data_source") == "XDF_FILE"
        and len(st.session_state.get("buf_time", [])) > 0
    )
    resp_title = (
        "SYNTHETIC DEMO RESPIRATION" if builtin_demo
        else ("WEBCAM MOTION PROXY" if DATA_SOURCE == "WEBCAM"
        else ("SIMULATOR RESP." if DATA_SOURCE == "SIMULATOR"
              else ("MAPPED RESP. · XDF" if DATA_SOURCE == "XDF_FILE"
                    else "MEASURED RESP. · LSL")))
    )
    fnirs_title = (
        "SYNTHETIC DEMO HbO₂/HbR · ARBITRARY UNITS" if builtin_demo
        else ("HEMODYNAMICS · LSL" if DATA_SOURCE == "LIVE_HARDWARE"
              else ("HEMODYNAMICS · XDF" if DATA_SOURCE == "XDF_FILE"
                    else "SIMULATED HEMODYNAMICS"))
    )
    orbit_title = (
        "PHASE ORBIT · SYNTHETIC DEMO" if builtin_demo
        else ("PHASE ORBIT · WEBCAM" if DATA_SOURCE == "WEBCAM"
              else ("PHASE ORBIT · LSL" if DATA_SOURCE == "LIVE_HARDWARE"
                    else ("PHASE ORBIT · XDF" if DATA_SOURCE == "XDF_FILE"
                          else "PHASE ORBIT · SIM")))
    )

    saved_xdf = (
        DATA_SOURCE == "XDF_FILE"
        and st.session_state.get("session_data_source") == "XDF_FILE"
        and len(st.session_state.get("buf_time", [])) > 0
    )
    if saved_xdf:
        t_idle = np.asarray(st.session_state["buf_time"], dtype=np.float64)
        resp_idle = np.asarray(st.session_state["buf_resp"], dtype=np.float64)
        hbo_idle = np.asarray(st.session_state["buf_hbo"], dtype=np.float64)
        hbr_idle = np.asarray(st.session_state["buf_hbr"], dtype=np.float64)
        plot_stride = max(1, len(t_idle) // 1200)
    else:
        t_idle = np.linspace(-BUFFER_SECONDS, 0, BUFFER_LEN)
        resp_idle = np.zeros(BUFFER_LEN)
        hbo_idle = np.zeros(BUFFER_LEN)
        hbr_idle = np.zeros(BUFFER_LEN)
        plot_stride = 1

    if DATA_SOURCE == "WEBCAM":
        cam_placeholder.info("Podgląd kamery pojawi się po uruchomieniu sesji.")
    elif DATA_SOURCE == "XDF_FILE":
        if saved_xdf and builtin_demo:
            cam_placeholder.info("Built-in deterministic synthetic demo is shown; its shared oscillation is constructed and is not physiological.")
        else:
            cam_placeholder.info("XDF playback follows recording timestamps; no webcam or pacer signal is added." if not saved_xdf else "Playback stopped or completed. The chart shows the imported XDF samples.")
    else:
        cam_placeholder.plotly_chart(_plot_breathing_guide(0), width="stretch", config={"displayModeBar": False})
    resp_placeholder.plotly_chart(_plot_respiratory_waveform(t_idle[::plot_stride], resp_idle[::plot_stride], title=resp_title), width="stretch", config={"displayModeBar": False})
    fnirs_placeholder.plotly_chart(_plot_fnirs_hemodynamics(t_idle[::plot_stride], hbo_idle[::plot_stride], hbr_idle[::plot_stride], title=fnirs_title), width="stretch", config={"displayModeBar": False})
    orbit_placeholder.plotly_chart(_plot_lissajous_orbit(hbo_idle[-min(len(hbo_idle), BUFFER_LEN):], resp_idle[-min(len(resp_idle), BUFFER_LEN):], title=orbit_title), width="stretch", config={"displayModeBar": False})
    idle_plv = st.session_state.get("current_plv", float("nan")) if saved_xdf else float("nan")
    if idle_plv is None:
        idle_plv = float("nan")
    gauge_placeholder.plotly_chart(_plot_plv_gauge(float(idle_plv)), width="stretch", config={"displayModeBar": False})

    with state_placeholder.container():
        if saved_xdf and builtin_demo:
            st.info(f"[SYNTHETIC DEMO · DESCRIPTIVE PLV: {idle_plv:.1f}%] — the shared oscillation is constructed in the fixture." if math.isfinite(float(idle_plv)) else "[STATUS: SYNTHETIC DEMO COMPLETE] — constructed samples remain visible")
        else:
            st.info("[STATUS: XDF PLAYBACK COMPLETE/STOPPED] — Imported samples remain visible" if saved_xdf else "[STATUS: STANDBY] — Ready to acquire data")
    with nvc_placeholder.container():
        st.caption("Hemodynamic signal relationship: awaiting samples")
    with insight_placeholder.container():
        st.caption(f"[SYSTEM] {st.session_state['last_ai_insight']}")
    with metrics_container.container():
        m1, m2 = st.columns(2)
        idle_window_plv = st.session_state.get("current_plv", float("nan"))
        idle_window_plv = float(idle_window_plv) if idle_window_plv is not None else float("nan")
        m1.metric("Current window PLV", f"{idle_window_plv:.1f}%" if math.isfinite(idle_window_plv) else "—")
        m2.metric("Analysis window", f"{ANALYSIS_WINDOW_SECONDS} s")
        m3, m4 = st.columns(2)
        if saved_xdf:
            observed_rate = estimate_sampling_rate_hz(t_idle)
            m3.metric("Sample rate", f"{observed_rate:.2f} Hz" if observed_rate is not None else "—")
            playback = st.session_state.get("xdf_playback")
            final_duration = st.session_state.get("session_active_duration", float(t_idle[-1] - t_idle[0]) if len(t_idle) > 1 else 0.0)
            m4.metric("Playback duration", f"{final_duration:.1f} s")
        else:
            m3.metric("Pacing target", f"{TARGET_RESP_BPM:.1f} bpm", help=f"Fixed reference: {TARGET_RESP_HZ:.3f} Hz")
            m4.metric("Analysis Band", "0.07–0.15 Hz")
        if saved_xdf:
            quality = summarize_signal_quality(
                t_idle,
                respiration=resp_idle,
                hbo2=hbo_idle,
                hbr=hbr_idle,
            )
            quality_source = "Deterministic synthetic demo" if builtin_demo else "Imported XDF"
            with st.expander("Signal quality and source", expanded=False):
                st.caption(f"Source: {quality_source}. Coverage and timing describe the displayed window only.")
                q1, q2, q3 = st.columns(3)
                q1.metric("Joint sample coverage", f"{quality['joint_coverage_pct']:.1f}%")
                q2.metric("Timestamp gaps > 3× median", str(quality["long_gap_count"]))
                q3.metric("Timestamp duration", f"{quality['duration_s']:.1f} s")
                channel_coverage = quality["channel_coverage_pct"]
                st.caption(
                    "Finite samples — "
                    f"respiration: {channel_coverage.get('respiration', 0.0):.1f}%; "
                    f"HbO₂: {channel_coverage.get('hbo2', 0.0):.1f}%; "
                    f"HbR: {channel_coverage.get('hbr', 0.0):.1f}%."
                )
    st.markdown("---")
    _render_diagnostics_panel()

def _show_camera_frame(placeholder: Any, frame: np.ndarray) -> None:
    """Update only the camera image without rebuilding charts or metrics."""
    frame_height, frame_width = frame.shape[:2]
    preview = cv2.resize(
        frame,
        (320, max(1, int(320 * frame_height / frame_width))),
    ) if frame_width > 320 else frame
    placeholder.image(
        preview,
        channels="BGR",
        output_format="JPEG",
        caption="Webcam chest-region motion proxy (live)",
    )


# ===========================================================================
# Active Real-Time Loop
# ===========================================================================
def _run_active_session() -> None:
    st.session_state["diagnostic_last_ui_refresh"] = datetime.now().astimezone().isoformat(timespec="seconds")
    session_started = st.session_state["session_start_monotonic"]
    data_source = st.session_state.get("session_data_source") or DATA_SOURCE
    xdf_source: Optional[XDFPlaybackSource] = st.session_state.get("xdf_playback")
    elapsed = xdf_source.position_s if data_source == "XDF_FILE" and xdf_source is not None else elapsed_seconds(session_started)
    if data_source == "XDF_FILE" and xdf_source is None:
        _record_diagnostic("ERROR", "Nie znaleziono źródła odtwarzania XDF; sesja została zatrzymana.")
        st.error("XDF playback source is missing; stopping this session.")
        _stop_session("Zatrzymano sesję bez źródła XDF.")
        st.rerun()
        return
    if data_source != "XDF_FILE" and duration_reached(session_started, session_duration):
        _stop_session("Sesja zakończyła się po osiągnięciu zaplanowanego czasu.")
        st.rerun()
        return

    sim: fNIRSSimulator = st.session_state["simulator"]
    lsl_session: Optional[SelectedLSLSession] = st.session_state.get("lsl_session")
    lsl_receiver: Optional[LSLBackgroundReceiver] = st.session_state.get("lsl_receiver")
    cam: Optional[CameraTracker] = st.session_state["camera"]
    estimator: ResonanceEstimator = st.session_state["estimator"]
    nvc_checker: HemoglobinCorrelationSummary = st.session_state["nvc_checker"]

    if data_source in ("SIMULATOR", "WEBCAM"):
        if sim is None:
            raise RuntimeError("The synthetic signal generator is missing from the active session.")
        if sim.worker_error:
            detail = sim.worker_traceback or "Original background stack was not available."
            raise RuntimeError(f"The synthetic signal generator failed: {sim.worker_error}\n{detail}")
        if elapsed > 1.0 and not sim.is_running:
            raise RuntimeError("The synthetic signal generator stopped without reporting an error.")

    # --- Fixed demonstration pacing reference ---
    _active_resp_hz: float = st.session_state.get("target_resp_hz", DEFAULT_RESP_HZ)

    ring_time: CircularRingBuffer = st.session_state["ring_time"]
    ring_resp: CircularRingBuffer = st.session_state["ring_resp"]
    ring_hbo: CircularRingBuffer = st.session_state["ring_hbo"]
    ring_hbr: CircularRingBuffer = st.session_state["ring_hbr"]

    live_data_fresh = True
    live_data_stale = False

    def _get_simulator_chunk(t_now: float) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        if sim is None:
            return {}, np.empty(0, dtype=np.float64)
        sample_data = sim.get_new_samples()
        timestamps = np.asarray(sample_data["timestamps"], dtype=np.float64)
        if timestamps.size == 0:
            last_receive = st.session_state.get("sim_last_receive_monotonic")
            if t_now > 3.0 and (last_receive is None or time.monotonic() - last_receive > 3.0):
                raise RuntimeError("The synthetic signal generator has not produced samples for more than 3 seconds.")
            return sample_data, timestamps
        st.session_state["sim_last_receive_monotonic"] = time.monotonic()
        dropped = int(sample_data.get("dropped_samples", 0))
        if dropped:
            for buffer in (ring_time, ring_resp, ring_hbo, ring_hbr):
                buffer.reset()
            st.session_state["sim_last_quality_warning"] = (
                f"Simulator sample buffer overflow dropped {dropped} samples; analysis buffers were reset."
            )
            _record_diagnostic("WARNING", st.session_state["sim_last_quality_warning"])
        offset = st.session_state.get("sim_clock_offset")
        if offset is None:
            offset = t_now - float(timestamps[-1])
            st.session_state["sim_clock_offset"] = offset
        return sample_data, timestamps + float(offset)

    def _ingest_samples() -> Tuple[Optional[np.ndarray], bool, bool]:
        """Sample the current source without rebuilding the dashboard."""
        nonlocal ring_time, ring_resp, ring_hbo, ring_hbr
        t_now = elapsed_seconds(st.session_state.get("session_start_monotonic"))
        frame = None
        fresh = True
        stale = False

        if data_source == "LIVE_HARDWARE":
            fresh = False
            if lsl_receiver is not None:
                chunk = lsl_receiver.get_chunk(max_samples=6000)
                live_warnings = list(chunk.get("quality_warnings", []))
                if chunk.get("receiver_error"):
                    live_warnings.append(str(chunk["receiver_error"]))
                timestamps = np.asarray(chunk.get("timestamps", []), dtype=np.float64)
                n_samples = int(timestamps.size)
                signal_break = any(
                    "gap(s) detected" in item
                    or "non-finite sample(s)" in item
                    or "queue overflow" in item
                    for item in live_warnings
                )
                if live_warnings:
                    st.session_state["lsl_last_quality_warnings"] = live_warnings
                    for warning in live_warnings:
                        severity = "ERROR" if chunk.get("receiver_error") and warning == str(chunk["receiver_error"]) else "WARNING"
                        _record_diagnostic(severity, warning)
                if signal_break:
                    ring_time.reset()
                    ring_resp.reset()
                    ring_hbo.reset()
                    ring_hbr.reset()
                    st.session_state["lsl_last_quality_warnings"] = live_warnings + [
                        "Analysis buffers cleared at the acquisition interruption; raw samples remain in the recording."
                    ]
                if n_samples:
                    st.session_state["lsl_last_receive_monotonic"] = time.monotonic()
                    observed_rate = lsl_session.fnirs_sample_rate_hz
                    wanted_capacity = min(
                        60000,
                        max(BUFFER_LEN, int(math.ceil(observed_rate * ANALYSIS_WINDOW_SECONDS)) + 2 if observed_rate > 0 else BUFFER_LEN),
                    )
                    if wanted_capacity > ring_time.capacity:
                        old_values = [buffer.get_all() for buffer in (ring_time, ring_resp, ring_hbo, ring_hbr)]
                        resized = [CircularRingBuffer(capacity=wanted_capacity) for _ in range(4)]
                        for buffer, values in zip(resized, old_values):
                            buffer.push_chunk(values)
                        st.session_state["ring_time"], st.session_state["ring_resp"], st.session_state["ring_hbo"], st.session_state["ring_hbr"] = resized
                        ring_time, ring_resp, ring_hbo, ring_hbr = resized
                    if observed_rate > 0 and observed_rate * ANALYSIS_WINDOW_SECONDS > ring_time.capacity:
                        st.session_state["lsl_last_quality_warnings"] = list(dict.fromkeys(
                            st.session_state.get("lsl_last_quality_warnings", [])
                            + [f"The bounded analysis buffer cannot hold {ANALYSIS_WINDOW_SECONDS} seconds at this outlet rate; live PLV remains unavailable."]
                        ))
                    if not signal_break:
                        origin = st.session_state.get("lsl_origin")
                        if origin is None:
                            origin = float(timestamps[0])
                            st.session_state["lsl_origin"] = origin
                        rel_time = timestamps - float(origin)
                        resp_values = np.asarray(chunk.get("Respiration", []), dtype=np.float64)
                        if resp_values.size != n_samples:
                            resp_values = np.full(n_samples, np.nan, dtype=np.float64)
                        ring_time.push_chunk(rel_time)
                        ring_resp.push_chunk(resp_values)
                        ring_hbo.push_chunk(np.asarray(chunk["HbO2"], dtype=np.float64))
                        ring_hbr.push_chunk(np.asarray(chunk["HbR"], dtype=np.float64))
                        fresh = True
            last_receive = st.session_state.get("lsl_last_receive_monotonic")
            stale = last_receive is None or time.monotonic() - last_receive > 2.0
        elif data_source == "XDF_FILE":
            if xdf_source is not None:
                chunk = xdf_source.get_chunk(max_samples=6000)
                timestamps = np.asarray(chunk.get("timestamps", []), dtype=np.float64)
                if chunk.get("quality_warnings"):
                    st.session_state["xdf_quality_warnings"] = list(chunk["quality_warnings"])
                    for warning in chunk["quality_warnings"]:
                        _record_diagnostic("WARNING", str(warning))
                if timestamps.size:
                    ring_time.push_chunk(timestamps)
                    ring_resp.push_chunk(np.asarray(chunk["Respiration"], dtype=np.float64))
                    ring_hbo.push_chunk(np.asarray(chunk["HbO2"], dtype=np.float64))
                    ring_hbr.push_chunk(np.asarray(chunk["HbR"], dtype=np.float64))
                    fresh = True
        elif data_source == "WEBCAM":
            if cam is not None and cam.is_running:
                frame = cam.get_frame_annotated(max_age_s=2.0)
            sim_data, sample_times = _get_simulator_chunk(t_now)
            if sample_times.size:
                resp_values = np.full(sample_times.size, np.nan, dtype=np.float64)
                if cam is not None and cam.is_running:
                    resp_obj: RespiratorySignal = cam.get_signal(120)
                    camera_times = np.asarray(resp_obj.timestamps, dtype=np.float64)
                    camera_values = np.asarray(resp_obj.detrended, dtype=np.float64)
                    wall_start = st.session_state.get("camera_wall_start")
                    if wall_start is not None and camera_times.size == camera_values.size:
                        valid_camera = np.isfinite(camera_times) & np.isfinite(camera_values)
                        if int(np.count_nonzero(valid_camera)) >= 2:
                            resp_values = np.interp(
                                sample_times,
                                camera_times[valid_camera] - float(wall_start),
                                camera_values[valid_camera],
                                left=np.nan,
                                right=np.nan,
                            )
                ring_time.push_chunk(sample_times)
                ring_resp.push_chunk(resp_values)
                ring_hbo.push_chunk(np.asarray(sim_data["HbO2"], dtype=np.float64))
                ring_hbr.push_chunk(np.asarray(sim_data["HbR"], dtype=np.float64))
        else:  # SIMULATOR
            sim_data, sample_times = _get_simulator_chunk(t_now)
            if sample_times.size:
                resp_rng = st.session_state.get("sim_resp_rng")
                if resp_rng is None:
                    resp_rng = np.random.default_rng(DEMO_SEED)
                    st.session_state["sim_resp_rng"] = resp_rng
                resp_values = (
                    np.sin(2 * np.pi * _active_resp_hz * sample_times)
                    + 0.15 * np.sin(4 * np.pi * _active_resp_hz * sample_times)
                    + resp_rng.normal(0, 0.03, size=sample_times.size)
                )
                ring_time.push_chunk(sample_times)
                ring_resp.push_chunk(resp_values)
                ring_hbo.push_chunk(np.asarray(sim_data["HbO2"], dtype=np.float64))
                ring_hbr.push_chunk(np.asarray(sim_data["HbR"], dtype=np.float64))
        return frame, fresh, stale

    annotated_frame, live_data_fresh, live_data_stale = _ingest_samples()

    buf_time = ring_time.get_all()
    buf_resp = ring_resp.get_all()
    buf_hbo = ring_hbo.get_all()
    buf_hbr = ring_hbr.get_all()

    st.session_state["buf_time"] = buf_time
    st.session_state["buf_resp"] = buf_resp
    st.session_state["buf_hbo"] = buf_hbo
    st.session_state["buf_hbr"] = buf_hbr

    valid_samples = len(buf_time)
    live_span_s = float(buf_time[-1] - buf_time[0]) if valid_samples > 1 else 0.0
    enough_samples = valid_samples >= 8 and live_span_s >= ANALYSIS_WINDOW_SECONDS

    low_resp_quality = False
    nvc_r = float("nan")

    if (data_source == "LIVE_HARDWARE" and (not live_data_fresh or live_data_stale)) or (
        data_source == "XDF_FILE" and not live_data_fresh
    ):
        current_plv = float("nan")
        state_label = "WAITING FOR NEW SAMPLES"
        state_color = "#4FC3F7"
        hbo_epoch = buf_hbo if len(buf_hbo) > 0 else np.zeros(10)
        r_epoch = buf_resp if len(buf_resp) > 0 else np.zeros(10)
        st.session_state["current_coherence"] = None
        st.session_state["current_dom_freq"] = float("nan")
    elif not enough_samples:
        current_plv = float("nan")
        state_label = (
            f"STARTING (BUFFERING {live_span_s:.1f}/{ANALYSIS_WINDOW_SECONDS:.0f} s)"
        )
        state_color = "#4FC3F7"
        hbo_epoch = buf_hbo if len(buf_hbo) > 0 else np.zeros(10)
        r_epoch = buf_resp if len(buf_resp) > 0 else np.zeros(10)
    else:
        win_len = min(valid_samples, ring_time.capacity) if data_source in ("LIVE_HARDWARE", "XDF_FILE") else min(valid_samples, BUFFER_LEN)
        r_epoch = buf_resp[-win_len:]
        hbo_epoch = buf_hbo[-win_len:]
        hbr_epoch = buf_hbr[-win_len:]
        t_epoch = buf_time[-win_len:]

        # Live LSL mode must use a measured respiratory channel. Never compare
        # fNIRS against the visual pacing reference as if it were breathing data.
        if data_source in ("LIVE_HARDWARE", "XDF_FILE") and not np.all(np.isfinite(r_epoch)):
            current_plv = float("nan")
            state_label = "INSUFFICIENT MEASURED RESPIRATION"
            state_color = "#4FC3F7"
            st.session_state["current_coherence"] = None
            st.session_state["current_dom_freq"] = float("nan")
        else:
            try:
                st.session_state["analysis_error"] = None
                res_state: ResonanceState = estimator.update(
                    resp_signal=r_epoch,
                    hbo2_signal=hbo_epoch,
                    hbr_signal=hbr_epoch,
                    timestamps_resp=t_epoch,
                    timestamps_fnirs=t_epoch,
                )
                if "INSUFFICIENT SIGNAL" in res_state.state_label:
                    current_plv = float("nan")
                    state_label = res_state.state_label
                    state_color = res_state.state_color
                else:
                    current_plv = float(res_state.plv_percent)
                if data_source in ("LIVE_HARDWARE", "XDF_FILE") and math.isfinite(current_plv):
                    state_label = "XDF REPLAY PLV (DESCRIPTIVE; NO STATE CLASSIFICATION)" if data_source == "XDF_FILE" else "LIVE PLV (DESCRIPTIVE; NO STATE CLASSIFICATION)"
                    state_color = "#4FC3F7"
                else:
                    state_label = res_state.state_label
                    state_color = res_state.state_color
                st.session_state["current_coherence"] = float(res_state.coherence_at_01hz)
                st.session_state["current_dom_freq"] = float(res_state.dominant_freq_hz)
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                logger.exception("Resonance estimator update failed")
                _record_diagnostic("ERROR", "Estymator koherencji nie mógł przetworzyć próbek.", exc)
                st.session_state["analysis_error"] = error_message
                current_plv = float("nan")
                state_label = "ANALYSIS ERROR"
                state_color = "#FF4444"
                st.session_state["current_coherence"] = None
                st.session_state["current_dom_freq"] = float("nan")

        # Heuristic warning for reduced webcam-derived chest-region motion.
        # This motion proxy cannot diagnose apnea or confirm absent breathing.
        if data_source == "WEBCAM" and elapsed > 5.0 and len(r_epoch) >= 40:
            baseline_std = float(np.std(r_epoch))
            recent_std = float(np.std(r_epoch[-30:]))
            if baseline_std > 1e-4 and recent_std < 0.25 * baseline_std:
                low_resp_quality = True

        # Summarize the descriptive HbO₂/HbR channel relationship.
        if data_source not in ("LIVE_HARDWARE", "XDF_FILE"):
            nvc_metrics = nvc_checker.update(hbo_epoch, hbr_epoch)
            nvc_r = float(nvc_metrics.get("anticorrelation_r", float("nan")))

    st.session_state["current_plv"] = current_plv
    st.session_state["current_state"] = state_label
    st.session_state["state_color"] = state_color
    st.session_state["nvc_r"] = nvc_r

    # 5. Periodically refresh the deterministic local insight
    now = time.time()
    if now - st.session_state["last_insight_time"] > 12.0 and math.isfinite(current_plv):
        st.session_state["last_insight_time"] = now
        try:
            # Deterministic local summary; no session data leaves this process.
            engine = FallbackDiagnosticsEngine()
            insight_state = build_session_insight_state(
                plv_pct=current_plv,
                phase_lag_ms=None,
                estimator_coherence=st.session_state.get("current_coherence"),
                state_label=state_label,
                pacing_reference_hz=_active_resp_hz if data_source != "XDF_FILE" else None,
            )
            st.session_state["last_ai_insight"] = engine.generate_realtime_insight(insight_state)
        except Exception as exc:
            logger.debug("Periodic local insight refresh deferred: %s", exc)
            _record_diagnostic("WARNING", "Nie udało się odświeżyć eksperymentalnego podsumowania sygnału.", exc)

    # 6. Render Visualizations with static keys and zero-lag downscaled preview
    builtin_demo = bool(st.session_state.get("session_is_synthetic_demo"))
    resp_title = (
        "Synthetic respiration channel (built-in demo)" if builtin_demo
        else ("Webcam ROI Luminance (uncalibrated motion proxy)" if data_source == "WEBCAM"
              else ("Synthetic Paced Respiration (0.1 Hz reference)" if data_source == "SIMULATOR"
                    else ("Imported XDF respiration channel (if mapped)" if data_source == "XDF_FILE"
                          else "Pacer Reference (not measured respiration; PLV disabled without an LSL respiration channel)")))
    )
    fnirs_title = (
        "Synthetic fNIRS-like channels (built-in demo; arbitrary units)" if builtin_demo
        else ("fNIRS HbO₂/HbR Signals (Live LSL; units per device metadata)" if data_source == "LIVE_HARDWARE"
              else ("Mapped fNIRS channels replayed from XDF (units from file metadata)" if data_source == "XDF_FILE"
                    else "Synthetic fNIRS-like HbO₂/HbR Signals (arbitrary units)"))
    )
    orbit_title = (
        "Phase orbit: constructed synthetic demo signals" if builtin_demo
        else ("Phase Orbit: Webcam Respiration vs Synthetic HbO2" if data_source == "WEBCAM"
              else ("Phase Orbit: Measured LSL Respiration vs Real HbO₂" if data_source == "LIVE_HARDWARE"
                    else ("Phase Orbit: User-mapped XDF channels" if data_source == "XDF_FILE"
                          else "Paired Signal Phase Orbit (Simulator)")))
    )

    if data_source == "WEBCAM" and annotated_frame is not None:
        _show_camera_frame(cam_placeholder, annotated_frame)
    elif data_source == "WEBCAM":
        cam_placeholder.warning("Brak świeżego obrazu z kamery. Trwa ponowne łączenie; sprawdź też, czy kamera nie jest zajęta przez inną aplikację.")
    elif data_source == "XDF_FILE":
        playback = st.session_state.get("xdf_playback")
        if playback is not None:
            cam_placeholder.info(f"XDF playback {elapsed:.1f}s / {playback.duration_s:.1f}s at {playback.speed:g}×. No pacer is mixed into the file data.")
    else:
        cam_placeholder.plotly_chart(_plot_breathing_guide(elapsed), width="stretch", config={"displayModeBar": False})

    plot_stride = max(1, len(buf_time) // 1200)
    resp_placeholder.plotly_chart(_plot_respiratory_waveform(buf_time[::plot_stride], buf_resp[::plot_stride], title=resp_title), width="stretch", config={"displayModeBar": False})
    fnirs_placeholder.plotly_chart(_plot_fnirs_hemodynamics(buf_time[::plot_stride], buf_hbo[::plot_stride], buf_hbr[::plot_stride], title=fnirs_title), width="stretch", config={"displayModeBar": False})
    orbit_placeholder.plotly_chart(_plot_lissajous_orbit(hbo_epoch, r_epoch, title=orbit_title), width="stretch", config={"displayModeBar": False})
    gauge_placeholder.plotly_chart(_plot_plv_gauge(current_plv), width="stretch", config={"displayModeBar": False})

    with state_placeholder.container():
        if state_label == "ANALYSIS ERROR":
            st.error(f"[ANALYSIS ERROR] {st.session_state.get('analysis_error', 'Unknown processing error')}")
        simulator_warning = st.session_state.get("sim_last_quality_warning")
        if simulator_warning:
            st.warning(simulator_warning)
        if low_resp_quality:
            st.warning("LOW CAMERA MOTION-PROXY VARIABILITY — wykryto spadek ruchu w obrazie klatki piersiowej. PLV pozostaje niezmienionym wynikiem estymatora; sprawdź kadr i jakość sygnału.")
        if "INSUFFICIENT MEASURED RESPIRATION" in state_label:
            st.info("PLV unavailable: this source needs an explicitly mapped measured respiration channel. No pacing signal is substituted.")
        if data_source == "LIVE_HARDWARE":
            for warning in st.session_state.get("lsl_last_quality_warnings", []):
                st.warning(warning)
        elif data_source == "XDF_FILE":
            for warning in st.session_state.get("xdf_quality_warnings", []):
                st.warning(warning)
            if builtin_demo and math.isfinite(current_plv):
                st.info(f"[SYNTHETIC DEMO · DESCRIPTIVE PLV: {current_plv:.1f}%] — the shared oscillation is constructed in the fixture.")
            elif "NO STATE CLASSIFICATION" in state_label:
                st.info(f"[XDF REPLAY · DESCRIPTIVE PLV: {current_plv:.1f}%] — signal statistic only; no physiological state is inferred.")
        elif "NO STATE CLASSIFICATION" in state_label:
            display_source = (
                "SYNTHETIC DEMO" if builtin_demo
                else ("XDF REPLAY" if data_source == "XDF_FILE"
                      else ("LIVE HARDWARE" if data_source == "LIVE_HARDWARE" else "SIMULATOR"))
            )
            st.info(f"[{display_source} · DESCRIPTIVE PLV: {current_plv:.1f}%] — signal statistic only; no physiological state is inferred.")
        elif "STARTING" in state_label:
            st.info(f"[STATUS: {state_label}] — Collecting the analysis window.")

    with nvc_placeholder.container():
        if not math.isfinite(nvc_r):
            st.markdown("**Hemodynamic signal relationship:** n/d (insufficient data)")
        elif nvc_r < -0.30:
            st.markdown(f"**Hemodynamic signal relationship:** inverse (*r* = `{nvc_r:.2f}`); review alongside acquisition quality and context")
        else:
            st.markdown(f"**Hemodynamic signal relationship:** (*r* = `{nvc_r:.2f}`); review alongside acquisition quality and context")

    with insight_placeholder.container():
        st.markdown(f"**Signal Summary:** {st.session_state['last_ai_insight']}")

    with metrics_container.container():
        current_window_plv = st.session_state.get("current_plv", float("nan"))
        current_window_plv = float(current_window_plv) if current_window_plv is not None else float("nan")
        measured_rate = estimate_sampling_rate_hz(buf_time)
        st.session_state["measured_sample_rate_hz"] = measured_rate
        c1, c2 = st.columns(2)
        c1.metric("Current window PLV", f"{current_window_plv:.1f}%" if math.isfinite(current_window_plv) else "—")
        c2.metric("Analysis window", f"{ANALYSIS_WINDOW_SECONDS} s")
        c3, c4 = st.columns(2)
        c3.metric("Sample rate", f"{measured_rate:.2f} Hz" if measured_rate is not None else "—")
        total_duration = xdf_source.duration_s if data_source == "XDF_FILE" and xdf_source is not None else session_duration
        if data_source == "XDF_FILE":
            progress_pct = min(100.0, max(0.0, elapsed / total_duration * 100.0)) if total_duration > 0 else 0.0
            c4.metric(
                "Playback progress",
                f"{progress_pct:.0f}%",
                help=f"{elapsed:.1f} s elapsed of {total_duration:.1f} s",
            )
        else:
            c4.metric(
                "Elapsed time",
                f"{int(elapsed)} s",
                help=f"Session duration: {session_duration} s",
            )

        quality = summarize_signal_quality(
            buf_time,
            respiration=buf_resp,
            hbo2=buf_hbo,
            hbr=buf_hbr,
        )
        quality_source = (
            "Deterministic synthetic demo" if builtin_demo
            else (st.session_state.get("session_data_source") or DATA_SOURCE)
        )
        with st.expander("Signal quality and source", expanded=False):
            st.caption(f"Source: {quality_source}. Coverage and timing describe the displayed window only.")
            q1, q2, q3 = st.columns(3)
            q1.metric("Joint sample coverage", f"{quality['joint_coverage_pct']:.1f}%")
            q2.metric("Timestamp gaps > 3× median", str(quality["long_gap_count"]))
            q3.metric("Timestamp duration", f"{quality['duration_s']:.1f} s")
            channel_coverage = quality["channel_coverage_pct"]
            st.caption(
                "Finite samples — "
                f"respiration: {channel_coverage.get('respiration', 0.0):.1f}%; "
                f"HbO₂: {channel_coverage.get('hbo2', 0.0):.1f}%; "
                f"HbR: {channel_coverage.get('hbr', 0.0):.1f}%."
            )
            if quality["length_mismatch"]:
                st.warning("Signal arrays have different lengths; coverage is computed over their shared prefix.")
            if quality["nonmonotonic_interval_count"]:
                st.warning(f"Found {quality['nonmonotonic_interval_count']} non-increasing timestamp interval(s).")

    st.markdown("---")
    _render_diagnostics_panel()

    tick = 1.0 / CAMERA_PREVIEW_FPS
    deadline = time.monotonic() + SLEEP_INTERVAL
    last_sample = time.monotonic()
    while True:
        loop_start = time.monotonic()
        if loop_start >= deadline or (data_source != "XDF_FILE" and duration_reached(session_started, session_duration)):
            break

        frame = None
        if loop_start - last_sample >= 1.0 / FS_DISPLAY:
            last_sample = loop_start
            frame, _, _ = _ingest_samples()
        elif data_source == "WEBCAM" and cam is not None and cam.is_running:
            frame = cam.get_frame_annotated(max_age_s=2.0)

        if data_source == "WEBCAM" and frame is not None:
            _show_camera_frame(cam_placeholder, frame)
        time.sleep(max(0.0, tick - (time.monotonic() - loop_start)))
        if data_source == "XDF_FILE" and xdf_source is not None and xdf_source.is_finished:
            break

    if data_source == "XDF_FILE" and xdf_source is not None and xdf_source.is_finished:
        _stop_session("Odtwarzanie pliku XDF dobiegło końca.")
        st.rerun()
        return
    if data_source != "XDF_FILE" and duration_reached(session_started, session_duration):
        _stop_session("Sesja zakończyła się po osiągnięciu zaplanowanego czasu.")
        st.rerun()
        return
    st.rerun()

# ===========================================================================
# Execution Dispatcher
# ===========================================================================
if st.session_state["session_active"]:
    try:
        _run_active_session()
    except Exception as exc:
        _record_diagnostic("ERROR", "Nieobsłużony błąd przerwał aktywną sesję; zapisano szczegóły.", exc)
        st.session_state["analysis_error"] = f"{type(exc).__name__}: {exc}"
        try:
            _stop_session("Sesja została bezpiecznie zatrzymana po błędzie.")
        except Exception as cleanup_exc:
            _record_diagnostic("ERROR", "Wystąpił dodatkowy błąd podczas zatrzymywania sesji.", cleanup_exc)
            st.session_state["session_active"] = False
            _cleanup_session_resources(recording_status="failed")
        st.rerun()
else:
    try:
        _render_idle_state()
    except Exception as exc:
        _record_diagnostic("ERROR", "Nieobsłużony błąd podczas odświeżania widoku.", exc)
        st.error(f"[DASHBOARD RENDER ERROR]: {type(exc).__name__}: {exc}")
        _render_diagnostics_panel()
