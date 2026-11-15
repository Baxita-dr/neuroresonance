"""Explicit LSL discovery, channel mapping, clock correction, and local capture.

This module deliberately does not infer physiological meaning from channel
names or units. The operator selects stream identities and channel indices,
then confirms that the selected HbO2/HbR channels are already derived
haemoglobin signals and that respiration is measured rather than paced.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.lsl_runtime import LSL_AVAILABLE as _LSL_AVAILABLE, StreamInlet, lsl_unavailable_message, resolve_streams


_LSL_CHANNEL_FORMAT_NAMES = {
    0: "undefined",
    1: "float32",
    2: "double64",
    3: "string",
    4: "int32",
    5: "int16",
    6: "int8",
    7: "int64",
}
_NUMERIC_LSL_FORMATS = {"float32", "double64", "int8", "int16", "int32", "int64"}


def normalize_lsl_channel_format(value: Any) -> str:
    """Return a readable LSL format name for pylsl strings or enum codes."""
    if isinstance(value, str):
        name = value.strip().casefold()
        if name.startswith("cf_"):
            name = name[3:]
        if name in _NUMERIC_LSL_FORMATS or name in {"string", "undefined"}:
            return name
        try:
            value = int(name)
        except (TypeError, ValueError):
            return name or "unknown"
    try:
        return _LSL_CHANNEL_FORMAT_NAMES.get(int(value), "unknown")
    except (TypeError, ValueError, OverflowError):
        return "unknown"


def is_numeric_lsl_channel_format(value: Any) -> bool:
    """LSL signal processing accepts numeric sample formats, never markers."""
    return normalize_lsl_channel_format(value) in _NUMERIC_LSL_FORMATS


def _value(info: Any, method_name: str, default: Any = "") -> Any:
    try:
        value = getattr(info, method_name)
        return value() if callable(value) else value
    except Exception:
        return default


def _stream_key(info: Any) -> str:
    """Prefer the outlet uid; use a core-metadata key only as a fallback."""
    uid = str(_value(info, "uid", "") or "").strip()
    if uid:
        return uid
    fields = (
        str(_value(info, "source_id", "") or ""),
        str(_value(info, "name", "") or ""),
        str(_value(info, "type", "") or ""),
        str(_value(info, "hostname", "") or ""),
        str(_value(info, "channel_count", "") or ""),
        str(_value(info, "nominal_srate", "") or ""),
    )
    return "|".join(fields)


def _core_metadata(info: Any) -> Dict[str, Any]:
    raw_channel_format = _value(info, "channel_format", "")
    channel_format = normalize_lsl_channel_format(raw_channel_format)
    return {
        "selector": _stream_key(info),
        "uid": str(_value(info, "uid", "") or ""),
        "name": str(_value(info, "name", "") or ""),
        "type": str(_value(info, "type", "") or ""),
        "source_id": str(_value(info, "source_id", "") or ""),
        "hostname": str(_value(info, "hostname", "") or ""),
        "channel_count": int(_value(info, "channel_count", 0) or 0),
        "nominal_srate_hz": float(_value(info, "nominal_srate", 0.0) or 0.0),
        "channel_format": channel_format,
        "channel_format_code": raw_channel_format,
        "numeric_samples": is_numeric_lsl_channel_format(raw_channel_format),
    }


def _channel_metadata(info: Any) -> List[Dict[str, Any]]:
    channels: List[Dict[str, Any]] = []
    try:
        node = info.desc().child("channels").child("channel")
        while node.name() == "channel":
            index = len(channels)
            label = node.child_value("label") or ""
            unit = node.child_value("unit") or ""
            channel_type = node.child_value("type") or ""
            channels.append({
                "index": index,
                "label": label,
                "unit": unit,
                "type": channel_type,
                "metadata_present": bool(label or unit or channel_type),
            })
            node = node.next_sibling()
    except Exception:
        channels = []
    count = int(_value(info, "channel_count", len(channels)) or len(channels))
    if not channels:
        channels = [
            {"index": i, "label": "", "unit": "", "type": "", "metadata_present": False}
            for i in range(count)
        ]
    return channels


def discover_lsl_streams(timeout_s: float = 1.0) -> List[Dict[str, Any]]:
    """Return available outlets with the core metadata supplied by LSL discovery."""
    if not _LSL_AVAILABLE or resolve_streams is None:
        raise RuntimeError(f"LSL stream discovery unavailable: {lsl_unavailable_message()}")
    infos = resolve_streams(wait_time=max(0.5, float(timeout_s)))
    records = [_core_metadata(info) for info in infos]
    records.sort(key=lambda row: (
        row["name"], row["type"], row["source_id"], row["hostname"], row["selector"]
    ))
    return records


def inspect_lsl_stream(selector: str, timeout_s: float = 3.0) -> Dict[str, Any]:
    """Open one selected outlet briefly and retrieve its complete channel metadata."""
    if not _LSL_AVAILABLE or resolve_streams is None or StreamInlet is None:
        raise RuntimeError(f"LSL metadata unavailable: {lsl_unavailable_message()}")
    infos = resolve_streams(wait_time=max(0.5, float(timeout_s)))
    matches = [info for info in infos if _stream_key(info) == selector]
    if len(matches) != 1:
        raise RuntimeError(
            f"Selected LSL identity resolved to {len(matches)} outlets; refresh and select one unique outlet."
        )
    inlet = StreamInlet(matches[0], max_buflen=10)
    try:
        inlet.open_stream(timeout=float(timeout_s))
        complete = inlet.info(timeout=float(timeout_s))
        record = _core_metadata(complete)
        record["channels"] = _channel_metadata(complete)
        record["metadata_complete"] = bool(record["channels"]) and all(
            item["label"] and item["unit"] and item["type"]
            for item in record["channels"]
        )
        record["channel_count_matches_metadata"] = len(record["channels"]) == record["channel_count"]
        return record
    finally:
        try:
            inlet.close_stream()
        except Exception:
            pass


class LocalSessionRecorder:
    """Append raw numeric samples and both timestamp domains to local CSV files."""

    def __init__(self, root: Path, metadata: Dict[str, Any]) -> None:
        self.session_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        self.directory = Path(root) / self.session_id
        self.directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        self._restrict_permissions(self.directory, directory=True)
        self.metadata_path = self.directory / "metadata.json"
        self.metadata = dict(metadata)
        self.metadata.update({
            "schema_version": 1,
            "session_id": self.session_id,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "ended_utc": None,
            "status": "recording",
            "sample_counts": {},
            "quality": {},
        })
        self._write_metadata()

    @staticmethod
    def _restrict_permissions(path: Path, *, directory: bool = False) -> None:
        """Apply owner-only POSIX permissions where the platform supports them."""
        if os.name == "posix":
            path.chmod(0o700 if directory else 0o600)

    @property
    def path(self) -> str:
        return str(self.directory.resolve())

    def record(
        self,
        role: str,
        samples: Sequence[Sequence[float]],
        raw_timestamps: Sequence[float],
        local_timestamps: Sequence[float],
        correction_s: float,
        channels: Sequence[Dict[str, Any]],
    ) -> None:
        values = np.asarray(samples, dtype=np.float64)
        raw_ts = np.asarray(raw_timestamps, dtype=np.float64).ravel()
        local_ts = np.asarray(local_timestamps, dtype=np.float64).ravel()
        if values.size == 0 or raw_ts.size == 0:
            return
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.shape[0] != raw_ts.size or local_ts.size != raw_ts.size:
            raise ValueError("LSL sample and timestamp counts do not match; refusing incomplete recording.")
        if values.shape[1] != len(channels):
            raise ValueError("LSL channel metadata does not match the sample width.")

        path = self.directory / f"{role}_raw.csv"
        write_header = not path.exists()
        # Keep untrusted outlet labels out of CSV headers. Full labels and units
        # are preserved in metadata.json, avoiding spreadsheet formula headers.
        channel_headers = [f"channel_{i}" for i, _item in enumerate(channels)]
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "a", newline="", encoding="utf-8") as handle:
            self._restrict_permissions(path)
            writer = csv.writer(handle)
            if write_header:
                writer.writerow([
                    "sample_index",
                    "timestamp_raw_lsl_s",
                    "timestamp_local_lsl_s",
                    "time_correction_s",
                    *channel_headers,
                ])
            start = int(self.metadata["sample_counts"].get(role, 0))
            for offset, row in enumerate(values):
                writer.writerow([
                    start + offset,
                    repr(float(raw_ts[offset])),
                    repr(float(local_ts[offset])),
                    repr(float(correction_s)),
                    *[repr(float(value)) for value in row],
                ])
        self.metadata["sample_counts"][role] = start + int(values.shape[0])

    def update_quality(self, quality: Dict[str, Any]) -> None:
        self.metadata["quality"] = quality

    def _write_metadata(self) -> None:
        temporary = self.metadata_path.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(self.metadata, ensure_ascii=False, indent=2, allow_nan=False))
        temporary.replace(self.metadata_path)
        self._restrict_permissions(self.metadata_path)

    def close(self, status: str, quality: Dict[str, Any]) -> None:
        self.metadata["ended_utc"] = datetime.now(timezone.utc).isoformat()
        self.metadata["status"] = status
        self.update_quality(quality)
        self._write_metadata()


class SelectedLSLSession:
    """A selected, explicitly mapped LSL acquisition session.

    Timestamps returned to the estimator are corrected into this host's LSL
    clock domain. The recorder keeps original sender timestamps alongside the
    correction and corrected timestamps. Large gaps are retained in the raw
    files, flagged, and never interpolated across during dual-stream alignment.
    """

    def __init__(
        self,
        fnirs_selector: str,
        channel_mapping: Dict[str, Optional[int]],
        respiration_selector: Optional[str] = None,
        respiration_channel_index: Optional[int] = None,
        confirmed_derived_hb_channels: bool = False,
        confirmed_measured_respiration: bool = False,
        recording_root: Optional[Path] = None,
        recording_enabled: bool = False,
        timeout_s: float = 3.0,
    ) -> None:
        self.fnirs_selector = str(fnirs_selector)
        self.respiration_selector = (
            str(respiration_selector) if respiration_selector else None
        )
        self.channel_mapping = {
            key: int(value) if value is not None else None
            for key, value in channel_mapping.items()
        }
        self.respiration_channel_index = (
            int(respiration_channel_index)
            if respiration_channel_index is not None
            else None
        )
        self.confirmed_derived_hb_channels = bool(confirmed_derived_hb_channels)
        self.confirmed_measured_respiration = bool(confirmed_measured_respiration)
        self.timeout_s = float(timeout_s)
        self.recording_root = recording_root
        self.recording_enabled = bool(recording_enabled)
        self._fnirs_inlet = None
        self._respiration_inlet = None
        self._fnirs_info = None
        self._respiration_info = None
        self._fnirs_channels: List[Dict[str, Any]] = []
        self._respiration_channels: List[Dict[str, Any]] = []
        self._fnirs_rate = 0.0
        self._observed_fnirs_rate = 0.0
        self._respiration_rate = 0.0
        self._clock_offsets: Dict[str, float] = {}
        self._last_clock_refresh: Dict[str, float] = {}
        self._last_timestamps: Dict[str, Optional[float]] = {"fnirs": None, "respiration": None}
        self._pending_fnirs_values = np.empty((0, 2), dtype=np.float64)
        self._pending_fnirs_raw_timestamps = np.empty(0, dtype=np.float64)
        self._pending_fnirs_timestamps = np.empty(0, dtype=np.float64)
        self._pending_resp_values = np.empty((0, 1), dtype=np.float64)
        self._pending_resp_raw_timestamps = np.empty(0, dtype=np.float64)
        self._pending_resp_timestamps = np.empty(0, dtype=np.float64)
        self._recorder: Optional[LocalSessionRecorder] = None
        self.last_quality_warnings: List[str] = []
        self.stats: Dict[str, Any] = {
            "total_samples": {"fnirs": 0, "respiration": 0},
            "gap_count": {"fnirs": 0, "respiration": 0},
            "nonfinite_sample_count": {"fnirs": 0, "respiration": 0},
            "disconnect_count": 0,
            "clock_correction_s": {},
            "observed_sample_rate_hz": {"fnirs": None, "respiration": None},
            "last_sample_local_s": {"fnirs": None, "respiration": None},
        }
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._fnirs_inlet is not None

    @property
    def fnirs_sample_rate_hz(self) -> float:
        return self._fnirs_rate if self._fnirs_rate > 0 else self._observed_fnirs_rate

    @property
    def recording_path(self) -> Optional[str]:
        return self._recorder.path if self._recorder is not None else None

    @property
    def stream_metadata(self) -> Dict[str, Any]:
        streams: Dict[str, Any] = {}
        if self._fnirs_info is not None:
            streams["fnirs"] = {
                **_core_metadata(self._fnirs_info),
                "channels": self._fnirs_channels,
            }
        if self._respiration_info is not None:
            streams["respiration"] = {
                **_core_metadata(self._respiration_info),
                "channels": self._respiration_channels,
            }
        return streams

    def _resolve(self, selector: str) -> Any:
        if not _LSL_AVAILABLE or resolve_streams is None:
            raise RuntimeError(f"Live LSL acquisition unavailable: {lsl_unavailable_message()}")
        infos = resolve_streams(wait_time=max(0.5, self.timeout_s))
        matches = [info for info in infos if _stream_key(info) == selector]
        if len(matches) != 1:
            raise RuntimeError(
                f"Selected LSL identity resolved to {len(matches)} outlets; refresh and select one unique outlet."
            )
        return matches[0]

    def _open_inlet(self, selector: str) -> Tuple[Any, Any, List[Dict[str, Any]]]:
        if StreamInlet is None:
            raise RuntimeError(f"Live LSL acquisition unavailable: {lsl_unavailable_message()}")
        inlet = StreamInlet(self._resolve(selector), max_buflen=360)
        try:
            inlet.open_stream(timeout=self.timeout_s)
            info = inlet.info(timeout=self.timeout_s)
            channels = _channel_metadata(info)
            return inlet, info, channels
        except Exception:
            try:
                inlet.close_stream()
            except Exception:
                pass
            raise

    @staticmethod
    def _validate_index(index: Optional[int], channels: Sequence[Dict[str, Any]], role: str) -> None:
        if index is None or index < 0 or index >= len(channels):
            raise ValueError(f"Select an explicit in-range LSL channel for {role}.")

    def connect(self) -> None:
        """Open only the explicitly selected outlets and validate the channel map."""
        if not self.fnirs_selector:
            raise ValueError("Select an LSL fNIRS outlet before starting the session.")
        hbo = self.channel_mapping.get("HbO2")
        hbr = self.channel_mapping.get("HbR")
        if hbo is None or hbr is None:
            raise ValueError("Select explicit LSL channel indices for both HbO2 and HbR.")
        if hbo == hbr:
            raise ValueError("HbO2 and HbR must map to different channel indices.")
        if not self.confirmed_derived_hb_channels:
            raise ValueError("Confirm that the selected channels are derived HbO2/HbR, not raw optical intensity or optical density.")

        separate_respiration = bool(
            self.respiration_selector and self.respiration_selector != self.fnirs_selector
        )
        if separate_respiration and self.respiration_channel_index is None:
            raise ValueError("Select an explicit channel in the separate respiration outlet.")
        if not separate_respiration:
            resp = self.channel_mapping.get("Respiration")
            if resp is not None and resp in (hbo, hbr):
                raise ValueError("Respiration must use a different channel from HbO2 and HbR.")
            if resp is not None and not self.confirmed_measured_respiration:
                raise ValueError("Confirm that the selected respiration channel is measured, not a pacing reference.")
        elif not self.confirmed_measured_respiration:
            raise ValueError("Confirm that the selected respiration outlet is a measured signal, not a pacing reference.")

        opened: List[Any] = []
        try:
            fnirs_inlet, fnirs_info, fnirs_channels = self._open_inlet(self.fnirs_selector)
            opened.append(fnirs_inlet)
            fnirs_format = _value(fnirs_info, "channel_format", "")
            if not is_numeric_lsl_channel_format(fnirs_format):
                raise ValueError(
                    f"Selected fNIRS outlet has {normalize_lsl_channel_format(fnirs_format)!r} samples; "
                    "signal plots require a numeric LSL stream. Select a numeric outlet, not a marker stream."
                )
            self._validate_index(hbo, fnirs_channels, "HbO2")
            self._validate_index(hbr, fnirs_channels, "HbR")
            resp = self.channel_mapping.get("Respiration")
            if not separate_respiration and resp is not None:
                self._validate_index(resp, fnirs_channels, "respiration")

            resp_inlet = None
            resp_info = None
            resp_channels: List[Dict[str, Any]] = []
            if separate_respiration:
                resp_inlet, resp_info, resp_channels = self._open_inlet(self.respiration_selector or "")
                opened.append(resp_inlet)
                resp_format = _value(resp_info, "channel_format", "")
                if not is_numeric_lsl_channel_format(resp_format):
                    raise ValueError(
                        f"Selected respiration outlet has {normalize_lsl_channel_format(resp_format)!r} samples; "
                        "measured respiration plots require a numeric LSL stream. Select a numeric outlet, not a marker stream."
                    )
                self._validate_index(self.respiration_channel_index, resp_channels, "respiration")

            self._fnirs_inlet, self._fnirs_info, self._fnirs_channels = fnirs_inlet, fnirs_info, fnirs_channels
            self._fnirs_rate = float(_value(fnirs_info, "nominal_srate", 0.0) or 0.0)
            self._respiration_inlet, self._respiration_info = resp_inlet, resp_info
            self._respiration_channels = resp_channels
            self._respiration_rate = float(_value(resp_info, "nominal_srate", 0.0) or 0.0) if resp_info is not None else self._fnirs_rate
            self._refresh_clock_correction("fnirs", self._fnirs_inlet, force=True)
            if self._respiration_inlet is not None:
                self._refresh_clock_correction("respiration", self._respiration_inlet, force=True)

            mapping_record = {
                "HbO2": self._mapped_channel_record(fnirs_channels, hbo),
                "HbR": self._mapped_channel_record(fnirs_channels, hbr),
                "Respiration": (
                    self._mapped_channel_record(
                        resp_channels if separate_respiration else fnirs_channels,
                        self.respiration_channel_index if separate_respiration else resp,
                    )
                    if (self.respiration_channel_index is not None if separate_respiration else resp is not None)
                    else None
                ),
            }
            metadata = {
                "acquisition": "LSL",
                "streams": self.stream_metadata,
                "channel_mapping": mapping_record,
                "operator_confirmations": {
                    "hb_channels_are_derived_hbo2_hbr": self.confirmed_derived_hb_channels,
                    "respiration_is_measured_not_paced": self.confirmed_measured_respiration,
                },
                "clock_policy": "Each outlet's raw LSL timestamps are retained; time_correction is added to map them to the receiving host's LSL clock.",
                "gap_policy": "Raw samples are retained. Gaps larger than three median sample intervals are flagged and are not bridged during separate-stream interpolation.",
                "unit_policy": "Units are recorded from LSL metadata when present; no optical-to-haemoglobin conversion is performed.",
            }
            if self.recording_enabled:
                local_data_root = Path(
                    os.environ.get("LOCALAPPDATA")
                    or (Path.home() / ".local" / "share")
                ) / "NeuroResonance" / "recordings"
                self._recorder = LocalSessionRecorder(
                    self.recording_root or local_data_root,
                    metadata,
                )
            self._connected = True
        except Exception:
            for inlet in opened:
                try:
                    inlet.close_stream()
                except Exception:
                    pass
            self._fnirs_inlet = None
            self._respiration_inlet = None
            self._connected = False
            raise

    @staticmethod
    def _mapped_channel_record(channels: Sequence[Dict[str, Any]], index: Optional[int]) -> Optional[Dict[str, Any]]:
        if index is None:
            return None
        return dict(channels[index])

    def _refresh_clock_correction(self, role: str, inlet: Any, force: bool = False) -> float:
        now = time.monotonic()
        if force or role not in self._clock_offsets or now - self._last_clock_refresh.get(role, 0.0) >= 5.0:
            offset = float(inlet.time_correction(timeout=self.timeout_s))
            if not math.isfinite(offset):
                raise RuntimeError(f"LSL time correction for {role} was not finite.")
            self._clock_offsets[role] = offset
            self._last_clock_refresh[role] = now
            self.stats["clock_correction_s"][role] = offset
        return self._clock_offsets[role]

    def _quality_for_timestamps(self, role: str, timestamps: np.ndarray, rate: float) -> List[str]:
        warnings: List[str] = []
        valid = timestamps[np.isfinite(timestamps)]
        previous = self._last_timestamps.get(role)
        combined = np.concatenate(([previous], valid)) if previous is not None and valid.size else valid
        if combined.size > 1:
            diffs = np.diff(combined)
            if np.any(diffs <= 0):
                warnings.append(f"{role}: timestamp order is not strictly increasing.")
            positive = diffs[diffs > 0]
            if positive.size:
                typical = float(np.median(positive))
                if rate > 0 and math.isfinite(rate):
                    typical = min(typical, 1.0 / rate) if typical > 0 else 1.0 / rate
                gap_count = int(np.count_nonzero(diffs > 3.0 * max(typical, 1e-9)))
                if gap_count:
                    self.stats["gap_count"][role] += gap_count
                    warnings.append(f"{role}: {gap_count} acquisition gap(s) detected; affected intervals will not be interpolated.")
        if valid.size:
            self._last_timestamps[role] = float(valid[-1])
            self.stats["last_sample_local_s"][role] = float(valid[-1])
        return warnings

    def _read_raw(self, role: str, inlet: Any, max_samples: int, rate: float, channels: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, List[str]]:
        samples, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=max_samples)
        raw_ts = np.asarray(timestamps, dtype=np.float64).ravel()
        if not samples or raw_ts.size == 0:
            self._refresh_clock_correction(role, inlet)
            return np.empty((0, len(channels))), raw_ts, raw_ts, self._clock_offsets[role], []
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, len(channels))
        if values.shape[0] != raw_ts.size:
            values = values.reshape(raw_ts.size, -1)
        if values.shape[1] != len(channels):
            raise RuntimeError(f"{role} sample width does not match the selected outlet metadata.")
        correction = self._refresh_clock_correction(role, inlet)
        local_ts = raw_ts + correction

        if role == "fnirs" and local_ts.size > 1:
            intervals = np.diff(local_ts)
            positive = intervals[np.isfinite(intervals) & (intervals > 0)]
            if positive.size:
                observed_rate = 1.0 / float(np.median(positive))
                if math.isfinite(observed_rate) and observed_rate > 0:
                    if role == "fnirs":
                        self._observed_fnirs_rate = observed_rate
                    self.stats["observed_sample_rate_hz"][role] = observed_rate

        if self._recorder is not None:
            self._recorder.record(role, values, raw_ts, local_ts, correction, channels)
        self.stats["total_samples"][role] += int(raw_ts.size)
        finite_rows = np.isfinite(raw_ts) & np.all(np.isfinite(values), axis=1)
        bad_count = int(raw_ts.size - np.count_nonzero(finite_rows))
        if bad_count:
            self.stats["nonfinite_sample_count"][role] += bad_count
        warnings = self._quality_for_timestamps(role, local_ts, rate)
        if bad_count:
            warnings.append(f"{role}: {bad_count} non-finite sample(s) excluded from analysis; raw rows remain recorded.")
        return values[finite_rows], raw_ts[finite_rows], local_ts[finite_rows], correction, warnings

    def _empty_chunk(self, warnings: Optional[List[str]] = None) -> Dict[str, Any]:
        self.last_quality_warnings = list(dict.fromkeys(warnings or []))
        if self._recorder is not None:
            self._recorder.update_quality(self.stats)
        return {
            "HbO2": np.empty(0),
            "HbR": np.empty(0),
            "Respiration": np.empty(0),
            "timestamps": np.empty(0),
            "timestamps_raw": np.empty(0),
            "time_correction_s": self._clock_offsets.get("fnirs", float("nan")),
            "quality_warnings": self.last_quality_warnings,
        }

    def pull_chunk(self, max_samples: int = 100) -> Dict[str, Any]:
        """Read a chunk, retaining raw timestamps and using corrected time for analysis."""
        if not self.is_connected:
            return self._empty_chunk(["LSL session is disconnected."])
        try:
            fnirs, fnirs_raw, fnirs_local, fnirs_offset, warnings = self._read_raw(
                "fnirs", self._fnirs_inlet, max_samples, self._fnirs_rate, self._fnirs_channels
            )
            separate = self._respiration_inlet is not None
            if not separate:
                hbo_idx = int(self.channel_mapping["HbO2"])
                hbr_idx = int(self.channel_mapping["HbR"])
                resp_idx = self.channel_mapping.get("Respiration")
                resp = fnirs[:, int(resp_idx)] if resp_idx is not None else np.empty(0)
                result = {
                    "HbO2": fnirs[:, hbo_idx],
                    "HbR": fnirs[:, hbr_idx],
                    "Respiration": resp,
                    "timestamps": fnirs_local,
                    "timestamps_raw": fnirs_raw,
                    "time_correction_s": fnirs_offset,
                    "quality_warnings": warnings,
                }
                self.last_quality_warnings = list(dict.fromkeys(warnings))
                if self._recorder is not None:
                    self._recorder.update_quality(self.stats)
                return result if fnirs_local.size else self._empty_chunk(warnings)

            respiration, respiration_raw, respiration_local, _, resp_warnings = self._read_raw(
                "respiration",
                self._respiration_inlet,
                max_samples,
                self._respiration_rate,
                self._respiration_channels,
            )
            warnings.extend(resp_warnings)
            if fnirs_local.size:
                hbo_idx = int(self.channel_mapping["HbO2"])
                hbr_idx = int(self.channel_mapping["HbR"])
                fnirs_values = fnirs[:, [hbo_idx, hbr_idx]]
                self._pending_fnirs_values, self._pending_fnirs_raw_timestamps, self._pending_fnirs_timestamps = self._merge_pending(
                    self._pending_fnirs_values, self._pending_fnirs_raw_timestamps, self._pending_fnirs_timestamps,
                    fnirs_values, fnirs_raw, fnirs_local,
                )
            if respiration_local.size:
                resp_idx = int(self.respiration_channel_index)
                self._pending_resp_values, self._pending_resp_raw_timestamps, self._pending_resp_timestamps = self._merge_pending(
                    self._pending_resp_values, self._pending_resp_raw_timestamps, self._pending_resp_timestamps,
                    respiration[:, [resp_idx]], respiration_raw, respiration_local,
                )
            result = self._align_dual_pending(warnings, fnirs_offset)
            if self._recorder is not None:
                self._recorder.update_quality(self.stats)
            return result
        except Exception as exc:
            self.stats["disconnect_count"] += 1
            self._connected = False
            return self._empty_chunk([f"LSL acquisition stopped: {type(exc).__name__}: {exc}"])

    @staticmethod
    def _merge_pending(
        old_values: np.ndarray,
        old_raw: np.ndarray,
        old_local: np.ndarray,
        new_values: np.ndarray,
        new_raw: np.ndarray,
        new_local: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.concatenate((old_values, new_values), axis=0)
        raw = np.concatenate((old_raw, new_raw))
        local = np.concatenate((old_local, new_local))
        if local.size == 0:
            return values, raw, local
        order = np.argsort(local)
        values, raw, local = values[order], raw[order], local[order]
        keep = np.concatenate(([True], np.diff(local) > 1e-9))
        values, raw, local = values[keep], raw[keep], local[keep]
        if local.size > 1:
            positive = np.diff(local)
            typical = float(np.median(positive[positive > 0])) if np.any(positive > 0) else 0.0
            if typical > 0:
                gaps = np.flatnonzero(positive > 3.0 * typical)
                if gaps.size:
                    cut = int(gaps[-1] + 1)
                    values, raw, local = values[cut:], raw[cut:], local[cut:]
        if local.size:
            keep_after = local[-1] - 360.0
            recent = local >= keep_after
            values, raw, local = values[recent], raw[recent], local[recent]
        return values, raw, local

    def _align_dual_pending(self, warnings: List[str], fnirs_offset: float) -> Dict[str, Any]:
        f_ts = self._pending_fnirs_timestamps
        r_ts = self._pending_resp_timestamps
        if f_ts.size < 1 or r_ts.size < 2:
            return self._empty_chunk(warnings)
        start, end = max(float(f_ts[0]), float(r_ts[0])), min(float(f_ts[-1]), float(r_ts[-1]))
        if end <= start:
            return self._empty_chunk(warnings)
        selected = (f_ts >= start) & (f_ts <= end)
        out_ts = f_ts[selected]
        if out_ts.size == 0:
            return self._empty_chunk(warnings)
        right = np.searchsorted(r_ts, out_ts, side="left")
        exact = (right < r_ts.size) & (np.abs(r_ts[np.minimum(right, r_ts.size - 1)] - out_ts) < 1e-9)
        left = np.maximum(right - 1, 0)
        right = np.minimum(right, r_ts.size - 1)
        if r_ts.size > 1:
            typical_resp_dt = float(np.median(np.diff(r_ts)))
        else:
            typical_resp_dt = 0.0
        interval = r_ts[right] - r_ts[left]
        valid = exact | ((right > left) & (interval > 0) & (interval <= 3.0 * max(typical_resp_dt, 1e-9)))
        if not np.any(valid):
            return self._empty_chunk(warnings + ["Separate streams have no safe timestamp overlap; no samples were interpolated."])
        out_ts = out_ts[valid]
        out_raw = self._pending_fnirs_raw_timestamps[selected][valid]
        f_values = self._pending_fnirs_values[selected][valid]
        left = left[valid]
        right = right[valid]
        resp_values = np.empty(out_ts.size, dtype=np.float64)
        same = left == right
        resp_values[same] = self._pending_resp_values[left[same], 0]
        interp = ~same
        if np.any(interp):
            frac = (out_ts[interp] - r_ts[left[interp]]) / (r_ts[right[interp]] - r_ts[left[interp]])
            resp_values[interp] = (
                self._pending_resp_values[left[interp], 0] * (1.0 - frac)
                + self._pending_resp_values[right[interp], 0] * frac
            )
        last_emitted = float(out_ts[-1])
        f_keep = self._pending_fnirs_timestamps > last_emitted
        self._pending_fnirs_values = self._pending_fnirs_values[f_keep]
        self._pending_fnirs_raw_timestamps = self._pending_fnirs_raw_timestamps[f_keep]
        self._pending_fnirs_timestamps = self._pending_fnirs_timestamps[f_keep]
        anchor = max(0, int(np.searchsorted(r_ts, last_emitted, side="right")) - 1)
        self._pending_resp_values = self._pending_resp_values[anchor:]
        self._pending_resp_raw_timestamps = self._pending_resp_raw_timestamps[anchor:]
        self._pending_resp_timestamps = self._pending_resp_timestamps[anchor:]
        self.stats["last_sample_local_s"]["fnirs"] = float(out_ts[-1])
        self.last_quality_warnings = list(dict.fromkeys(warnings))
        return {
            "HbO2": f_values[:, 0],
            "HbR": f_values[:, 1],
            "Respiration": resp_values,
            "timestamps": out_ts,
            "timestamps_raw": out_raw,
            "time_correction_s": fnirs_offset,
            "quality_warnings": self.last_quality_warnings,
        }

    def close_recording(self, status: str = "completed") -> None:
        if self._recorder is not None:
            self._recorder.close(status, self.stats)
            self._recorder = None

    def disconnect(self) -> None:
        self._connected = False
        for inlet in (self._fnirs_inlet, self._respiration_inlet):
            if inlet is not None:
                try:
                    inlet.close_stream()
                except Exception:
                    pass
        self._fnirs_inlet = None
        self._respiration_inlet = None
        self._pending_fnirs_values = np.empty((0, 2), dtype=np.float64)
        self._pending_fnirs_raw_timestamps = np.empty(0, dtype=np.float64)
        self._pending_fnirs_timestamps = np.empty(0, dtype=np.float64)
        self._pending_resp_values = np.empty((0, 1), dtype=np.float64)
        self._pending_resp_raw_timestamps = np.empty(0, dtype=np.float64)
        self._pending_resp_timestamps = np.empty(0, dtype=np.float64)
