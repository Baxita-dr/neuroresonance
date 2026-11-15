"""Loading and timestamp-paced replay for LabRecorder XDF files."""

from __future__ import annotations

import io
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.dashboard_metrics import estimate_sampling_rate_hz


_NUMERIC_XDF_FORMATS = {"float32", "double64", "int8", "int16", "int32", "int64"}
MAX_XDF_FILE_BYTES = 100 * 1024 * 1024
MAX_XDF_STREAMS = 256
MAX_XDF_SAMPLES_PER_STREAM = 1_000_000
MAX_XDF_CHANNELS_PER_STREAM = 64
MAX_XDF_TOTAL_SAMPLES = 5_000_000
MAX_XDF_TOTAL_SAMPLE_VALUES = 10_000_000


def _first(value: Any, default: Any = None) -> Any:
    while isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    return default if value is None else value


def _text(value: Any, default: str = "") -> str:
    value = _first(value, default)
    return str(value) if value is not None else default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(_first(value, default))
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _channel_metadata(info: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
    desc = _first(info.get("desc"), {})
    channels_node = _first(desc.get("channels", {}), {}) if isinstance(desc, dict) else {}
    channel_nodes = channels_node.get("channel", []) if isinstance(channels_node, dict) else []
    if isinstance(channel_nodes, dict):
        channel_nodes = [channel_nodes]
    result: List[Dict[str, Any]] = []
    for index in range(count):
        node = channel_nodes[index] if index < len(channel_nodes) and isinstance(channel_nodes[index], dict) else {}
        result.append({
            "index": index,
            "label": _text(node.get("label"), ""),
            "unit": _text(node.get("unit"), ""),
            "type": _text(node.get("type"), ""),
        })
    return result


def load_xdf_recording(content: bytes) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse XDF bytes with pyxdf and return normalized stream records."""
    if not content:
        raise ValueError("The selected XDF file is empty.")
    if len(content) > MAX_XDF_FILE_BYTES:
        raise ValueError(f"XDF files are limited to {MAX_XDF_FILE_BYTES // (1024 * 1024)} MiB.")
    try:
        import pyxdf
    except ImportError as exc:
        raise RuntimeError("XDF import requires pyxdf. Install project dependencies with `python -m pip install -r requirements.txt`.") from exc

    try:
        raw_streams, header = pyxdf.load_xdf(
            io.BytesIO(content),
            synchronize_clocks=True,
            dejitter_timestamps=True,
        )
    except Exception as exc:
        raise ValueError(f"Could not read this XDF file: {type(exc).__name__}: {exc}") from exc

    streams: List[Dict[str, Any]] = []
    if len(raw_streams) > MAX_XDF_STREAMS:
        raise ValueError(f"XDF contains {len(raw_streams)} streams; the limit is {MAX_XDF_STREAMS}.")
    total_sample_values = 0
    total_sample_rows = 0
    for index, raw in enumerate(raw_streams):
        info = raw.get("info", {})
        values = np.asarray(raw.get("time_series", []))
        if values.ndim > 1 and values.shape[1] > MAX_XDF_CHANNELS_PER_STREAM:
            raise ValueError(f"XDF stream {index} exceeds the {MAX_XDF_CHANNELS_PER_STREAM}-channel limit.")
        if values.ndim > 0 and values.shape[0] > MAX_XDF_SAMPLES_PER_STREAM:
            raise ValueError(f"XDF stream {index} exceeds the {MAX_XDF_SAMPLES_PER_STREAM}-sample limit.")
        total_sample_rows += int(values.shape[0]) if values.ndim > 0 else 0
        if total_sample_rows > MAX_XDF_TOTAL_SAMPLES:
            raise ValueError(f"XDF exceeds the total {MAX_XDF_TOTAL_SAMPLES}-sample limit.")
        total_sample_values += int(values.size)
        if total_sample_values > MAX_XDF_TOTAL_SAMPLE_VALUES:
            raise ValueError(f"XDF exceeds the total {MAX_XDF_TOTAL_SAMPLE_VALUES}-value limit.")
        timestamps = np.asarray(raw.get("time_stamps", []), dtype=np.float64).ravel()
        channel_format = _text(info.get("channel_format"), "").strip().casefold()
        numeric_format = channel_format in _NUMERIC_XDF_FORMATS
        if not channel_format:
            numeric_format = values.dtype.kind in "biuf"
        numeric = False
        if numeric_format:
            try:
                values = np.asarray(values, dtype=np.float64)
                numeric = True
            except (TypeError, ValueError):
                numeric = False
        if values.ndim == 1 and numeric:
            values = values.reshape(-1, 1)
        channel_count = int(_number(info.get("channel_count", values.shape[1] if values.ndim > 1 else 0)))
        if values.ndim > 1:
            channel_count = max(channel_count, int(values.shape[1]))
        sample_count = int(values.shape[0]) if values.ndim > 0 else 0
        if timestamps.size != sample_count:
            raise ValueError(
                f"XDF stream {index} has {sample_count} sample rows but {timestamps.size} timestamps; refusing to misalign them."
            )
        finite_times = timestamps[np.isfinite(timestamps)]
        duration = float(np.max(finite_times) - np.min(finite_times)) if finite_times.size > 1 else 0.0
        streams.append({
            "index": index,
            "name": _text(info.get("name"), f"Stream {index}"),
            "type": _text(info.get("type"), ""),
            "source_id": _text(info.get("source_id"), ""),
            "channel_format": _text(info.get("channel_format"), ""),
            "channel_count": channel_count,
            "nominal_srate_hz": _number(info.get("nominal_srate"), 0.0),
            "sample_count": sample_count,
            "duration_s": max(0.0, duration),
            "channels": _channel_metadata(info, channel_count),
            "is_numeric": bool(
                numeric
                and values.ndim == 2
                and values.shape[1] >= channel_count
                and channel_count > 0
            ),
            "time_series": values,
            "timestamps": timestamps,
        })
    return streams, header if isinstance(header, dict) else {}


def _numeric_stream_arrays(stream: Dict[str, Any], role: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not stream.get("is_numeric"):
        raise ValueError(f"The selected {role} stream is not numeric and cannot be plotted as a signal.")
    values = np.asarray(stream["time_series"], dtype=np.float64)
    timestamps = np.asarray(stream["timestamps"], dtype=np.float64).ravel()
    if values.ndim != 2 or len(values) != timestamps.size:
        raise ValueError(f"The selected {role} stream has incompatible sample and timestamp arrays.")
    warnings: List[str] = []
    finite = np.isfinite(timestamps)
    if not np.all(finite):
        warnings.append(f"{role}: excluded {int(np.count_nonzero(~finite))} sample(s) with invalid XDF timestamps.")
        values, timestamps = values[finite], timestamps[finite]
    if timestamps.size < 2:
        raise ValueError(f"The selected {role} stream needs at least two timestamped samples.")
    order = np.argsort(timestamps, kind="stable")
    if not np.array_equal(order, np.arange(order.size)):
        warnings.append(f"{role}: sorted samples by their XDF timestamps before playback.")
        values, timestamps = values[order], timestamps[order]
    keep = np.concatenate(([True], np.diff(timestamps) > 0.0))
    if not np.all(keep):
        warnings.append(f"{role}: excluded {int(np.count_nonzero(~keep))} duplicate/non-increasing timestamp sample(s).")
        values, timestamps = values[keep], timestamps[keep]
    if timestamps.size < 2:
        raise ValueError(f"The selected {role} stream has fewer than two unique timestamps.")
    intervals = np.diff(timestamps)
    typical_interval = float(np.median(intervals))
    gap_count = int(np.count_nonzero(intervals > 3.0 * typical_interval)) if typical_interval > 0 else 0
    if gap_count:
        warnings.append(
            f"{role}: {gap_count} timestamp gap(s) exceed three median sample intervals; playback preserves the gaps."
        )
    return values, timestamps, warnings


def _align_respiration(
    target_timestamps: np.ndarray,
    source_timestamps: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    """Interpolate only short, locally bracketed intervals; leave gaps as NaN."""
    output = np.full(target_timestamps.size, np.nan, dtype=np.float64)
    valid = np.isfinite(source_timestamps) & np.isfinite(source_values)
    source_timestamps = source_timestamps[valid]
    source_values = source_values[valid]
    if source_timestamps.size < 2:
        return output
    order = np.argsort(source_timestamps, kind="stable")
    source_timestamps, source_values = source_timestamps[order], source_values[order]
    keep = np.concatenate(([True], np.diff(source_timestamps) > 0.0))
    source_timestamps, source_values = source_timestamps[keep], source_values[keep]
    intervals = np.diff(source_timestamps)
    if intervals.size == 0:
        return output
    typical = float(np.median(intervals[intervals > 0]))
    if not math.isfinite(typical) or typical <= 0:
        return output
    right = np.searchsorted(source_timestamps, target_timestamps, side="left")
    exact = (right < source_timestamps.size)
    exact[exact] &= np.isclose(source_timestamps[right[exact]], target_timestamps[exact], rtol=0.0, atol=1e-9)
    if np.any(exact):
        output[exact] = source_values[right[exact]]
    between = (~exact) & (right > 0) & (right < source_timestamps.size)
    indices = np.flatnonzero(between)
    if indices.size:
        left_idx = right[indices] - 1
        right_idx = right[indices]
        gaps = source_timestamps[right_idx] - source_timestamps[left_idx]
        safe = (gaps > 0) & (gaps <= 3.0 * typical)
        indices, left_idx, right_idx, gaps = indices[safe], left_idx[safe], right_idx[safe], gaps[safe]
        fraction = (target_timestamps[indices] - source_timestamps[left_idx]) / gaps
        output[indices] = source_values[left_idx] + fraction * (source_values[right_idx] - source_values[left_idx])
    return output


class XDFPlaybackSource:
    """Replay mapped numeric XDF samples against a monotonic playback clock."""

    def __init__(
        self,
        streams: Sequence[Dict[str, Any]],
        *,
        fnirs_stream_index: int,
        hbo2_channel_index: int,
        hbr_channel_index: int,
        respiration_stream_index: Optional[int] = None,
        respiration_channel_index: Optional[int] = None,
        speed: float = 1.0,
    ) -> None:
        if not math.isfinite(float(speed)) or float(speed) <= 0:
            raise ValueError("Playback speed must be a positive finite number.")
        try:
            fnirs_stream = streams[int(fnirs_stream_index)]
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("Select an available numeric XDF stream.") from exc
        fnirs_values, fnirs_timestamps, warnings = _numeric_stream_arrays(fnirs_stream, "fNIRS")
        channel_count = fnirs_values.shape[1]
        hbo_index, hbr_index = int(hbo2_channel_index), int(hbr_channel_index)
        if hbo_index < 0 or hbr_index < 0 or hbo_index >= channel_count or hbr_index >= channel_count:
            raise ValueError("Choose valid HbO₂ and HbR channel indices from the selected XDF stream.")
        if hbo_index == hbr_index:
            raise ValueError("HbO₂ and HbR must use different XDF channel indices.")

        origin = float(fnirs_timestamps[0])
        self.timestamps = fnirs_timestamps - origin
        self.hbo2 = fnirs_values[:, hbo_index]
        self.hbr = fnirs_values[:, hbr_index]
        self.respiration = np.full(self.timestamps.size, np.nan, dtype=np.float64)
        if respiration_stream_index is not None:
            try:
                resp_stream = streams[int(respiration_stream_index)]
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError("Select an available numeric XDF respiration stream.") from exc
            if respiration_channel_index is None:
                raise ValueError("Choose a channel index for the measured respiration stream.")
            if int(respiration_stream_index) == int(fnirs_stream_index):
                if int(respiration_channel_index) < 0 or int(respiration_channel_index) >= channel_count:
                    raise ValueError("Choose a valid measured respiration channel index.")
                self.respiration = fnirs_values[:, int(respiration_channel_index)]
            else:
                resp_values, resp_timestamps, resp_warnings = _numeric_stream_arrays(resp_stream, "respiration")
                resp_index = int(respiration_channel_index)
                if resp_index < 0 or resp_index >= resp_values.shape[1]:
                    raise ValueError("Choose a valid measured respiration channel index.")
                warnings.extend(resp_warnings)
                self.respiration = _align_respiration(
                    fnirs_timestamps,
                    resp_timestamps,
                    resp_values[:, resp_index],
                )
        self.duration_s = float(self.timestamps[-1])
        self.sample_rate_hz = estimate_sampling_rate_hz(self.timestamps) or 0.0
        self.quality_warnings = warnings
        self.speed = float(speed)
        self._started_monotonic: Optional[float] = None
        self._cursor = 0

    def start(self, now_monotonic: Optional[float] = None) -> None:
        self._cursor = 0
        self._started_monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)

    @property
    def position_s(self) -> float:
        if self._started_monotonic is None:
            return 0.0
        return min(self.duration_s, max(0.0, (time.monotonic() - self._started_monotonic) * self.speed))

    def position_at(self, now_monotonic: float) -> float:
        if self._started_monotonic is None:
            return 0.0
        return min(self.duration_s, max(0.0, (float(now_monotonic) - self._started_monotonic) * self.speed))

    @property
    def is_finished(self) -> bool:
        return self._started_monotonic is not None and self._cursor >= self.timestamps.size and self.position_s >= self.duration_s

    def get_chunk(self, max_samples: int = 6000, *, now_monotonic: Optional[float] = None) -> Dict[str, Any]:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive.")
        if self._started_monotonic is None:
            return self._empty_chunk()
        position = self.position_s if now_monotonic is None else self.position_at(now_monotonic)
        # Floating-point subtraction of monotonic clock values can place a
        # sample boundary a few ulps ahead of the computed playback position.
        due = int(np.searchsorted(self.timestamps, position + max(1e-9, abs(position) * 1e-12), side="right"))
        end = min(due, self._cursor + int(max_samples))
        start = self._cursor
        self._cursor = end
        return {
            "timestamps": self.timestamps[start:end].copy(),
            "HbO2": self.hbo2[start:end].copy(),
            "HbR": self.hbr[start:end].copy(),
            "Respiration": self.respiration[start:end].copy(),
            "quality_warnings": list(self.quality_warnings),
        }

    @staticmethod
    def _empty_chunk() -> Dict[str, Any]:
        empty = np.empty(0, dtype=np.float64)
        return {"timestamps": empty, "HbO2": empty, "HbR": empty, "Respiration": empty, "quality_warnings": []}

    def stop(self) -> None:
        self._started_monotonic = None
        self._cursor = 0
