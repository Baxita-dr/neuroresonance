from pathlib import Path

import numpy as np
import pytest

from src.xdf_playback import XDFPlaybackSource, _align_respiration, load_xdf_recording


def _stream(index, timestamps, values, *, name="test"):
    values = np.asarray(values, dtype=float)
    return {
        "index": index,
        "name": name,
        "is_numeric": True,
        "time_series": values,
        "timestamps": np.asarray(timestamps, dtype=float),
        "channels": [{"index": i, "label": f"ch{i}", "unit": "", "type": ""} for i in range(values.shape[1])],
    }


def test_minimal_xdf_sample_loads_numeric_signal_and_marker_streams():
    pytest.importorskip("pyxdf")
    sample = Path(__file__).resolve().parents[1] / "examples" / "minimal.xdf"

    streams, _header = load_xdf_recording(sample.read_bytes())

    assert len(streams) == 2
    assert streams[0]["is_numeric"]
    assert streams[0]["sample_count"] == 9
    assert streams[0]["channel_count"] == 3
    assert streams[1]["is_numeric"] is False
    assert streams[1]["sample_count"] == 9


def test_official_clock_reset_file_loads_with_ordered_playback_and_gap_warning():
    fixture = Path(__file__).resolve().parent / "fixtures" / "xdf" / "clock_resets.xdf"

    streams, _header = load_xdf_recording(fixture.read_bytes())
    eeg = next(item for item in streams if item["name"] == "BioSemi")
    playback = XDFPlaybackSource(
        streams,
        fnirs_stream_index=eeg["index"],
        hbo2_channel_index=0,
        hbr_channel_index=1,
    )

    assert eeg["sample_count"] > 20_000
    assert playback.duration_s > 500.0
    assert np.all(np.diff(playback.timestamps) > 0.0)
    assert any("timestamp gap(s)" in warning for warning in playback.quality_warnings)
    playback.start(now_monotonic=50.0)
    first = playback.get_chunk(now_monotonic=50.0, max_samples=10)
    assert first["timestamps"].size == 1
    assert first["HbO2"].size == 1


def test_official_empty_and_string_marker_streams_are_never_signal_streams():
    fixture = Path(__file__).resolve().parent / "fixtures" / "xdf" / "empty_streams.xdf"

    streams, _header = load_xdf_recording(fixture.read_bytes())

    by_name = {item["name"]: item for item in streams}
    assert by_name["Empty data stream: test stream 0 counter"]["is_numeric"]
    assert by_name["Empty data stream: test stream 0 counter"]["sample_count"] == 0
    assert by_name["Empty marker stream: test stream 0 counter"]["is_numeric"] is False
    assert by_name["ctrl"]["is_numeric"] is False


def test_official_two_channel_string_marker_file_keeps_marker_separate():
    fixture = Path(__file__).resolve().parent / "fixtures" / "xdf" / "twochannel_string_marker.xdf"

    streams, _header = load_xdf_recording(fixture.read_bytes())

    by_name = {item["name"]: item for item in streams}
    assert by_name["SendDataC"]["channel_format"] == "double64"
    assert by_name["SendDataC"]["is_numeric"]
    assert by_name["SendMarker"]["channel_format"] == "string"
    assert by_name["SendMarker"]["is_numeric"] is False


def test_playback_returns_only_due_samples_and_preserves_mapped_channels():
    stream = _stream(
        0,
        [5.0, 5.1, 5.2, 5.3],
        [[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400]],
    )
    playback = XDFPlaybackSource(
        [stream],
        fnirs_stream_index=0,
        hbo2_channel_index=0,
        hbr_channel_index=1,
        respiration_stream_index=0,
        respiration_channel_index=2,
        speed=2.0,
    )
    playback.start(now_monotonic=100.0)

    first = playback.get_chunk(now_monotonic=100.0)
    second = playback.get_chunk(now_monotonic=100.1)
    third = playback.get_chunk(now_monotonic=100.1)

    np.testing.assert_allclose(first["timestamps"], [0.0])
    np.testing.assert_allclose(first["HbO2"], [1.0])
    np.testing.assert_allclose(first["HbR"], [10.0])
    np.testing.assert_allclose(first["Respiration"], [100.0])
    np.testing.assert_allclose(second["timestamps"], [0.1, 0.2])
    np.testing.assert_allclose(second["HbO2"], [2.0, 3.0])
    assert third["timestamps"].size == 0


def test_separate_respiration_aligns_short_gaps_and_keeps_long_gaps_missing():
    aligned = _align_respiration(
        np.array([0.05, 2.0, 10.0]),
        np.array([0.0, 0.1, 0.2, 0.3, 10.0]),
        np.array([0.0, 1.0, 2.0, 3.0, 10.0]),
    )

    assert aligned[0] == pytest.approx(0.5)
    assert np.isnan(aligned[1])
    assert aligned[2] == pytest.approx(10.0)


def test_playback_rejects_duplicate_hbo2_and_hbr_mapping():
    stream = _stream(0, [0.0, 0.1], [[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="different XDF channel indices"):
        XDFPlaybackSource(
            [stream],
            fnirs_stream_index=0,
            hbo2_channel_index=1,
            hbr_channel_index=1,
        )
