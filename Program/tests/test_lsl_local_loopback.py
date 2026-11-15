"""Local pylsl outlet-to-dashboard receiver integration, not a hardware test."""

import threading
import time
import uuid

import numpy as np
import pytest

from src.infrastructure.lsl_receiver import LSLBackgroundReceiver
from src.lsl_integration import SelectedLSLSession, discover_lsl_streams
from src.lsl_runtime import LSL_AVAILABLE, StreamInfo, StreamOutlet, local_clock


pytestmark = pytest.mark.skipif(not LSL_AVAILABLE, reason="pylsl/liblsl runtime is unavailable")


def test_local_pylsl_outlet_reaches_background_receiver_with_mapped_channels(tmp_path):
    stream_name = f"NeuroResonance local loopback {uuid.uuid4().hex}"
    info = StreamInfo(
        stream_name,
        "NeuroResonanceLoopbackTest",
        3,
        100.0,
        "float32",
        uuid.uuid4().hex,
    )
    channels = info.desc().append_child("channels")
    for label in ("HbO2", "HbR", "Respiration"):
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("unit", "synthetic-test-unit")
        channel.append_child_value("type", "local loopback test")
    outlet = StreamOutlet(info)
    session = None
    receiver = None
    sender = None
    sender_errors = []
    try:
        streams = discover_lsl_streams(timeout_s=2.0)
        selected = next(item for item in streams if item["name"] == stream_name)
        session = SelectedLSLSession(
            fnirs_selector=selected["selector"],
            channel_mapping={"HbO2": 0, "HbR": 1, "Respiration": 2},
            confirmed_derived_hb_channels=True,
            confirmed_measured_respiration=True,
            recording_root=tmp_path,
            timeout_s=3.0,
        )
        session.connect()
        receiver = LSLBackgroundReceiver(session, poll_interval_s=0.005)
        receiver.start()

        sender_done = threading.Event()

        def send_samples():
            try:
                for index in range(40):
                    outlet.push_sample(
                        [float(index), float(-index), float(index) / 100.0],
                        timestamp=local_clock(),
                    )
                    time.sleep(0.005)
            except BaseException as exc:  # propagate worker failures to the test thread
                sender_errors.append(exc)
            finally:
                sender_done.set()

        sender = threading.Thread(target=send_samples, daemon=True)
        sender.start()
        collected = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and len(collected) < 20:
            chunk = receiver.get_chunk(max_samples=100)
            collected.extend(zip(chunk["HbO2"], chunk["HbR"], chunk["Respiration"], chunk["timestamps"]))
            if sender_done.is_set() and len(collected) >= 20:
                break
            time.sleep(0.005)

        assert len(collected) >= 20
        sender.join(timeout=2.0)
        assert not sender.is_alive()
        assert not sender_errors
        values = np.asarray(collected, dtype=np.float64)
        assert np.all(values[:, 0] == -values[:, 1])
        assert np.allclose(values[:, 2], values[:, 0] / 100.0)
        assert np.all(np.diff(values[:, 3]) > 0)
    finally:
        if sender is not None and sender.is_alive():
            sender.join(timeout=2.0)
        if receiver is not None:
            assert receiver.stop(timeout_s=3.0)
        if session is not None:
            session.close_recording("completed")
            session.disconnect()
        del outlet


def test_local_int16_signal_stream_is_read_and_string_markers_are_classified(tmp_path):
    signal_name = f"NeuroResonance int16 loopback {uuid.uuid4().hex}"
    marker_name = f"NeuroResonance marker loopback {uuid.uuid4().hex}"
    signal_info = StreamInfo(signal_name, "SignalLoopbackTest", 3, 50.0, "int16", uuid.uuid4().hex)
    marker_info = StreamInfo(marker_name, "Markers", 1, 0.0, "string", uuid.uuid4().hex)
    signal_outlet = StreamOutlet(signal_info)
    marker_outlet = StreamOutlet(marker_info)
    session = None
    try:
        streams = discover_lsl_streams(timeout_s=2.0)
        numeric = next(item for item in streams if item["name"] == signal_name)
        marker = next(item for item in streams if item["name"] == marker_name)
        assert numeric["channel_format"] == "int16"
        assert numeric["numeric_samples"] is True
        assert marker["channel_format"] == "string"
        assert marker["numeric_samples"] is False

        session = SelectedLSLSession(
            fnirs_selector=numeric["selector"],
            channel_mapping={"HbO2": 0, "HbR": 1, "Respiration": 2},
            confirmed_derived_hb_channels=True,
            confirmed_measured_respiration=True,
            recording_root=tmp_path,
            timeout_s=3.0,
        )
        session.connect()
        for index in range(20):
            signal_outlet.push_sample([index, -index, index * 2], timestamp=local_clock())
        time.sleep(0.15)
        chunk = session.pull_chunk(max_samples=100)

        assert chunk["timestamps"].size >= 10
        np.testing.assert_array_equal(chunk["HbO2"], -chunk["HbR"])
        np.testing.assert_array_equal(chunk["Respiration"], chunk["HbO2"] * 2)
    finally:
        if session is not None:
            session.close_recording("completed")
            session.disconnect()
        del marker_outlet
        del signal_outlet
