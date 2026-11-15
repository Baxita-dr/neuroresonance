import threading
import time

import numpy as np
import pytest

from src.infrastructure.lsl_receiver import LSLBackgroundReceiver


class FakeSession:
    def __init__(self, chunk=None, error=None):
        self.is_connected = True
        self.chunk = chunk or {
            "timestamps": np.array([1.0, 1.1, 1.2]),
            "timestamps_raw": np.array([0.9, 1.0, 1.1]),
            "HbO2": np.array([10.0, 11.0, 12.0]),
            "HbR": np.array([-1.0, -2.0, -3.0]),
            "Respiration": np.array([0.2, 0.3, 0.4]),
            "time_correction_s": 0.1,
            "quality_warnings": [],
        }
        self.error = error
        self.first_pull = threading.Event()
        self._sent = False

    def pull_chunk(self, max_samples):
        self.first_pull.set()
        if self.error is not None:
            raise self.error
        if self._sent:
            return {**self.chunk, "timestamps": np.array([]), "timestamps_raw": np.array([]),
                    "HbO2": np.array([]), "HbR": np.array([]), "Respiration": np.array([])}
        self._sent = True
        return self.chunk


def _read_until(receiver, expected_min, timeout_s=1.0):
    deadline = time.monotonic() + timeout_s
    chunk = receiver.get_chunk()
    while chunk["timestamps"].size < expected_min and time.monotonic() < deadline:
        time.sleep(0.002)
        chunk = receiver.get_chunk()
    return chunk


def test_receiver_delivers_timestamped_samples_from_background_thread():
    session = FakeSession()
    receiver = LSLBackgroundReceiver(session, poll_interval_s=0.002)
    receiver.start()
    try:
        assert session.first_pull.wait(1.0)
        chunk = _read_until(receiver, expected_min=3)
        assert np.allclose(chunk["timestamps"], [1.0, 1.1, 1.2])
        assert np.allclose(chunk["timestamps_raw"], [0.9, 1.0, 1.1])
        assert np.allclose(chunk["HbO2"], [10.0, 11.0, 12.0])
        assert np.allclose(chunk["HbR"], [-1.0, -2.0, -3.0])
        assert np.allclose(chunk["Respiration"], [0.2, 0.3, 0.4])
        assert chunk["time_correction_s"] == pytest.approx(0.1)
    finally:
        assert receiver.stop()


def test_receiver_reports_overflow_and_keeps_newest_samples():
    session = FakeSession()
    receiver = LSLBackgroundReceiver(session, queue_capacity=2, poll_interval_s=0.002)
    receiver.start()
    try:
        assert session.first_pull.wait(1.0)
        chunk = _read_until(receiver, expected_min=2)
        assert np.allclose(chunk["timestamps"], [1.1, 1.2])
        assert chunk["dropped_samples"] == 1
        assert any("queue overflow" in warning for warning in chunk["quality_warnings"])
    finally:
        assert receiver.stop()


def test_receiver_surfaces_worker_errors_to_the_dashboard():
    session = FakeSession(error=OSError("device inlet closed"))
    receiver = LSLBackgroundReceiver(session, poll_interval_s=0.002)
    receiver.start()
    try:
        assert session.first_pull.wait(1.0)
        deadline = time.monotonic() + 1.0
        while receiver.error is None and time.monotonic() < deadline:
            time.sleep(0.002)
        chunk = receiver.get_chunk()
        assert "device inlet closed" in chunk["receiver_error"]
        assert any("device inlet closed" in warning for warning in chunk["quality_warnings"])
    finally:
        assert receiver.stop()


def test_receiver_refuses_to_start_disconnected_session():
    session = FakeSession()
    session.is_connected = False
    with pytest.raises(RuntimeError, match="disconnected"):
        LSLBackgroundReceiver(session).start()
