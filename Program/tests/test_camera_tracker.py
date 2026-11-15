"""
tests/test_camera_tracker.py — Unit Tests for Camera Motion Tracker & ROI Annotations
"""

import numpy as np
import pytest
import time
from src.camera_tracker import CameraTracker, RespiratorySignal


def test_camera_tracker_initialization():
    """Verify default initialisation parameters and buffer allocations."""
    tracker = CameraTracker(fps_target=20, buffer_seconds=5)
    assert tracker._fps_target == 20
    assert tracker._roi_y_frac == (0.40, 0.85)
    assert tracker._roi_x_frac == (0.25, 0.75)
    assert not tracker.is_running


def test_camera_tracker_annotated_frame_no_crash():
    """Verify get_frame_annotated correctly computes ROI coordinates without NameError."""
    tracker = CameraTracker(fps_target=20, buffer_seconds=5)
    # Inject a synthetic frame into internal state
    test_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    with tracker._lock:
        tracker._current_frame = test_frame
        tracker._raw_buffer.append(100.0)

    annotated = tracker.get_frame_annotated()
    assert annotated is not None
    assert annotated.shape == (480, 640, 3)
    # The chest ROI box draws green (0, 255, 136) at the rectangle boundary
    # Check that green pixels exist in the annotated frame
    has_green = np.any((annotated[:, :, 0] == 0) & (annotated[:, :, 1] == 255) & (annotated[:, :, 2] == 136))
    assert has_green


def test_camera_tracker_hides_stale_preview_frame():
    """An old frame is not returned as if the camera were still live."""
    tracker = CameraTracker(fps_target=20, buffer_seconds=5)
    with tracker._lock:
        tracker._current_frame = np.full((120, 160, 3), 128, dtype=np.uint8)
        tracker._last_frame_monotonic = time.monotonic() - 5.0

    assert tracker.get_frame_annotated(max_age_s=2.0) is None


def test_camera_tracker_reopens_stalled_capture(monkeypatch):
    """A failed capture handle is replaced so the background loop can recover."""
    class MockCapture:
        def __init__(self, opened):
            self.opened = opened
            self.released = False
            self.properties = {}

        def isOpened(self):
            return self.opened

        def set(self, prop, value):
            self.properties[prop] = value
            return True

        def get(self, prop):
            return 20.0

        def release(self):
            self.released = True

    old_capture = MockCapture(True)
    new_capture = MockCapture(True)
    tracker = CameraTracker(fps_target=20, buffer_seconds=5)
    tracker._cap = old_capture
    tracker._capture_target = 0
    monkeypatch.setattr("src.camera_tracker.cv2.VideoCapture", lambda *_: new_capture)

    assert tracker._reopen_capture() is True
    assert old_capture.released is True
    assert tracker._cap is new_capture


def test_camera_tracker_get_signal_empty_and_populated():
    """Verify get_signal returns valid RespiratorySignal dataclass."""
    tracker = CameraTracker(fps_target=20, buffer_seconds=5)
    sig_empty = tracker.get_signal()
    assert isinstance(sig_empty, RespiratorySignal)
    assert len(sig_empty.raw) == 0

    with tracker._lock:
        tracker._raw_buffer.extend([1.0, 2.0, 3.0])
        tracker._ts_buffer.extend([0.1, 0.2, 0.3])
        tracker._filtered_buffer.extend([0.05, 0.10, 0.15])

    sig = tracker.get_signal(2)
    assert len(sig.raw) == 2


def test_camera_tracker_dominant_frequency_with_nan_padding():
    """Verify dominant_frequency_hz handles initial NaN padding without returning NaN or 0.1 default (L2)."""
    fs = 30.0
    duration_s = 30.0
    t = np.linspace(0, duration_s, int(fs * duration_s))
    sine_025 = np.sin(2 * np.pi * 0.25 * t)  # 0.25 Hz target oscillation
    # Inject initial NaN padding into filtered array
    filtered_with_nan = np.concatenate(([np.nan, np.nan], sine_025))
    raw_with_nan = np.concatenate(([np.nan, np.nan], sine_025))

    sig = RespiratorySignal(
        raw=raw_with_nan,
        filtered=filtered_with_nan,
        timestamps=np.linspace(0, duration_s, len(filtered_with_nan)),
        sampling_rate=fs,
    )
    dom_f = sig.dominant_frequency_hz()
    assert np.isfinite(dom_f)
    assert np.isclose(dom_f, 0.25, atol=0.03)


def test_camera_tracker_custom_source_configuration():
    """Verify camera tracker accepts device indices and stream URLs."""
    tracker_default = CameraTracker()
    assert tracker_default.camera_source == 0

    tracker_idx = CameraTracker(camera_source=1)
    assert tracker_idx.camera_source == 1

    tracker_str_idx = CameraTracker(camera_source="2")
    assert tracker_str_idx.camera_source == "2"

    stream_url = "http://192.168.1.100:8080/video"
    tracker_stream = CameraTracker(camera_source=stream_url)
    assert tracker_stream.camera_source == stream_url


def test_camera_tracker_invalid_source_raises_runtime_error(monkeypatch):
    """Verify start() raises descriptive RuntimeError on unreachable video source without delay."""
    # Test unopenable index instantaneously
    tracker = CameraTracker(camera_source=9999)
    with pytest.raises(RuntimeError) as exc_info:
        tracker.start()
    assert "Cannot open video source (device index 9999)" in str(exc_info.value)

    # Test unopenable stream with mocked VideoCapture for sub-millisecond execution
    class MockFailedCap:
        def isOpened(self):
            return False
        def release(self):
            pass

    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: MockFailedCap())
    tracker_stream = CameraTracker(camera_source="http://192.168.1.100:8080/video")
    with pytest.raises(RuntimeError) as exc_info2:
        tracker_stream.start()
    assert "network stream 'http://192.168.1.100:8080/video'" in str(exc_info2.value)


