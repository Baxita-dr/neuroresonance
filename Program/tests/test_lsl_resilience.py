"""
tests/test_lsl_resilience.py — LSL Streamer & Hardware Bridge Resilience Tests
"""

import time
import numpy as np
import pytest
from src.lsl_streamer import fNIRSSimulator, LSLBridge


def test_fnirs_simulator_lifecycle_and_synthetic_channels():
    """Verify the simulator creates example channels; not physiological validity."""
    sim = fNIRSSimulator(sampling_rate=20.0, mayer_wave_freq=0.1)
    sim.start()
    time.sleep(0.5)  # Wait for background thread to buffer samples

    data = sim.get_latest(n_samples=20)
    sim.stop()

    assert "HbO2" in data and "HbR" in data
    assert len(data["HbO2"]) > 0
    assert len(data["HbR"]) > 0
    assert sim.STREAM_TYPE == "fNIRS-SIM"
    assert "Mayer" in sim.STREAM_NAME

    # The example generator uses bounded arbitrary synthetic units; this is
    # only a sanity check on the chosen simulation scale, not physiology.
    assert np.all(np.abs(data["HbO2"]) < 10.0)
    assert np.all(np.abs(data["HbR"]) < 10.0)


def test_fnirs_simulator_returns_each_new_sample_once_in_timestamp_order():
    sim = fNIRSSimulator(sampling_rate=1.0, seed=7)
    with sim._lock:
        for index in range(4):
            sim._hbo2_buf.append(float(index))
            sim._hbr_buf.append(float(-index))
            sim._ts_buf.append(float(index) / 10.0)
            sim._total_generated += 1

    chunk = sim.get_new_samples()
    empty = sim.get_new_samples()

    assert np.array_equal(chunk["sample_indices"], [0, 1, 2, 3])
    assert np.array_equal(chunk["HbO2"], [0.0, 1.0, 2.0, 3.0])
    assert np.array_equal(chunk["timestamps"], [0.0, 0.1, 0.2, 0.3])
    assert chunk["dropped_samples"] == 0
    assert empty["timestamps"].size == 0


def test_fnirs_simulator_reports_samples_lost_to_its_bounded_buffer():
    sim = fNIRSSimulator(sampling_rate=1.0, seed=7)
    with sim._lock:
        for index in range(sim._buffer_size + 3):
            sim._hbo2_buf.append(float(index))
            sim._hbr_buf.append(float(-index))
            sim._ts_buf.append(float(index))
            sim._total_generated += 1

    chunk = sim.get_new_samples()

    assert chunk["dropped_samples"] == 3
    assert chunk["timestamps"].size == sim._buffer_size
    assert chunk["sample_indices"][0] == 3
    assert chunk["timestamps"][0] == 3.0


def test_lsl_bridge_timeout_and_offline_safety():
    """Verify LSLBridge does not hang or raise uncaught exceptions when no device is found."""
    # Attempt connecting to non-existent hardware stream with low timeout
    bridge = LSLBridge(stream_name="NonExistent_Cortivision_Stream_XYZ", timeout=0.3, auto_reconnect=False)
    success = bridge.connect()

    assert success is False
    assert bridge.is_connected is False

    # Pull chunk should return safe empty dict without crashing
    chunk = bridge.pull_chunk(max_samples=50)
    assert len(chunk["HbO2"]) == 0
    assert len(chunk["HbR"]) == 0
    assert len(chunk["timestamps"]) == 0

    bridge.disconnect()


def test_lsl_bridge_aligns_two_streams_by_timestamps():
    """Align a separate respiration inlet to the fNIRS sample times."""
    class FakeInlet:
        def __init__(self, samples, timestamps):
            self.samples = samples
            self.timestamps = timestamps

        def pull_chunk(self, timeout=0.0, max_samples=100):
            return self.samples, self.timestamps

        def close_stream(self):
            pass

    bridge = LSLBridge(
        respiration_stream_name="BreathStream",
        auto_reconnect=False,
    )
    bridge._connected = True
    bridge._inlet = FakeInlet(
        samples=[[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]],
        timestamps=[1.0, 1.1, 1.2],
    )
    bridge._respiration_inlet = FakeInlet(
        samples=[[0.0], [1.0], [2.0], [3.0], [4.0]],
        timestamps=[0.95, 1.05, 1.15, 1.25, 1.35],
    )
    bridge._channel_names = ["HbO2", "HbR"]
    bridge._respiration_channel_names = ["Respiration"]

    aligned = bridge.pull_chunk(max_samples=20)

    assert np.allclose(aligned["timestamps"], [1.0, 1.1, 1.2])
    assert np.allclose(aligned["HbO2"], [1.0, 2.0, 3.0])
    assert np.allclose(aligned["HbR"], [-1.0, -2.0, -3.0])
    assert np.allclose(aligned["Respiration"], [0.5, 1.5, 2.5])
    bridge.disconnect()
