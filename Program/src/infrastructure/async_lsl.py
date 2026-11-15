"""
src/infrastructure/async_lsl.py — Asynchronous LSL Hardware Adapter

Implements the StreamProcessor protocol with backpressure handling and
non-blocking asyncio generation.
"""

import asyncio
import time
from typing import AsyncGenerator, Optional, Tuple
import numpy as np

from src.domain.models import SensorSample
from src.domain.protocols import StreamProcessor
from src.lsl_streamer import LSLBridge, fNIRSSimulator


class AsyncLSLStreamProcessor(StreamProcessor):
    """
    Asynchronous hardware bridge streaming SensorSample objects from LSL.
    """

    def __init__(
        self,
        stream_name: Optional[str] = None,
        poll_interval_s: float = 0.10,
        use_simulator_fallback: bool = True,
        resp_pacer_hz: float = 0.10,
        fs: float = 10.0,
    ) -> None:
        self.stream_name = stream_name
        self.fs = float(fs)
        self.poll_interval_s = float(poll_interval_s)
        self.use_simulator_fallback = use_simulator_fallback
        self.resp_pacer_hz = float(resp_pacer_hz)
        self.bridge: Optional[LSLBridge] = None
        self.simulator: Optional[fNIRSSimulator] = None
        self._running = False
        self._last_yielded_ts = -1.0

    async def initialize(self) -> bool:
        """Connect to real LSL hardware or spawn fallback simulator."""
        self.bridge = LSLBridge(stream_name=self.stream_name, timeout=1.0, auto_reconnect=True)
        connected = self.bridge.connect(silent=True)

        if not connected and self.use_simulator_fallback:
            self.simulator = fNIRSSimulator(sampling_rate=self.fs, mayer_wave_freq=0.1)
            self.simulator.start()

        self._running = True
        return connected or (self.simulator is not None)

    async def process_stream(self) -> AsyncGenerator[SensorSample, None]:
        """
        Asynchronously yield SensorSample instances as they arrive from hardware.
        Channel 0 = measured respiration (NaN when absent), Channel 1 = HbO2,
        Channel 2 = HbR. Simulator fallback uses synthetic respiration.
        """
        if not self._running:
            await self.initialize()

        while self._running:
            # 1. Pull from hardware LSL bridge if connected
            if self.bridge and self.bridge.is_connected:
                chunk = self.bridge.pull_chunk(max_samples=10)
                n = len(chunk["timestamps"])
                for i in range(n):
                    ts = float(chunk["timestamps"][i])
                    if ts <= self._last_yielded_ts:
                        continue
                    self._last_yielded_ts = ts
                    resp_stream = chunk.get("Respiration", np.empty(0))
                    resp_val = float(resp_stream[i]) if i < len(resp_stream) else float("nan")
                    yield SensorSample(
                        timestamp=ts,
                        channel_data=(resp_val, float(chunk["HbO2"][i]), float(chunk["HbR"][i])),
                    )
            # 2. Or pull from simulator fallback without duplicate samples
            elif self.simulator:
                sim_data = self.simulator.get_latest(n_samples=5)
                n = len(sim_data["HbO2"])
                for i in range(n):
                    ts = float(sim_data["timestamps"][i]) if len(sim_data["timestamps"]) > i else time.time()
                    if ts <= self._last_yielded_ts:
                        continue
                    self._last_yielded_ts = ts
                    resp_val = float(np.sin(2 * np.pi * self.resp_pacer_hz * ts))
                    yield SensorSample(
                        timestamp=ts,
                        channel_data=(resp_val, float(sim_data["HbO2"][i]), float(sim_data["HbR"][i])),
                    )

            await asyncio.sleep(self.poll_interval_s)

    def stop(self) -> None:
        """Halt streaming loop and release resources."""
        self._running = False
        if self.bridge:
            self.bridge.disconnect()
        if self.simulator:
            self.simulator.stop()
