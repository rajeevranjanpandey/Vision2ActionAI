"""Directional haptic output.

Haptics carry the safety signal because they survive traffic noise, do not compete with
the user's echolocation, and reach awareness ~120 ms faster than speech. Three ERM/LRA
actuators on the shoulder strap: left, centre, right.

The driver talks to a microcontroller over serial with a 3-byte frame. A ``NullHaptics``
fallback keeps the pipeline runnable on a laptop with no hardware attached.
"""

from __future__ import annotations

import logging
import struct
import time
from typing import Protocol

from ..config import HapticsConfig
from ..types import AlertLevel, Direction

logger = logging.getLogger(__name__)

_CHANNEL_INDEX = {Direction.LEFT: 0, Direction.CENTRE: 1, Direction.RIGHT: 2}


class HapticDriver(Protocol):
    def pulse(self, channel: int, intensity: float, duration_ms: int) -> None: ...
    def close(self) -> None: ...


class SerialHaptics:
    """3-byte frame: [channel, intensity 0-255, duration_ms // 4]."""

    def __init__(self, port: str = "/dev/ttyTHS0", baud: int = 115200) -> None:
        import serial  # pyserial; optional dependency

        self._serial = serial.Serial(port, baud, timeout=0.05)

    def pulse(self, channel: int, intensity: float, duration_ms: int) -> None:
        frame = struct.pack(
            "BBB",
            channel & 0xFF,
            int(max(0.0, min(1.0, intensity)) * 255),
            min(255, max(1, duration_ms // 4)),
        )
        self._serial.write(frame)

    def close(self) -> None:
        self._serial.close()


class NullHaptics:
    def pulse(self, channel: int, intensity: float, duration_ms: int) -> None:
        logger.debug("haptic ch=%d i=%.2f d=%dms", channel, intensity, duration_ms)

    def close(self) -> None:
        pass


class HapticFeedback:
    """Level -> pulse pattern mapping."""

    def __init__(self, cfg: HapticsConfig, driver: HapticDriver | None = None) -> None:
        self.cfg = cfg
        if driver is not None:
            self.driver = driver
        elif not cfg.enabled:
            self.driver = NullHaptics()
        else:
            try:
                self.driver = SerialHaptics()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Haptic hardware not found (%s); using null driver", exc)
                self.driver = NullHaptics()

    def emit(self, level: AlertLevel, direction: Direction) -> None:
        """Fire the pattern. Called on the fast path, so it must not block meaningfully."""
        channel = _CHANNEL_INDEX[direction]
        patterns = {
            AlertLevel.INFO: (1, 0.35, 60),
            AlertLevel.WARN: (2, 0.60, 80),
            AlertLevel.URGENT: (4, 1.00, 60),
            AlertLevel.DEGRADED: (1, 0.50, 400),
        }
        pulses, intensity, duration = patterns.get(level, (1, 0.4, 60))
        rate = self.cfg.urgent_pattern_hz if level is AlertLevel.URGENT else self.cfg.warn_pattern_hz
        gap_s = max(0.0, 1.0 / max(rate, 0.5) - duration / 1000.0)

        for i in range(pulses):
            self.driver.pulse(channel, intensity, duration)
            if i < pulses - 1:
                time.sleep(min(gap_s, 0.08))

    def close(self) -> None:
        self.driver.close()
