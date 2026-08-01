"""Post-incident safety channel.

Everything in this package runs on an **asynchronous channel** that is structurally
separate from the 10 Hz fast path. The fast path exists to stop a collision happening;
this package exists for the case where that failed -- the wearer is on the ground, or
the device has lost the world entirely -- and someone needs to be told.

It reads the IMU that ``perception/calibration.py`` already streams for ground-plane
pitch, so it adds a sensor cost of zero and a fast-path latency cost of zero.
"""

from .escalation import (
    ContactAlert,
    EscalationState,
    FallEscalation,
    TrustedContact,
)
from .fall import FallDetector, FallEvent, ImuSample

__all__ = [
    "ContactAlert",
    "EscalationState",
    "FallDetector",
    "FallEscalation",
    "FallEvent",
    "ImuSample",
    "TrustedContact",
]
