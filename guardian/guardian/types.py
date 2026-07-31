"""Core data structures shared by every stage of the pipeline.

These are deliberately plain dataclasses: the fast path allocates them at 10 Hz and
anything heavier (pydantic, attrs validators) shows up in the latency budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class AlertLevel(str, Enum):
    """Ordered by escalation. Comparison is done via ``ORDER`` below."""

    NONE = "none"
    INFO = "info"
    WARN = "warn"
    URGENT = "urgent"
    DEGRADED = "degraded"


ALERT_ORDER = {
    AlertLevel.NONE: 0,
    AlertLevel.INFO: 1,
    AlertLevel.WARN: 2,
    AlertLevel.URGENT: 3,
    AlertLevel.DEGRADED: 4,
}


class Direction(str, Enum):
    LEFT = "left"
    CENTRE = "centre"
    RIGHT = "right"


@dataclass(slots=True)
class Detection:
    """A single 2D detection in image space."""

    xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str

    @property
    def cx(self) -> float:
        return 0.5 * (self.xyxy[0] + self.xyxy[2])

    @property
    def cy(self) -> float:
        return 0.5 * (self.xyxy[1] + self.xyxy[3])

    @property
    def bottom(self) -> float:
        """Bottom edge -- the ground-contact point used for range estimation."""
        return self.xyxy[3]


@dataclass(slots=True)
class DepthResult:
    """Depth Anything V2 output after metric rescaling."""

    relative: np.ndarray        # HxW, unitless inverse-depth as produced by the model
    metric: np.ndarray          # HxW, metres
    scale: float                # recovered metric scale factor
    scale_confidence: float     # 0..1, ground-plane inlier ratio


@dataclass(slots=True)
class Observation:
    """A detection lifted into the 3D ego frame.

    Ego frame: +x right, +y up, +z forward, origin at the camera, z on the ground plane.
    """

    x_m: float
    z_m: float
    width_m: float
    class_name: str
    score: float
    detection: Detection
    mask: Optional[np.ndarray] = None


@dataclass(slots=True)
class Track:
    """A temporally associated 3D object with velocity."""

    track_id: int
    x_m: float
    z_m: float
    vx_mps: float
    vz_mps: float
    width_m: float
    class_name: str
    hits: int = 0
    age: int = 0
    time_since_update: int = 0
    covariance_trace: float = 0.0

    @property
    def range_m(self) -> float:
        return float(np.hypot(self.x_m, self.z_m))

    @property
    def confirmed(self) -> bool:
        return self.hits >= 3


@dataclass(slots=True)
class Hazard:
    """A track that the forecast says will breach the ego cylinder."""

    track: Track
    ttc_s: float
    closest_approach_m: float
    time_of_closest_approach_s: float
    direction: Direction
    severity: float             # 0..1, monotonic in urgency


@dataclass(slots=True)
class Alert:
    """The single arbitrated output of the fast path for one tick."""

    level: AlertLevel
    direction: Direction
    hazard: Optional[Hazard]
    utterance: str
    ttc_s: float = float("inf")
    timestamp_s: float = 0.0
    advisory: Optional[str] = None   # slow-path VLM enrichment, may be None


@dataclass(slots=True)
class FrameResult:
    """Everything produced for one processed frame. Used by the recorder and evaluator."""

    frame_index: int
    timestamp_s: float
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    hazards: list[Hazard] = field(default_factory=list)
    alert: Optional[Alert] = None
    depth_scale: float = 1.0
    stage_latency_ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        return float(sum(self.stage_latency_ms.values()))
