"""Alert arbitration.

The single hardest product constraint in assistive tech: **at most one thing is said at
a time**, and it must be the right thing. Users abandon devices that chatter. This module
turns a ranked hazard list into at most one alert per tick, with hysteresis to stop
flicker and cooldowns to stop chatter, and it is the only place allowed to emit alerts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import PolicyConfig, RiskConfig
from ..types import Alert, AlertLevel, Direction, Hazard


@dataclass(slots=True)
class _Pending:
    track_id: int
    level: AlertLevel
    frames: int


class AlertPolicy:
    """Hysteresis + cooldown arbitration over ranked hazards."""

    def __init__(self, policy: PolicyConfig, risk: RiskConfig) -> None:
        self.policy = policy
        self.risk = risk
        self._pending: _Pending | None = None
        self._last_emit_s: float = -1e9
        self._last_level = AlertLevel.NONE
        self._last_track_id: int | None = None

    def reset(self) -> None:
        self._pending = None
        self._last_emit_s = -1e9
        self._last_level = AlertLevel.NONE
        self._last_track_id = None

    def decide(
        self,
        hazards: list[Hazard],
        now_s: float,
        degraded: bool = False,
        advisory: str | None = None,
    ) -> Alert | None:
        """Return the alert to speak this tick, or None for silence."""
        if degraded:
            # Loud failure. Silence is the one unacceptable failure mode.
            self._last_emit_s = now_s
            self._last_level = AlertLevel.DEGRADED
            return Alert(
                level=AlertLevel.DEGRADED,
                direction=Direction.CENTRE,
                hazard=None,
                utterance="Guardian degraded. Rely on your cane.",
                timestamp_s=now_s,
            )

        if not hazards:
            self._pending = None
            self._last_level = AlertLevel.NONE
            self._last_track_id = None
            return None

        # assess() already sorts, but the policy must not depend on its caller for the
        # single most safety-relevant decision it makes: which hazard gets the voice.
        hazard = max(hazards, key=lambda h: (h.severity, -h.ttc_s))

        level = self._level_for(hazard)
        if level is AlertLevel.NONE:
            self._pending = None
            return None

        if not self._confirm(hazard, level):
            return None

        if not self._cooldown_elapsed(hazard, level, now_s):
            return None

        self._last_emit_s = now_s
        self._last_level = level
        self._last_track_id = hazard.track.track_id
        return Alert(
            level=level,
            direction=hazard.direction,
            hazard=hazard,
            utterance=utterance_for(hazard, level),
            ttc_s=hazard.ttc_s,
            timestamp_s=now_s,
            advisory=advisory,
        )

    # ----------------------------------------------------------------- private

    def _level_for(self, hazard: Hazard) -> AlertLevel:
        if hazard.ttc_s <= self.risk.ttc_urgent_s:
            return AlertLevel.URGENT
        if hazard.ttc_s <= self.risk.ttc_warn_s:
            return AlertLevel.WARN
        if hazard.closest_approach_m <= self.risk.ego_radius_m:
            return AlertLevel.INFO
        return AlertLevel.NONE

    def _confirm(self, hazard: Hazard, level: AlertLevel) -> bool:
        """Require N consecutive frames before speaking -- except when urgent.

        Hysteresis buys precision at the cost of latency. At urgent TTC we cannot afford
        the latency, so we trade back the other way and accept the occasional false
        urgent alert. Missing a real one is far worse.
        """
        if level is AlertLevel.URGENT:
            self._pending = _Pending(hazard.track.track_id, level, self.policy.hysteresis_frames)
            return True

        tid = hazard.track.track_id
        if self._pending is not None and self._pending.track_id == tid:
            self._pending.frames += 1
            self._pending.level = level
        else:
            self._pending = _Pending(tid, level, 1)
        return self._pending.frames >= self.policy.hysteresis_frames

    def _cooldown_elapsed(self, hazard: Hazard, level: AlertLevel, now_s: float) -> bool:
        cooldown = (
            self.policy.urgent_cooldown_s
            if level is AlertLevel.URGENT
            else self.policy.alert_cooldown_s
        )
        elapsed = now_s - self._last_emit_s

        # Escalation to a strictly higher level bypasses the cooldown: if a warning
        # becomes urgent, waiting out the timer could cost the user the collision.
        escalating = level is AlertLevel.URGENT and self._last_level is not AlertLevel.URGENT
        if escalating:
            return True

        # Repeating the same hazard needs the full cooldown; a *new* hazard needs half.
        if self._last_track_id is not None and hazard.track.track_id != self._last_track_id:
            cooldown *= 0.5
        return elapsed >= cooldown


def utterance_for(hazard: Hazard, level: AlertLevel) -> str:
    """Short, unambiguous, direction-first speech.

    Direction first because it is actionable before the sentence finishes. "Left, post"
    lets the user start moving on the first syllable; "there is a post on your left"
    wastes 1.2 s of a 1.5 s window.
    """
    direction = {
        Direction.LEFT: "Left",
        Direction.RIGHT: "Right",
        Direction.CENTRE: "Ahead",
    }[hazard.direction]

    obj = _friendly(hazard.track.class_name)
    distance = hazard.track.range_m

    if level is AlertLevel.URGENT:
        return f"{direction}. Stop. {obj}."
    if level is AlertLevel.WARN:
        if math.isfinite(hazard.ttc_s):
            return f"{direction}, {obj}, {distance:.0f} metres, closing."
        return f"{direction}, {obj}, {distance:.0f} metres."
    return f"{obj} {direction.lower()}, {distance:.0f} metres."


def _friendly(class_name: str) -> str:
    """Map detector labels to words a pedestrian actually uses."""
    mapping = {
        "person": "person",
        "bicycle": "bike",
        "motorcycle": "motorbike",
        "car": "car",
        "bus": "bus",
        "truck": "truck",
        "traffic light": "crossing signal",
        "fire hydrant": "hydrant",
        "stop sign": "sign post",
        "parking meter": "post",
        "bench": "bench",
        "potted plant": "planter",
        "chair": "chair",
        "dining table": "table",
    }
    return mapping.get(class_name, class_name.replace("_", " "))
