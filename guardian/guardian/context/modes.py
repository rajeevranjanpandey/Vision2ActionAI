"""Contexts and the risk profiles they select."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from ..config import RiskConfig


class Mode(str, Enum):
    SIDEWALK = "sidewalk"
    CROSSING = "crossing"
    TRANSIT_PLATFORM = "transit_platform"


# Higher rank = more conservative. Used by the switching rule: escalating is instant,
# relaxing has to be earned.
_RANK = {Mode.SIDEWALK: 0, Mode.CROSSING: 1, Mode.TRANSIT_PLATFORM: 2}


def conservatism_rank(mode: Mode) -> int:
    return _RANK[mode]


@dataclass(frozen=True, slots=True)
class RiskProfile:
    """Per-context overrides on the geometric kernel.

    ``hard_floor_m`` is the piece that has no analogue in the sidewalk profile: a
    distance from a detected platform edge inside which an alert fires on geometry alone,
    with no learned suppression and no TTC test. It is the same shape of rule as the
    imminent-contact veto -- a case where the system stops reasoning and just warns.
    """

    mode: Mode
    cone_half_width_m: float
    cone_widen_per_m: float
    ttc_warn_s: float
    ttc_urgent_s: float
    min_closing_speed_mps: float
    hard_floor_m: float | None
    vocabulary: tuple[str, ...]
    rationale: str


PROFILES: dict[Mode, RiskProfile] = {
    Mode.SIDEWALK: RiskProfile(
        mode=Mode.SIDEWALK,
        cone_half_width_m=0.55,
        cone_widen_per_m=0.08,
        ttc_warn_s=3.0,
        ttc_urgent_s=1.5,
        min_closing_speed_mps=0.15,
        hard_floor_m=None,
        vocabulary=("cyclist", "person", "post", "step"),
        rationale="The validated default. Every other profile is described relative to it.",
    ),
    Mode.CROSSING: RiskProfile(
        mode=Mode.CROSSING,
        cone_half_width_m=0.75,
        cone_widen_per_m=0.12,
        ttc_warn_s=4.0,
        ttc_urgent_s=2.0,
        min_closing_speed_mps=0.25,
        hard_floor_m=None,
        vocabulary=("vehicle", "turning car", "cyclist", "kerb"),
        rationale=(
            "Closing speeds are vehicular, so the warning horizon lengthens and the "
            "corridor widens to cover a turning vehicle's swept path. The raised "
            "min-closing-speed suppresses the stationary queue the user is walking past."
        ),
    ),
    Mode.TRANSIT_PLATFORM: RiskProfile(
        mode=Mode.TRANSIT_PLATFORM,
        cone_half_width_m=0.85,
        cone_widen_per_m=0.10,
        ttc_warn_s=4.0,
        ttc_urgent_s=2.0,
        min_closing_speed_mps=0.10,
        hard_floor_m=1.2,
        vocabulary=("platform edge", "train", "gap", "person"),
        rationale=(
            "The dangerous object is the floor. A 1.2 m hard geometric floor from the "
            "detected edge fires regardless of TTC, closing speed, or the learned head."
        ),
    ),
}


def profile_for(mode: Mode) -> RiskProfile:
    return PROFILES[mode]


def apply_profile(base: RiskConfig, profile: RiskProfile) -> RiskConfig:
    """Return a ``RiskConfig`` with the profile's overrides applied.

    Guarded, not trusted: a profile may only *widen* the corridor and *lengthen* the
    warning horizon relative to the validated sidewalk defaults. A context that tried to
    shrink either would be a context that quietly deleted validated safety margin, so the
    override is clamped instead of applied.
    """
    sidewalk = PROFILES[Mode.SIDEWALK]
    return replace(
        base,
        cone_half_width_m=max(profile.cone_half_width_m, sidewalk.cone_half_width_m),
        cone_widen_per_m=max(profile.cone_widen_per_m, sidewalk.cone_widen_per_m),
        ttc_warn_s=max(profile.ttc_warn_s, sidewalk.ttc_warn_s),
        ttc_urgent_s=max(profile.ttc_urgent_s, sidewalk.ttc_urgent_s),
        min_closing_speed_mps=profile.min_closing_speed_mps,
    )
