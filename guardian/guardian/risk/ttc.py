"""Risk cone geometry, time-to-collision, and closest-approach forecasting.

This module is the safety kernel. It has no model dependencies, no I/O, and no
randomness -- it is pure geometry so it can be unit-tested exhaustively and reasoned
about in a safety case. Everything upstream is best-effort perception; everything here
must be exactly right.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import RiskConfig
from ..types import Direction, Hazard, Track


def cone_half_width_at(z_m: float, cfg: RiskConfig) -> float:
    """Half-width of the risk corridor at forward distance ``z_m``.

    The corridor widens with distance because heading uncertainty grows: a 5-degree
    heading error is 9 cm at 1 m and 87 cm at 10 m.
    """
    return cfg.cone_half_width_m + cfg.cone_widen_per_m * max(z_m, 0.0)


def in_risk_cone(track: Track, cfg: RiskConfig) -> bool:
    """Is the object currently inside the corridor the user is walking into?"""
    if track.z_m <= 0 or track.z_m > cfg.cone_range_m:
        return False
    half = cone_half_width_at(track.z_m, cfg) + 0.5 * track.width_m
    return abs(track.x_m) <= half


def closing_speed(track: Track) -> float:
    """Rate of decrease of range, in m/s. Positive means approaching."""
    r = track.range_m
    if r < 1e-6:
        return 0.0
    return -(track.x_m * track.vx_mps + track.z_m * track.vz_mps) / r


def closest_approach(track: Track, cfg: RiskConfig) -> tuple[float, float]:
    """Analytic closest point of approach under constant velocity.

    Returns ``(distance_m, time_s)`` clamped to the forecast horizon. Solving this in
    closed form rather than rolling out saves ~40x the compute in the fast path.
    """
    px, pz = track.x_m, track.z_m
    vx, vz = track.vx_mps, track.vz_mps
    v2 = vx * vx + vz * vz
    if v2 < 1e-9:
        return float(math.hypot(px, pz)), 0.0

    t = -(px * vx + pz * vz) / v2
    t = float(np.clip(t, 0.0, cfg.horizon_s))
    cx, cz = px + vx * t, pz + vz * t
    return float(math.hypot(cx, cz)), t


def time_to_collision(track: Track, cfg: RiskConfig) -> float:
    """Time until the object's extent breaches the ego cylinder.

    Solves |p + v t| = R for the effective radius R = ego_radius + half object width.
    Returns +inf when no breach occurs within the horizon.
    """
    radius = cfg.ego_radius_m + 0.5 * track.width_m
    px, pz = track.x_m, track.z_m
    vx, vz = track.vx_mps, track.vz_mps

    a = vx * vx + vz * vz
    b = 2.0 * (px * vx + pz * vz)
    c = px * px + pz * pz - radius * radius

    if c <= 0.0:
        return 0.0  # already inside the cylinder
    if a < 1e-9:
        return float("inf")  # relative motion is static

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return float("inf")  # trajectory misses the cylinder

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    candidates = [t for t in (t1, t2) if t >= 0.0]
    if not candidates:
        return float("inf")
    ttc = min(candidates)
    return ttc if ttc <= cfg.horizon_s else float("inf")


def direction_of(track: Track, cfg: RiskConfig) -> Direction:
    """Which haptic channel should fire."""
    half = cone_half_width_at(track.z_m, cfg)
    if track.x_m < -half * 0.5:
        return Direction.LEFT
    if track.x_m > half * 0.5:
        return Direction.RIGHT
    return Direction.CENTRE


def severity_of(ttc_s: float, distance_m: float, cfg: RiskConfig) -> float:
    """Monotone 0..1 urgency score used to rank competing hazards.

    Time dominates distance: an object 6 m away closing at 4 m/s outranks one 2 m away
    that is stationary, because only the first has a shrinking avoidance window.
    """
    if not math.isfinite(ttc_s):
        time_term = 0.0
    else:
        time_term = float(np.clip(1.0 - ttc_s / max(cfg.horizon_s, 1e-6), 0.0, 1.0))
    dist_term = float(np.clip(1.0 - distance_m / max(cfg.cone_range_m, 1e-6), 0.0, 1.0))
    return float(np.clip(0.75 * time_term + 0.25 * dist_term, 0.0, 1.0))


def assess(tracks: list[Track], cfg: RiskConfig) -> list[Hazard]:
    """Turn tracks into ranked hazards. This is the whole predictive claim in one call."""
    hazards: list[Hazard] = []
    for track in tracks:
        if not track.confirmed:
            continue
        if track.z_m <= 0.0 or track.z_m > cfg.cone_range_m:
            continue

        ttc = time_to_collision(track, cfg)
        cpa_dist, cpa_time = closest_approach(track, cfg)
        approaching = closing_speed(track) >= cfg.min_closing_speed_mps

        breaches = math.isfinite(ttc) and ttc <= cfg.ttc_warn_s
        static_blocker = in_risk_cone(track, cfg) and track.range_m <= cfg.ego_radius_m * 2.0

        # A static object directly in the path still matters even with zero closing
        # speed if the user is the one moving -- ego compensation folds their walking
        # speed into vz, so this branch only catches the standing-still edge case.
        if not (breaches and approaching) and not static_blocker:
            continue

        hazards.append(
            Hazard(
                track=track,
                ttc_s=ttc,
                closest_approach_m=cpa_dist,
                time_of_closest_approach_s=cpa_time,
                direction=direction_of(track, cfg),
                severity=severity_of(ttc, track.range_m, cfg),
            )
        )

    hazards.sort(key=lambda h: (-h.severity, h.ttc_s))
    return hazards
