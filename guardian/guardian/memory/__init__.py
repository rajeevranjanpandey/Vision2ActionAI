"""Persistent hazard memory -- roadmap item 1.

The fast path re-derives the same pothole from raw pixels on every pass. That is correct
for a cyclist and wasteful for a kerb: static geometry does not need 10 Hz re-discovery,
it needs to be *remembered*.

Two things live here, and the split matters:

* :class:`HazardMemory` is a local, single-user store of geo-pinned static hazards with
  a visual fingerprint, built up by the **slow path** (the VLM tags each hazard static or
  dynamic). It is read by the fast path as a cheap dictionary lookup -- a bounded number
  of float comparisons, no model, no allocation -- so the 90 ms budget is untouched.
* :class:`RepeatAlertGate` is the part that actually moves the metric. A hazard the user
  has already walked past safely, several times, on this exact route, does not need the
  same full warning every pass. The gate downgrades that repeat to a short known-hazard
  earcon -- and it is bound by the same asymmetry as the learned head: it may only
  downgrade non-urgent geometry, and never a hazard the memory calls dynamic.

MVP boundary from the roadmap: single-user local cache. There is no cross-user sync here
on purpose -- sharing another person's route history is a consent problem, not an
engineering one, and it is deliberately v2.
"""

from .hazard_map import (
    HazardMemory,
    HazardObservation,
    HazardPrior,
    RepeatAlertGate,
    StaticHazard,
    haversine_m,
)
from .route_sim import RoutePass, simulate_repeat_route

__all__ = [
    "HazardMemory",
    "HazardObservation",
    "HazardPrior",
    "RepeatAlertGate",
    "RoutePass",
    "StaticHazard",
    "haversine_m",
    "simulate_repeat_route",
]
