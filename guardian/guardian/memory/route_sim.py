"""Repeat-route simulator for the hazard-memory metric.

The roadmap's kill-or-keep number for item 1 is *false alarms per kilometre on repeat
routes, relative to first-pass FA/km*. That comparison needs a route walked more than
once, with the same street furniture in the same places and realistically noisy GPS,
which is exactly what this generates.

What is simulated is the **encounter stream**, not pixels: each pass produces a sequence
of hazard encounters with a TTC and a slow-path static/dynamic tag. That is the level the
gate operates at, so it is the level the metric should be measured at.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hazard_map import (
    FINGERPRINT_DIM,
    HazardMemory,
    HazardObservation,
    MemoryParams,
    RepeatAlertGate,
)

# One degree of latitude is ~111.32 km; the routes here are short and straight enough
# that a flat-earth step in latitude is exact to well under the GPS noise floor.
_M_PER_DEG_LAT = 111_320.0

STATIC_FURNITURE = [
    ("pothole", 0.45),
    ("a_board", 0.55),
    ("bollard", 0.35),
    ("kerb_drop", 0.70),
    ("scaffold_leg", 0.60),
]


@dataclass(slots=True)
class Encounter:
    """One fast-path detection event.

    ``lat``/``lon`` is the *walker's* pose at the moment of detection, which is what the
    memory is queried with; ``hazard_lat``/``hazard_lon`` is where the hazard itself was
    estimated to be, which is what gets written back. Conflating the two makes the
    heading cone meaningless -- a pin sitting on top of you is neither ahead nor behind.
    """

    t_s: float
    lat: float
    lon: float
    hazard_lat: float
    hazard_lon: float
    heading_deg: float
    class_name: str
    ttc_s: float
    is_static: bool
    truth_id: int


@dataclass
class RoutePass:
    index: int
    length_km: float
    encounters: list[Encounter] = field(default_factory=list)


def _fingerprint(rng: np.random.Generator, seed_vec: np.ndarray, jitter: float) -> np.ndarray:
    v = seed_vec + rng.normal(0.0, jitter, size=FINGERPRINT_DIM)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def build_route(
    n_static: int = 7,
    n_passes: int = 6,
    length_km: float = 1.2,
    dynamic_per_km: float = 4.0,
    gps_sigma_m: float = 5.0,
    seed: int = 0,
) -> tuple[list[RoutePass], dict[int, np.ndarray]]:
    """A straight north-bound route walked ``n_passes`` times at 1.4 m/s."""
    rng = np.random.default_rng(seed)
    base_lat, base_lon = 51.5074, -0.1278

    truth = []
    prints: dict[int, np.ndarray] = {}
    for i in range(n_static):
        cls, radius = STATIC_FURNITURE[i % len(STATIC_FURNITURE)]
        frac = (i + 0.5) / n_static
        truth.append(
            {
                "id": i,
                "lat": base_lat + frac * length_km * 1000.0 / _M_PER_DEG_LAT,
                "lon": base_lon,
                "class_name": cls,
                "radius_m": radius,
            }
        )
        seed_vec = rng.normal(size=FINGERPRINT_DIM)
        prints[i] = seed_vec / float(np.linalg.norm(seed_vec))

    passes: list[RoutePass] = []
    walk_speed = 1.4
    for p in range(n_passes):
        rp = RoutePass(index=p, length_km=length_km)
        for t in truth:
            # A static hazard in the walking corridor is approached, so it generates a
            # burst of encounters as the range closes -- that burst is the repeat load.
            t0 = (t["id"] + 0.5) / n_static * (length_km * 1000.0 / walk_speed)
            for k, ttc in enumerate((3.4, 2.6, 1.9)):
                # The walker is ttc * speed metres short of the hazard, heading north.
                range_m = ttc * walk_speed
                rp.encounters.append(
                    Encounter(
                        t_s=t0 + k * 0.7,
                        lat=t["lat"] - range_m / _M_PER_DEG_LAT
                        + rng.normal(0.0, gps_sigma_m) / _M_PER_DEG_LAT,
                        lon=t["lon"] + rng.normal(0.0, gps_sigma_m) / _M_PER_DEG_LAT,
                        hazard_lat=t["lat"] + rng.normal(0.0, gps_sigma_m) / _M_PER_DEG_LAT,
                        hazard_lon=t["lon"] + rng.normal(0.0, gps_sigma_m) / _M_PER_DEG_LAT,
                        heading_deg=0.0 + rng.normal(0.0, 4.0),
                        class_name=str(t["class_name"]),
                        ttc_s=ttc,
                        is_static=True,
                        truth_id=int(t["id"]),
                    )
                )
        n_dyn = rng.poisson(dynamic_per_km * length_km)
        for j in range(int(n_dyn)):
            frac = rng.uniform(0.05, 0.95)
            rp.encounters.append(
                Encounter(
                    t_s=frac * (length_km * 1000.0 / walk_speed),
                    lat=base_lat + frac * length_km * 1000.0 / _M_PER_DEG_LAT,
                    lon=base_lon + rng.normal(0.0, gps_sigma_m) / _M_PER_DEG_LAT,
                    hazard_lat=base_lat + frac * length_km * 1000.0 / _M_PER_DEG_LAT,
                    hazard_lon=base_lon,
                    heading_deg=rng.normal(0.0, 4.0),
                    class_name="person",
                    ttc_s=float(rng.uniform(1.1, 3.2)),
                    is_static=False,
                    truth_id=-1 - j,
                )
            )
        rp.encounters.sort(key=lambda e: e.t_s)
        passes.append(rp)
    return passes, prints


def simulate_repeat_route(
    n_static: int = 7,
    n_passes: int = 6,
    length_km: float = 1.2,
    seed: int = 0,
    params: MemoryParams | None = None,
) -> dict[str, object]:
    """Walk the route ``n_passes`` times and score alerts per km, with and without memory.

    Only *non-urgent repeat announcements of established static pins* can be gated, so
    the urgent-encounter count is reported alongside and must be identical in both arms:
    if it is not, the gate has eaten a safety alert and the feature is dead.
    """
    passes, prints = build_route(
        n_static=n_static, n_passes=n_passes, length_km=length_km, seed=seed
    )
    rng = np.random.default_rng(seed + 991)

    memory = HazardMemory(params)
    gate = RepeatAlertGate()
    rows: list[dict[str, float]] = []

    for rp in passes:
        gate.reset_pass()
        baseline = gated = urgent_base = urgent_gated = 0
        seen_this_pass: set[str] = set()

        for enc in rp.encounters:
            hazard_id = None
            prior = 0.0
            established = False
            if enc.is_static:
                hits = memory.query(enc.lat, enc.lon, enc.heading_deg)
                match = next(
                    (h for h in hits if h.hazard.class_name == enc.class_name), None
                )
                if match is not None:
                    hazard_id = match.hazard.id
                    prior = match.prior
                    established = match.established

            decision = gate.decide(hazard_id, prior, enc.ttc_s, established, enc.t_s)
            baseline += 1
            gated += 1 if decision.announce else 0
            if enc.ttc_s <= gate.urgent_ttc_s:
                urgent_base += 1
                urgent_gated += 1 if decision.announce else 0

            # The slow path tags a hazard once per approach, at 0.5-1 Hz, when it is
            # close enough to fingerprint reliably -- not once per fast-path encounter.
            if enc.is_static and enc.ttc_s <= 2.0:
                obs = HazardObservation(
                    lat=enc.hazard_lat,
                    lon=enc.hazard_lon,
                    class_name=enc.class_name,
                    fingerprint=_fingerprint(rng, prints[enc.truth_id], 0.10),
                    t_s=rp.index * 1000.0 + enc.t_s,
                )
                stored = memory.observe(obs)
                seen_this_pass.add(stored.id)

        rows.append(
            {
                "pass": rp.index + 1,
                "baseline_per_km": baseline / rp.length_km,
                "memory_per_km": gated / rp.length_km,
                "urgent_baseline": float(urgent_base),
                "urgent_memory": float(urgent_gated),
                "pins": float(len(memory)),
                "established_pins": float(sum(1 for h in memory.hazards if h.established)),
            }
        )

    first = rows[0]["memory_per_km"]
    later = float(np.mean([r["memory_per_km"] for r in rows[1:]])) if len(rows) > 1 else first
    base_later = float(np.mean([r["baseline_per_km"] for r in rows[1:]])) if len(rows) > 1 else \
        rows[0]["baseline_per_km"]
    # The honest headline is the *paired* comparison on the same pass: both arms saw the
    # identical encounter list, so the dynamic-hazard count cancels. Comparing the last
    # pass against the first pass instead mixes in Poisson variation in pedestrian
    # traffic and can report a "reduction" of either sign for reasons unrelated to memory.
    settled = [r for r in rows if r["established_pins"] > 0]
    paired = settled or rows
    paired_base = float(np.mean([r["baseline_per_km"] for r in paired]))
    paired_mem = float(np.mean([r["memory_per_km"] for r in paired]))
    last = rows[-1]
    return {
        "rows": rows,
        "first_pass_per_km": first,
        "repeat_pass_per_km": later,
        "baseline_repeat_per_km": base_later,
        "paired_reduction_settled": (paired_base - paired_mem) / paired_base if paired_base else 0.0,
        "paired_reduction_last_pass": (
            (last["baseline_per_km"] - last["memory_per_km"]) / last["baseline_per_km"]
            if last["baseline_per_km"] else 0.0
        ),
        "reduction_vs_first_pass": (first - later) / first if first else 0.0,
        "reduction_vs_baseline": (base_later - later) / base_later if base_later else 0.0,
        "urgent_preserved": all(
            r["urgent_baseline"] == r["urgent_memory"] for r in rows
        ),
    }
