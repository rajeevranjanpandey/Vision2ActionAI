"""Geo-pinned static-hazard store with a visual fingerprint.

Association is deliberately conjunctive: a stored pin is only re-identified when the GPS
fix agrees *and* the appearance fingerprint agrees *and* the class agrees. Consumer GPS
under buildings is routinely 8-15 m out, so position alone would merge a kerb with a
bollard on the far pavement; appearance alone would merge every A-board in the city.

Nothing in this module stores an image. A fingerprint is a short unit-norm descriptor,
which is what lets the whole cache sit in a JSON file the user can delete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

FINGERPRINT_DIM = 16
_EARTH_R_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres. Accurate to well under a metre at city scale."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return float(2.0 * _EARTH_R_M * math.asin(min(1.0, math.sqrt(a))))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return float((math.degrees(math.atan2(y, x)) + 360.0) % 360.0)


def angular_delta_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two headings, in [0, 180]."""
    return float(abs((a - b + 180.0) % 360.0 - 180.0))


def normalise(vec: Iterable[float]) -> np.ndarray:
    v = np.asarray(list(vec), dtype=np.float64).ravel()
    if v.size != FINGERPRINT_DIM:
        raise ValueError(f"fingerprint must be {FINGERPRINT_DIM}-D, got {v.size}")
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


# --------------------------------------------------------------------------- records


@dataclass(slots=True)
class HazardObservation:
    """One slow-path sighting of something the VLM called static."""

    lat: float
    lon: float
    class_name: str
    fingerprint: np.ndarray
    radius_m: float = 0.6
    t_s: float = 0.0
    gps_accuracy_m: float = 8.0

    def __post_init__(self) -> None:
        self.fingerprint = normalise(self.fingerprint)


@dataclass(slots=True)
class StaticHazard:
    """A remembered pin. Position is a running mean weighted by GPS accuracy."""

    id: str
    lat: float
    lon: float
    class_name: str
    fingerprint: np.ndarray
    radius_m: float = 0.6
    confirmations: int = 1
    contradictions: int = 0
    first_seen_s: float = 0.0
    last_seen_s: float = 0.0
    _weight: float = 1.0

    @property
    def confidence(self) -> float:
        """Laplace-smoothed hit rate. A single sighting is never trusted at 1.0 --
        one detection of a delivery van parked on the pavement must not become a
        permanent map feature."""
        return float(
            (self.confirmations + 1.0) / (self.confirmations + self.contradictions + 3.0)
        )

    @property
    def established(self) -> bool:
        """Enough independent passes to be treated as route furniture."""
        return self.confirmations >= 3 and self.contradictions <= self.confirmations

    def merge(self, obs: HazardObservation) -> None:
        w = 1.0 / max(obs.gps_accuracy_m, 1.0) ** 2
        total = self._weight + w
        self.lat = (self.lat * self._weight + obs.lat * w) / total
        self.lon = (self.lon * self._weight + obs.lon * w) / total
        self._weight = total
        # Fingerprints are averaged then renormalised: appearance drifts with season and
        # light, and a frozen first-sighting descriptor stops matching by winter.
        self.fingerprint = normalise(
            0.85 * self.fingerprint + 0.15 * obs.fingerprint
        )
        self.radius_m = max(self.radius_m, obs.radius_m)
        self.confirmations += 1
        self.last_seen_s = obs.t_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "lat": self.lat,
            "lon": self.lon,
            "class_name": self.class_name,
            "fingerprint": [round(float(v), 6) for v in self.fingerprint],
            "radius_m": self.radius_m,
            "confirmations": self.confirmations,
            "contradictions": self.contradictions,
            "first_seen_s": self.first_seen_s,
            "last_seen_s": self.last_seen_s,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StaticHazard":
        return cls(
            id=str(d["id"]),
            lat=float(d["lat"]),
            lon=float(d["lon"]),
            class_name=str(d["class_name"]),
            fingerprint=normalise(d["fingerprint"]),
            radius_m=float(d.get("radius_m", 0.6)),
            confirmations=int(d.get("confirmations", 1)),
            contradictions=int(d.get("contradictions", 0)),
            first_seen_s=float(d.get("first_seen_s", 0.0)),
            last_seen_s=float(d.get("last_seen_s", 0.0)),
        )


@dataclass(slots=True)
class HazardPrior:
    """What the fast path receives: a pin, where it is, and how much to trust it."""

    hazard: StaticHazard
    distance_m: float
    bearing_offset_deg: float
    prior: float

    @property
    def established(self) -> bool:
        return self.hazard.established


# ---------------------------------------------------------------------------- store


@dataclass
class MemoryParams:
    merge_radius_m: float = 18.0
    fingerprint_min_cos: float = 0.72
    lookahead_m: float = 25.0
    heading_cone_deg: float = 60.0
    # Consumer GNSS pose noise. Bearing uncertainty to a pin scales as ~sigma/range, so a
    # fixed cone is wrong: at four metres a five-metre pose error can put a pin that is
    # directly ahead anywhere in the hemisphere. The cone widens as range shrinks.
    pose_sigma_m: float = 5.0
    forget_after_s: float = 90.0 * 24 * 3600.0
    min_confidence: float = 0.25
    cell_deg: float = 0.0009  # ~100 m; keeps the candidate scan O(1) per query


class HazardMemory:
    """Local, single-user store of static hazards.

    Lookup is a spatial-hash scan over at most nine ~100 m cells, so a query costs a few
    microseconds regardless of how many hazards the user has accumulated. That bound is
    the reason the fast path is allowed to read it at all.
    """

    def __init__(self, params: MemoryParams | None = None) -> None:
        self.params = params or MemoryParams()
        self._hazards: dict[str, StaticHazard] = {}
        self._cells: dict[tuple[int, int], set[str]] = {}
        self._next_id = 1

    # ------------------------------------------------------------------ indexing

    def _cell_of(self, lat: float, lon: float) -> tuple[int, int]:
        c = self.params.cell_deg
        return (int(math.floor(lat / c)), int(math.floor(lon / c)))

    def _index(self, h: StaticHazard) -> None:
        self._cells.setdefault(self._cell_of(h.lat, h.lon), set()).add(h.id)

    def _deindex(self, h: StaticHazard) -> None:
        cell = self._cell_of(h.lat, h.lon)
        bucket = self._cells.get(cell)
        if bucket:
            bucket.discard(h.id)
            if not bucket:
                del self._cells[cell]

    def _nearby_ids(self, lat: float, lon: float) -> list[str]:
        ci, cj = self._cell_of(lat, lon)
        out: list[str] = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                out.extend(self._cells.get((ci + di, cj + dj), ()))
        return out

    # ------------------------------------------------------------------- mutation

    def observe(self, obs: HazardObservation) -> StaticHazard:
        """Fold a sighting in, merging with an existing pin when all three tests agree."""
        match = self._match(obs)
        if match is not None:
            self._deindex(match)
            match.merge(obs)
            self._index(match)
            return match

        hz = StaticHazard(
            id=f"h{self._next_id:04d}",
            lat=obs.lat,
            lon=obs.lon,
            class_name=obs.class_name,
            fingerprint=obs.fingerprint,
            radius_m=obs.radius_m,
            first_seen_s=obs.t_s,
            last_seen_s=obs.t_s,
            _weight=1.0 / max(obs.gps_accuracy_m, 1.0) ** 2,
        )
        self._next_id += 1
        self._hazards[hz.id] = hz
        self._index(hz)
        return hz

    def _match(self, obs: HazardObservation) -> StaticHazard | None:
        p = self.params
        best: StaticHazard | None = None
        best_d = float("inf")
        for hid in self._nearby_ids(obs.lat, obs.lon):
            h = self._hazards[hid]
            if h.class_name != obs.class_name:
                continue
            d = haversine_m(h.lat, h.lon, obs.lat, obs.lon)
            if d > p.merge_radius_m:
                continue
            if cosine(h.fingerprint, obs.fingerprint) < p.fingerprint_min_cos:
                continue
            if d < best_d:
                best, best_d = h, d
        return best

    def contradict(self, hazard_id: str) -> None:
        """Record a pass where the pin should have been visible and was not.

        Contradiction is how a repaired pavement leaves the map. Without it a memory
        layer is a ratchet that only ever accumulates phantom hazards.
        """
        h = self._hazards.get(hazard_id)
        if h is not None:
            h.contradictions += 1

    def prune(self, now_s: float) -> int:
        """Drop stale or discredited pins. Returns the number removed."""
        p = self.params
        doomed = [
            h for h in self._hazards.values()
            if h.confidence < p.min_confidence or now_s - h.last_seen_s > p.forget_after_s
        ]
        for h in doomed:
            self._deindex(h)
            del self._hazards[h.id]
        return len(doomed)

    # ---------------------------------------------------------------------- query

    def query(self, lat: float, lon: float, heading_deg: float) -> list[HazardPrior]:
        """Pins ahead of the walker, nearest first.

        Only what is in front matters -- a pothole already passed is not a prior, it is
        trivia, and feeding it forward is how a memory layer becomes a nuisance.
        """
        p = self.params
        out: list[HazardPrior] = []
        for hid in self._nearby_ids(lat, lon):
            h = self._hazards[hid]
            d = haversine_m(lat, lon, h.lat, h.lon)
            if d > p.lookahead_m:
                continue
            off = angular_delta_deg(bearing_deg(lat, lon, h.lat, h.lon), heading_deg)
            cone = min(150.0, p.heading_cone_deg * (1.0 + p.pose_sigma_m / max(d, 1.0)))
            if off > cone:
                continue
            # Prior decays with distance and with off-axis angle; confidence scales it.
            geometry = (1.0 - d / p.lookahead_m) * (1.0 - off / cone)
            out.append(
                HazardPrior(
                    hazard=h,
                    distance_m=d,
                    bearing_offset_deg=off,
                    prior=float(np.clip(h.confidence * geometry, 0.0, 1.0)),
                )
            )
        out.sort(key=lambda hp: hp.distance_m)
        return out

    def __len__(self) -> int:
        return len(self._hazards)

    @property
    def hazards(self) -> list[StaticHazard]:
        return sorted(self._hazards.values(), key=lambda h: h.id)

    # ------------------------------------------------------------------ cache i/o

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "hazards": [h.to_dict() for h in self.hazards]}

    @classmethod
    def from_dict(cls, d: dict[str, Any], params: MemoryParams | None = None) -> "HazardMemory":
        mem = cls(params)
        for raw in d.get("hazards", []):
            h = StaticHazard.from_dict(raw)
            mem._hazards[h.id] = h
            mem._index(h)
            n = int(h.id[1:]) if h.id[1:].isdigit() else 0
            mem._next_id = max(mem._next_id, n + 1)
        return mem


# ------------------------------------------------------------------------- the gate


@dataclass(slots=True)
class GateDecision:
    announce: bool
    style: str      # "full" | "known" | "suppressed"
    reason: str
    prior: float = 0.0


class RepeatAlertGate:
    """Downgrades repeat warnings about established static pins.

    The safety contract is the same one the learned head lives under, restated for a map
    instead of a model:

    * urgent geometry is never touched -- a pin cannot talk the device out of a 1.2 s TTC;
    * only *established* pins (three-plus confirmations, not contradicted) qualify;
    * the first encounter on any pass is always announced in full. The saving comes from
      the second and third announcement of the same kerb during one approach, which is
      what actually generates the repeat-route false-alarm load.
    """

    def __init__(
        self,
        min_prior: float = 0.22,
        urgent_ttc_s: float = 1.5,
        re_announce_after_s: float = 45.0,
    ) -> None:
        self.min_prior = float(min_prior)
        self.urgent_ttc_s = float(urgent_ttc_s)
        self.re_announce_after_s = float(re_announce_after_s)
        self._last_announced: dict[str, float] = {}

    def reset_pass(self) -> None:
        self._last_announced.clear()

    def decide(
        self,
        hazard_id: str | None,
        prior: float,
        ttc_s: float,
        established: bool,
        t_s: float,
    ) -> GateDecision:
        if ttc_s <= self.urgent_ttc_s:
            return GateDecision(True, "full", "urgent geometry: gate does not apply", prior)
        if hazard_id is None or not established or prior < self.min_prior:
            return GateDecision(True, "full", "unknown or unestablished hazard", prior)

        last = self._last_announced.get(hazard_id)
        if last is None or t_s - last > self.re_announce_after_s:
            self._last_announced[hazard_id] = t_s
            return GateDecision(True, "full", "first encounter this pass", prior)

        return GateDecision(False, "known", "established pin already announced", prior)
