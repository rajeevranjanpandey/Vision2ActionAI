"""Keyword scorer over slow-path VLM tags, with an asymmetric switching rule."""

from __future__ import annotations

from dataclasses import dataclass, field

from .modes import Mode, conservatism_rank

# Weights are hand-set and deliberately readable. A learned scene classifier would be
# more accurate and would also be a second thing that can fail silently at 0.5 Hz; the
# cost of being wrong here is bounded by the switching rule, not by the scorer.
KEYWORDS: dict[Mode, dict[str, float]] = {
    Mode.TRANSIT_PLATFORM: {
        "platform": 3.0,
        "platform edge": 4.0,
        "yellow line": 3.0,
        "tactile strip": 2.5,
        "railway": 2.5,
        "train": 2.0,
        "subway": 2.0,
        "track": 2.0,
        "station": 1.5,
        "turnstile": 1.0,
    },
    Mode.CROSSING: {
        "crosswalk": 3.5,
        "zebra crossing": 3.5,
        "pedestrian crossing": 3.5,
        "traffic light": 2.5,
        "kerb": 1.5,
        "curb": 1.5,
        "road": 1.5,
        "junction": 2.0,
        "traffic": 1.5,
        "vehicle": 1.0,
    },
    Mode.SIDEWALK: {
        "sidewalk": 3.0,
        "pavement": 3.0,
        "footpath": 2.5,
        "shopfront": 1.5,
        "park": 1.5,
        "pedestrians": 1.0,
        "corridor": 1.0,
        "aisle": 1.0,
    },
}

MIN_SCORE = 2.0  # below this the scene is "unrecognised", not "sidewalk by default"


@dataclass(slots=True)
class ContextObservation:
    t_s: float
    tags: tuple[str, ...]
    scores: dict[Mode, float]
    proposal: Mode | None


@dataclass(slots=True)
class ContextState:
    mode: Mode = Mode.SIDEWALK
    since_s: float = 0.0
    pending: Mode | None = None
    pending_since_s: float = 0.0
    switches: int = 0
    escalations: int = 0
    relaxations: int = 0
    history: list[tuple[float, Mode, str]] = field(default_factory=list)


def score_tags(tags: list[str] | tuple[str, ...]) -> dict[Mode, float]:
    text = " ".join(t.lower() for t in tags)
    out: dict[Mode, float] = {}
    for mode, table in KEYWORDS.items():
        out[mode] = float(sum(w for kw, w in table.items() if kw in text))
    return out


class ContextClassifier:
    """Turns a stream of VLM tag sets into a stable mode.

    Escalation (toward a more conservative profile) is immediate: one credible mention of
    a platform edge is enough, because the cost of being early is a wider cone and the
    cost of being late is a fall onto a track. Relaxation requires ``dwell_s`` of
    sustained disagreement with the current mode, which is what stops a single
    misclassified frame from dropping the platform floor while the user is still on the
    platform.
    """

    def __init__(self, dwell_s: float = 6.0, min_score: float = MIN_SCORE) -> None:
        self.dwell_s = float(dwell_s)
        self.min_score = float(min_score)
        self.state = ContextState()

    def observe(self, tags: list[str] | tuple[str, ...], t_s: float) -> ContextObservation:
        scores = score_tags(tags)
        best = max(scores, key=lambda m: scores[m])
        proposal: Mode | None = best if scores[best] >= self.min_score else None

        # A tie between a conservative and a relaxed reading resolves conservatively.
        tied = [m for m, s in scores.items() if abs(s - scores[best]) < 1e-9]
        if proposal is not None and len(tied) > 1:
            proposal = max(tied, key=conservatism_rank)

        if proposal is None:
            # Unrecognised: hold the current mode. Falling back to a relaxed default on
            # a low-confidence frame is precisely the failure this rule exists to stop.
            self.state.pending = None
            return ContextObservation(t_s, tuple(tags), scores, None)

        cur = self.state.mode
        if proposal == cur:
            self.state.pending = None
            return ContextObservation(t_s, tuple(tags), scores, proposal)

        if conservatism_rank(proposal) > conservatism_rank(cur):
            self._switch(proposal, t_s, "escalate: conservative modes apply immediately")
            self.state.escalations += 1
            return ContextObservation(t_s, tuple(tags), scores, proposal)

        # Relaxation path.
        if self.state.pending != proposal:
            self.state.pending = proposal
            self.state.pending_since_s = t_s
        elif t_s - self.state.pending_since_s >= self.dwell_s:
            self._switch(proposal, t_s, f"relax: {self.dwell_s:.0f}s sustained agreement")
            self.state.relaxations += 1
        return ContextObservation(t_s, tuple(tags), scores, proposal)

    def _switch(self, mode: Mode, t_s: float, reason: str) -> None:
        self.state.mode = mode
        self.state.since_s = t_s
        self.state.pending = None
        self.state.switches += 1
        self.state.history.append((t_s, mode, reason))

    @property
    def mode(self) -> Mode:
        return self.state.mode
