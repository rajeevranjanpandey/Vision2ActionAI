"""Mode-aware context switching -- roadmap item 4.

One risk-cone geometry cannot be right for a train platform, a signalised crossing, and
a supermarket aisle. On a platform, a metre of lateral error is a fall onto the track; in
an aisle, the same cone fires at every shopper and the device gets switched off.

The design constraint is that this must not become a second perception system. The
classifier is a keyword scorer over tags the slow-path VLM is already producing at
0.5-1 Hz -- no new model, no new inference cost, and nothing on the fast path except
reading a mode flag and the profile it selects.

The safety asymmetry is in the switching rule, not in the classifier:

* moving to a **more conservative** mode happens on the first plausible observation;
* moving to a **less conservative** mode requires sustained agreement (``dwell_s``);
* an unrecognised scene falls back to sidewalk, never to the relaxed indoor profile.

That way a misclassification costs false alarms, which is recoverable, instead of costing
the platform-edge margin, which is not.
"""

from .classifier import ContextClassifier, ContextObservation, ContextState
from .modes import Mode, RiskProfile, conservatism_rank, profile_for

__all__ = [
    "ContextClassifier",
    "ContextObservation",
    "ContextState",
    "Mode",
    "RiskProfile",
    "conservatism_rank",
    "profile_for",
]
