"""Federated near-miss feedback -- roadmap item 5.

The head is trained on simulated physics. Deployment surfaces failure modes simulation
cannot anticipate, and the obvious fix -- collect the video -- is not available: a
wearer's camera sees bystanders' faces and their own daily route, and that must not be
centralised at any price.

So the wearer flags the event, not the footage. Pressing the button (or saying "that was
a near miss") emits the 8-frame x 14-feature window the head *already computed* for that
track, a label, and nothing else. No frames, no audio, no coordinates, no timestamps
finer than the hour.

Two modules:

* :mod:`nearmiss` -- the on-device log, with a redaction pass and an explicit assertion
  that a payload contains only the declared numeric fields. Privacy claims in a thesis
  need a test behind them, not a paragraph.
* :mod:`federated` -- clipped, noised FedAvg over model deltas. Clipping bounds any one
  client's influence (which is also the poisoning defence); Gaussian noise on the sum
  gives the DP story its teeth.

MVP boundary from the roadmap: feature-vector export plus a manual retrain cycle. The
production federated infrastructure is v2; what is here is the algorithm and its audit.
"""

from .federated import (
    AggregationReport,
    ClientUpdate,
    aggregate,
    clip_delta,
    delta_norm,
)
from .nearmiss import (
    NearMissLog,
    NearMissRecord,
    RedactionError,
    assert_privacy_safe,
)
from .retrain import RetrainRound, simulate_retrain

__all__ = [
    "AggregationReport",
    "ClientUpdate",
    "NearMissLog",
    "NearMissRecord",
    "RedactionError",
    "RetrainRound",
    "aggregate",
    "assert_privacy_safe",
    "clip_delta",
    "delta_norm",
    "simulate_retrain",
]
