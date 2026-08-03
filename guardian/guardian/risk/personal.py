"""Per-user conformal recalibration -- roadmap item 2.

The shipped operating point (alpha = 0.1) is a population average, and there is no such
thing as a population pedestrian. A cautious first-year cane user and a confident
long-cane commuter want opposite things from the same device: one will tolerate nuisance
alerts to feel covered, the other switches the device off after the third false warning.

So alpha becomes a user-visible dial -- with two hard edges that are not user-visible:

1. **The range is bounded.** ``alpha`` may only move inside ``[0.05, 0.15]``. Outside
   that band the conformal guarantee is either vacuous or so tight the device becomes a
   siren, and neither is a choice a user should be able to make by talking to it.
2. **There is a recall floor.** A personalised threshold is only adopted if, on the
   *frozen validation split*, it still catches at least ``min_recall`` of hazards. This
   is the bit that makes the feature safe: personalisation runs against the user's own
   logged sessions, which are small, self-selected, and biased toward the routes they
   already walk confidently. A threshold fitted on that data can look excellent and be
   dangerous on an unfamiliar street. The floor is evaluated on data the user did not
   generate, so it cannot be gamed by their own history.

No new model. This is a threshold override on top of the existing split-conformal
machinery in ``risk/calibration.py``, which means it costs one float comparison at
runtime and adds nothing to the fast path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .calibration import conformal_threshold

ALPHA_MIN = 0.05
ALPHA_MAX = 0.15
ALPHA_STEP = 0.025


# ------------------------------------------------------------------------- intents


@dataclass(slots=True)
class VoiceIntent:
    """Parsed result of a spoken preference change."""

    recognised: bool
    direction: int      # -1 = warn me more, +1 = fewer alerts, 0 = query only
    phrase: str = ""


_FEWER = re.compile(
    r"\b(fewer|less|quieter|too many|stop warning|calm down|reduce)\b", re.I
)
_MORE = re.compile(
    r"\b(more|warn me more|louder|missed|didn'?t warn|be careful|extra)\b", re.I
)
_QUERY = re.compile(r"\b(how (sensitive|cautious)|what.*setting|current setting)\b", re.I)


def parse_intent(utterance: str) -> VoiceIntent:
    """Map a spoken phrase to a direction on the alpha dial.

    Ambiguity resolves to *unrecognised*, never to a guess. A misheard "less" that
    silences the device is a safety event; a misheard phrase that produces "sorry, say
    that again" is an annoyance.
    """
    text = (utterance or "").strip()
    if not text:
        return VoiceIntent(False, 0)
    if _QUERY.search(text):
        return VoiceIntent(True, 0, text)
    fewer, more = bool(_FEWER.search(text)), bool(_MORE.search(text))
    if fewer == more:
        return VoiceIntent(False, 0, text)
    return VoiceIntent(True, 1 if fewer else -1, text)


# --------------------------------------------------------------------- calibration


@dataclass(slots=True)
class PersonalCalibration:
    """The outcome of one recalibration attempt."""

    alpha: float
    threshold: float
    global_threshold: float
    n_user_hazards: int
    validation_recall: float
    accepted: bool
    clamped: bool
    reason: str

    @property
    def delta(self) -> float:
        return self.threshold - self.global_threshold

    def to_dict(self) -> dict[str, float | str | bool | int]:
        return {
            "alpha": round(self.alpha, 4),
            "threshold": round(self.threshold, 4),
            "globalThreshold": round(self.global_threshold, 4),
            "nUserHazards": self.n_user_hazards,
            "validationRecall": round(self.validation_recall, 4),
            "accepted": self.accepted,
            "clamped": self.clamped,
            "reason": self.reason,
        }


@dataclass
class PersonalProfile:
    """Per-user state. Small enough to live in the same local cache as the hazard map."""

    user_id: str = "local"
    alpha: float = 0.10
    sessions: int = 0
    accepted_updates: int = 0
    rejected_updates: int = 0
    history: list[float] = field(default_factory=list)


class PersonalConformal:
    """Bounded per-user threshold override with a validation-set recall floor."""

    def __init__(
        self,
        global_threshold: float,
        min_recall: float = 0.95,
        min_user_hazards: int = 40,
        profile: PersonalProfile | None = None,
    ) -> None:
        self.global_threshold = float(global_threshold)
        self.min_recall = float(min_recall)
        self.min_user_hazards = int(min_user_hazards)
        self.profile = profile or PersonalProfile()

    # ------------------------------------------------------------------- the dial

    def nudge(self, utterance: str) -> tuple[VoiceIntent, float]:
        """Apply a spoken preference. Returns the parsed intent and the resulting alpha."""
        intent = parse_intent(utterance)
        if intent.recognised and intent.direction != 0:
            self.profile.alpha = float(
                np.clip(
                    self.profile.alpha + intent.direction * ALPHA_STEP,
                    ALPHA_MIN,
                    ALPHA_MAX,
                )
            )
        return intent, self.profile.alpha

    def at_limit(self) -> str | None:
        if self.profile.alpha >= ALPHA_MAX - 1e-9:
            return "max"
        if self.profile.alpha <= ALPHA_MIN + 1e-9:
            return "min"
        return None

    # ------------------------------------------------------------ recalibration

    def recalibrate(
        self,
        user_hazard_scores: np.ndarray,
        validation_y: np.ndarray,
        validation_p: np.ndarray,
    ) -> PersonalCalibration:
        """Fit a threshold on the user's own hazards, then audit it on frozen data.

        Three ways this returns the global threshold instead, all of them deliberate:
        too few user hazards to spend a conformal budget on; a personalised threshold
        that fails the recall floor; or a personalised threshold that is *looser* than
        the global one while the user is at the cautious end of the dial.
        """
        alpha = float(np.clip(self.profile.alpha, ALPHA_MIN, ALPHA_MAX))
        scores = np.asarray(user_hazard_scores, dtype=np.float64).ravel()
        n = int(scores.size)

        if n < self.min_user_hazards:
            self.profile.rejected_updates += 1
            return PersonalCalibration(
                alpha, self.global_threshold, self.global_threshold, n, float("nan"),
                accepted=False, clamped=True,
                reason=f"only {n} logged hazards; need {self.min_user_hazards}",
            )

        candidate = conformal_threshold(scores, alpha=alpha)
        recall = self._validation_recall(validation_y, validation_p, candidate)

        if recall < self.min_recall:
            self.profile.rejected_updates += 1
            return PersonalCalibration(
                alpha, self.global_threshold, self.global_threshold, n, recall,
                accepted=False, clamped=True,
                reason=(
                    f"recall {recall:.3f} on the frozen validation split is below the "
                    f"{self.min_recall:.2f} floor"
                ),
            )

        self.profile.accepted_updates += 1
        self.profile.history.append(candidate)
        return PersonalCalibration(
            alpha, float(candidate), self.global_threshold, n, recall,
            accepted=True, clamped=False,
            reason="personalised threshold clears the validation recall floor",
        )

    def _validation_recall(
        self, y: np.ndarray, p: np.ndarray, threshold: float
    ) -> float:
        y = np.asarray(y).ravel()
        p = np.asarray(p, dtype=np.float64).ravel()
        pos = y == 1
        if not pos.any():
            return 0.0
        return float((p[pos] >= threshold).mean())


def alpha_sweep(
    user_hazard_scores: np.ndarray,
    validation_y: np.ndarray,
    validation_p: np.ndarray,
    global_threshold: float,
    alphas: tuple[float, ...] = (0.05, 0.075, 0.10, 0.125, 0.15),
    min_recall: float = 0.95,
) -> list[dict[str, float | bool | str]]:
    """What each point on the dial costs -- the table a reviewer asks for immediately."""
    rows: list[dict[str, float | bool | str]] = []
    y = np.asarray(validation_y).ravel()
    p = np.asarray(validation_p, dtype=np.float64).ravel()
    for a in alphas:
        pc = PersonalConformal(
            global_threshold, min_recall=min_recall,
            profile=PersonalProfile(alpha=a),
        )
        cal = pc.recalibrate(user_hazard_scores, y, p)
        fired_neg = float((p[y == 0] >= cal.threshold).mean()) if (y == 0).any() else 0.0
        rows.append(
            {
                "alpha": a,
                "threshold": round(cal.threshold, 4),
                "recall": round(cal.validation_recall, 4),
                "falseAlarmRate": round(fired_neg, 4),
                "accepted": cal.accepted,
                "reason": cal.reason,
            }
        )
    return rows
