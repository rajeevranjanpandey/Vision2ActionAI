"""Calibration and distribution-free risk control.

A probability that is not calibrated is a number that cannot enter a decision rule. A
head that outputs 0.9 must be wrong one time in ten, or the alert policy's thresholds
mean nothing and the false-alarm budget cannot be reasoned about.

Three things live here:

1. **Temperature scaling** (Guo et al., 2017) -- a one-parameter post-hoc fix for the
   overconfidence that focal loss plus ensembling still leaves behind. One parameter
   means it cannot overfit a small held-out split, which is the whole point.
2. **Expected calibration error + reliability bins** -- the diagnostic that proves it
   worked, reported with equal-mass bins so a spike of easy negatives cannot hide a
   badly calibrated head in the region that matters.
3. **Split conformal risk control** -- the part that makes this defensible rather than
   merely tuned. Given a held-out calibration split, it returns a threshold with a
   finite-sample, distribution-free guarantee on the miss rate: at most alpha of true
   hazards fall below it, with no assumption about the head being correct.

The conformal layer is what lets the safety case say something stronger than "it scored
well on our test set". It bounds the failure mode we actually care about -- a missed
hazard -- rather than average accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------- temperature scaling


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def nll(y: np.ndarray, p: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_temperature(y: np.ndarray, p: np.ndarray,
                    lo: float = 0.25, hi: float = 6.0, iters: int = 60) -> float:
    """Golden-section search for the temperature minimising held-out NLL.

    The NLL in log-temperature is unimodal for a fixed set of logits, so a derivative-
    free line search is both sufficient and immune to the step-size tuning that a
    gradient version would need.
    """
    z = _logit(p)
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = np.log(lo), np.log(hi)
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)

    def obj(log_t: float) -> float:
        return nll(y, 1.0 / (1.0 + np.exp(-z / np.exp(log_t))))

    fc, fd = obj(c), obj(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = obj(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = obj(d)
    return float(np.exp(0.5 * (a + b)))


def apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-_logit(p) / max(temperature, 1e-6)))


# ------------------------------------------------------------------------ diagnostics


@dataclass
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    empirical_rate: float


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[ReliabilityBin]:
    """Equal-mass bins. Equal-width bins put 90% of this dataset in the first bucket."""
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    if len(p) == 0:
        return []
    order = np.argsort(p)
    y, p = y[order], p[order]
    edges = np.array_split(np.arange(len(p)), min(n_bins, len(p)))
    bins: list[ReliabilityBin] = []
    for idx in edges:
        if len(idx) == 0:
            continue
        bins.append(
            ReliabilityBin(
                lower=float(p[idx[0]]),
                upper=float(p[idx[-1]]),
                count=int(len(idx)),
                mean_confidence=float(p[idx].mean()),
                empirical_rate=float(y[idx].mean()),
            )
        )
    return bins


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = reliability_curve(y, p, n_bins)
    n = float(len(np.asarray(p).ravel())) or 1.0
    return float(sum(b.count / n * abs(b.mean_confidence - b.empirical_rate) for b in bins))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2))


# ------------------------------------------------------------- conformal risk control


@dataclass
class ConformalThreshold:
    """A score threshold with a finite-sample miss-rate guarantee."""

    threshold: float
    alpha: float
    n_calibration: int
    empirical_miss_rate: float
    empirical_alarm_rate: float

    @property
    def guarantee(self) -> str:
        return (
            f"P(miss) <= {self.alpha:.2f} at 1-alpha coverage, "
            f"from n={self.n_calibration} exchangeable calibration hazards"
        )


def conformal_threshold(scores_positive: np.ndarray, alpha: float = 0.1) -> float:
    """Split-conformal threshold: the finite-sample-corrected alpha-quantile.

    ``scores_positive`` are the head's scores on calibration examples that really were
    hazards. Taking the ``floor(alpha (n+1)) / n`` empirical quantile guarantees that at
    most an alpha fraction of *future* exchangeable hazards score below the threshold --
    no Gaussianity, no model-correctness assumption, valid at n = 40.
    """
    s = np.sort(np.asarray(scores_positive, dtype=np.float64).ravel())
    n = len(s)
    if n == 0:
        return 0.0
    k = int(np.floor(alpha * (n + 1))) - 1
    if k < 0:
        return float(max(s[0] - 1e-6, 0.0))  # not enough data to spend any budget
    k = min(k, n - 1)
    return float(s[k])


def calibrate_alert_threshold(y: np.ndarray, p: np.ndarray,
                              alpha: float = 0.1) -> ConformalThreshold:
    """Pick the firing threshold from the miss-rate budget instead of from an F1 sweep.

    Ordering matters here: the safety requirement ("miss at most alpha of hazards") sets
    the threshold, and the false-alarm rate is then *reported*, not optimised. Doing it
    the other way round -- maximising F1 and quoting the recall that falls out -- is how
    assistive devices end up with a great benchmark number and a user who was not warned.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    thr = conformal_threshold(p[y == 1], alpha=alpha)
    fired = p >= thr
    miss = float(((y == 1) & ~fired).sum() / max((y == 1).sum(), 1))
    alarm = float(((y == 0) & fired).sum() / max((y == 0).sum(), 1))
    return ConformalThreshold(
        threshold=thr,
        alpha=alpha,
        n_calibration=int((y == 1).sum()),
        empirical_miss_rate=miss,
        empirical_alarm_rate=alarm,
    )


def risk_control_curve(y: np.ndarray, p: np.ndarray,
                       alphas: tuple[float, ...] = (0.02, 0.05, 0.1, 0.15, 0.2)
                       ) -> list[ConformalThreshold]:
    """The operating curve a reviewer will ask for: what does each safety budget cost?"""
    return [calibrate_alert_threshold(y, p, a) for a in alphas]


@dataclass
class CalibrationArtefact:
    """Everything the runtime needs to turn a raw head score into a decision."""

    temperature: float
    threshold: float
    alpha: float
    ece_before: float
    ece_after: float
    brier_after: float

    def to_dict(self) -> dict[str, float]:
        return {
            "temperature": self.temperature,
            "threshold": self.threshold,
            "alpha": self.alpha,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "brier_after": self.brier_after,
        }


def calibrate(y: np.ndarray, p_raw: np.ndarray, alpha: float = 0.1) -> CalibrationArtefact:
    """Full post-hoc pipeline: temperature, then conformal threshold on scaled scores."""
    temperature = fit_temperature(y, p_raw)
    p_cal = apply_temperature(p_raw, temperature)
    thr = calibrate_alert_threshold(y, p_cal, alpha=alpha)
    return CalibrationArtefact(
        temperature=temperature,
        threshold=thr.threshold,
        alpha=alpha,
        ece_before=expected_calibration_error(y, p_raw),
        ece_after=expected_calibration_error(y, p_cal),
        brier_after=brier_score(y, p_cal),
    )
