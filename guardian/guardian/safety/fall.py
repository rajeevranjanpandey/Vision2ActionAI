"""IMU fall / impact classifier.

Design note, because this is the part a reviewer will push on: fall detection is not a
detection problem, it is a *false-positive* problem. Sensitivity is easy -- a 2 g
threshold catches essentially every fall and also every time the wearer drops into a
chair. The adoption gate is false alarms per week, because a device that phones a
family member when its wearer sat down heavily gets switched off within a fortnight.

So the classifier is a **conjunctive** three-phase template rather than a single
threshold, mirroring the accepted threshold-based literature (Bourke's SVM template,
Kangas' phase decomposition) rather than a learned model we have no field data to train:

1. **Free-fall (optional).** Signal-vector magnitude drops below ~0.6 g -- the body is
   accelerating downward under gravity. Present in trips and syncope, largely absent in
   a controlled sit-down.
2. **Impact (required).** A peak above ~2.4 g within 800 ms of that dip.
3. **Post-impact posture + stillness (required).** The gravity vector has rotated by
   tens of degrees relative to the pre-event orientation, and the wearer is *still*.
   This is the discriminator that a stumble-and-recover or a heavy sit fails: both
   produce impacts, but the wearer stays upright and keeps moving.

The rule is: impact AND stillness AND (free-fall OR posture change). Requiring
stillness is what buys the specificity; making free-fall and posture alternatives is
what keeps sensitivity on falls the wearer breaks with their hands.

Everything is pure numpy and runs on a 50 Hz stream in a few microseconds per sample.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..config import FallConfig

G = 9.80665


@dataclass(slots=True)
class ImuSample:
    """One IMU reading in the device frame.

    Accelerometer is in **g** (so gravity alone reads magnitude 1.0) and gyroscope in
    rad/s. The wearable's driver already normalises to these units for the ground-plane
    pitch estimator, so nothing new is needed on the sensor side.
    """

    t_s: float
    ax: float
    ay: float
    az: float
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0

    @property
    def svm(self) -> float:
        """Signal-vector magnitude, in g."""
        return float(math.sqrt(self.ax * self.ax + self.ay * self.ay + self.az * self.az))

    @property
    def gyro_magnitude(self) -> float:
        return float(math.sqrt(self.gx * self.gx + self.gy * self.gy + self.gz * self.gz))

    def acc_vector(self) -> np.ndarray:
        return np.array([self.ax, self.ay, self.az], dtype=np.float64)


class Phase(str, Enum):
    IDLE = "idle"
    CONFIRMING = "confirming"   # impact seen, waiting out the stillness window
    REFRACTORY = "refractory"   # just emitted, ignore the aftershocks


@dataclass(slots=True)
class FallEvent:
    """A confirmed fall, with the evidence that produced it.

    The evidence fields are not decoration: the trusted-contact message quotes them, and
    a wearer who cancels a false positive gives us a labelled negative with its features
    already attached (which is exactly the payload roadmap item 5 wants).
    """

    t_s: float                  # timestamp of the impact peak
    confidence: float           # 0..1
    peak_g: float
    freefall_ms: float
    tilt_deg: float             # gravity-vector rotation, pre vs post
    post_impact_std_g: float
    gyro_peak_rps: float

    @property
    def summary(self) -> str:
        return (
            f"impact {self.peak_g:.1f} g, free-fall {self.freefall_ms:.0f} ms, "
            f"posture change {self.tilt_deg:.0f} deg, then still"
        )


@dataclass(slots=True)
class DetectorTrace:
    """Per-sample diagnostic output, used by the evaluation harness and the web replay."""

    t_s: float
    svm: float
    phase: Phase
    event: FallEvent | None = None


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(cos))


class FallDetector:
    """Streaming three-phase fall classifier.

    Feed it samples in time order with :meth:`update`; it returns a :class:`FallEvent`
    on the sample where a fall is confirmed and ``None`` otherwise.
    """

    def __init__(self, cfg: FallConfig | None = None) -> None:
        self.cfg = cfg or FallConfig()
        history_s = max(
            4.0,
            (self.cfg.impact_window_ms + self.cfg.stillness_window_ms) / 1000.0 + 2.0,
        )
        self._maxlen = max(16, int(history_s * self.cfg.sample_rate_hz))
        self._buf: deque[ImuSample] = deque(maxlen=self._maxlen)
        self._phase = Phase.IDLE
        self._impact_t: float = 0.0
        self._impact_peak: float = 0.0
        self._freefall_ms: float = 0.0
        self._pre_gravity: np.ndarray = np.array([0.0, 0.0, 1.0])
        self._refractory_until: float = -1e9
        self.traces: list[DetectorTrace] = []

    # ------------------------------------------------------------------ properties

    @property
    def phase(self) -> Phase:
        return self._phase

    def reset(self) -> None:
        self._buf.clear()
        self._phase = Phase.IDLE
        self._refractory_until = -1e9
        self.traces.clear()

    # --------------------------------------------------------------------- stream

    def update(self, sample: ImuSample, record_trace: bool = False) -> FallEvent | None:
        self._buf.append(sample)
        event: FallEvent | None = None

        if self._phase is Phase.REFRACTORY:
            if sample.t_s >= self._refractory_until:
                self._phase = Phase.IDLE

        if self._phase is Phase.IDLE:
            if sample.svm >= self.cfg.impact_g:
                self._begin_confirmation(sample)
        elif self._phase is Phase.CONFIRMING:
            # Track the true peak of the impact, which may be a sample or two later.
            self._impact_peak = max(self._impact_peak, sample.svm)
            elapsed_ms = (sample.t_s - self._impact_t) * 1000.0
            if elapsed_ms >= self.cfg.stillness_window_ms:
                event = self._adjudicate(sample)
                self._phase = Phase.REFRACTORY
                self._refractory_until = sample.t_s + self.cfg.refractory_s

        if record_trace:
            self.traces.append(DetectorTrace(sample.t_s, sample.svm, self._phase, event))
        return event

    def run(self, samples: list[ImuSample], record_trace: bool = False) -> list[FallEvent]:
        """Convenience wrapper for offline evaluation."""
        return [e for s in samples if (e := self.update(s, record_trace)) is not None]

    # ----------------------------------------------------------------- internals

    def _begin_confirmation(self, sample: ImuSample) -> None:
        self._phase = Phase.CONFIRMING
        self._impact_t = sample.t_s
        self._impact_peak = sample.svm
        window_s = self.cfg.impact_window_ms / 1000.0
        pre = [s for s in self._buf if 0.0 < sample.t_s - s.t_s <= window_s]
        self._freefall_ms = self._longest_freefall_ms(pre)
        # Reference orientation: the second before the free-fall dip, where the wearer was
        # still upright and gravity was the only thing the accelerometer saw.
        settled = [
            s for s in self._buf
            if window_s < sample.t_s - s.t_s <= window_s + 1.0 and abs(s.svm - 1.0) < 0.35
        ]
        if not settled:
            settled = [s for s in self._buf if sample.t_s - s.t_s > window_s]
        self._pre_gravity = (
            np.mean([s.acc_vector() for s in settled], axis=0)
            if settled else np.array([0.0, 0.0, 1.0])
        )

    def _longest_freefall_ms(self, window: list[ImuSample]) -> float:
        dt_ms = 1000.0 / self.cfg.sample_rate_hz
        best = run = 0
        for s in window:
            run = run + 1 if s.svm <= self.cfg.freefall_g else 0
            best = max(best, run)
        return best * dt_ms

    def _adjudicate(self, now: ImuSample) -> FallEvent | None:
        post = [
            s for s in self._buf
            if 0.15 <= s.t_s - self._impact_t <= self.cfg.stillness_window_ms / 1000.0
        ]
        if len(post) < 3:
            return None

        svm = np.array([s.svm for s in post])
        std = float(svm.std())
        post_gravity = np.mean([s.acc_vector() for s in post], axis=0)
        tilt = _angle_between(self._pre_gravity, post_gravity)
        gyro_peak = max(
            (s.gyro_magnitude for s in self._buf
             if abs(s.t_s - self._impact_t) <= self.cfg.impact_window_ms / 1000.0),
            default=0.0,
        )

        still = std <= self.cfg.stillness_std_g
        freefell = self._freefall_ms >= self.cfg.freefall_min_ms
        toppled = tilt >= self.cfg.posture_change_deg

        if not (still and (freefell or toppled)):
            return None

        confidence = self._score(self._impact_peak, self._freefall_ms, tilt, std)
        if confidence < self.cfg.min_confidence:
            return None

        return FallEvent(
            t_s=self._impact_t,
            confidence=confidence,
            peak_g=self._impact_peak,
            freefall_ms=self._freefall_ms,
            tilt_deg=tilt,
            post_impact_std_g=std,
            gyro_peak_rps=gyro_peak,
        )

    def _score(self, peak_g: float, freefall_ms: float, tilt_deg: float, std_g: float) -> float:
        """Bounded evidence score in 0..1.

        Deliberately a transparent weighted sum rather than a learned probability: with
        no field dataset, a logistic regression here would be false precision, and this
        number is going to be read out to a caregiver.
        """
        impact = np.clip((peak_g - self.cfg.impact_g) / 1.6, 0.0, 1.0)
        ff = np.clip(freefall_ms / 300.0, 0.0, 1.0)
        posture = np.clip(tilt_deg / 90.0, 0.0, 1.0)
        quiet = np.clip(1.0 - std_g / max(self.cfg.stillness_std_g, 1e-6), 0.0, 1.0)
        score = 0.30 * impact + 0.20 * ff + 0.30 * posture + 0.20 * quiet
        return float(np.clip(0.35 + 0.65 * score, 0.0, 1.0))


@dataclass(slots=True)
class FallDetectionReport:
    """Scoring summary over a labelled set of IMU episodes."""

    sensitivity: float
    specificity: float
    false_positives: int
    false_negatives: int
    n_falls: int
    n_adls: int
    mean_confidence: float
    median_detect_latency_ms: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def false_positives_per_week(self) -> float:
        """Projected nuisance alerts per week.

        Assumes the literature-standard activity budget of ~120 fall-like daily events
        (sit-downs, stair descents, stumbles, jolts on transit) for an active walker.
        """
        rate = 1.0 - self.specificity
        return float(rate * 120.0 * 7.0)
