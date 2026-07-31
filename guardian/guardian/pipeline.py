"""The two-path orchestrator.

Fast path (10 Hz, hard latency budget):
    frame -> detect -> depth -> [risk-cone gate] -> SAM 2 -> backproject -> track
          -> assess -> policy -> haptic + earcon

Slow path (0.5-1 Hz, best effort, fully decoupled):
    frame -> VLM -> cached advisory string

The gate between depth and SAM 2 is what makes this run on an Orin: SAM 2 only ever sees
the handful of detections whose coarse position already puts them in the risk corridor.

If any fast-path stage overruns ``policy.degraded_after_ms``, the pipeline emits a
DEGRADED alert. Silent failure is the one unacceptable outcome for a safety device.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator, Optional

import numpy as np

from .audio.haptics import HapticFeedback
from .audio.tts import Speaker
from .config import GuardianConfig
from .language.vlm import SceneNarrator
from .perception.backprojection import backproject, depth_at_detection
from .perception.depth import DepthEstimator
from .perception.detector import Detector
from .perception.segmenter import Segmenter
from .risk import ttc as risk
from .risk.policy import AlertPolicy
from .tracking.tracker import Tracker
from .types import Alert, Detection, FrameResult, Observation

logger = logging.getLogger(__name__)


class GuardianPipeline:
    """End-to-end predictive assistance pipeline."""

    def __init__(self, cfg: GuardianConfig, headless: bool = False) -> None:
        self.cfg = cfg
        self.headless = headless

        self.detector = Detector(cfg.detector)
        self.depth = DepthEstimator(cfg.camera, cfg.depth)
        self.segmenter = Segmenter(cfg.segmenter)
        self.tracker = Tracker(cfg.tracking)
        self.policy = AlertPolicy(cfg.policy, cfg.risk)
        self.narrator = SceneNarrator(cfg.vlm)

        self.speaker: Optional[Speaker] = None
        self.haptics: Optional[HapticFeedback] = None
        if not headless:
            self.speaker = Speaker(cfg.tts)
            self.haptics = HapticFeedback(cfg.haptics)

        self._frame_index = 0
        self._last_ts: Optional[float] = None
        self._last_vlm_submit = 0.0

    # ------------------------------------------------------------------ public

    def start(self) -> None:
        self.narrator.start()
        if self.speaker:
            self.speaker.start()

    def stop(self) -> None:
        self.narrator.stop()
        if self.speaker:
            self.speaker.stop()
        if self.haptics:
            self.haptics.close()

    def reset(self) -> None:
        self.tracker.reset()
        self.policy.reset()
        self._frame_index = 0
        self._last_ts = None

    def process(
        self,
        frame_rgb: np.ndarray,
        timestamp_s: Optional[float] = None,
        ego_speed_mps: float = 0.0,
        yaw_rate_rps: float = 0.0,
        pitch_rad: Optional[float] = None,
    ) -> FrameResult:
        """Run one fast-path tick. Returns everything needed for logging and evaluation."""
        now = time.monotonic() if timestamp_s is None else timestamp_s
        dt = 1.0 / self.cfg.camera.fps if self._last_ts is None else max(now - self._last_ts, 1e-3)
        self._last_ts = now

        latency: dict[str, float] = {}

        with _timed(latency, "detect"):
            detections = self.detector(frame_rgb)

        with _timed(latency, "depth"):
            depth_result = self.depth(frame_rgb, pitch_rad)

        with _timed(latency, "gate"):
            gated = self._gate_by_coarse_risk(detections, depth_result.metric)

        with _timed(latency, "segment"):
            masks = self.segmenter.segment(frame_rgb, gated) if gated else {}

        with _timed(latency, "backproject"):
            observations = self._lift(detections, gated, masks, depth_result.metric)

        with _timed(latency, "track"):
            tracks = self.tracker.step(observations, dt, ego_speed_mps, yaw_rate_rps)

        with _timed(latency, "assess"):
            hazards = risk.assess(tracks, self.cfg.risk)

        elapsed_ms = sum(latency.values())
        degraded = elapsed_ms > self.cfg.policy.degraded_after_ms

        # Depth scale collapse means our metres are not metres. Treat as degraded rather
        # than quietly emitting confident nonsense.
        if depth_result.scale_confidence < 0.15:
            logger.debug("Ground-plane confidence collapsed (%.2f)", depth_result.scale_confidence)
            degraded = True

        alert = self.policy.decide(
            hazards, now, degraded=degraded, advisory=self.narrator.advisory()
        )
        if alert is not None:
            self._emit(alert)

        self._maybe_submit_to_vlm(frame_rgb, now)

        self._frame_index += 1
        return FrameResult(
            frame_index=self._frame_index,
            timestamp_s=now,
            detections=detections,
            tracks=tracks,
            hazards=hazards,
            alert=alert,
            depth_scale=depth_result.scale,
            stage_latency_ms=latency,
        )

    def handle_question(self, frame_rgb: np.ndarray, question: str) -> str:
        """User-initiated query. Runs on the slow path; never blocks the fast path."""
        answer = self.narrator.answer(frame_rgb, question)
        if self.speaker:
            self.speaker.say(answer)
        return answer

    # ----------------------------------------------------------------- private

    def _gate_by_coarse_risk(
        self, detections: list[Detection], metric_depth: np.ndarray
    ) -> list[Detection]:
        """Cheap pre-filter deciding who is worth a SAM 2 prompt.

        Uses the box-patch depth (no mask yet) to place each detection roughly, keeps
        those inside a generously widened corridor, and ranks by proximity. Being
        generous here is deliberate: a missed gate is a missed hazard, whereas an extra
        prompt only costs a few milliseconds.
        """
        scored: list[tuple[float, Detection]] = []
        fx, _, cx, _ = self.cfg.camera.intrinsics
        for det in detections:
            rng = depth_at_detection(det, metric_depth, None)
            if rng is None or rng > self.cfg.risk.cone_range_m:
                continue
            x_m = (det.cx - cx) * rng / fx
            half = risk.cone_half_width_at(rng, self.cfg.risk) * 1.8 + 0.5
            if abs(x_m) > half:
                continue
            scored.append((rng, det))

        scored.sort(key=lambda pair: pair[0])
        return [det for _, det in scored[: self.cfg.segmenter.max_prompts_per_frame]]

    def _lift(
        self,
        detections: list[Detection],
        gated: list[Detection],
        masks: dict[int, np.ndarray],
        metric_depth: np.ndarray,
    ) -> list[Observation]:
        """Back-project everything; gated detections get their SAM 2 mask attached."""
        mask_by_id = {id(gated[i]): m for i, m in masks.items() if i < len(gated)}
        out: list[Observation] = []
        for det in detections:
            mask = mask_by_id.get(id(det))
            rng = depth_at_detection(det, metric_depth, mask)
            if rng is None:
                continue
            out.append(backproject(det, rng, self.cfg.camera, mask))
        return out

    def _emit(self, alert: Alert) -> None:
        """Haptic first, earcon second, speech last -- strictly in latency order."""
        if self.haptics:
            self.haptics.emit(alert.level, alert.direction)
        if self.speaker:
            self.speaker.earcon(alert.level, alert.direction)
            text = alert.utterance
            if alert.advisory and alert.level.value in ("info",):
                text = f"{text} {alert.advisory}"
            self.speaker.say(text, alert.level)
        logger.info("[%s] %s", alert.level.value, alert.utterance)

    def _maybe_submit_to_vlm(self, frame_rgb: np.ndarray, now: float) -> None:
        if not self.cfg.vlm.enabled:
            return
        if now - self._last_vlm_submit < self.cfg.vlm.interval_s:
            return
        self._last_vlm_submit = now
        self.narrator.submit(frame_rgb)


@contextmanager
def _timed(sink: dict[str, float], key: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        sink[key] = (time.perf_counter() - start) * 1000.0
