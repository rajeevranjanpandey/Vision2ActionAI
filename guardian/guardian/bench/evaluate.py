"""Offline evaluation over GuardianBench."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..config import GuardianConfig
from ..pipeline import GuardianPipeline
from ..types import Alert
from .dataset import Clip, GuardianBench
from .metrics import BenchmarkReport, ClipScore, aggregate, score_clip

logger = logging.getLogger(__name__)


def evaluate_clip(pipeline: GuardianPipeline, clip: Clip) -> ClipScore:
    """Replay one clip through the pipeline and score its alerts."""
    pipeline.reset()
    alerts: list[tuple[int, Alert]] = []

    for frame_index, frame_rgb in clip.frames():
        imu = clip.imu_at(frame_index)
        result = pipeline.process(
            frame_rgb,
            timestamp_s=frame_index / clip.fps,
            ego_speed_mps=imu.speed_mps,
            yaw_rate_rps=imu.yaw_rate_rps,
            pitch_rad=imu.pitch_rad,
        )
        if result.alert is not None:
            alerts.append((frame_index, result.alert))

    return score_clip(clip, alerts)


def evaluate(
    cfg: GuardianConfig,
    dataset_root: str | Path,
    split: str = "test",
    output: str | Path | None = None,
) -> BenchmarkReport:
    """Full-split evaluation. Runs headless: no audio, no haptics."""
    bench = GuardianBench(dataset_root)
    pipeline = GuardianPipeline(cfg, headless=True)
    pipeline.start()

    scores: list[ClipScore] = []
    try:
        for clip in bench.iter_split(split):
            logger.info("Evaluating %s (%.1f s, %d hazards)",
                        clip.clip_id, clip.duration_s, len(clip.hazards))
            scores.append(evaluate_clip(pipeline, clip))
    finally:
        pipeline.stop()

    report = aggregate(scores)
    if output is not None:
        payload = {
            "split": split,
            "summary": report.summary(),
            "per_clip": [
                {
                    "clip_id": s.clip_id,
                    "detected": s.detected,
                    "missed": s.missed,
                    "false_alarms": s.false_alarms,
                    "distance_m": s.distance_m,
                    "lead_times_s": s.lead_times_s,
                }
                for s in scores
            ],
        }
        Path(output).write_text(json.dumps(payload, indent=2))
        logger.info("Wrote %s", output)

    return report


def ablate(cfg: GuardianConfig, dataset_root: str | Path, split: str = "val") -> dict[str, dict]:
    """The ablation table the paper needs.

    Each row removes one component and re-measures lead time and false alarms. This is
    what turns "we built a system" into "we showed which parts matter".
    """
    import copy

    variants: dict[str, GuardianConfig] = {}

    full = copy.deepcopy(cfg)
    variants["full"] = full

    no_sam = copy.deepcopy(cfg)
    no_sam.segmenter.enabled = False
    variants["no_sam2"] = no_sam

    no_metric = copy.deepcopy(cfg)
    no_metric.depth.scale_mode = "none"
    variants["no_metric_scale"] = no_metric

    no_forecast = copy.deepcopy(cfg)
    no_forecast.risk.horizon_s = 0.3          # collapses prediction to reaction
    no_forecast.risk.ttc_warn_s = 0.25
    no_forecast.risk.ttc_urgent_s = 0.2
    variants["no_forecast"] = no_forecast

    no_vlm = copy.deepcopy(cfg)
    no_vlm.vlm.enabled = False
    variants["no_vlm"] = no_vlm

    results: dict[str, dict] = {}
    for name, variant in variants.items():
        logger.info("--- ablation: %s ---", name)
        results[name] = evaluate(variant, dataset_root, split).summary()
    return results
