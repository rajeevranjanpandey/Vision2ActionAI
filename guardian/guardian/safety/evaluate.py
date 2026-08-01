"""Scoring harness for the fall channel.

The headline number is **not** sensitivity. It is false positives per week, because
that is the metric the roadmap declared as the kill-or-keep gate: nobody keeps a device
that cries wolf to their family.
"""

from __future__ import annotations

import numpy as np

from ..config import FallConfig
from .fall import FallDetectionReport, FallDetector
from .simulate_imu import ImuEpisode


def evaluate(episodes: list[ImuEpisode], cfg: FallConfig | None = None) -> FallDetectionReport:
    cfg = cfg or FallConfig()
    tp = fp = tn = fn = 0
    confidences: list[float] = []
    latencies: list[float] = []
    per_class: dict[str, dict[str, float]] = {}

    for ep in episodes:
        det = FallDetector(cfg)
        events = det.run(ep.samples)
        fired = len(events) > 0
        bucket = per_class.setdefault(ep.label, {"n": 0.0, "fired": 0.0})
        bucket["n"] += 1
        bucket["fired"] += 1.0 if fired else 0.0

        if ep.is_fall:
            if fired:
                tp += 1
                confidences.append(events[0].confidence)
                latencies.append((events[0].t_s - ep.onset_s) * 1000.0
                                 + cfg.stillness_window_ms)
            else:
                fn += 1
        else:
            if fired:
                fp += 1
            else:
                tn += 1

    n_falls, n_adls = tp + fn, fp + tn
    for label, b in per_class.items():
        b["rate"] = b["fired"] / b["n"] if b["n"] else 0.0

    return FallDetectionReport(
        sensitivity=tp / n_falls if n_falls else 0.0,
        specificity=tn / n_adls if n_adls else 0.0,
        false_positives=fp,
        false_negatives=fn,
        n_falls=n_falls,
        n_adls=n_adls,
        mean_confidence=float(np.mean(confidences)) if confidences else 0.0,
        median_detect_latency_ms=float(np.median(latencies)) if latencies else 0.0,
        per_class=per_class,
    )


def threshold_sweep(
    episodes: list[ImuEpisode],
    impact_values: list[float],
    base: FallConfig | None = None,
) -> list[dict[str, float]]:
    """Sensitivity / specificity as the impact threshold moves.

    The operating point is chosen off this curve, not off the best F1 -- the shipping
    rule is "highest specificity at sensitivity >= 0.95", because a missed fall still
    leaves the wearer with a phone, while a false alarm costs the family's trust.
    """
    rows: list[dict[str, float]] = []
    for g in impact_values:
        cfg = FallConfig(**{**(base or FallConfig()).__dict__, "impact_g": g})
        rep = evaluate(episodes, cfg)
        rows.append({
            "impact_g": g,
            "sensitivity": rep.sensitivity,
            "specificity": rep.specificity,
            "false_positives_per_week": rep.false_positives_per_week,
        })
    return rows
