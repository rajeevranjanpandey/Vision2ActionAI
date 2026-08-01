#!/usr/bin/env python3
"""Score the fall channel and emit the numbers the project page displays.

    PYTHONPATH=. python scripts/eval_fall.py

Writes ``src/data/fall_detection.json``: the operating-point report, the per-activity
breakdown, the impact-threshold sweep, and a handful of decimated IMU traces the browser
replays through a TypeScript port of the same classifier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from guardian.config import FallConfig
from guardian.safety.evaluate import evaluate, threshold_sweep
from guardian.safety.fall import FallDetector
from guardian.safety.simulate_imu import generate_dataset, generate_episode

TRACE_CLASSES = [
    ("fall_forward", "Forward fall, hands not out"),
    ("fall_slow_slump", "Slow slump down a wall"),
    ("sit_heavy", "Dropping into a chair"),
    ("stumble_recover", "Stumble, recovered"),
    ("stairs_descent", "Descending stairs"),
]


def trace_for(label: str, seed: int, cfg: FallConfig) -> dict:
    rng = np.random.default_rng(seed)
    ep = generate_episode(label, rng)
    det = FallDetector(cfg)
    events = det.run(ep.samples, record_trace=True)
    stride = 2  # 50 Hz -> 25 Hz for transport; the classifier still ran at 50 Hz
    return {
        "label": label,
        "isFall": ep.is_fall,
        "onsetS": round(ep.onset_s, 3),
        "fired": bool(events),
        "event": (
            {
                "tS": round(events[0].t_s, 3),
                "confidence": round(events[0].confidence, 4),
                "peakG": round(events[0].peak_g, 3),
                "freefallMs": round(events[0].freefall_ms, 1),
                "tiltDeg": round(events[0].tilt_deg, 1),
                "postImpactStdG": round(events[0].post_impact_std_g, 4),
                "summary": events[0].summary,
            }
            if events else None
        ),
        "samples": [
            {"t": round(s.t_s, 3), "svm": round(s.svm, 4)}
            for s in ep.samples[::stride]
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="../src/data/fall_detection.json")
    args = ap.parse_args()

    cfg = FallConfig()
    episodes = generate_dataset(n_per_class=args.per_class, seed=args.seed)
    report = evaluate(episodes, cfg)
    sweep = threshold_sweep(episodes, [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.2, 3.6], cfg)

    payload = {
        "generatedBy": "guardian/scripts/eval_fall.py",
        "provenance": "simulated IMU episodes -- not worn-sensor field data",
        "seed": args.seed,
        "episodesPerClass": args.per_class,
        "config": {
            "sampleRateHz": cfg.sample_rate_hz,
            "freefallG": cfg.freefall_g,
            "impactG": cfg.impact_g,
            "postureChangeDeg": cfg.posture_change_deg,
            "stillnessWindowMs": cfg.stillness_window_ms,
            "stillnessStdG": cfg.stillness_std_g,
            "countdownS": cfg.countdown_s,
        },
        "report": {
            "sensitivity": round(report.sensitivity, 4),
            "specificity": round(report.specificity, 4),
            "falsePositives": report.false_positives,
            "falseNegatives": report.false_negatives,
            "nFalls": report.n_falls,
            "nAdls": report.n_adls,
            "meanConfidence": round(report.mean_confidence, 4),
            "medianDetectLatencyMs": round(report.median_detect_latency_ms, 1),
            "falsePositivesPerWeek": round(report.false_positives_per_week, 2),
        },
        "perClass": [
            {"label": k, "n": int(v["n"]), "fireRate": round(v["rate"], 4)}
            for k, v in sorted(report.per_class.items())
        ],
        "sweep": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
            for row in sweep
        ],
        "traces": [
            {**trace_for(label, 100 + i, cfg), "title": title}
            for i, (label, title) in enumerate(TRACE_CLASSES)
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    r = payload["report"]
    print(f"sensitivity {r['sensitivity']:.3f}  specificity {r['specificity']:.3f}  "
          f"FP/week {r['falsePositivesPerWeek']:.1f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
