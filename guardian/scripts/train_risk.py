#!/usr/bin/env python3
"""Train, calibrate, and ablate the learned risk head end to end.

    python scripts/train_risk.py --clips 800 --seed 0 --out artifacts/risk_head.npz

Runs in ~15 s on a laptop CPU with no downloads. Emits:

* ``artifacts/risk_head.npz``   -- the deep ensemble plus its calibration constants
* ``artifacts/report.json``     -- metrics, ablations, reliability bins, risk-control curve
* ``src/data/results.json``     -- the same report, wired into the project web page

Everything is seeded, so the numbers on the page are the numbers this script produced.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from guardian.bench.simeval import run_ablations
from guardian.config import RiskConfig
from guardian.risk.calibration import (
    calibrate,
    expected_calibration_error,
    reliability_curve,
    risk_control_curve,
)
from guardian.risk.learned import LearnedRiskHead
from guardian.train.features import WINDOW, build_windows
from guardian.train.simulate import generate_dataset
from guardian.train.trainer import (
    TrainConfig,
    average_precision,
    group_split,
    roc_auc,
    save_ensemble,
    train_ensemble,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="conformal miss-rate budget for the firing threshold")
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out", type=Path, default=Path("artifacts/risk_head.npz"))
    ap.add_argument("--report", type=Path, default=Path("artifacts/report.json"))
    ap.add_argument("--web", type=Path, default=Path("../src/data/results.json"))
    args = ap.parse_args()

    cfg = RiskConfig()
    t0 = time.time()

    # 1 -- data ------------------------------------------------------------------
    print(f"simulating {args.clips} clips (seed {args.seed}) ...")
    train_clips = generate_dataset(args.clips, cfg, seed=args.seed)
    # Held-out evaluation clips come from a *different seed stream*, so no scenario
    # sample is shared with training even by coincidence.
    eval_clips = generate_dataset(max(200, args.clips // 3), cfg, seed=args.seed + 10_000)

    X, y, groups = build_windows(
        [c.features for c in train_clips],
        [c.labels for c in train_clips],
        WINDOW,
        [c.clip_id for c in train_clips],
    )
    print(f"  {X.shape[0]} windows, dim {X.shape[1]}, positive rate {y.mean():.3f}")

    # 2 -- fit -------------------------------------------------------------------
    tcfg = TrainConfig(seed=args.seed, ensemble=args.ensemble, epochs=args.epochs)
    models, report, holdout = train_ensemble(X, y, groups, tcfg)
    print(f"  ensemble of {len(models)} fitted in {time.time() - t0:.1f}s, "
          f"val AUC {report.best_auc:.4f}")

    # 3 -- calibrate on a split the models never saw ------------------------------
    y_val, p_val = holdout["y"], holdout["p"]
    # Halve the validation split: one half sets temperature + threshold, the other
    # reports them. Calibrating and reporting on the same rows is self-congratulation.
    cal_mask, test_mask = group_split(
        np.array([f"v{i // 64}" for i in range(len(y_val))], dtype=object), 0.5, seed=args.seed
    )
    artefact = calibrate(y_val[cal_mask], p_val[cal_mask], alpha=args.alpha)
    print(f"  temperature {artefact.temperature:.3f}, threshold {artefact.threshold:.3f}, "
          f"ECE {artefact.ece_before:.4f} -> {artefact.ece_after:.4f}")

    save_ensemble(
        str(_ensure(args.out)),
        models,
        {
            "temperature": artefact.temperature,
            "threshold": artefact.threshold,
            "window": WINDOW,
            "alpha": args.alpha,
            "seed": args.seed,
        },
    )
    head = LearnedRiskHead.from_npz(args.out)

    # 4 -- ablations on unseen clips ----------------------------------------------
    print("running ablations ...")
    arms = run_ablations(eval_clips, head, cfg)
    for arm in arms:
        print(f"  {arm.name:34s} recall {arm.recall:.3f}  "
              f"median lead {arm.median_lead_time_s:.2f}s  "
              f"p10 {arm.p10_lead_time_s:.2f}s  FA/km {arm.false_alarms_per_km:.2f}")

    from guardian.risk.calibration import apply_temperature

    p_test = apply_temperature(p_val[test_mask], artefact.temperature)
    y_test = y_val[test_mask]

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "train": {
            "clips": args.clips,
            "windows": int(X.shape[0]),
            "feature_dim": int(X.shape[1]),
            "positive_rate": round(float(y.mean()), 4),
            "ensemble": len(models),
            "val_auc": round(float(report.best_auc), 4),
            "val_ap": round(float(average_precision(y_test, p_test)), 4),
            "fit_seconds": round(time.time() - t0, 1),
        },
        "calibration": {
            **{k: round(float(v), 4) for k, v in artefact.to_dict().items()},
            "auc_holdout": round(float(roc_auc(y_test, p_test)), 4),
            "ece_holdout": round(float(expected_calibration_error(y_test, p_test)), 4),
            "reliability": [
                {
                    "confidence": round(b.mean_confidence, 4),
                    "empirical": round(b.empirical_rate, 4),
                    "count": b.count,
                }
                for b in reliability_curve(y_test, p_test)
            ],
            "risk_control": [
                {
                    "alpha": t.alpha,
                    "threshold": round(t.threshold, 4),
                    "miss_rate": round(t.empirical_miss_rate, 4),
                    "alarm_rate": round(t.empirical_alarm_rate, 4),
                    "n": t.n_calibration,
                }
                for t in risk_control_curve(y_test, p_test)
            ],
        },
        "ablations": [arm.to_dict() for arm in arms],
        "lead_time_hist": _lead_time_hist(arms),
    }

    _ensure(args.report).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.report}")
    web = Path(__file__).resolve().parent.parent / args.web
    _ensure(web).write_text(json.dumps(payload, indent=2))
    print(f"wrote {web}")
    return 0


def _lead_time_hist(arms) -> list[dict]:
    """Histogram of lead times per arm, in 0.5 s buckets, for the web figure."""
    edges = np.arange(0.0, 3.5, 0.5)
    out = []
    for i, edge in enumerate(edges):
        row: dict[str, float | str] = {"bucket": f"{edge:.1f}-{edge + 0.5:.1f}s"}
        for arm in arms:
            lt = np.array(arm.lead_times_s)
            row[arm.name] = int(((lt >= edge) & (lt < edge + 0.5)).sum()) if lt.size else 0
        out.append(row)
        if i > 8:
            break
    return out


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
