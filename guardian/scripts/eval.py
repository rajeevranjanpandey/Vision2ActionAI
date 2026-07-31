"""GuardianBench evaluation entry point.

    python scripts/eval.py --config configs/default.yaml --data data/guardianbench --split test
    python scripts/eval.py --ablate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardian.bench.evaluate import ablate, evaluate  # noqa: E402
from guardian.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate AI Guardian on GuardianBench")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/guardianbench")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="artifacts/report.json")
    ap.add_argument("--ablate", action="store_true", help="run the full ablation table")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    if args.ablate:
        results = ablate(cfg, args.data, args.split)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        header = f"{'variant':<18}{'lead p50':>10}{'lead p10':>10}{'recall':>9}{'FA/km':>9}"
        print("\n" + header)
        print("-" * len(header))
        for name, s in results.items():
            print(f"{name:<18}{s['median_lead_time_s']:>10.2f}{s['p10_lead_time_s']:>10.2f}"
                  f"{s['recall']:>9.3f}{s['false_alarms_per_km']:>9.2f}")
        return

    report = evaluate(cfg, args.data, args.split, args.out)
    print("\n" + report.render())


if __name__ == "__main__":
    main()
