#!/usr/bin/env python3
"""Export the trained ensemble to JSON so the project page can run it in the browser.

The web demo is not a mock-up: it evaluates the same weights, the same temperature, and
the same conformal threshold this repository trained, over the same scenario physics.
If the demo disagrees with the paper, one of them is wrong and we want to see it.

    PYTHONPATH=. python scripts/export_web_head.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from guardian.train.trainer import load_ensemble


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=Path("artifacts/risk_head.npz"))
    ap.add_argument("--out", type=Path, default=Path("../src/data/risk_head.json"))
    ap.add_argument("--members", type=int, default=3,
                    help="members to ship; the browser only needs enough spread for the "
                         "abstention band, and every member costs bandwidth")
    ap.add_argument("--precision", type=int, default=4)
    args = ap.parse_args()

    models, meta = load_ensemble(str(args.weights))
    keep = models[: args.members]

    payload = {
        "temperature": float(meta.get("temperature", np.array(1.0))),
        "threshold": float(meta.get("threshold", np.array(0.5))),
        "window": int(meta.get("window", np.array(8))),
        "members": [
            {
                "W": [np.round(w, args.precision).tolist() for w in m.W],
                "b": [np.round(b, args.precision).tolist() for b in m.b],
            }
            for m in keep
        ],
    }

    out = Path(__file__).resolve().parent.parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(keep)} members)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
