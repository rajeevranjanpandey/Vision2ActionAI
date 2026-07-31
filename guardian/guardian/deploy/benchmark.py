"""Per-stage latency and thermal benchmarking on the target device.

Report p50/p95/p99 per stage, not the mean. The mean hides exactly the tail that causes a
missed warning, and the tail is the safety-relevant number.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np

from ..config import load_config
from ..pipeline import GuardianPipeline

logger = logging.getLogger(__name__)


def read_jetson_thermals() -> dict[str, float]:
    """Read Jetson thermal zones. Returns empty dict off-target."""
    zones: dict[str, float] = {}
    base = Path("/sys/devices/virtual/thermal")
    if not base.exists():
        return zones
    for zone in sorted(base.glob("thermal_zone*")):
        try:
            name = (zone / "type").read_text().strip()
            temp = int((zone / "temp").read_text().strip()) / 1000.0
            zones[name] = temp
        except OSError:
            continue
    return zones


def read_power_mw() -> float | None:
    """Total module power from the INA3221 rail, if present."""
    for path in Path("/sys/bus/i2c/drivers/ina3221").rglob("in1_input"):
        try:
            return float(path.read_text().strip())
        except OSError:
            continue
    return None


def run(config_path: str, frames: int = 300, warmup: int = 30) -> dict:
    cfg = load_config(config_path)
    pipeline = GuardianPipeline(cfg, headless=True)
    pipeline.start()

    h, w = cfg.camera.height, cfg.camera.width
    rng = np.random.default_rng(0)
    synthetic = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)

    per_stage: dict[str, list[float]] = {}
    totals: list[float] = []

    try:
        for i in range(frames + warmup):
            result = pipeline.process(synthetic, timestamp_s=i / cfg.camera.fps)
            if i < warmup:
                continue
            for stage, ms in result.stage_latency_ms.items():
                per_stage.setdefault(stage, []).append(ms)
            totals.append(result.total_latency_ms)
    finally:
        pipeline.stop()

    def stats(values: list[float]) -> dict[str, float]:
        arr = np.array(values)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()),
        }

    report = {
        "frames": frames,
        "stages": {name: stats(v) for name, v in per_stage.items()},
        "total": stats(totals),
        "achieved_hz": 1000.0 / max(float(np.median(totals)), 1e-6),
        "thermals_c": read_jetson_thermals(),
        "power_mw": read_power_mw(),
    }
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Benchmark the AI Guardian fast path")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--out", default="artifacts/latency.json")
    args = ap.parse_args()

    report = run(args.config, args.frames)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"\nfast path: {report['achieved_hz']:.1f} Hz "
          f"(p99 {report['total']['p99']:.1f} ms)\n")
    for stage, s in sorted(report["stages"].items(), key=lambda kv: -kv[1]["p50"]):
        print(f"  {stage:<12} p50 {s['p50']:6.1f}  p95 {s['p95']:6.1f}  p99 {s['p99']:6.1f} ms")
    if report["thermals_c"]:
        print("\nthermals:", ", ".join(f"{k} {v:.1f}C" for k, v in report["thermals_c"].items()))


if __name__ == "__main__":
    main()
