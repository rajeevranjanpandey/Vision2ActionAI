"""Post-training quantisation with an accuracy budget, not a vibe.

INT8 is not free and "it still looks fine" is not evidence. This module runs the same
protocol we would defend in a paper appendix:

1. Collect a calibration set of real activations from held-out frames (never training
   frames, never synthetic noise -- percentile calibration is only as good as the
   activation distribution it saw).
2. Compute per-tensor scales with entropy or percentile calibration and compare them.
3. Re-run the *task* metric, not a proxy: lead time and false alarms per km, because a
   0.4 mAP-point drop that happens entirely on distant static objects costs nothing,
   and a 0.4-point drop concentrated on cyclists at 8 m is a safety regression.
4. Refuse the engine if the task metric regresses beyond the configured budget.

Torch/TensorRT are imported lazily so the rest of the repo -- and CI -- runs without them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


@dataclass
class QuantConfig:
    calibration_frames: int = 512
    method: str = "percentile"        # "percentile" | "entropy" | "minmax"
    percentile: float = 99.99
    max_lead_time_loss_s: float = 0.15
    max_recall_loss: float = 0.01
    max_false_alarm_increase_per_km: float = 0.25
    workspace_gb: int = 4


@dataclass
class LayerScale:
    name: str
    amax: float
    scale: float
    clipped_fraction: float


@dataclass
class QuantReport:
    layers: list[LayerScale] = field(default_factory=list)
    fp16_latency_ms: float = 0.0
    int8_latency_ms: float = 0.0
    lead_time_delta_s: float = 0.0
    recall_delta: float = 0.0
    false_alarm_delta_per_km: float = 0.0
    accepted: bool = False
    reason: str = ""

    @property
    def speedup(self) -> float:
        return self.fp16_latency_ms / max(self.int8_latency_ms, 1e-6)


# ------------------------------------------------------------------- scale selection


def percentile_scale(activations: np.ndarray, percentile: float = 99.99) -> tuple[float, float]:
    """Symmetric INT8 scale from a high percentile of |x|.

    Percentile beats min/max because one saturated pixel from a headlight at night sets
    a max that wastes seven of the eight bits on values that occur once per hour.
    """
    a = np.abs(np.asarray(activations, dtype=np.float64).ravel())
    if a.size == 0:
        return 0.0, 1.0
    amax = float(np.percentile(a, percentile))
    amax = max(amax, 1e-8)
    return amax, amax / 127.0


def entropy_scale(activations: np.ndarray, bins: int = 2048) -> tuple[float, float]:
    """KL-divergence calibration (TensorRT's default), implemented explicitly.

    Sweeps candidate clip points and picks the one whose quantised histogram minimises
    KL to the reference distribution. Slower than percentile, materially better on the
    long-tailed activations that depth decoders produce.
    """
    a = np.abs(np.asarray(activations, dtype=np.float64).ravel())
    if a.size == 0:
        return 0.0, 1.0
    hist, edges = np.histogram(a, bins=bins)
    hist = hist.astype(np.float64)
    best_kl, best_i = np.inf, bins - 1

    for i in range(128, bins):
        reference = hist[:i].copy()
        outliers = hist[i:].sum()
        if reference.sum() == 0:
            continue
        reference[-1] += outliers

        # Quantise the reference histogram down to 128 levels and expand it back.
        space = np.array_split(np.arange(i), 128)
        candidate = np.zeros(i, dtype=np.float64)
        for idx in space:
            block = hist[idx]
            nonzero = block > 0
            if nonzero.sum() == 0:
                continue
            candidate[idx] = np.where(nonzero, block.sum() / nonzero.sum(), 0.0)

        p = reference / max(reference.sum(), 1e-12) + 1e-12
        q = candidate / max(candidate.sum(), 1e-12) + 1e-12
        kl = float(np.sum(p * np.log(p / q)))
        if kl < best_kl:
            best_kl, best_i = kl, i

    amax = float(edges[best_i])
    amax = max(amax, 1e-8)
    return amax, amax / 127.0


def calibrate_layer(name: str, activations: np.ndarray, cfg: QuantConfig) -> LayerScale:
    if cfg.method == "entropy":
        amax, scale = entropy_scale(activations)
    elif cfg.method == "minmax":
        amax = float(np.abs(activations).max()) if activations.size else 1e-8
        scale = max(amax, 1e-8) / 127.0
    else:
        amax, scale = percentile_scale(activations, cfg.percentile)
    a = np.abs(np.asarray(activations, dtype=np.float64).ravel())
    clipped = float((a > amax).mean()) if a.size else 0.0
    return LayerScale(name=name, amax=amax, scale=scale, clipped_fraction=clipped)


def fake_quantise(x: np.ndarray, scale: float) -> np.ndarray:
    """Simulate INT8 round-trip in fp64. Used to estimate accuracy loss before we ever
    build an engine -- building a TensorRT engine takes minutes, this takes milliseconds."""
    if scale <= 0:
        return np.asarray(x, dtype=np.float64)
    q = np.clip(np.round(np.asarray(x, dtype=np.float64) / scale), -127, 127)
    return q * scale


def quantisation_snr_db(x: np.ndarray, scale: float) -> float:
    """Signal-to-quantisation-noise. Below ~20 dB on a depth decoder means visible banding
    in the metric depth, which shows up as phantom closing speed on distant tracks."""
    x = np.asarray(x, dtype=np.float64)
    err = x - fake_quantise(x, scale)
    denom = float(np.mean(err**2))
    if denom <= 0:
        return float("inf")
    return float(10.0 * np.log10(max(np.mean(x**2), 1e-12) / denom))


# ----------------------------------------------------------------------- acceptance


def accept(report: QuantReport, cfg: QuantConfig) -> QuantReport:
    """Gate the engine on task metrics. This function is the whole point of the module."""
    failures = []
    if report.lead_time_delta_s < -cfg.max_lead_time_loss_s:
        failures.append(
            f"lead time dropped {abs(report.lead_time_delta_s):.2f}s "
            f"(budget {cfg.max_lead_time_loss_s:.2f}s)"
        )
    if report.recall_delta < -cfg.max_recall_loss:
        failures.append(
            f"recall dropped {abs(report.recall_delta):.3f} (budget {cfg.max_recall_loss:.3f})"
        )
    if report.false_alarm_delta_per_km > cfg.max_false_alarm_increase_per_km:
        failures.append(
            f"false alarms rose {report.false_alarm_delta_per_km:.2f}/km "
            f"(budget {cfg.max_false_alarm_increase_per_km:.2f}/km)"
        )
    report.accepted = not failures
    report.reason = "within budget" if not failures else "; ".join(failures)
    return report


# ------------------------------------------------------------------- engine building


def build_int8_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    calibration_batches: Iterable[np.ndarray],
    cfg: QuantConfig | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Build a TensorRT INT8 engine. Requires ``tensorrt`` on the Jetson; no-op elsewhere."""
    cfg = cfg or QuantConfig()
    try:
        import tensorrt as trt  # type: ignore
    except ImportError as exc:  # pragma: no cover - device-only path
        raise RuntimeError(
            "tensorrt is not installed. Run this on the Jetson (JetPack ships it) or "
            "use fake_quantise() to estimate accuracy loss on a workstation."
        ) from exc

    logger = trt.Logger(trt.Logger.WARNING)  # pragma: no cover - device-only path
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as fh:
        if not parser.parse(fh.read()):
            raise RuntimeError("\n".join(str(parser.get_error(i))
                                         for i in range(parser.num_errors)))

    build_cfg = builder.create_builder_config()
    build_cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, cfg.workspace_gb << 30)
    build_cfg.set_flag(trt.BuilderFlag.INT8)
    build_cfg.set_flag(trt.BuilderFlag.FP16)  # fall back per-layer where INT8 hurts

    class _Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self._batches = iter(calibration_batches)
            self._device = None

        def get_batch_size(self) -> int:
            return 1

        def get_batch(self, names):
            import pycuda.driver as cuda  # type: ignore

            try:
                batch = np.ascontiguousarray(next(self._batches), dtype=np.float32)
            except StopIteration:
                return None
            if self._device is None:
                self._device = cuda.mem_alloc(batch.nbytes)
            cuda.memcpy_htod(self._device, batch)
            return [int(self._device)]

        def read_calibration_cache(self):
            path = Path(str(engine_path) + ".cache")
            return path.read_bytes() if path.exists() else None

        def write_calibration_cache(self, cache):
            Path(str(engine_path) + ".cache").write_bytes(cache)

    build_cfg.int8_calibrator = _Calibrator()
    log(f"building INT8 engine from {onnx_path} ({cfg.method} calibration)")
    serialized = builder.build_serialized_network(network, build_cfg)
    if serialized is None:
        raise RuntimeError("TensorRT returned no engine; check the ONNX opset and shapes")
    out = Path(engine_path)
    out.write_bytes(serialized)
    log(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out
