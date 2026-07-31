"""Export the perception stack to ONNX and build INT8 TensorRT engines for Orin.

Calibration data must come from GuardianBench, not COCO. INT8 calibration on the wrong
distribution is the classic way to lose 6 mAP on the deployment target while every
desktop benchmark still looks fine.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from ..config import GuardianConfig, load_config

logger = logging.getLogger(__name__)


def export_detector_onnx(cfg: GuardianConfig, out_dir: Path) -> Path:
    """Export RT-DETR / YOLOv10 to ONNX via the ultralytics exporter."""
    from ultralytics import RTDETR, YOLO

    loader = RTDETR if "rtdetr" in cfg.detector.weights.lower() else YOLO
    model = loader(cfg.detector.weights)
    path = model.export(
        format="onnx",
        imgsz=cfg.detector.input_size,
        dynamic=False,
        simplify=True,
        opset=17,
    )
    target = out_dir / "detector.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    Path(path).replace(target)
    logger.info("Detector ONNX -> %s", target)
    return target


def export_depth_onnx(cfg: GuardianConfig, out_dir: Path) -> Path:
    """Export Depth Anything V2 to ONNX with a fixed input size."""
    import torch
    from transformers import AutoModelForDepthEstimation

    model = AutoModelForDepthEstimation.from_pretrained(cfg.depth.model).eval()
    size = cfg.depth.input_size
    dummy = torch.randn(1, 3, size, size)

    target = out_dir / "depth.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(target),
        input_names=["pixel_values"],
        output_names=["predicted_depth"],
        opset_version=17,
        do_constant_folding=True,
    )
    logger.info("Depth ONNX -> %s", target)
    return target


class Int8Calibrator:
    """TensorRT entropy calibrator fed from real GuardianBench frames."""

    def __init__(self, frames: list[np.ndarray], cache: Path, batch: int = 8) -> None:
        import tensorrt as trt  # noqa: F401  (import validates availability early)

        self.frames = frames
        self.cache = cache
        self.batch = batch
        self.index = 0

    def get_batch_size(self) -> int:
        return self.batch

    def get_batch(self, _names):  # noqa: ANN001
        if self.index + self.batch > len(self.frames):
            return None
        batch = np.stack(self.frames[self.index : self.index + self.batch])
        self.index += self.batch
        return [np.ascontiguousarray(batch, dtype=np.float32)]

    def read_calibration_cache(self):
        return self.cache.read_bytes() if self.cache.exists() else None

    def write_calibration_cache(self, cache):  # noqa: ANN001
        self.cache.write_bytes(cache)


def build_engine(onnx_path: Path, engine_path: Path, int8: bool = True, workspace_gb: int = 4) -> Path:
    """Build a TensorRT engine. Requires the JetPack TensorRT python bindings."""
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    if not parser.parse(onnx_path.read_bytes()):
        errors = [parser.get_error(i) for i in range(parser.num_errors)]
        raise RuntimeError(f"ONNX parse failed: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    config.set_flag(trt.BuilderFlag.FP16)
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned None")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    logger.info("Engine -> %s", engine_path)
    return engine_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Export AI Guardian models for Jetson Orin")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--no-int8", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out)

    detector_onnx = export_detector_onnx(cfg, out_dir)
    depth_onnx = export_depth_onnx(cfg, out_dir)

    try:
        build_engine(detector_onnx, out_dir / "detector.engine", int8=not args.no_int8)
        build_engine(depth_onnx, out_dir / "depth.engine", int8=not args.no_int8)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "TensorRT build skipped (%s). Run this step on the Jetson itself -- engines "
            "are not portable across devices or TensorRT versions.",
            exc,
        )


if __name__ == "__main__":
    main()
