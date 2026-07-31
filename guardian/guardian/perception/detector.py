"""Object detection front-end: RT-DETR or YOLOv10.

Three backends share one interface so the Orin deployment can swap the ultralytics
PyTorch path for a TensorRT engine without touching the pipeline.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from ..config import DetectorConfig
from ..types import Detection

logger = logging.getLogger(__name__)

# COCO names, indexed by class id. Only the navigation-relevant subset is kept by
# ``DetectorConfig.keep_classes``.
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


class DetectorBackend(Protocol):
    def infer(self, frame: np.ndarray) -> list[Detection]: ...


class Detector:
    """Thin dispatcher over the configured backend."""

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self._keep = set(cfg.keep_classes) if cfg.keep_classes else None
        self._backend = self._build_backend()

    def _build_backend(self) -> DetectorBackend:
        if self.cfg.backend == "ultralytics":
            return _UltralyticsBackend(self.cfg)
        if self.cfg.backend in ("onnx", "tensorrt"):
            return _OnnxBackend(self.cfg)
        raise ValueError(f"Unknown detector backend '{self.cfg.backend}'")

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        dets = self._backend.infer(frame)
        out = []
        for d in dets:
            if d.score < self.cfg.conf_threshold:
                continue
            if self._keep is not None and d.class_id not in self._keep:
                continue
            out.append(d)
        return out


class _UltralyticsBackend:
    """RT-DETR / YOLOv10 through the ultralytics API. Used for training and desktop dev."""

    def __init__(self, cfg: DetectorConfig) -> None:
        from ultralytics import RTDETR, YOLO  # imported lazily: heavy

        self.cfg = cfg
        loader = RTDETR if "rtdetr" in cfg.weights.lower() else YOLO
        self.model = loader(cfg.weights)
        self.names: dict[int, str] = getattr(self.model, "names", {}) or {}

    def infer(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            imgsz=self.cfg.input_size,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            verbose=False,
        )
        out: list[Detection] = []
        for res in results:
            boxes = res.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            for box, score, class_id in zip(xyxy, conf, cls):
                out.append(
                    Detection(
                        xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        score=float(score),
                        class_id=int(class_id),
                        class_name=self.names.get(int(class_id), _coco_name(int(class_id))),
                    )
                )
        return out


class _OnnxBackend:
    """ONNX Runtime / TensorRT execution provider. This is the deployment path."""

    def __init__(self, cfg: DetectorConfig) -> None:
        import onnxruntime as ort  # lazy

        providers = (
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            if cfg.backend == "tensorrt"
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.cfg = cfg
        self.session = ort.InferenceSession(cfg.weights, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        logger.info("Detector running on %s", self.session.get_providers()[0])

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        import cv2

        h, w = frame.shape[:2]
        size = self.cfg.input_size
        scale = min(size / w, size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = resized
        chw = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return chw[None], scale, (nw, nh)

    def infer(self, frame: np.ndarray) -> list[Detection]:
        blob, scale, _ = self._preprocess(frame)
        raw = self.session.run(None, {self.input_name: blob})[0]
        raw = np.squeeze(raw)
        out: list[Detection] = []
        # Expected export layout: [N, 6] = x1, y1, x2, y2, score, class_id
        for row in np.atleast_2d(raw):
            if row.shape[0] < 6:
                continue
            x1, y1, x2, y2, score, class_id = row[:6]
            out.append(
                Detection(
                    xyxy=(
                        float(x1) / scale,
                        float(y1) / scale,
                        float(x2) / scale,
                        float(y2) / scale,
                    ),
                    score=float(score),
                    class_id=int(class_id),
                    class_name=_coco_name(int(class_id)),
                )
            )
        return out


def _coco_name(class_id: int) -> str:
    if 0 <= class_id < len(COCO_NAMES):
        return COCO_NAMES[class_id]
    return f"class_{class_id}"
