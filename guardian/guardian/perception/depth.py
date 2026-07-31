"""Depth Anything V2 wrapper producing metric depth."""

from __future__ import annotations

import logging

import numpy as np

from ..config import CameraConfig, DepthConfig
from ..types import DepthResult
from .calibration import GroundPlaneCalibrator

logger = logging.getLogger(__name__)


class DepthEstimator:
    """Relative depth from Depth Anything V2, rescaled to metres each frame."""

    def __init__(self, camera: CameraConfig, cfg: DepthConfig) -> None:
        self.cfg = cfg
        self.calibrator = GroundPlaneCalibrator(camera, cfg)
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._processor = AutoImageProcessor.from_pretrained(self.cfg.model)
        self._model = AutoModelForDepthEstimation.from_pretrained(
            self.cfg.model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if torch.cuda.is_available():
            self._model = self._model.to("cuda")
        self._model.eval()
        logger.info("Loaded depth model %s", self.cfg.model)

    def __call__(self, frame_rgb: np.ndarray, pitch_rad: float | None = None) -> DepthResult:
        import torch

        self._ensure_loaded()
        assert self._model is not None and self._processor is not None

        inputs = self._processor(images=frame_rgb, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda", dtype=self._model.dtype if v.is_floating_point() else v.dtype)
                      for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)

        depth = outputs.predicted_depth  # (1, h, w) inverse relative depth
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1).float(),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        relative = depth.detach().cpu().numpy()

        estimate = self.calibrator.estimate(relative, pitch_rad)
        metric = self.calibrator.apply(relative, estimate.scale)
        return DepthResult(
            relative=relative,
            metric=metric,
            scale=estimate.scale,
            scale_confidence=estimate.confidence,
        )
