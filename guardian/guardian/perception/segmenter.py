"""SAM 2, prompted sparsely.

Running SAM 2 over the full frame at 30 fps on an Orin is not realistic, and it is not
necessary. We prompt it only with detections that already entered the risk cone, and we
reuse its memory bank as the tracker's appearance identity signal so we do not pay for a
second tracker on top.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import SegmenterConfig
from ..types import Detection

logger = logging.getLogger(__name__)


class Segmenter:
    """On-demand SAM 2 segmentation with a bounded prompt budget per frame."""

    def __init__(self, cfg: SegmenterConfig) -> None:
        self.cfg = cfg
        self._predictor = None
        self._initialised = False

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def _ensure_loaded(self) -> None:
        if self._predictor is not None or not self.cfg.enabled:
            return
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self._predictor = SAM2ImagePredictor.from_pretrained(self.cfg.model)
            logger.info("Loaded SAM 2 %s", self.cfg.model)
        except Exception as exc:  # noqa: BLE001 - segmentation is optional, never fatal
            logger.warning("SAM 2 unavailable (%s); running without masks", exc)
            self.cfg.enabled = False

    def segment(
        self,
        frame_rgb: np.ndarray,
        detections: list[Detection],
    ) -> dict[int, np.ndarray]:
        """Return ``{detection_index: boolean mask}`` for at most ``max_prompts_per_frame``.

        Detections are expected to be pre-sorted by risk; we take the head of the list.
        """
        self._ensure_loaded()
        if not self.cfg.enabled or self._predictor is None or not detections:
            return {}

        import torch

        budget = detections[: self.cfg.max_prompts_per_frame]
        boxes = np.array([d.xyxy for d in budget], dtype=np.float32)

        try:
            with torch.inference_mode():
                self._predictor.set_image(frame_rgb)
                masks, scores, _ = self._predictor.predict(
                    box=boxes,
                    multimask_output=False,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("SAM 2 predict failed: %s", exc)
            return {}

        masks = np.atleast_3d(np.squeeze(masks))
        if masks.ndim == 2:
            masks = masks[None]
        out: dict[int, np.ndarray] = {}
        for i in range(min(len(budget), masks.shape[0])):
            out[i] = masks[i].astype(bool)
        return out

    @staticmethod
    def mask_depth(mask: Optional[np.ndarray], metric_depth: np.ndarray) -> Optional[float]:
        """Median depth over the mask -- far more robust than a box-centre sample.

        A box centre on a lamp post samples the wall behind it. The mask does not.
        """
        if mask is None or not mask.any():
            return None
        values = metric_depth[mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        return float(np.median(values))
