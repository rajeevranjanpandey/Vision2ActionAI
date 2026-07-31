"""Lift 2D detections + metric depth into the 3D ego frame.

Ego frame convention: origin at the camera, +x right, +y up, +z forward, with the
ground plane at y = -camera.height_m after pitch de-rotation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import CameraConfig
from ..types import Detection, Observation


def depth_at_detection(
    detection: Detection,
    metric_depth: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Robust range for one detection.

    Prefers the SAM 2 mask median. Falls back to the lower-central patch of the box,
    which is where a ground-standing object's own surface is, rather than the box centre
    which frequently lands on background through a gap in the object.
    """
    if mask is not None and mask.any():
        values = metric_depth[mask]
        values = values[np.isfinite(values)]
        if values.size:
            return float(np.median(values))

    h, w = metric_depth.shape
    x1, y1, x2, y2 = detection.xyxy
    cx = int(np.clip(0.5 * (x1 + x2), 0, w - 1))
    # Sample at 70% down the box: below the visual centre, above the ground contact.
    cy = int(np.clip(y1 + 0.7 * (y2 - y1), 0, h - 1))
    half = max(2, int(0.08 * (x2 - x1)))
    patch = metric_depth[
        max(0, cy - half) : min(h, cy + half + 1),
        max(0, cx - half) : min(w, cx + half + 1),
    ]
    patch = patch[np.isfinite(patch)]
    if patch.size == 0:
        return None
    # 25th percentile: we want the *nearest* consistent surface, not the mean including
    # background bleed. Under-estimating range is the safe direction of error.
    return float(np.percentile(patch, 25))


def backproject(
    detection: Detection,
    range_m: float,
    camera: CameraConfig,
    mask: Optional[np.ndarray] = None,
) -> Observation:
    """Convert a detection + range into an ego-frame observation."""
    fx, fy, cx, cy = camera.intrinsics

    # Ray through the box centre in camera coordinates.
    u, v = detection.cx, detection.cy
    x_cam = (u - cx) * range_m / fx
    y_cam = (v - cy) * range_m / fy
    z_cam = range_m

    # De-rotate the camera pitch so z is horizontal forward distance.
    pitch = camera.pitch_rad
    c, s = np.cos(-pitch), np.sin(-pitch)
    z_ego = c * z_cam - s * y_cam
    x_ego = x_cam

    # Physical width from the box width at that range.
    box_w_px = detection.xyxy[2] - detection.xyxy[0]
    width_m = float(box_w_px * range_m / fx)

    return Observation(
        x_m=float(x_ego),
        z_m=float(max(z_ego, 0.05)),
        width_m=float(np.clip(width_m, 0.1, 6.0)),
        class_name=detection.class_name,
        score=detection.score,
        detection=detection,
        mask=mask,
    )


def observations_from_frame(
    detections: list[Detection],
    metric_depth: np.ndarray,
    camera: CameraConfig,
    masks: Optional[dict[int, np.ndarray]] = None,
) -> list[Observation]:
    """Batch conversion. Detections without a usable depth sample are dropped."""
    masks = masks or {}
    out: list[Observation] = []
    for i, det in enumerate(detections):
        mask = masks.get(i)
        rng = depth_at_detection(det, metric_depth, mask)
        if rng is None:
            continue
        out.append(backproject(det, rng, camera, mask))
    return out
