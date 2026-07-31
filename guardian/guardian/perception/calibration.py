"""Metric scale recovery for Depth Anything V2.

This is the part reviewers attack first. Depth Anything V2 is a *relative* depth model:
its output is affine-ambiguous inverse depth. Distances in metres do not come out of it.

We recover metric scale every frame from geometry we already know: the camera is mounted
at a fixed height on the user's body, and the IMU gives us pitch. Fit a plane to the
lower image region with RANSAC, then solve for the scale factor that puts that plane at
the known camera height. Report the inlier ratio as confidence -- when confidence
collapses (stairs, kerb drop, crowd occluding the ground) the pipeline must widen its
safety margins rather than trust the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CameraConfig, DepthConfig


@dataclass(slots=True)
class ScaleEstimate:
    scale: float
    confidence: float
    inliers: int
    plane_normal: np.ndarray


class GroundPlaneCalibrator:
    """IMU-aided ground-plane RANSAC that turns relative depth into metres."""

    def __init__(self, camera: CameraConfig, cfg: DepthConfig) -> None:
        self.camera = camera
        self.cfg = cfg
        self._scale_ema: float | None = None
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ public

    def estimate(self, relative_depth: np.ndarray, pitch_rad: float | None = None) -> ScaleEstimate:
        """Return the multiplicative factor mapping ``relative_depth`` to metres."""
        if self.cfg.scale_mode == "none":
            return ScaleEstimate(1.0, 1.0, 0, np.array([0.0, 1.0, 0.0]))
        if self.cfg.scale_mode == "fixed":
            return ScaleEstimate(self._scale_ema or 1.0, 0.5, 0, np.array([0.0, 1.0, 0.0]))

        pitch = self.camera.pitch_rad if pitch_rad is None else pitch_rad
        points = self._sample_ground_candidates(relative_depth, pitch)
        if points.shape[0] < 50:
            return ScaleEstimate(self._scale_ema or 1.0, 0.0, 0, np.array([0.0, 1.0, 0.0]))

        normal, offset, inlier_mask = self._ransac_plane(points)
        inliers = int(inlier_mask.sum())
        confidence = inliers / float(points.shape[0])

        # The fitted plane sits at |offset| in *relative* units below the camera.
        # We know it must sit at camera.height_m in metres.
        if abs(offset) < 1e-6:
            return ScaleEstimate(self._scale_ema or 1.0, 0.0, inliers, normal)
        raw_scale = self.camera.height_m / abs(offset)
        scale = self._smooth(raw_scale, confidence)
        return ScaleEstimate(scale, confidence, inliers, normal)

    def apply(self, relative_depth: np.ndarray, scale: float) -> np.ndarray:
        """Convert model output to metric depth in metres.

        Depth Anything V2 emits *inverse* relative depth (larger = nearer), so metric
        depth is scale / disparity. We clamp to a sane operating range: below 0.3 m the
        wearable cannot help anyway, and beyond 40 m the estimate is noise.
        """
        disparity = np.maximum(relative_depth, 1e-4)
        metric = scale / disparity
        return np.clip(metric, 0.3, 40.0)

    # ----------------------------------------------------------------- private

    def _sample_ground_candidates(self, depth: np.ndarray, pitch: float) -> np.ndarray:
        """Back-project a sparse sample of the lower image band into relative 3D."""
        h, w = depth.shape
        # Horizon row shifts with pitch; only sample below it.
        horizon = int(np.clip(self.camera.cy + self.camera.fy * np.tan(pitch), 0, h - 1))
        top = min(h - 1, max(horizon + 10, int(h * 0.55)))
        step = max(1, (h - top) // 48)

        ys = np.arange(top, h, step)
        xs = np.arange(0, w, max(1, w // 64))
        gy, gx = np.meshgrid(ys, xs, indexing="ij")
        d = depth[gy, gx].astype(np.float64)

        valid = d > 1e-4
        gx, gy, d = gx[valid], gy[valid], d[valid]
        z = 1.0 / d  # relative range
        x = (gx - self.camera.cx) * z / self.camera.fx
        y = (gy - self.camera.cy) * z / self.camera.fy
        pts = np.stack([x, y, z], axis=1)
        return _rotate_pitch(pts, -pitch)

    def _ransac_plane(self, points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Fit y = const-ish plane. Returns (unit normal, signed offset, inlier mask)."""
        best_inliers = np.zeros(points.shape[0], dtype=bool)
        best_normal = np.array([0.0, 1.0, 0.0])
        best_offset = float(np.median(points[:, 1]))
        n = points.shape[0]
        # Threshold is metres-equivalent only after scaling; use a relative proxy.
        median_range = float(np.median(points[:, 2])) or 1.0
        thresh = self.cfg.ransac_inlier_m / max(median_range, 1e-3)

        for _ in range(self.cfg.ransac_iterations):
            idx = self._rng.choice(n, size=3, replace=False)
            p0, p1, p2 = points[idx]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            normal = normal / norm
            # Reject planes that are not roughly horizontal after de-rotation.
            if abs(normal[1]) < 0.85:
                continue
            offset = float(np.dot(normal, p0))
            dist = np.abs(points @ normal - offset)
            inliers = dist < thresh
            if inliers.sum() > best_inliers.sum():
                best_inliers, best_normal, best_offset = inliers, normal, offset

        if best_inliers.sum() >= 3:
            # Least-squares refit on the inlier set.
            pts = points[best_inliers]
            centroid = pts.mean(axis=0)
            _, _, vh = np.linalg.svd(pts - centroid)
            best_normal = vh[-1] / (np.linalg.norm(vh[-1]) + 1e-12)
            best_offset = float(np.dot(best_normal, centroid))
        return best_normal, best_offset, best_inliers

    def _smooth(self, raw_scale: float, confidence: float) -> float:
        """EMA the scale, weighted by confidence so bad fits barely move it."""
        alpha = self.cfg.scale_smoothing
        effective = alpha + (1.0 - alpha) * (1.0 - confidence)
        effective = float(np.clip(effective, 0.0, 0.995))
        if self._scale_ema is None:
            self._scale_ema = raw_scale
        else:
            self._scale_ema = effective * self._scale_ema + (1.0 - effective) * raw_scale
        return float(self._scale_ema)


def _rotate_pitch(points: np.ndarray, pitch: float) -> np.ndarray:
    """Rotate about the x-axis to bring the camera frame level with the ground."""
    c, s = np.cos(pitch), np.sin(pitch)
    rot = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return points @ rot.T
