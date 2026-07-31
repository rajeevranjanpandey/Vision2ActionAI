"""GuardianBench: egocentric pedestrian navigation with hazard-onset labels.

No existing dataset works here. VizWiz is static photos, Cityscapes is car-mounted,
nuScenes is a vehicle. None of them capture a blind pedestrian's camera height, gait
oscillation, cane sweep occlusion, or -- critically -- *when a warning should have been
given*. That last label is what makes predictive evaluation possible, and it is the
dataset's real contribution.

Annotation protocol: certified O&M instructors mark, per hazard, the frame at which a
sighted guide would have intervened. Two instructors per clip; disagreements over 0.4 s
are adjudicated by a third. Report Krippendorff's alpha on onset timing.

On-disk layout:
    root/
      clips/<clip_id>/frames/000000.jpg ...
      clips/<clip_id>/imu.npy            # (N, 4) = t, speed_mps, yaw_rate_rps, pitch_rad
      clips/<clip_id>/meta.json          # fps, route, weather, participant id (hashed)
      annotations/<clip_id>.json         # hazard onsets + contact events
      splits.json                        # {"train": [...], "val": [...], "test": [...]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


@dataclass(slots=True)
class HazardLabel:
    """One annotated hazard within a clip."""

    hazard_id: str
    category: str               # vehicle | pedestrian | fixed_obstacle | overhead | drop | surface
    onset_frame: int            # first frame a warning should have been given
    contact_frame: Optional[int]  # frame the hazard entered the 1 m ego cylinder
    direction: str              # left | centre | right
    severity: str               # nuisance | injury | serious
    notes: str = ""


@dataclass(slots=True)
class ImuSample:
    t: float
    speed_mps: float
    yaw_rate_rps: float
    pitch_rad: float


@dataclass(slots=True)
class Clip:
    clip_id: str
    root: Path
    fps: float
    frame_paths: list[Path]
    imu: np.ndarray
    hazards: list[HazardLabel]
    meta: dict

    @property
    def duration_s(self) -> float:
        return len(self.frame_paths) / max(self.fps, 1e-6)

    @property
    def distance_m(self) -> float:
        """Path length from IMU speed. Denominator for false-alarms-per-km."""
        if self.imu.size == 0:
            return 0.0
        t, speed = self.imu[:, 0], self.imu[:, 1]
        return float(np.trapezoid(speed, t)) if t.size > 1 else 0.0

    def imu_at(self, frame_index: int) -> ImuSample:
        """Nearest IMU sample for a frame. IMU runs faster than the camera."""
        if self.imu.size == 0:
            return ImuSample(frame_index / self.fps, 0.0, 0.0, 0.0)
        t = frame_index / self.fps
        i = int(np.argmin(np.abs(self.imu[:, 0] - t)))
        row = self.imu[i]
        return ImuSample(float(row[0]), float(row[1]), float(row[2]), float(row[3]))

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        import cv2

        for i, path in enumerate(self.frame_paths):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            yield i, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class GuardianBench:
    """Loader over the on-disk layout described in the module docstring."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"GuardianBench root not found: {self.root}. "
                "See scripts/record.py to collect clips."
            )
        splits_path = self.root / "splits.json"
        self.splits: dict[str, list[str]] = (
            json.loads(splits_path.read_text()) if splits_path.exists() else {}
        )

    def clip_ids(self, split: str = "test") -> list[str]:
        if split in self.splits:
            return self.splits[split]
        return sorted(p.name for p in (self.root / "clips").iterdir() if p.is_dir())

    def load(self, clip_id: str) -> Clip:
        clip_dir = self.root / "clips" / clip_id
        meta = json.loads((clip_dir / "meta.json").read_text())
        frame_paths = sorted((clip_dir / "frames").glob("*.jpg"))

        imu_path = clip_dir / "imu.npy"
        imu = np.load(imu_path) if imu_path.exists() else np.zeros((0, 4))

        ann_path = self.root / "annotations" / f"{clip_id}.json"
        hazards: list[HazardLabel] = []
        if ann_path.exists():
            for raw in json.loads(ann_path.read_text()).get("hazards", []):
                hazards.append(
                    HazardLabel(
                        hazard_id=raw["hazard_id"],
                        category=raw["category"],
                        onset_frame=int(raw["onset_frame"]),
                        contact_frame=(
                            int(raw["contact_frame"]) if raw.get("contact_frame") is not None else None
                        ),
                        direction=raw.get("direction", "centre"),
                        severity=raw.get("severity", "injury"),
                        notes=raw.get("notes", ""),
                    )
                )

        return Clip(
            clip_id=clip_id,
            root=clip_dir,
            fps=float(meta.get("fps", 30.0)),
            frame_paths=frame_paths,
            imu=imu,
            hazards=hazards,
            meta=meta,
        )

    def iter_split(self, split: str = "test") -> Iterator[Clip]:
        for clip_id in self.clip_ids(split):
            yield self.load(clip_id)
