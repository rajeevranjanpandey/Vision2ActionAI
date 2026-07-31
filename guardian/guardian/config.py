"""Typed configuration loaded from ``configs/*.yaml``.

Every stage takes its own sub-config so units can be tested in isolation without
constructing the whole pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Type, TypeVar

import yaml


@dataclass
class CameraConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    fx: float = 912.0
    fy: float = 912.0
    cx: float = 640.0
    cy: float = 360.0
    height_m: float = 1.50
    pitch_rad: float = -0.10

    @property
    def intrinsics(self) -> tuple[float, float, float, float]:
        return (self.fx, self.fy, self.cx, self.cy)


@dataclass
class DetectorConfig:
    backend: str = "ultralytics"
    weights: str = "rtdetr-l.pt"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.5
    input_size: int = 640
    keep_classes: list[int] = field(default_factory=list)


@dataclass
class DepthConfig:
    model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    input_size: int = 518
    scale_mode: str = "ground_plane"
    ransac_iterations: int = 120
    ransac_inlier_m: float = 0.06
    scale_smoothing: float = 0.85


@dataclass
class SegmenterConfig:
    enabled: bool = True
    model: str = "facebook/sam2-hiera-tiny"
    max_prompts_per_frame: int = 4
    memory_frames: int = 7


@dataclass
class TrackingConfig:
    max_age_frames: int = 12
    min_hits: int = 3
    association_radius_m: float = 1.2
    process_noise: float = 0.6
    measurement_noise: float = 0.25


@dataclass
class RiskConfig:
    cone_half_width_m: float = 0.55
    cone_widen_per_m: float = 0.08
    cone_range_m: float = 12.0
    ego_radius_m: float = 1.0
    horizon_s: float = 3.0
    forecast_dt: float = 0.1
    ttc_warn_s: float = 3.0
    ttc_urgent_s: float = 1.5
    min_closing_speed_mps: float = 0.15


@dataclass
class PolicyConfig:
    fast_path_hz: int = 10
    alert_cooldown_s: float = 2.5
    urgent_cooldown_s: float = 0.8
    hysteresis_frames: int = 2
    degraded_after_ms: float = 400.0


@dataclass
class VlmConfig:
    enabled: bool = True
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_new_tokens: int = 64
    interval_s: float = 1.5
    advisory_only: bool = True


@dataclass
class SttConfig:
    enabled: bool = True
    model: str = "small.en"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    vad: bool = True


@dataclass
class TtsConfig:
    voice: str = "en_US-lessac-medium"
    rate: float = 1.15
    earcon_sample_rate: int = 22050


@dataclass
class HapticsConfig:
    enabled: bool = True
    channels: int = 3
    urgent_pattern_hz: float = 12.0
    warn_pattern_hz: float = 4.0


@dataclass
class GuardianConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    vlm: VlmConfig = field(default_factory=VlmConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    haptics: HapticsConfig = field(default_factory=HapticsConfig)


T = TypeVar("T")


def _build(cls: Type[T], data: dict[str, Any] | None) -> T:
    """Recursively construct a dataclass, ignoring unknown keys loudly-but-safely."""
    if not data:
        return cls()  # type: ignore[call-arg]
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        ftype = known[key].type
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _build(ftype, value)  # type: ignore[arg-type]
        else:
            kwargs[key] = value
    return cls(**kwargs)  # type: ignore[call-arg]


def load_config(path: str | Path) -> GuardianConfig:
    """Load and validate a YAML config file."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = GuardianConfig(
        camera=_build(CameraConfig, raw.get("camera")),
        detector=_build(DetectorConfig, raw.get("detector")),
        depth=_build(DepthConfig, raw.get("depth")),
        segmenter=_build(SegmenterConfig, raw.get("segmenter")),
        tracking=_build(TrackingConfig, raw.get("tracking")),
        risk=_build(RiskConfig, raw.get("risk")),
        policy=_build(PolicyConfig, raw.get("policy")),
        vlm=_build(VlmConfig, raw.get("vlm")),
        stt=_build(SttConfig, raw.get("stt")),
        tts=_build(TtsConfig, raw.get("tts")),
        haptics=_build(HapticsConfig, raw.get("haptics")),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: GuardianConfig) -> None:
    if cfg.risk.ttc_urgent_s >= cfg.risk.ttc_warn_s:
        raise ValueError("ttc_urgent_s must be strictly less than ttc_warn_s")
    if cfg.camera.height_m <= 0:
        raise ValueError("camera.height_m must be positive; it sets the metric scale")
    if not 0.0 <= cfg.depth.scale_smoothing < 1.0:
        raise ValueError("depth.scale_smoothing must be in [0, 1)")
    if cfg.vlm.enabled and not cfg.vlm.advisory_only:
        raise ValueError(
            "vlm.advisory_only=False is not permitted: the VLM must never gate a "
            "safety alert. See README 'two-path design'."
        )
