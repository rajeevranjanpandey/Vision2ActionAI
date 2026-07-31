"""AI Guardian -- predictive mobility assistance for visually impaired pedestrians.

Heavy dependencies (torch, ultralytics, transformers) are pulled in lazily so that the
risk kernel, tracker, and benchmark scorer can be imported and tested on a machine with
nothing but numpy and scipy installed.
"""

from typing import TYPE_CHECKING, Any

from .config import GuardianConfig, load_config
from .types import Alert, AlertLevel, Detection, Direction, FrameResult, Hazard, Track

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import GuardianPipeline

__version__ = "0.1.0"

__all__ = [
    "GuardianConfig",
    "GuardianPipeline",
    "load_config",
    "Alert",
    "AlertLevel",
    "Detection",
    "Direction",
    "FrameResult",
    "Hazard",
    "Track",
]


def __getattr__(name: str) -> Any:
    if name == "GuardianPipeline":
        from .pipeline import GuardianPipeline

        return GuardianPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
