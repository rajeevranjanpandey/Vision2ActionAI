"""Learning components for the predictive risk head.

Everything in here is numpy-only on purpose. The geometric kernel in ``risk/`` is the
safety floor; this package learns a *residual* on top of it, and it must be trainable,
reproducible, and testable on a laptop with no GPU and no downloaded weights. Torch is
used only in ``deploy/`` for exporting the perception backbones.
"""

from .features import FEATURE_NAMES, WINDOW, build_windows, track_features
from .simulate import Sequence, generate_dataset

__all__ = [
    "FEATURE_NAMES",
    "WINDOW",
    "build_windows",
    "track_features",
    "Sequence",
    "generate_dataset",
]
