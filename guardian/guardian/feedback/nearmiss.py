"""On-device near-miss log: feature windows in, nothing identifying out."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from ..train.features import D, FEATURE_NAMES, WINDOW

# The complete set of keys an exported payload may contain. Anything else is a leak, and
# the assertion below is what turns that sentence into a test.
ALLOWED_KEYS = frozenset(
    {"schema", "window", "label", "context", "hour_bucket", "device_hash", "head_version"}
)

VALID_LABELS = ("near_miss", "false_alarm", "correct_alert")


class RedactionError(RuntimeError):
    """Raised when a payload contains a field that must never leave the device."""


def device_hash(device_id: str, salt: str = "guardian-v1") -> str:
    """Stable pseudonym. Not reversible, and not stable across salts, so a leaked log
    cannot be joined against another release's logs."""
    return hashlib.sha256(f"{salt}:{device_id}".encode()).hexdigest()[:16]


@dataclass(slots=True)
class NearMissRecord:
    """One flagged event.

    ``window`` is the exact ``(WINDOW, D)`` tensor the head consumed, so a retrain sees
    what the model saw -- not a reconstruction of it, which is the usual way feedback
    loops end up training on a different distribution than they deployed on.
    """

    window: np.ndarray
    label: str
    context: str = "sidewalk"
    hour_bucket: int = 0
    device: str = "local"
    head_version: str = "0.1.0"

    def __post_init__(self) -> None:
        arr = np.asarray(self.window, dtype=np.float32)
        if arr.shape != (WINDOW, D):
            raise ValueError(f"window must be {(WINDOW, D)}, got {arr.shape}")
        if self.label not in VALID_LABELS:
            raise ValueError(f"label must be one of {VALID_LABELS}, got {self.label!r}")
        if not 0 <= self.hour_bucket <= 23:
            raise ValueError("hour_bucket must be an hour of day")
        self.window = arr

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": {"window": [WINDOW, D], "features": list(FEATURE_NAMES)},
            "window": [[round(float(v), 5) for v in row] for row in self.window],
            "label": self.label,
            "context": self.context,
            # Hour of day, not a timestamp: enough to study dusk failures, useless for
            # reconstructing when someone leaves the house.
            "hour_bucket": int(self.hour_bucket),
            "device_hash": device_hash(self.device),
            "head_version": self.head_version,
        }

    @property
    def payload_bytes(self) -> int:
        return len(json.dumps(self.to_payload()).encode())


def assert_privacy_safe(payload: dict[str, Any]) -> None:
    """Fail loudly if a payload grew a field it should not have.

    This runs before every export. It is cheap, and it is the only thing standing between
    a well-meaning future patch that attaches "just the thumbnail, for debugging" and a
    dataset of bystanders' faces.
    """
    extra = set(payload) - ALLOWED_KEYS
    if extra:
        raise RedactionError(f"payload contains disallowed fields: {sorted(extra)}")
    blob = json.dumps(payload).lower()
    for banned in ("jpeg", "jpg", "png", "base64", "image", "frame_uri", "lat", "lon",
                   "gps", "audio", "transcript"):
        if banned in blob:
            raise RedactionError(f"payload smells of raw media or location: {banned!r}")


@dataclass
class NearMissLog:
    """Append-only local log with a bounded size."""

    max_records: int = 500
    records: list[NearMissRecord] = field(default_factory=list)

    def flag(self, record: NearMissRecord) -> NearMissRecord:
        self.records.append(record)
        if len(self.records) > self.max_records:
            # Drop oldest. A log that fills up and then silently stops recording is a
            # log that misses the failure mode that only appears in month three.
            self.records = self.records[-self.max_records:]
        return record

    def export(self) -> list[dict[str, Any]]:
        out = []
        for r in self.records:
            payload = r.to_payload()
            assert_privacy_safe(payload)
            out.append(payload)
        return out

    def export_bytes(self) -> int:
        return len(json.dumps(self.export()).encode())

    def as_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """``(X, y)`` for retraining: flattened windows and a hazard label.

        ``near_miss`` is a positive the head missed; ``false_alarm`` is a negative it
        fired on; ``correct_alert`` is a confirmed positive kept as an anchor so the
        retrain does not drift toward only-the-mistakes.
        """
        if not self.records:
            return np.zeros((0, WINDOW * D), np.float32), np.zeros((0,), np.float32)
        X = np.stack([r.window.reshape(-1) for r in self.records]).astype(np.float32)
        y = np.array(
            [1.0 if r.label in ("near_miss", "correct_alert") else 0.0
             for r in self.records],
            dtype=np.float32,
        )
        return X, y

    def counts(self) -> dict[str, int]:
        return {lab: sum(1 for r in self.records if r.label == lab) for lab in VALID_LABELS}

    def extend(self, records: Iterable[NearMissRecord]) -> None:
        for r in records:
            self.flag(r)
