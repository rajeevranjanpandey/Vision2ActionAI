"""Synthetic IMU episode generator for the fall channel.

Same honesty caveat as the risk head: these are physics-shaped synthetic traces, not
worn-sensor recordings. They exist so the classifier's *decision structure* can be
regression-tested and its specificity argued about before any field data exists. Every
number the site reports from these traces is labelled as simulated.

The episode classes are chosen to be the hard ones -- the confusable activities of daily
living that a naive 2 g threshold gets wrong:

* ``fall_forward`` / ``fall_sideways`` / ``fall_backward`` -- free-fall, impact, topple, still
* ``fall_slow_slump``  -- no free-fall dip (the wearer slid down a wall) but does topple
* ``sit_heavy``        -- impact, no free-fall, stays upright, settles still  (classic FP)
* ``stumble_recover``  -- big impact and free-fall, but keeps walking          (classic FP)
* ``stairs_descent``   -- repeated 1.8 g heel strikes, upright, never still    (classic FP)
* ``device_bump``      -- sharp jolt from a doorframe knock, wearer unaffected
* ``walk``             -- baseline gait
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .fall import ImuSample

FALL_CLASSES = ("fall_forward", "fall_sideways", "fall_backward", "fall_slow_slump")
ADL_CLASSES = ("sit_heavy", "stumble_recover", "stairs_descent", "device_bump", "walk")


@dataclass(slots=True)
class ImuEpisode:
    label: str          # class name
    is_fall: bool
    samples: list[ImuSample]
    onset_s: float      # ground-truth impact time (falls only; -1 otherwise)

    @property
    def duration_s(self) -> float:
        return self.samples[-1].t_s - self.samples[0].t_s if self.samples else 0.0


def _gait(t: float, amp: float = 0.22, hz: float = 1.9) -> np.ndarray:
    """Upright walking: gravity on +z with a vertical bob and a little lateral sway."""
    return np.array([
        0.10 * math.sin(2 * math.pi * hz * t * 0.5),
        0.06 * math.sin(2 * math.pi * hz * t + 1.0),
        1.0 + amp * math.sin(2 * math.pi * hz * t),
    ])


def _rotate_to(vec: np.ndarray, tilt_deg: float, azimuth_deg: float) -> np.ndarray:
    """Rotate the gravity direction by ``tilt_deg`` about an axis at ``azimuth_deg``."""
    t = math.radians(tilt_deg)
    a = math.radians(azimuth_deg)
    axis = np.array([math.cos(a), math.sin(a), 0.0])
    v = vec
    return (
        v * math.cos(t)
        + np.cross(axis, v) * math.sin(t)
        + axis * float(np.dot(axis, v)) * (1 - math.cos(t))
    )


def generate_episode(label: str, rng: np.random.Generator, fs: float = 50.0) -> ImuEpisode:
    dt = 1.0 / fs
    duration = 8.0
    n = int(duration * fs)
    t0 = 3.0 + float(rng.uniform(-0.4, 0.4))     # event onset
    samples: list[ImuSample] = []
    is_fall = label in FALL_CLASSES
    noise = 0.035

    tilt_final = {
        "fall_forward": float(rng.uniform(62, 88)),
        "fall_sideways": float(rng.uniform(70, 95)),
        "fall_backward": float(rng.uniform(60, 85)),
        "fall_slow_slump": float(rng.uniform(55, 75)),
        "sit_heavy": float(rng.uniform(4, 14)),
        "stumble_recover": float(rng.uniform(3, 12)),
        "stairs_descent": float(rng.uniform(2, 8)),
        "device_bump": float(rng.uniform(1, 6)),
        "walk": float(rng.uniform(0, 4)),
    }[label]
    azimuth = float(rng.uniform(0, 360))

    peak_g = {
        "fall_forward": float(rng.uniform(2.9, 4.6)),
        "fall_sideways": float(rng.uniform(2.8, 4.2)),
        "fall_backward": float(rng.uniform(3.0, 4.8)),
        "fall_slow_slump": float(rng.uniform(2.5, 3.2)),
        "sit_heavy": float(rng.uniform(2.4, 3.0)),
        "stumble_recover": float(rng.uniform(2.6, 3.6)),
        "stairs_descent": float(rng.uniform(1.7, 2.2)),
        "device_bump": float(rng.uniform(2.5, 3.4)),
        "walk": 0.0,
    }[label]

    freefall_ms = {
        "fall_forward": float(rng.uniform(180, 380)),
        "fall_sideways": float(rng.uniform(160, 340)),
        "fall_backward": float(rng.uniform(200, 400)),
        "fall_slow_slump": 0.0,
        "sit_heavy": 0.0,
        "stumble_recover": float(rng.uniform(120, 220)),
        "stairs_descent": 0.0,
        "device_bump": 0.0,
        "walk": 0.0,
    }[label]

    ff_start = t0 - freefall_ms / 1000.0

    for i in range(n):
        t = i * dt
        gyro = np.array([0.0, 0.0, 0.0])

        if t < ff_start - 0.05:
            acc = _gait(t)
        elif freefall_ms > 0.0 and ff_start <= t < t0:
            acc = np.array([0.0, 0.0, float(rng.uniform(0.15, 0.45))])
            gyro = np.array([rng.normal(0, 1.5), rng.normal(0, 1.5), rng.normal(0, 0.8)])
        elif t0 <= t < t0 + 0.16:
            # Impact: a short half-sine spike along the current gravity direction.
            frac = (t - t0) / 0.16
            mag = 1.0 + (peak_g - 1.0) * math.sin(math.pi * frac)
            acc = _rotate_to(np.array([0.0, 0.0, mag]), tilt_final * frac, azimuth)
            gyro = np.array([rng.normal(0, 4.0), rng.normal(0, 4.0), rng.normal(0, 2.0)])
        else:
            after = t - (t0 + 0.16)
            if label in ("fall_forward", "fall_sideways", "fall_backward", "fall_slow_slump"):
                acc = _rotate_to(np.array([0.0, 0.0, 1.0]), tilt_final, azimuth)
                acc = acc + rng.normal(0, 0.02, 3)          # motionless on the ground
            elif label == "sit_heavy":
                settle = math.exp(-after / 0.35) * 0.30 * math.sin(2 * math.pi * 3.0 * after)
                acc = _rotate_to(np.array([0.0, 0.0, 1.0 + settle]), tilt_final, azimuth)
                # Seated but alive: small torso motion keeps the variance above quiet.
                acc = acc + np.array([0.0, 0.0, 0.16 * math.sin(2 * math.pi * 0.7 * after)])
            elif label == "stumble_recover":
                acc = _gait(t, amp=0.34, hz=2.2)             # back to walking immediately
            elif label == "stairs_descent":
                acc = _gait(t, amp=0.42, hz=2.4)
                if math.sin(2 * math.pi * 2.4 * t) > 0.96:
                    acc = acc * 1.7
            elif label == "device_bump":
                acc = _gait(t, amp=0.26, hz=2.0)
            else:
                acc = _gait(t)

        acc = acc + rng.normal(0, noise, 3)
        samples.append(
            ImuSample(
                t_s=t,
                ax=float(acc[0]), ay=float(acc[1]), az=float(acc[2]),
                gx=float(gyro[0]), gy=float(gyro[1]), gz=float(gyro[2]),
            )
        )

    if label == "stairs_descent":
        # A staircase is many near-threshold heel strikes, not one event.
        for i, s in enumerate(samples):
            if 1.0 < s.t_s < 7.0 and i % 21 == 0:
                samples[i] = ImuSample(s.t_s, s.ax, s.ay, s.az * peak_g / 1.0,
                                       s.gx, s.gy, s.gz)

    return ImuEpisode(label=label, is_fall=is_fall, samples=samples,
                      onset_s=t0 if is_fall else -1.0)


def generate_dataset(
    n_per_class: int = 40, seed: int = 0, fs: float = 50.0
) -> list[ImuEpisode]:
    """Balanced-by-class episode set. Falls are rarer than ADLs in life; the report
    reweights to a realistic activity budget rather than faking the prior here."""
    rng = np.random.default_rng(seed)
    episodes: list[ImuEpisode] = []
    for label in FALL_CLASSES + ADL_CLASSES:
        for _ in range(n_per_class):
            episodes.append(generate_episode(label, rng, fs))
    rng.shuffle(episodes)  # type: ignore[arg-type]
    return episodes
