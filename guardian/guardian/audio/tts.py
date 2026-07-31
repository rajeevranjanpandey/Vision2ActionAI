"""Speech and earcon output.

Two channels with different latency budgets:
- **Earcons** are synthesised numerically and play in <20 ms. Safety alerts lead with an
  earcon; a spoken sentence arrives 300-800 ms later, which is a lifetime at 1.5 s TTC.
- **Speech** (Piper) carries the detail after the earcon has already bought a reaction.
"""

from __future__ import annotations

import logging
import queue
import threading

import numpy as np

from ..config import TtsConfig
from ..types import AlertLevel, Direction

logger = logging.getLogger(__name__)


class Speaker:
    """Non-blocking audio out with barge-in: a new urgent alert cancels current speech."""

    def __init__(self, cfg: TtsConfig) -> None:
        self.cfg = cfg
        self._voice = None
        self._queue: "queue.Queue[tuple[AlertLevel, str]]" = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="tts", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def say(self, text: str, level: AlertLevel = AlertLevel.INFO) -> None:
        if level is AlertLevel.URGENT:
            self._drain()  # barge-in
        try:
            self._queue.put_nowait((level, text))
        except queue.Full:
            logger.debug("Dropping speech, queue full: %s", text)

    def earcon(self, level: AlertLevel, direction: Direction) -> None:
        """Play the pre-speech tone immediately on the calling thread."""
        tone = synth_earcon(level, direction, self.cfg.earcon_sample_rate)
        _play(tone, self.cfg.earcon_sample_rate)

    # ----------------------------------------------------------------- private

    def _drain(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _ensure_voice(self) -> None:
        if self._voice is not None:
            return
        try:
            from piper.voice import PiperVoice

            self._voice = PiperVoice.load(self.cfg.voice)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Piper unavailable (%s); speech disabled, earcons still active", exc)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                _level, text = self._queue.get(timeout=0.3)
            except queue.Empty:
                continue
            self._ensure_voice()
            if self._voice is None:
                logger.info("[speech] %s", text)
                continue
            try:
                chunks = [
                    np.frombuffer(c, dtype=np.int16)
                    for c in self._voice.synthesize_stream_raw(text)
                ]
                if chunks:
                    _play(np.concatenate(chunks).astype(np.float32) / 32768.0,
                          self._voice.config.sample_rate)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Speech synthesis failed: %s", exc)


def synth_earcon(level: AlertLevel, direction: Direction, sample_rate: int) -> np.ndarray:
    """Numerically synthesised stereo earcon.

    Pitch encodes urgency, stereo pan encodes direction, repetition rate encodes level.
    All three are learnable in minutes and require no language processing by the user,
    which is why the earcon leads and the sentence follows.
    """
    specs = {
        AlertLevel.INFO: (660.0, 1, 0.10),
        AlertLevel.WARN: (880.0, 2, 0.09),
        AlertLevel.URGENT: (1320.0, 3, 0.07),
        AlertLevel.DEGRADED: (330.0, 1, 0.60),
    }
    freq, pulses, dur = specs.get(level, (660.0, 1, 0.10))

    gap = np.zeros(int(sample_rate * 0.045), dtype=np.float32)
    t = np.linspace(0.0, dur, int(sample_rate * dur), endpoint=False, dtype=np.float32)
    # Raised-cosine envelope: square edges click and are fatiguing over a full day.
    envelope = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(t.size) / max(t.size - 1, 1)))
    pulse = (np.sin(2.0 * np.pi * freq * t) * envelope * 0.35).astype(np.float32)

    segments: list[np.ndarray] = []
    for i in range(pulses):
        if i:
            segments.append(gap)
        segments.append(pulse)
    mono = np.concatenate(segments)

    pan = {Direction.LEFT: 0.15, Direction.CENTRE: 0.5, Direction.RIGHT: 0.85}[direction]
    left = mono * float(np.sqrt(1.0 - pan))
    right = mono * float(np.sqrt(pan))
    return np.stack([left, right], axis=1)


def _play(samples: np.ndarray, sample_rate: int) -> None:
    try:
        import sounddevice as sd

        sd.play(samples, sample_rate, blocking=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Audio output unavailable: %s", exc)
