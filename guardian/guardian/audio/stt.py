"""Whisper streaming intent capture.

The user talks to the wearable; this turns speech into an intent. Runs on its own thread
with VAD so we are not transcribing traffic noise for 8 hours a day.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..config import SttConfig

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BLOCK_S = 0.5


@dataclass(slots=True)
class Intent:
    intent: str
    target: str
    transcript: str


class SpeechListener:
    """Microphone -> faster-whisper -> intent callback."""

    def __init__(self, cfg: SttConfig, on_intent: Callable[[Intent], None]) -> None:
        self.cfg = cfg
        self.on_intent = on_intent
        self._model = None
        self._audio: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=32)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if not self.cfg.enabled:
            return
        self._threads = [
            threading.Thread(target=self._capture_loop, name="stt-capture", daemon=True),
            threading.Thread(target=self._transcribe_loop, name="stt-decode", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

    # ----------------------------------------------------------------- private

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.cfg.model, device=self.cfg.device, compute_type=self.cfg.compute_type
        )
        logger.info("Loaded Whisper %s (%s)", self.cfg.model, self.cfg.compute_type)

    def _capture_loop(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:  # noqa: BLE001
            logger.warning("Audio input unavailable: %s", exc)
            return

        block = int(SAMPLE_RATE * BLOCK_S)

        def callback(indata, _frames, _time, status):  # noqa: ANN001
            if status:
                logger.debug("audio status: %s", status)
            try:
                self._audio.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass  # drop; the newest audio matters more than a backlog

        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=block, callback=callback
        ):
            self._stop.wait()

    def _transcribe_loop(self) -> None:
        buffer: list[np.ndarray] = []
        silence_blocks = 0

        while not self._stop.is_set():
            try:
                chunk = self._audio.get(timeout=0.5)
            except queue.Empty:
                continue

            speaking = float(np.sqrt(np.mean(chunk**2))) > 0.012
            if speaking:
                buffer.append(chunk)
                silence_blocks = 0
                continue

            if buffer:
                silence_blocks += 1
                # ~1 s of trailing silence closes the utterance.
                if silence_blocks >= 2:
                    audio = np.concatenate(buffer)
                    buffer, silence_blocks = [], 0
                    self._handle(audio)

    def _handle(self, audio: np.ndarray) -> None:
        try:
            self._ensure_loaded()
            assert self._model is not None
            segments, _info = self._model.transcribe(
                audio, language="en", vad_filter=self.cfg.vad, beam_size=1
            )
            transcript = " ".join(s.text for s in segments).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Transcription failed: %s", exc)
            return

        if len(transcript) < 3:
            return
        self.on_intent(classify(transcript))


def classify(transcript: str) -> Intent:
    """Rule-first intent classification.

    Deliberately not a model call: these six intents cover >90% of utterances in O&M
    pilot logs, and a 2 ms rule beats a 900 ms VLM round-trip for a user mid-stride.
    Anything unmatched falls through to the VLM as a free-form question.
    """
    text = transcript.lower().strip().rstrip("?.")

    if any(k in text for k in ("read", "what does it say", "sign say")):
        return Intent("READ", _after(text, ("read", "the")), transcript)
    if any(k in text for k in ("where is", "find", "locate")):
        return Intent("FIND", _after(text, ("where is", "find", "locate", "the")), transcript)
    if any(k in text for k in ("take me", "navigate", "how do i get")):
        return Intent("NAVIGATE", _after(text, ("take me to", "navigate to", "how do i get to")), transcript)
    if any(k in text for k in ("quieter", "louder", "repeat", "more warnings", "fewer")):
        return Intent("SETTING", text, transcript)
    if any(k in text for k in ("what's", "what is", "describe", "around me", "ahead")):
        return Intent("DESCRIBE", "", transcript)
    return Intent("UNKNOWN", "", transcript)


def _after(text: str, keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in text:
            text = text.split(key, 1)[1]
    return text.strip()


def parse_intent_json(raw: str) -> Optional[Intent]:
    """Parse a VLM-produced intent JSON blob, tolerating fenced output."""
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return Intent(str(data.get("intent", "UNKNOWN")), str(data.get("target", "")), raw)
