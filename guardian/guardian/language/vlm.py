"""Slow-path Vision-Language Model (Qwen2.5-VL / LLaVA).

Runs in its own thread at 0.5-1 Hz. It can never block, raise, or suppress a fast-path
alert -- ``AlertPolicy`` reads only the latest cached advisory string and treats it as
decoration. If the VLM thread dies, the safety system is unaffected.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from ..config import VlmConfig
from .prompts import QUERY_SYSTEM, SCENE_CONTEXT_SYSTEM

logger = logging.getLogger(__name__)


class SceneNarrator:
    """Background VLM worker with a single-slot latest-frame queue."""

    def __init__(self, cfg: VlmConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._processor = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._advisory: Optional[str] = None
        self._advisory_ts: float = 0.0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ public

    def start(self) -> None:
        if not self.cfg.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="vlm", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def submit(self, frame_rgb: np.ndarray) -> None:
        """Offer the newest frame. Older unprocessed frames are dropped, by design."""
        if not self.cfg.enabled:
            return
        with self._lock:
            self._latest_frame = frame_rgb

    def advisory(self, max_age_s: float = 6.0) -> Optional[str]:
        """Latest advisory, or None if stale. Stale context is worse than none."""
        with self._lock:
            if self._advisory is None:
                return None
            if time.monotonic() - self._advisory_ts > max_age_s:
                return None
            return self._advisory

    def answer(self, frame_rgb: np.ndarray, question: str) -> str:
        """Synchronous user-initiated query. Called from the STT handler, not the loop."""
        self._ensure_loaded()
        if self._model is None:
            return "Scene description is unavailable right now."
        return self._generate(frame_rgb, QUERY_SYSTEM, question) or "I can't tell from here."

    # ----------------------------------------------------------------- private

    def _loop(self) -> None:
        while not self._stop.is_set():
            start = time.monotonic()
            frame = None
            with self._lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame
                    self._latest_frame = None

            if frame is not None:
                try:
                    text = self._generate(
                        frame, SCENE_CONTEXT_SYSTEM, "Describe navigation-relevant context."
                    )
                    if text and text.strip().upper() != "NOTHING":
                        with self._lock:
                            self._advisory = text.strip()
                            self._advisory_ts = time.monotonic()
                except Exception as exc:  # noqa: BLE001 - advisory path must never crash the app
                    logger.warning("VLM inference failed: %s", exc)

            elapsed = time.monotonic() - start
            self._stop.wait(max(0.0, self.cfg.interval_s - elapsed))

    def _ensure_loaded(self) -> None:
        if self._model is not None or not self.cfg.enabled:
            return
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq

            self._processor = AutoProcessor.from_pretrained(self.cfg.model)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.cfg.model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self._model.eval()
            logger.info("Loaded VLM %s", self.cfg.model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VLM unavailable (%s); slow path disabled", exc)
            self.cfg.enabled = False

    def _generate(self, frame_rgb: np.ndarray, system: str, user: str) -> Optional[str]:
        import torch
        from PIL import Image

        self._ensure_loaded()
        if self._model is None or self._processor is None:
            return None

        image = Image.fromarray(frame_rgb)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": user}],
            },
        ]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
            )
        text = self._processor.batch_decode(
            output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        return text.strip()
