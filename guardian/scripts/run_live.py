"""Live wearable loop.

    python scripts/run_live.py --config configs/default.yaml --source 0
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardian.audio.stt import Intent, SpeechListener  # noqa: E402
from guardian.config import load_config  # noqa: E402
from guardian.pipeline import GuardianPipeline  # noqa: E402
from guardian.risk.forecast import rollout  # noqa: E402
from guardian.types import FrameResult  # noqa: E402

logger = logging.getLogger("run_live")


def open_source(source: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source '{source}'")
    return cap


def draw_overlay(frame_bgr, result: FrameResult, cfg) -> None:
    """Sighted-observer debug view. The user never sees this; researchers live in it."""
    for det in result.detections:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (90, 90, 90), 1)

    hazard_ids = {h.track.track_id for h in result.hazards}
    for track in result.tracks:
        colour = (0, 0, 255) if track.track_id in hazard_ids else (0, 200, 0)
        label = f"#{track.track_id} {track.class_name} {track.range_m:.1f}m v={track.vz_mps:+.1f}"
        cv2.putText(frame_bgr, label, (12, 60 + 18 * (track.track_id % 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    hud = f"{result.total_latency_ms:5.1f} ms | scale {result.depth_scale:.3f}"
    cv2.putText(frame_bgr, hud, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if result.alert:
        cv2.putText(frame_bgr, f"{result.alert.level.value.upper()}: {result.alert.utterance}",
                    (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for hazard in result.hazards[:3]:
        path = rollout(hazard.track, cfg.risk)
        logger.debug("hazard %s predicted end (%.1f, %.1f)",
                     hazard.track.track_id, path[-1, 1], path[-1, 2])


def main() -> None:
    ap = argparse.ArgumentParser(description="AI Guardian live loop")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--source", default="0")
    ap.add_argument("--display", action="store_true", help="show the debug overlay window")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    pipeline = GuardianPipeline(cfg)
    pipeline.start()

    latest_frame = {"rgb": None}

    def on_intent(intent: Intent) -> None:
        logger.info("intent=%s target=%r (%s)", intent.intent, intent.target, intent.transcript)
        frame = latest_frame["rgb"]
        if frame is None:
            return
        if intent.intent in ("DESCRIBE", "FIND", "READ", "UNKNOWN"):
            pipeline.handle_question(frame, intent.transcript)

    listener = SpeechListener(cfg.stt, on_intent)
    listener.start()

    cap = open_source(args.source, cfg.camera.width, cfg.camera.height, cfg.camera.fps)
    interval = 1.0 / max(cfg.policy.fast_path_hz, 1)
    next_tick = time.monotonic()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            now = time.monotonic()
            if now < next_tick:
                # Camera runs at 30 fps; the fast path runs at 10 Hz. Extra frames are
                # dropped rather than queued -- a stale frame is worse than no frame.
                continue
            next_tick = now + interval

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            latest_frame["rgb"] = frame_rgb

            # TODO: replace with real IMU integration on the wearable.
            result = pipeline.process(frame_rgb, ego_speed_mps=1.3, yaw_rate_rps=0.0)

            if args.display:
                draw_overlay(frame_bgr, result, cfg)
                cv2.imshow("AI Guardian", frame_bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        listener.stop()
        pipeline.stop()


if __name__ == "__main__":
    main()
