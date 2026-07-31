# AI Guardian

**A Vision–Language–Action model for predictive assistance in real-world mobility of visually impaired individuals.**

AI Guardian does not describe the scene. It predicts it. The system estimates the
ego-relative *future* state of every hazard 1.5–3 s ahead and emits an action only
when predicted time-to-collision (TTC) drops below a safety threshold.

```
Camera (30 fps) + IMU
   |
   +-- RT-DETR / YOLOv10 -----> 2D detections
   +-- Depth Anything V2 ------> relative depth -> metric depth (ground-plane + IMU scale)
   +-- SAM 2 (prompted) -------> temporally persistent masks / instance identity
                    |
              3D track states (x, z, vx, vz) in the ego frame
                    |
        +-----------+------------------+
   FAST PATH (10 Hz)             SLOW PATH (0.5-1 Hz)
   TTC + risk field              Qwen2.5-VL / LLaVA
   haptic + earcon               semantic context, disambiguation
   latency budget < 150 ms       latency budget < 2000 ms
                    |
              Action arbitration policy
                    |
        Bone-conduction audio  <--  Whisper (user intent in)
```

The **two-path split is the systems contribution**: a VLM at 1–2 tok/s can never sit in a
safety loop. Safety alerts are deterministic geometry; language only enriches them.

## Layout

```
guardian/
  config.py               typed config, loaded from configs/*.yaml
  types.py                Detection / Track / Hazard / Alert dataclasses
  perception/
    detector.py           RT-DETR / YOLOv10 wrapper (ultralytics or ONNX/TensorRT)
    depth.py              Depth Anything V2 + metric rescaling
    calibration.py        IMU-aided ground-plane RANSAC -> metric scale
    segmenter.py          SAM 2, prompted only inside the risk cone
    backprojection.py     pixel + metric depth -> 3D ego coordinates
  tracking/
    kalman.py             constant-velocity 3D Kalman filter
    tracker.py            greedy 3D association + track lifecycle
  risk/
    ttc.py                risk cone, closing speed, time-to-collision
    forecast.py           1.5-3 s ego-relative trajectory rollout
    policy.py             hysteresis + arbitration -> at most one alert
  language/
    vlm.py                slow-path VLM (Qwen2.5-VL / LLaVA)
    prompts.py            system prompts for the slow path
  audio/
    stt.py                Whisper streaming intent capture
    tts.py                Piper TTS + earcon synthesis
    haptics.py            directional haptic driver
  pipeline.py             async two-path orchestrator
  bench/
    dataset.py            GuardianBench loader (hazard-onset labels)
    metrics.py            lead time, false alarms/km, alert->avoidance
    evaluate.py           offline evaluation entry point
  deploy/
    export.py             ONNX -> TensorRT INT8 export for Orin
    benchmark.py          per-stage latency + thermals
scripts/
  run_live.py             live wearable loop
  eval.py                 GuardianBench evaluation
  record.py               data collection for GuardianBench
tests/                    unit tests for the safety-critical math
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On Jetson Orin, install the JetPack builds of torch/torchvision and TensorRT from
NVIDIA's index instead of the PyPI wheels pinned here.

## Run

```bash
# live loop (webcam or CSI camera)
python scripts/run_live.py --config configs/default.yaml --source 0

# offline evaluation against GuardianBench
python scripts/eval.py --config configs/default.yaml --split test

# export INT8 engines for Orin
python -m guardian.deploy.export --config configs/orin.yaml
```

## Hardware

Jetson **Orin NX 16 GB** minimum. Nano will not run this stack. Budget ~10 W sustained
or the wearable becomes hot and heavy. Camera at 1.4–1.6 m mounting height, ~65° HFOV,
global shutter strongly preferred (rolling shutter + gait = depth artifacts).

## Metrics that matter

| Metric | Definition | Target |
| --- | --- | --- |
| Lead time | seconds between alert and hazard entering the 1 m ego cylinder | ≥ 2.0 s median |
| False alarms / km | alerts with no hazard within 1.5 m over the next 3 s | ≤ 1.5 |
| Alert→avoidance | fraction of alerts followed by successful avoidance | ≥ 0.9 |
| Fast-path p99 | frame-in to haptic-out latency | ≤ 150 ms |

mAP is reported only as a component diagnostic. It is not the headline number.

## Safety

This is a **supplement to** a white cane or guide dog, never a replacement. The system
must degrade loudly: if any fast-path stage stalls beyond its budget, `pipeline.py`
emits a continuous degraded-mode tone rather than silence. Silent failure is the one
unacceptable outcome.

Data collection requires IRB approval and informed consent from participants; faces and
license plates in GuardianBench are blurred at ingest (`scripts/record.py --anonymize`).
