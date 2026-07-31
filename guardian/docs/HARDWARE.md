# Hardware and deployment

## Bill of materials

| Component | Part | Notes |
| --- | --- | --- |
| Compute | Jetson Orin NX 16 GB | 8 GB works only with INT8 and the small VLM |
| Camera | Global-shutter RGB, 120° HFOV, 1280x720 @ 30 fps | Rolling shutter smears during gait; do not use a webcam |
| IMU | 6-axis, 100 Hz+ | Required for pitch; ego-motion compensation degrades badly without it |
| Audio out | Bone-conduction headset | Never occlude the ear canal — ambient hearing is a primary mobility sense |
| Audio in | Beam-forming mic array | Street noise is the hard case |
| Haptics | 3x LRA (left / centre / right) on a shoulder strap | Direction without occupying the audio channel |
| Power | 20 000 mAh USB-C PD | ~4 h at 15 W; battery weight dominates the wearability budget |

Total worn mass should stay under 1.2 kg. Above that, participants stop wearing it
regardless of how good the warnings are, and the study measures nothing.

## Camera mounting

Chest mount at 1.40–1.55 m, pitched down 5–8°. Chest beats head-mount here: the view
does not swing with head turns, which keeps ego-motion compensation tractable and stops
the risk cone from sweeping the pavement every time the user looks around.

Measure `camera.height_m` per participant. It feeds the ground-plane scale recovery
directly — a 10 cm error is a ~7 % range error, which is a ~7 % lead-time error.

## Jetson setup

```bash
# Max clocks, fans on. Thermal throttling silently costs frames.
sudo nvpmodel -m 0
sudo jetson_clocks

# TensorRT engines must be built on the device. They are not portable across
# devices or TensorRT versions.
python -m guardian.deploy.export --config configs/orin.yaml --out artifacts/

# Verify the fast path holds 10 Hz under thermal load before trusting any result.
python -m guardian.deploy.benchmark --config configs/orin.yaml --frames 600
```

## Latency budget (Orin NX 16 GB, INT8)

| Stage | Budget | Notes |
| --- | --- | --- |
| Detect (RT-DETR-l) | 25 ms | INT8 TensorRT |
| Depth (DA-V2-Small) | 30 ms | 518x518, INT8 |
| Gate | < 1 ms | Pure arithmetic |
| SAM 2 (tiny, <= 4 prompts) | 25 ms | Only gated detections — this is why the gate exists |
| Backproject + track | 5 ms | CPU |
| Risk + policy | 2 ms | CPU |
| **Total** | **< 90 ms** | Leaves headroom inside the 100 ms tick |

The VLM runs on a separate thread at 0.5–1 Hz and is explicitly excluded from this
budget. If it ever blocks the fast path, that is a bug, not a tuning problem.

## Degraded mode

The device announces its own failure rather than going quiet. Triggers:

- Fast path exceeds `policy.degraded_after_ms` (default 400 ms)
- Ground-plane scale confidence below 0.15 — metres are no longer metres
- Camera frame timeout
- Thermal throttle detected

Response: "Guardian degraded. Rely on your cane." Haptics continue for any hazard the
geometry can still support. A safety device that fails silently is worse than no device,
because the user has already adapted their behaviour to trust it.
