# AI Guardian — Vision2ActionAI

**A Vision-Language-Action system for predictive mobility assistance for visually impaired pedestrians.**

Repository: <https://github.com/rajeevranjanpandey/Vision2ActionAI>

---

## 1. Problem Statement

Existing assistive vision systems are **descriptive**: they tell a blind user what is
in front of them *now* ("a person, two metres ahead"). By the time a description is
spoken, a cyclist travelling at 5 m/s has already covered 8 metres. Description is
too late for avoidance.

The unsolved problem is **predictive assistance**: estimating, 1.5–3 seconds ahead of
contact, which of the many moving objects in an egocentric scene will actually breach
the walker's personal space — and warning about *only* those. Two failure modes bound
the problem:

- **Misses** are unacceptable (a missed hazard is an injury).
- **False alarms** are almost as bad: a device that warns constantly gets switched off,
  and alarm fatigue destroys the trust a real warning depends on.

The research question is therefore: *can a learned temporal model reduce false alarms
without ever removing a warning the physics says is necessary?*

## 2. What This Repository Contains

Two halves, both complete and runnable.

### `guardian/` — the Python research system

| Area | Modules |
|---|---|
| Perception | `perception/detector.py` (RT-DETR / YOLOv10), `depth.py` (Depth Anything V2), `segmenter.py` (SAM 2), `calibration.py` (ground-plane RANSAC for metric scale), `backprojection.py` |
| Tracking | `tracking/kalman.py`, `tracking/tracker.py` — Hungarian-matched 3D tracks with ego-motion compensation |
| Risk | `risk/ttc.py` (geometric time-to-collision kernel), `risk/learned.py` (deep-ensemble head + fusion), `risk/calibration.py` (temperature scaling + split conformal), `risk/policy.py` (hysteresis + alert arbitration) |
| Learning | `train/simulate.py` (physics scenario generator), `train/features.py` (14-D features, 8-frame window), `train/trainer.py` (pure-numpy MLP ensemble, focal loss) |
| Evaluation | `bench/metrics.py`, `bench/simeval.py`, `bench/evaluate.py` — recall, median/p10 lead time, false alarms per km, ablation arms |
| Interaction | `audio/tts.py`, `audio/haptics.py`, `audio/stt.py` (Whisper), `language/vlm.py` (LLaVA / Qwen2.5-VL) |
| Deployment | `deploy/export.py`, `deploy/quantize.py`, `deploy/benchmark.py` — Jetson Orin latency and thermal profiling |
| Tests | 75 passing tests under `guardian/tests/` |

### `src/` — the browser presentation layer

A TanStack Start site that is not a brochure: it re-implements the trained ensemble in
TypeScript (`src/lib/riskHead.ts`) and replays the training scenarios at 10 Hz
(`src/components/guardian/LiveDecisionDemo.tsx`), alongside a results dashboard
(`ResultsDashboard.tsx`) driven by the real `src/data/results.json` emitted by
`scripts/train_risk.py`.

## 3. Current Features

- **Two-path architecture.** A 10 Hz geometric fast path (detect → depth → track → TTC →
  alert) with a ~90 ms budget, and a 0.5–1 Hz VLM slow path for semantic context.
- **Metric depth from a monocular camera.** Relative depth is anchored to metres via
  ground-plane RANSAC plus IMU pitch, so TTC is in real seconds.
- **Learned risk head with an asymmetric safety contract.** A 5-member MLP ensemble may
  *escalate* freely, but may *suppress* only when the geometry is non-urgent, the
  ensemble agrees, and the pessimistic upper confidence bound is low. Urgent geometry
  can never be vetoed by the model.
- **Calibrated, conformal outputs.** Temperature scaling (ECE 0.0156 → 0.0065) and
  split-conformal thresholds give a distribution-free miss-rate budget at a chosen α.
- **Ablation harness.** Baseline TTC vs head-only vs fused vs oracle. The shipping arm
  keeps recall at 1.0 and removes all late alerts while cutting false alarms per km
  against the baseline.
- **Live browser demo.** Same weights, same physics, no server round-trip.

## 4. Pros and Cons

**Pros**
- Predictive rather than descriptive — warnings arrive with usable lead time.
- Safety argument is structural, not empirical: a mispredicting head degrades the system
  to the constant-velocity baseline, never to silence.
- Outputs are calibrated and conformally controlled, so the miss budget is a dial, not a hope.
- Runs on a single edge board; no cloud dependency in the safety loop.
- Fully reproducible: seeded scripts regenerate every number shown on the site.

**Cons / honest limitations**
- All current metrics come from **simulated physics**, not road data. They demonstrate
  the mechanism, not field performance.
- No user study with blind participants yet — the strongest possible reviewer objection.
- Monocular metric depth degrades on slopes, stairs, and textureless ground.
- SAM 2 is latency-expensive; it is gated by coarse risk rather than run on every frame.
- The VLM path can hallucinate; it is deliberately excluded from any safety decision.
- Poor weather, night, and glass/reflective surfaces are untested.

## 5. Scope

**In scope:** outdoor and indoor pedestrian navigation, dynamic obstacle avoidance,
static obstacle detection in the walking corridor, audio/haptic alerting, scene
question-answering on request, edge deployment on Jetson Nano/Orin.

**Out of scope (deliberately):** road crossing decisions, traffic-signal interpretation as
an authoritative instruction, replacement of a white cane or guide dog, medical or
navigation-grade positioning.

## 6. Constraints

- **Latency:** end-to-end fast path ≤ 100 ms; the alert must precede contact by ≥ 1.5 s.
- **Power/thermal:** must sustain the loop on Orin without throttling in a wearable form factor.
- **Safety:** no learned component may reduce an urgent geometric warning.
- **Privacy:** no raw video leaves the device; the VLM path runs locally.
- **Data:** no public egocentric hazard-onset dataset exists with the needed labels,
  which is why simulation currently bootstraps the head.

## 7. Future Development

1. **GuardianBench** — collect and label a real egocentric hazard-onset dataset
   (hazard clips + matched safe clips), and re-report every ablation on it.
2. **User study** with blind and low-vision participants: trust, alarm tolerance,
   preferred earcon vocabulary, and lead-time preference.
3. **On-device continual adaptation** to a user's gait and typical routes without
   storing raw frames.
4. **Multi-hazard arbitration** — currently the most severe hazard wins; a richer
   spatial-audio scene could convey two.
5. **Quantised deployment** (INT8/TensorRT) with an accuracy-vs-latency Pareto report.
6. **Stair, kerb, and drop-off detection** as a dedicated geometric channel.
7. **Failure-case telemetry** with on-device-only aggregation for model improvement.

## 8. Reproducing the Results

```sh
cd guardian
pip install -r requirements.txt
pytest tests/                                    # 75 tests
PYTHONPATH=. python scripts/train_risk.py        # ~40 s, writes src/data/results.json
PYTHONPATH=. python scripts/export_web_head.py   # writes src/data/risk_head.json
```

Web app:

```sh
npm i
npm run dev
```

## 9. Tech Stack

YOLOv10 / RT-DETR · Depth Anything V2 · SAM 2 · LLaVA / Qwen2.5-VL · Whisper ·
NumPy/SciPy risk stack · Jetson Nano / Orin · TanStack Start · React · TypeScript ·
Tailwind CSS · Recharts

## 10. Ethical Note

This system is an assistive aid, not a safety-certified device. It supplements a cane or
guide dog and must never be presented as a replacement. Every number published on the
project site states whether it came from simulation or from road data.
