# Research protocol

## Why this needs a protocol at all

The system makes claims about a safety-critical device used by people with disabilities.
Two things follow. First, the evaluation has to measure *warning timing*, not detection
accuracy, because a perfectly accurate warning delivered 0.3 s before impact is a
failure. Second, no result involving blind participants is publishable, or ethical,
without IRB approval and disability-community involvement in the study design.

## Claim under test

> A predictive two-path architecture warns of pedestrian-scale hazards with a median
> lead time above 2.0 s and fewer than 2 false alarms per kilometre, on a wearable
> Jetson Orin NX at 10 Hz.

The claim is falsifiable in three independent ways: lead time, false alarm rate, and
on-device latency. All three are reported; none is allowed to be traded away silently.

## Metric definitions

| Metric | Definition | Target |
| --- | --- | --- |
| Lead time (median) | Seconds between the alert and the hazard entering the 1 m ego cylinder | > 2.0 s |
| Lead time (p10) | Same, tenth percentile — the tail is the safety-relevant number | > 1.0 s |
| Recall | Fraction of O&M-labelled hazards that produced a correctly directed alert | > 0.92 |
| False alarms / km | Non-degraded alerts with no hazard within a 4 s window, per km walked | < 2.0 |
| Fast-path p99 | 99th percentile end-to-end fast-path latency on-device | < 100 ms |

Reporting the mean instead of p95/p99 latency is how systems papers hide the exact
tail that causes a missed warning. Do not do it.

## Baselines

1. **Reactive distance alarm** — ultrasonic-style threshold on current range. This is
   what commercial devices do; it is the baseline the lead-time claim must beat.
2. **Detection + depth, no forecasting** — the "obvious" deep-learning system. Isolates
   the contribution of trajectory forecasting rather than better perception.
3. **VLM-only narration** (Qwen2.5-VL at 1 Hz, no fast path) — isolates whether a
   general-purpose VLM alone can act as a safety device. It cannot; the point is to
   show *how badly* on the lead-time tail.
4. **Ours, ablated** — see the ablation table below.

## Ablation table

Produced by `python scripts/eval.py --ablate`.

| Variant | Removes | Question it answers |
| --- | --- | --- |
| `full` | — | — |
| `no_sam2` | Sparse segmentation | Do mask-accurate depth samples matter, or is the box enough? |
| `no_metric_scale` | Ground-plane scale recovery | How much does relative-depth ambiguity cost in lead time? |
| `no_forecast` | 3 s trajectory rollout | The core claim: is prediction better than reaction? |
| `no_vlm` | Slow-path semantics | Does the language path affect safety, or only usefulness? |

`no_forecast` is the row a reviewer will read first. If it does not move lead time
substantially, the paper has no contribution.

## Human study

**Stage 1 — sighted proxy (n = 12).** Blindfolded participants on a closed course.
Purpose is debugging the alert vocabulary and timing, not evidence. Report it as such.

**Stage 2 — blind and low-vision participants (n = 15+), IRB-approved.** Recruited
through O&M instructors and consumer organisations, compensated at professional rates,
with an instructor present at all times. Participants keep their own primary mobility
aid throughout; the device is strictly additive.

Measures: NASA-TLX cognitive load, alert trust and annoyance on a 7-point scale,
semi-structured exit interviews, and objective route-completion timing.

**Non-negotiable safety conditions:**
- The device never replaces a cane or guide dog, and the consent form says so plainly.
- Closed course before any street environment.
- Instructor can halt any trial instantly.
- Degraded-mode audio ("Guardian degraded, rely on your cane") is announced during
  training so participants have heard it before it matters.

## Participatory design

Blind users are co-designers, not test subjects. At minimum: a blind co-author or paid
advisory panel involved before the alert vocabulary is frozen, review of the earcon set
by people who will actually hear it under traffic noise, and a published statement of
who was involved and how. A paper about assistive technology written entirely by
sighted researchers should expect to be asked about this, and should have an answer.

## Threats to validity

- **Depth scale drift.** Ground-plane RANSAC fails on stairs, steep camber, and dense
  crowds. Report scale confidence distributions and the degraded-mode trigger rate;
  do not quietly drop those frames.
- **Dataset bias.** Clips collected in one city, one season, one weather band do not
  generalise. Stratify splits by route and weather, and report per-stratum results.
- **Annotation subjectivity.** "When should a warning have been given" is a judgement
  call. Two independent O&M annotators, adjudication over 0.4 s disagreement, and a
  published Krippendorff's alpha.
- **Novelty effect.** Trust ratings after 20 minutes are not trust. Multi-session
  studies or an explicit limitation statement.
- **Latency measured off-target.** Desktop numbers are irrelevant. All latency comes
  from the Orin, thermally soaked, on battery.

## Reproducibility checklist

- [ ] Config file, git commit, and model hashes recorded in every result JSON
- [ ] Seeds fixed for calibration frame sampling
- [ ] TensorRT engines rebuilt on the target device (engines are not portable)
- [ ] Latency reported as p50/p95/p99, from the Orin, under thermal load
- [ ] GuardianBench annotations released with inter-annotator agreement
- [ ] Failure cases published, including at least one near-miss the system missed
