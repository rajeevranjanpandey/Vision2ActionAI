# AI Guardian — Next Sprint Roadmap

Five scoped additions, framed PRD-style: problem, solution, MVP boundary, and the
kill-or-keep metric.

**Structural invariant across all five:** none of them touch the fast path's safety
loop. They sit in the slow path, the calibration layer, or a new async channel. New
features do not get to add latency risk to the 90 ms budget just because they are
valuable.

---

## 1. Persistent Hazard Memory (crowdsourced static-hazard layer)

**Problem.** The fast path re-detects the same pothole or A-board every single pass.
Static hazards don't need 10 Hz geometry — they need to be remembered.

**Solution.** The slow-path VLM tags each hazard as static vs. dynamic. Static ones get
geo-pinned (GPS + visual fingerprint) and pushed to a shared map, so the fast path
pre-loads a risk-corridor prior before the camera even confirms it.

**MVP scope.** Single-user local cache first. No cross-user sharing — that is a v2
privacy/consent problem. Out of scope: real-time multi-user sync.

**Metric.** False alarms/km on repeat routes drops measurably vs. first-pass FA/km.

---

## 2. Per-user Conformal Recalibration

**Problem.** The shipped alpha = 0.1 operating point is a population average. A cautious
user and a confident long-cane user have different tolerances for nuisance alerts vs.
missed hazards.

**Solution.** Let the user shift alpha within a bounded range (0.05–0.15) by voice
("fewer alerts" / "warn me more"), recalibrated against their own logged sessions rather
than the global calibration set.

**MVP scope.** A bounded slider on top of the existing split-conformal machinery. No new
model — a per-user threshold override with a hard floor so recall never drops below the
validated minimum.

**Metric.** Self-reported "trust the device" score plus retention at 30/60/90 days.

---

## 3. Fall Detection + Trusted-Contact Alert

**Problem.** Degraded mode protects the user's awareness, but nothing protects them if a
hazard results in a fall or the device disconnects entirely.

**Solution.** An IMU-based fall/impact classifier — reusing the IMU already present for
ground-plane scale — triggers a countdown-then-SMS/push to a pre-registered contact with
last-known GPS, in the spirit of Apple Watch fall detection.

**MVP scope.** Single trusted contact, cellular/SMS fallback, no app required on their
end for v1. Out of scope: full caregiver dashboard.

**Metric.** False-positive fall rate per week. This is the adoption gate — nobody keeps a
device that cries wolf to their family.

---

## 4. Mode-Aware Context Switching (indoor / transit / crossing)

**Problem.** One risk-cone geometry does not fit a train platform, a signalized crosswalk,
and a grocery aisle equally well. The false-alarm/lead-time tradeoff should shift with
context.

**Solution.** A lightweight scene classifier piggybacking on the existing slow-path VLM
call (no new model) sets a context flag that swaps risk-cone parameters and vocabulary.
Platform edge gets a hard geometric floor like the current imminent-contact veto; the
grocery aisle relaxes it.

**MVP scope.** Three contexts: outdoor sidewalk, crossing, transit platform. Out of scope:
full indoor SLAM.

**Metric.** FA/km broken out by context. The platform-edge case must be at least as
conservative as today — an improved average is not sufficient evidence.

---

## 5. Federated Near-Miss Feedback Loop

**Problem.** The risk head is trained on GuardianBench's 73 hazard clips plus simulated
physics. Real deployment surfaces failure modes simulation cannot anticipate, and raw
wearer video cannot be centralized (bystander faces, route privacy).

**Solution.** An on-device flag — "that was a near-miss" / "that was a false alarm", voice
or button — emits a small feature-vector log (the same 8-frame / 14-feature window already
computed), not video. Logs are aggregated federated-style to retrain the ensemble without
raw footage leaving the device.

**MVP scope.** Feature-vector export plus a manual retrain cycle. Full federated averaging
infrastructure is v2.

**Metric.** Late-alert count on held-out real-world clips trends toward 0 — the same bar
the shipped model is already held to.
