# AI Guardian — Simulation Validation Protocols (v2 features)

Every protocol below follows the shape already established by the risk-head validation:
holdout metrics, ablation, honest failure reporting. Each is a **scripted scenario**, a
**step-by-step trace** of what the simulation emits at each moment, and **explicit
pass/fail criteria**. Run each scenario as an isolated simulation before integration
testing.

## Cross-cutting rule (read first)

Log **confident-wrong answers** as a separately tracked failure class from
**low-confidence / no-answer** outcomes. For a blind user, a wrong answer delivered
with confidence is categorically more dangerous than an honest "I don't know" or
"try again". Any simulation run that does not separate these two failure types is not
producing the number that matters.

---

## 1. Traffic signal & crosswalk reading

**Setup.** Synthetic street scene, camera approaching a signalized crosswalk at walking
pace (1.2 m/s), scripted vehicle that either stops, proceeds straight, or makes an
unprotected turn into the crosswalk.

**Trace.**

| t | Scenario | Simulation output |
| --- | --- | --- |
| 0 s | Crosswalk lines enter frame at ~15 m | `crosswalk_lines: detected (conf 0.xx)` → `crossing_mode = ON`; audio "Crosswalk ahead." |
| 1 s | Pedestrian signal enters frame | `pedestrian_signal: detected` → class ∈ {walk, dont_walk, changing}; audio matches state, e.g. "Signal: walk." |
| 2 s (A) | Clean crossing, no vehicle hazard | Risk cone widened to crossing-mode width, no TTC breach → **silence** after the initial announcement |
| 2 s (B) | Turning vehicle closes on the crosswalk path | Fast-path TTC veto fires **independently of signal state**; log signal-state output and alert-arbitration output side by side and assert the alert is NOT suppressed |
| 3 s (C) | Crosswalk lines, no `pedestrian_signal` class in frame | "No signal here — listen for traffic before crossing." fires **once**, not per frame |
| 4 s (D) | Signal occluded by tree/glare, classifier below threshold | System does not guess — falls back to branch C's utterance rather than asserting a state |

**Pass/fail.**
- **Branch B is non-negotiable:** zero tolerance for suppressed alerts under any signal
  state. A single failure fails the feature regardless of aggregate accuracy.
- Signal-state accuracy ≥ 95 % on clean frames; ambiguous frames route to the fallback
  utterance. False-confident-state rate target: **0**.
- Utterance repetition: each state announced once per crossing approach, not per frame.

---

## 2. "Find my way back" breadcrumb navigation

**Setup.** Scripted walk with 3 tagged waypoints (bus stop → café entrance → store), then
a return trip from the store to the bus stop, with one deliberate wrong-turn injection.

**Trace.**
1. **Waypoint tagging** — voice "remember this spot" + label →
   `waypoint_saved: {gps, label, timestamp}`, confirmed audibly: "Saved: café entrance."
2. **Retrace request** — "guide me back to the bus stop" → lookup succeeds, bearing
   computed from current GPS to target.
3. **Straight segment** — continuous straight-tone haptic pulses while bearing error is
   within tolerance (±15°).
4. **Turn segment** — haptic channel switches to left-pulse pattern at the correct
   *distance before* the turn, not at the turn (lead distance, analogous to lead time in
   the hazard system).
5. **Wrong-turn injection** — bearing error grows past tolerance → distinct "off-course"
   pulse, never silent continuation pointing at a now-wrong relative bearing.
6. **Arrival** — within 3 m of the waypoint GPS: guidance stops, audio "Arrived: bus stop."

**Pass/fail.**
- Waypoint recall accuracy 100 % on the exact saved set; no false matches between
  similarly-labeled waypoints.
- Bearing-guidance latency: haptic update lag < 1 s so a walking-pace user cannot
  overshoot a turn.
- Off-course detection triggers within a bounded deviation distance — the user must not
  have to be badly lost first.

---

## 3. "Where did I leave it" — indoor object memory

**Setup.** Scripted single-room indoor environment; object placed and tagged; user moves
elsewhere in the room and requests retrieval; a visually similar distractor is present.

**Trace.**
1. **Tagging** — "remember my keys are here" with camera on the object → SAM 2 mask,
   visual fingerprint extracted, tagged with **IMU-derived relative position** (not GPS).
   Confirmation: "Saved: keys."
2. **User moves** — dead-reckoned position updates via IMU.
3. **Retrieval request** — "Where are my keys?" → fingerprint match against the current
   frame (if in view) and/or stored relative position (if not).
4. **Distractor test** — similar small metallic item in frame; log match confidence for
   both and confirm the tagged object wins.
5. **Guidance output** — haptic pulse toward last-known relative bearing, with an explicit
   caveat if IMU drift has accumulated: "Approximate direction, may have drifted."

**Pass/fail.**
- Fingerprint precision ≥ 90 % correct-object match under distractor conditions — false
  matches actively mislead, which is worse than "no match found".
- Drift-awareness: degraded confidence must be flagged past a defined time/distance
  threshold rather than confidently pointing at a stale position.
- "Not found" is a valid, clearly spoken output. Never a silent failure.

---

## 4. Live scene & object description on demand

**Setup.** Scripted static and dynamic scenes (cluttered desk, street scene, a moving
hazard already known to the fast path) with an on-demand query mid-scenario.

**Trace.**
1. **Idle** — no query: slow-path VLM is not narrating (this mode is distinct from
   continuous hazard enrichment — no double-narration).
2. **Query triggered** — "what's in front of me": current frame(s) to VLM, response
   within the slow path's existing budget (0.5–1 Hz class, not instant).
3. **Concurrent hazard test** — fire the query while a fast-path alert is active. Log both
   timestamps and confirm the hazard alert is neither delayed nor interrupted.
4. **Response quality** — compare VLM output against scripted ground-truth labels. An
   omission of a safety-relevant object the fast path is actively alerting on is a
   **correctness** failure, not merely incompleteness.

**Pass/fail.**
- Zero measured latency impact on concurrent fast-path alerts — test explicitly, never
  assume.
- Description completeness against scripted ground truth, target defined per scene
  complexity.
- No contradiction between a simultaneous hazard alert and the description ("hazard
  ahead" vs "clear path") — any such conflict is a failure.

---

## 5. Currency & document reading

**Setup.** Scripted currency notes (multiple denominations, worn/folded), printed
documents (menu, mail, medication label), varying lighting and angle.

**Trace.**
1. **Clean condition** — flat, well-lit item: OCR/VLM read returned, confidence logged.
2. **Degraded condition** — folded/worn note, angled document, partial occlusion: either a
   correct read with appropriately lower confidence, or an explicit "can't read clearly,
   try repositioning". Never a confident wrong answer.
3. **Medication label test (highest stakes)** — similar-looking dosage numerals
   ("10mg" vs "100mg"). Log confidence score, exact digit string, and any ambiguity flag.
4. **Currency denomination distractor** — two denominations of similar colour/size;
   confirm discrimination and log the confusion matrix.

**Pass/fail.**
- Medication/dosage-numeral accuracy is a **hard gate, not an average**: any misread of
  this content class blocks release regardless of overall accuracy.
- Confidence-gated fallback: below threshold the system asks for repositioning rather
  than guessing.
- Currency confusion rate ≈ 0 for common denomination pairs in the deployment region.

---

## 6. Places & errands (Google Places API)

**Setup.** Scripted queries ("nearest pharmacy", "is this restaurant open") against a
mocked Places response set, including a no-connectivity condition and a hand-off to
breadcrumb navigation.

**Trace.**
1. **Query issued** — API request constructed, mock response returned (name, distance,
   open/closed).
2. **Result read aloud** — spoken output matches the mocked payload exactly; no
   hallucinated detail beyond what the API returned.
3. **No-connectivity branch** — timeout/failure produces an explicit spoken failure
   ("No connection — can't check nearby places right now"), never silence and never a
   stale cached answer presented as current.
4. **Navigate hand-off** — "navigate there" passes destination coordinates to the
   breadcrumb/retrace module (Feature 2's code path). No duplicate navigation logic; the
   haptic behaviour must be identical to a manually-tagged waypoint.
5. **Stale data check** — mocked "open" status that is outdated: phrase as "listed as
   open", never an unconditional guarantee.

**Pass/fail.**
- No fabricated detail beyond the API payload (pass-through correctness).
- Degraded/no-connectivity always produces a clear spoken failure — never silent, never
  stale-as-current.
- Hand-off reuses the existing tested module; no separate untested guidance logic.

---

## 7. Dynamic rerouting around reported hazards

**Setup.** Scripted route with one previously-tagged static hazard (sidewalk closure) on
the direct path, and an alternate route available.

**Trace.**
1. **Hazard tagging (prior session)** — slow-path VLM classifies an obstruction as static
   and geo-pins it locally.
2. **New route request** — a route that would normally pass through the tagged location.
3. **Hazard check** — local hazard cache queried during route proposal; tagged hazard
   flagged.
4. **Rerouting output** — alternate path suggested with a spoken reason ("Rerouting —
   reported obstruction on direct path"), never a silent detour.
5. **Stale-hazard test** — significant elapsed time since tagging: the system must either
   allow user override ("hazard may be cleared, proceed anyway?") or apply a decay/expiry
   policy, rather than treating the pin as permanent.
6. **Live contradiction test** — fast path observes the obstruction is gone: the live
   geometric observation takes precedence. The cache informs route planning and never
   overrides live perception.

**Pass/fail.**
- Live perception always wins over cached static-hazard data on conflict — verify
  explicitly; same "geometry has final say" principle as the signal feature.
- Rerouting always carries a spoken reason; no silent path changes.
- Defined expiry/reconfirmation policy; measure false-persistent-hazard rate over
  simulated time.

---

## 8. Human-in-the-loop fallback

**Setup.** Scripted queries at varying VLM confidence, including a genuinely ambiguous
case (unusual sign, foreign-language label) and connectivity loss during an active human
session.

**Trace.**
1. **High-confidence query** — VLM answers directly, no escalation offered (confirms the
   fallback is not over-triggering on easy cases, which would erode trust in AI-first mode).
2. **Low-confidence query** — confidence below the co-design-panel threshold: "I'm not
   confident — connect to a volunteer?"
3. **User accepts** — handoff sequence: current frame/context passed to a mocked
   human-agent interface, session state logged.
4. **User declines** — system still gives its best AI answer with an explicit uncertainty
   caveat, rather than refusing to answer.
5. **Connectivity loss mid-session** — clear spoken notification ("Volunteer
   disconnected") and fallback to AI-only mode. Never a silent hang where the user is
   unknowingly talking to no one.

**Pass/fail.**
- Escalation threshold calibrated against the co-design panel's tolerance, not an
  arbitrary default; log offer rate against a target range.
- Zero silent failures on connectivity loss during an active session — the equivalent of
  the hazard system's "silent failure is the one unacceptable outcome" rule.
- The decline path must still produce a usable, caveated answer — never a dead end.

---

## 9. Fall detection + trusted contact alert

**Setup.** Scripted IMU traces for (a) a genuine hard fall, (b) false-positive-prone
events (device dropped, sitting down abruptly, running), (c) a fall followed by user
cancellation inside the countdown window.

**Trace.**
1. **Genuine fall** — IMU spike matches the fall signature: classifier fires, countdown
   begins ("Fall detected. Alerting [contact] in 15 seconds. Say cancel to stop.").
2. **No response** — countdown completes, SMS/push dispatched with current GPS to the
   single registered trusted contact; confirmation logged.
3. **User cancels** — spoken "cancel" mid-countdown: countdown halted, nothing sent,
   confirmation spoken ("Cancelled.").
4. **False-positive (a)** — device set down hard: classifier must not trigger, or must
   trigger only an easily-cancelled short countdown rather than immediate dispatch.
5. **False-positive (b)** — jogging, sitting down quickly: log false trigger rate across a
   batch of traces. This number determines whether users tolerate the feature at all.
6. **No-connectivity fall** — genuine fall with no signal: clear local audio indication
   that the alert could **not** be sent.

**Pass/fail.**
- False-positive rate on high-motion non-fall traces is the primary release-blocking
  metric; define an explicit target (e.g. < X per week of normal use) before shipping —
  an unreliable fall alert is worse than none.
- Cancellation must be reliable and fast, including under street-noise audio conditions.
- The no-connectivity case must never present as a successful alert. Hard release gate,
  not a tunable threshold.
