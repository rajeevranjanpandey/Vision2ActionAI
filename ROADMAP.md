# AI Guardian — Roadmap

Ten features. Three releases. One rule.

**The rule:** nothing outside the fast path is allowed to add latency to the 10 Hz
geometric safety loop. Every feature below is slow-path, calibration-layer, or
asynchronous by construction. Geometry keeps an absolute veto; semantic and signal
readings are advisory only, always.

---

## v1 — safety core

Nothing else ships until the safety loop is bulletproof. This is the trust foundation.

### 1. Predictive hazard alerts — *built*
Forecast collisions 1.5–3 s ahead via fast-path TTC over forecast trajectories, rather
than describing what is already present. The spine everything else attaches to.
**Kill metric:** median lead time < 1.5 s, or false alarms > 2 / km.

### 2. Traffic signal & crosswalk reading — *specified*
Detect crosswalk geometry and read walk / don't-walk state. Advisory only: a "walk"
reading may never suppress a geometric hold, and a turning vehicle always wins.
Ambiguity routes to the slow-path VLM with a "no signal, listen for traffic" fallback.
**Kill metric:** one case of a signal reading suppressing a geometric hold.

### 10. Fall detection + trusted contact — *built*
Three-phase IMU signature (free-fall → impact → stillness) on the inertial stream that
already supplies ground-plane pitch. Spoken cancel window, then SMS with GPS.
**Kill metric:** > ~1 false alert per week in the field.

---

## v2 — independence

Daily-life utility: beat the incumbents where they have nothing (3, 4), match them where
they do (5, 6).

### 3. Find my way back — *specified*
Voice-tagged waypoints retraced in reverse on the existing three-channel haptic belt.
**Kill metric:** tagging costs more effort than retracing returns.

### 4. Where did I leave it — *specified*
One visual + spatial fingerprint per object (reusing SAM 2), recalled as a directional
haptic pulse and a distance. No persistent object memory exists in any shipping app.
**Kill metric:** recall accuracy < ~80 % in a lived-in room.

### 6. Currency & document reading — *specified*
Mail, menus, labels, banknotes, read aloud. Table stakes; omission makes the device feel
incomplete next to free apps.

### 5. Live scene description — *specified*
Tap-to-ask slow-path narration as an explicit user-facing mode.
**Kill metric:** answer latency > ~3 s.

---

## v3 — world-connected

Riskiest to ship half-working: needs external API reliability and privacy infrastructure.

### 7. Places & errands — *specified*
Nearest pharmacy / ATM / opening hours via Google Places, then a haptic handoff that
takes the last twenty metres to the actual door.
**Kill metric:** API reliability below what someone can plan an errand around.

### 8. Rerouting around known hazards — *specified*
Remembered hazards change the route, not just the alert. Single-user only until consent
and privacy infrastructure exists — no crowd sharing before then.
**Kill metric:** detours costing more walking than the hazard costs risk.

### 9. Human-in-the-loop fallback — *specified*
On model uncertainty, admit it and connect a sighted assistant rather than guess.
**Kill metric:** escalation rate too high to afford, or too low to be honest.

### 11. Live translation — *specified*
Two inputs, one feature: the microphone translates speech spoken to the user and public
announcements; the camera translates signage, menus and printed notices. Slow path only.
Low confidence offers "ask them to repeat" or a human assistant rather than guessing.
**Kill metric:** a mistranslated safety-relevant instruction, or latency long enough that
the conversation has already moved on.

---

## Environment realism (applies to every simulation)

Simulations run inside a shared street condition set — crowd density, ambient dB, light,
dust/rain on the lens, and gait motion — which derives a camera-trust and a mic-trust
number. Every feature degrades against those two numbers: an occluded signal head falls
back to hold, a glared banknote asks for a retake, an 88 dB platform shortens speech and
adds haptics, and lost visual landmarks fall back to dead reckoning with the drift stated
out loud.

---


## Shipped supporting work

Persistent hazard memory, per-user conformal recalibration, mode-aware context switching,
and the federated near-miss loop are implemented under `guardian/memory/`,
`guardian/risk/personal.py`, `guardian/context/`, and `guardian/feedback/`. They surface
in the web build behind feature 8's detail disclosure, since they are the machinery that
makes rerouting and personalisation possible.

---

## Validation protocols

Each v2 feature has a scripted, isolated simulation protocol — setup, step-by-step trace,
and explicit pass/fail gates — in [`VALIDATION_PROTOCOLS.md`](./VALIDATION_PROTOCOLS.md).
Run each scenario alone before integration testing.

The shared rule across all of them: log **confident-wrong answers** as a separately
tracked failure class from **low-confidence / no-answer** outcomes. For a blind user a
wrong answer delivered with confidence is categorically more dangerous than an honest
"I don't know", and a run that does not split those two numbers is not measuring the
thing that matters.

---

## Next session — simulation environment build-out

[`ENVIRONMENT_SPEC.md`](./ENVIRONMENT_SPEC.md) defines the eight environment classes every
feature must be validated in (street, transit, indoor residential, retail/public, mixed
indoor↔outdoor transition, park/plaza, construction zone, and a fall-physics IMU rig),
the sensor-accurate rendering and ground-truth-channel requirements, the
feature→environment mapping, and the build order: A variants → H physics rig (parallel) →
C residential → E transition → D/F/G.

Standing rule: never validate a feature only inside the environment it was designed
around.

---

## Next session — presentation & accessibility rebuild

[`PRESENTATION_SPEC.md`](./PRESENTATION_SPEC.md) defines the next pass: reorder the page
into a trust arc (hook → problem → mechanism → live proof → principles → roadmap → one
CTA), promote "Nothing is allowed to slow the safety loop" to the hero headline, rename
engineer-speak features to benefit-first names, migrate to a HUD palette (`#0B1220` base,
`#FF7A1A` reserved strictly for real hazard states, desaturated green-teal for safe,
`#F5F6F7` text), swap shadows for 1px hairlines, replace emoji glyphs with a single-weight
line icon set, add a looping sense→predict→veto→guide hero SVG, and convert confidence /
urgency readouts to arc gauges.

Hard gate: the simulation must be fully keyboard-operable and screen-reader narratable,
and every shipped colour pair must pass WCAG AA 4.5:1 by measurement. A demo a blind
visitor cannot experience is the worst possible inconsistency for this product.
