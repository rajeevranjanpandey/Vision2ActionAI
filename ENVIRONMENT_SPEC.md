# AI Guardian — Simulation Environment Specification

**Status: in progress — the eight classes are wired into the dashboard as selectable simulation contexts (`src/lib/environments.ts`); A and H run against built code, the rest are spec-backed presets.**

Street-view is only one of at least eight environment classes this system must be
validated in. A blind user's day moves through indoor rooms, transit interiors, retail
aisles, and mixed indoor/outdoor transitions — each with a different sensor problem (GPS
denial, clutter density, lighting extremes, crowd occlusion). Treating "the environment"
as a single street scene is why assistive-vision demos look good and then fail the first
time someone opens a cabinet or walks into a station.

Reference human model for the rendered walker / ragdoll rig:
<https://hmthanh.github.io/3d-human-model/>

---

## Part 1 — General 3D environment principles

**Sensor-accurate rendering, not just visual fidelity.** Depth Anything V2 and SAM 2
respond to texture, edge contrast, and material properties, not geometry alone. Flat,
untextured game-pack assets under-stress depth and segmentation. Every environment needs
PBR materials with real roughness/reflectivity, especially:

- Wet/reflective surfaces (rain-slicked crosswalk lines, glossy floors) — break monocular
  depth assumptions.
- Glass and reflective storefronts — false depth returns, phantom detections.
- Low-texture surfaces (blank walls, uniform carpet) — where depth degrades specifically.

**Ground-truth channels, rendered separately from the RGB pass:**

- Per-pixel metric depth (evaluate the depth model against truth, not plausibility).
- Instance segmentation masks (SAM 2 evaluation).
- 3D boxes + velocity vectors for every dynamic actor (TTC / Kalman evaluation).
- A **hazard-onset semantic label** baked into the scene script — an "O&M instructor"
  marker for the exact frame a sighted guide would intervene. This is what makes lead
  time measurable in simulation the same way GuardianBench measures it.

**IMU noise and drift are modeled, not assumed clean.** Ground-plane RANSAC and indoor
dead reckoning both depend on IMU quality. Inject drift, bias, and vibration noise matched
to the actual wearable's IMU spec — otherwise object memory and breadcrumb nav validate
against a sensor that does not exist in the field.

**Camera degradation modes are first-class scenario variables:** rolling shutter during
gait, lens flare/glare (low sun at crosswalks, oncoming headlights), fog/rain particulate,
condensation, and motion blur at walking and jogging pace.

**Dynamic actors need behavior trees, not only scripted splines.** A car on an identical
pre-baked path teaches nothing. Actors need probabilistic behavior — yield/don't-yield
branching, variable reaction time, occlusion-then-reveal — so one environment produces a
distribution of outcomes. The existing "pedestrian who yields" hard case is the norm to
generalise, not the exception.

**Crowd density is an explicit tunable.** Most failure modes occur at moderate-to-high
density, where many actors must be tracked simultaneously and audio alerts compete with
ambient noise. Empty-street simulation dramatically under-tests this.

---

## Part 2 — Environment catalog

### A. Outdoor street & intersection (exists, expand variants)
Sub-variants: 4-way signalized, T-junction, unmarked/unsignalized, midblock, roundabout-
adjacent, raised/tabled crosswalk.
Lighting: midday high sun, overcast (flat low-contrast — harder for depth than it looks),
dusk/dawn low sun into camera, night streetlight-only, night with headlight glare.
Weather/surface: dry, wet (reflective lines, puddle false depth), light fog, heavy rain.
Ground plane: flat asphalt, cambered road (breaks RANSAC assumptions), curb cut vs none,
gravel shoulder, potholes.

### B. Transit environments (new)
- **Platform edge (train/metro)** — highest-consequence environment in the system.
  Hazard onset here is *distance to platform edge*, not TTC to a moving object. Needs
  crowd-density variants, train arrival/departure actor, tactile edge strip as a
  detectable class, PA-system noise floor for alert-audibility testing.
- **Bus interior/boarding** — door-closing timing hazard, step-up geometry, crowded aisle.
- **Escalator/stairs** — step-edge depth discontinuity is a nasty monocular-depth case;
  dedicated scene.

### C. Indoor residential (new — Feature 3)
Rooms: kitchen, living room, bedroom, hallway/entryway (where keys and bags land).
Clutter: sparse / moderate / high (distractor stress test — many similar small objects).
Lighting: window daylight (harsh dynamic range), artificial warm/cool (affects appearance
matching), low-light/lamp-only.
**GPS denial is modeled fully** — loss and multipath, forcing pure IMU dead reckoning.
That is the actual stress condition for Feature 3.

### D. Indoor retail / public buildings (new — Features 4, 5, 6, 8)
- **Grocery aisle** — shelving as obstacle and as description target, product labels for
  OCR, moving cart/shopper actors.
- **Pharmacy counter** — medication-label reading (Feature 5's highest-stakes case);
  varied font sizes, angled presentation, glare off pill-bottle plastic.
- **Restaurant interior** — menu reading, table/chair density, ambient noise floor.
- **Public-building lobby** — signage reading, form/document handling for Feature 6.

### E. Mixed indoor/outdoor transition (new — Feature 2)
Bus stop → café entrance → store interior as one continuous scene with GPS → no-GPS → GPS
transition, threshold detection, and a deliberate wrong-turn branch injected mid-route.
Breadcrumb nav's hardest failure is exactly at the boundary where positioning quality
changes.

### F. Park / open plaza (new — Features 1, 4, 7)
Unstructured space with no path edges: tests the risk cone and description systems against
open geometry rather than corridor-like streets. Benches, low walls, grass/gravel
transitions, off-leash dogs as an unpredictable actor class.

### G. Construction / obstruction zone (new — Feature 7)
Sidewalk closure with signage, temporary fencing, marked detour — where the persistent
hazard cache and live-perception override actually get exercised. Needs a
**"hazard resolved"** variant (fencing removed) as a separate scene state to test
stale-cache-vs-live-perception conflict directly.

### H. Fall / impact physical scenarios (new — Feature 9)
Not a camera scene — a physics rig for IMU trace generation.
- Ragdoll / mocap-driven fall physics across surfaces: hard pavement, grass, stairs, wet
  floor.
- Explicit false-positive motion library: sitting abruptly, dropping the device on a table,
  jogging, jumping, fast stair descent, being jostled in a crowd.
- Curb step-off and stair-miss as the realistic near-fall band between a fall and gait.

---

## Part 3 — Feature-to-environment mapping

| Feature | Primary environment(s) | Required scenario variants |
| --- | --- | --- |
| 1. Traffic signal & crosswalk reading | A | All intersection sub-types; day/night/glare; wet lines; occluded signal; unprotected-turn branch |
| 2. Breadcrumb navigation | E, A, C | GPS→indoor transition; wrong-turn injection; multi-waypoint chained retrace |
| 3. Indoor object memory | C | High-clutter distractors; GPS-denied drift over time; low-light retrieval |
| 4. Live scene description | D, F, C | Concurrent fast-path hazard during query (run in A or D with a live hazard actor) |
| 5. Currency & document reading | D (pharmacy, restaurant) + tabletop lighting rig | Degraded/angled/folded conditions; medication near-miss digit pairs; denomination distractors |
| 6. Places & errands | A, D (POI-dense urban block) | No-connectivity branch; stale open/closed status; hand-off into E for final approach |
| 7. Dynamic rerouting | G, A | Stale-vs-live conflict (hazard-resolved state); alternate route vs no-alternate dead end |
| 8. Human-in-the-loop fallback | D, F, A (ambiguous variants) | Foreign-language signage; non-standard sign geometry; connectivity loss mid-session |
| 9. Fall detection | H (physics rig) | Full false-positive library; genuine falls across all surfaces; no-connectivity dispatch failure |

---

## Part 4 — Build priority

1. **Environment A variants** — partly built from GuardianBench street work; extend
   lighting/weather/intersection coverage first, since Feature 1 depends on it entirely and
   is the most safety-critical new feature.
2. **Environment H (physics rig)** — build in parallel; separate pipeline, blocks on
   nothing.
3. **Environment C (residential)** — needed for Feature 3; its clutter/lighting assets are
   reused by Feature 4's indoor description testing.
4. **Environment E (mixed transition)** — once C and A exist, it is a stitched combination
   plus a GPS-denial layer.
5. **Environments D, F, G** — lower individual safety stakes, build last, but each is
   required before its feature (5, 6, 7, 8) can be validated at all. Do not skip to
   integration testing without them.

**Standing rule:** never validate a feature only inside the environment it was designed
around. Run Feature 1's crosswalk logic in environment F's unstructured plaza, where there
is no clean crosswalk-line geometry — that is usually where the failure mode a
designed-for-purpose scene was built to hide shows up.
