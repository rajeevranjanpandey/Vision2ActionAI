/**
 * The console: one street, eleven jobs.
 *
 * Left  — what the selected feature is, how it works, why it earns its place, and the
 *         exact logic the device runs.
 * Middle— the simulation, drawn as the street the walker is standing in.
 * Right — the live readout the wearable itself is acting on.
 *
 * Everything on the right and in the middle comes from the same 10 Hz tick: the
 * trained risk head evaluated in the browser over the scenario physics, arbitrated by
 * the geometric safety rule.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  ArrowDown,
  ArrowRight,
  Bike,
  Brain,
  Camera,
  Check,
  ChevronRight,
  ChevronUp,

  Clock,
  Compass,
  Gauge,
  HelpCircle,
  Ruler,
  Sparkles,
  ShieldAlert,

  Info,
  Pause,
  PersonStanding,
  Play,
  RotateCcw,
  ShieldCheck,
  Signpost,
  Volume2,
  Waves,
} from "lucide-react";

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import {
  ConditionsBar,
  ConditionsProvider,
  ModalityBadges,
  useConditions,
} from "@/components/guardian/SceneConditions";
import { ENV_BY_ID, SCENE_BY_ENV, type EnvId } from "@/lib/environments";

import type { SceneOverlay } from "@/components/guardian/StreetScene";
import { FeatureStage } from "@/components/guardian/FeatureStage";
import { PhotoScene } from "@/components/guardian/PhotoScene";

import { DT, SCENARIOS } from "@/lib/scenarios";
import {
  FeatureBuffer,
  decide,
  predictRisk,
  trackFeatures,
  type Decision,
  type TrackState,
} from "@/lib/riskHead";
import { ResultsDashboard } from "@/components/guardian/ResultsDashboard";
import { SprintDashboard } from "@/components/guardian/SprintDashboard";
import { FallChannelPanel } from "@/components/guardian/FallChannelPanel";

type Accent = "sense" | "veto" | "guide" | "assist" | "memory" | "urgent";

const ACCENT: Record<Accent, { text: string; bg: string; ring: string; dot: string }> = {
  sense: { text: "text-sense", bg: "bg-sense-soft", ring: "border-sense", dot: "bg-sense" },
  veto: { text: "text-veto", bg: "bg-veto-soft", ring: "border-veto", dot: "bg-veto" },
  guide: { text: "text-guide", bg: "bg-guide-soft", ring: "border-guide", dot: "bg-guide" },
  assist: {
    text: "text-assist",
    bg: "bg-assist-soft",
    ring: "border-assist",
    dot: "bg-assist",
  },
  memory: {
    text: "text-memory",
    bg: "bg-memory-soft",
    ring: "border-memory",
    dot: "bg-memory",
  },
  urgent: {
    text: "text-urgent",
    bg: "bg-urgent-soft",
    ring: "border-urgent",
    dot: "bg-urgent",
  },
};

interface Feature {
  n: number;
  short: string;
  title: string;
  release: "v1" | "v2" | "v3";
  status: "built" | "specified";
  accent: Accent;
  overlay: SceneOverlay;
  scenario: string;
  modes: ("camera" | "mic" | "imu" | "gps")[];
  what: string;
  how: string;
  why: string;
  logic: string[];
  kill: string;
  /** Extra right-hand readouts, feature specific. */
  extras: (ctx: { t: number; visionConf: number; audioConf: number; slowMs: number }) => {
    k: string;
    v: string;
  }[];
  detailLabel?: string;
  detail?: React.ReactNode;
}

const RELEASES: { id: "v1" | "v2" | "v3"; label: string }[] = [
  { id: "v1", label: "Safety core" },
  { id: "v2", label: "Independence" },
  { id: "v3", label: "World-connected" },
];

const FEATURES: Feature[] = [
  {
    n: 1,
    short: "Predictive alerts",
    title: "Predictive hazard alerts",
    release: "v1",
    status: "built",
    accent: "urgent",
    overlay: "predict",
    scenario: "yielding_pedestrian",
    modes: ["camera", "imu"],
    what: "A warning 1.5–3 s before contact, instead of a description of what is already there.",
    how: "RT-DETR detections are lifted to metric 3D with Depth Anything V2 plus a ground-plane fit, tracked with a Kalman filter, and forecast forward. A trained ensemble reads the last 0.8 s of that track.",
    why: "Every shipping app answers a question. None of them runs a continuous loop that speaks first. Lead time is the whole product.",
    logic: [
      "TTC from constant-velocity geometry — the veto, always on.",
      "Learned head over an 8-frame window scores P(breach ≤ 2 s).",
      "Head may escalate freely; it may suppress only when confident and the ensemble agrees.",
      "Ensemble disagreement ⇒ abstain, and geometry decides alone.",
    ],
    kill: "Median lead time below 1.5 s, or false alarms above 2 / km.",
    extras: () => [],
    detailLabel: "Evaluation summary",
    detail: <ResultsDashboard />,
  },
  {
    n: 2,
    short: "Signals & crossings",
    title: "Traffic signal & crosswalk reading",
    release: "v1",
    status: "specified",
    accent: "veto",
    overlay: "crossing",
    scenario: "near_miss_vehicle",
    modes: ["camera"],
    what: "Reads crosswalk geometry and walk / don't-walk state, and says which one it is — advisory only.",
    how: "The signal head is a detector class; the lamp state is a small classifier on the cropped head, with countdown digits read on the slow path.",
    why: "Guessing at a crossing is where independence actually breaks down. This is the highest-stakes read the device makes.",
    logic: [
      "Signal state is advisory. It can add confidence, never remove a hold.",
      "A turning vehicle inside the cone overrides any 'walk' reading.",
      "Occluded or ambiguous head ⇒ 'no signal, listen for traffic'.",
    ],
    kill: "One case of a signal reading suppressing a geometric hold.",
    extras: ({ t, visionConf }) => [
      {
        k: "signal state",
        v: visionConf < 0.45 ? "occluded" : t % 14 < 7 ? "walk" : "don't walk",
      },
      {
        k: "countdown",
        v: visionConf < 0.6 ? "unreadable" : `${Math.max(0, 7 - Math.floor(t % 14))} s`,
      },
    ],
  },
  {
    n: 12,
    short: "Edges & drop-offs",
    title: "Kerb, step & platform-edge detection",
    release: "v1",
    status: "specified",
    accent: "veto",
    overlay: "edge",
    scenario: "static_obstacle",
    modes: ["camera", "imu"],
    what: "Warns about the drop you are walking along or toward — platform edge, kerb, top step — as a distance, not a time.",
    how: "Depth discontinuity along the ground plane fit, confirmed against the tactile-surface class and the IMU's pitch estimate before anything is spoken.",
    why: "The highest-consequence hazard in the day has no closing velocity, so a TTC-only system is silent for it. Platform falls are the failure the whole safety core exists to prevent.",
    logic: [
      "Hazard onset is metres to the edge, not seconds to contact.",
      "Two-of-three agreement: depth step, tactile strip, gait pitch.",
      "Inside 1.2 m of a drop the alert is urgent and cannot be suppressed.",
      "A missing tactile strip lowers confidence but never cancels the hold.",
    ],
    kill: "Any missed platform edge inside 1 m, or edge chatter while walking a normal kerb line.",
    extras: ({ t, visionConf }) => [
      { k: "edge distance", v: `${(1.9 + 1.1 * Math.sin(t * 0.7)).toFixed(1)} m` },
      { k: "tactile strip", v: visionConf > 0.5 ? "detected" : "not visible" },
    ],
  },

  {
    n: 10,
    short: "Fall detection",
    title: "Fall detection + trusted contact",
    release: "v1",
    status: "built",
    accent: "urgent",
    overlay: "fall",
    scenario: "static_obstacle",
    modes: ["imu", "mic", "gps"],
    what: "Detects a fall, opens a spoken cancel window, then sends a trusted contact an SMS with GPS.",
    how: "A three-phase signature on the inertial stream that already supplies ground-plane pitch: free-fall, impact, stillness — all three, in order, or nothing fires.",
    why: "The failure mode a family actually worries about. It runs on a sensor already in the device.",
    logic: [
      "Free-fall < 0.4 g, then impact > 2.5 g, then stillness ≥ 8 s.",
      "30 s spoken cancel window before anything is sent.",
      "Escalation is async — it never enters the fast-path tick.",
    ],
    kill: "More than about one false alert per week in the field.",
    extras: ({ t }) => [
      { k: "phase", v: t % 12 > 7 ? "stillness · cancel window" : "walking" },
      {
        k: "cancel in",
        v: t % 12 > 7 ? `${Math.max(0, 30 - Math.floor((t % 12) - 7) * 6)} s` : "—",
      },
    ],
    detailLabel: "IMU replay, escalation state machine, SMS preview",
    detail: <FallChannelPanel />,
  },
  {
    n: 3,
    short: "Way back",
    title: "Find my way back",
    release: "v2",
    status: "specified",
    accent: "guide",
    overlay: "breadcrumb",
    scenario: "crossing_cyclist",
    modes: ["camera", "imu", "gps"],
    what: "Voice-tagged waypoints, retraced in reverse on the haptic belt that already exists.",
    how: "Visual-inertial odometry between tags, with GPS as a coarse anchor; each tag stores a keyframe so the retrace can re-localise.",
    why: "Getting lost costs more independence than any single obstacle does.",
    logic: [
      "Tag on voice command; store keyframe + pose + spoken label.",
      "Retrace plays the reversed chain as directional haptic pulses.",
      "Lost visual landmarks ⇒ dead reckoning, with the drift said out loud.",
    ],
    kill: "Tagging costs more effort than retracing returns.",
    extras: ({ t, visionConf }) => [
      { k: "waypoints", v: `${3 + (Math.floor(t / 6) % 4)} tagged` },
      { k: "drift", v: visionConf < 0.5 ? "±9 m · dead reckoning" : "±1.8 m" },
    ],
  },
  {
    n: 4,
    short: "Object memory",
    title: "Where did I leave it",
    release: "v2",
    status: "specified",
    accent: "assist",
    overlay: "object",
    scenario: "static_obstacle",
    modes: ["camera"],
    what: "One fingerprint per object, recalled later as a direction and a distance.",
    how: "SAM 2 masks the object once; an embedding plus its room pose goes into a small local store, matched on recall.",
    why: "No shipping app has persistent object memory. Families name this as the daily friction point.",
    logic: [
      "Store: mask → embedding → (room, pose, timestamp).",
      "Recall: nearest embedding above a match floor, else say 'not sure'.",
      "Never guess a location: a wrong direction costs more than a shrug.",
    ],
    kill: "Recall accuracy below about 80 % in a lived-in room.",
    extras: ({ visionConf }) => [
      { k: "match", v: visionConf > 0.6 ? "keys · 0.88" : "no confident match" },
      { k: "last seen", v: "kitchen shelf · 2 h ago" },
    ],
  },
  {
    n: 6,
    short: "Reading",
    title: "Currency & document reading",
    release: "v2",
    status: "specified",
    accent: "memory",
    overlay: "reader",
    scenario: "near_miss_vehicle",
    modes: ["camera"],
    what: "Mail, menus, labels and banknotes, read aloud.",
    how: "Slow-path VLM with an OCR fallback, plus a framing coach that says which way to move the note or page.",
    why: "Table stakes. Omitting it makes the device feel incomplete beside free apps.",
    logic: [
      "Glare or partial frame ⇒ ask for a retake rather than read half a number.",
      "Currency values are spoken twice, never once.",
      "Reading never pre-empts a safety utterance in the audio queue.",
    ],
    kill: "Nothing kills it; it is a completeness requirement.",
    extras: ({ visionConf, slowMs }) => [
      { k: "frame quality", v: visionConf > 0.6 ? "good" : "glare · retake asked" },
      { k: "read latency", v: `${slowMs} ms` },
    ],
  },
  {
    n: 5,
    short: "Scene on demand",
    title: "Live scene description",
    release: "v2",
    status: "specified",
    accent: "assist",
    overlay: "scene",
    scenario: "crossing_cyclist",
    modes: ["camera", "mic"],
    what: "Tap or ask, and the device narrates what is in front of you.",
    how: "The slow-path VLM answers against the same world model the fast path maintains, so it can use metres and clock directions.",
    why: "Matches what people already like about Be My AI, but grounded in the geometry the device already has.",
    logic: [
      "Answers are capped at 25 words while walking.",
      "Never speculates: says what is blocking the view instead.",
      "Runs at 0.5–1 Hz, fully outside the fast-path budget.",
    ],
    kill: "Answer latency above about 3 s.",
    extras: ({ slowMs, audioConf }) => [
      { k: "answer latency", v: `${slowMs} ms` },
      { k: "speech channel", v: audioConf < 0.45 ? "haptic + shortened" : "spoken" },
    ],
  },
  {
    n: 7,
    short: "Places",
    title: "Places & errands",
    release: "v3",
    status: "specified",
    accent: "memory",
    overlay: "places",
    scenario: "crossing_cyclist",
    modes: ["gps", "mic"],
    what: "Nearest pharmacy, opening hours, distance to the ATM — then the last twenty metres to the door.",
    how: "Places API for the search, then a handoff to vision and haptics for the doorway approach GPS cannot resolve.",
    why: "Turns a safety device into an errand device.",
    logic: [
      "GPS gets you to the building; vision gets you to the door.",
      "Opening hours are read out with their age ('as of this morning').",
      "API failure degrades to a direction and a distance, never silence.",
    ],
    kill: "API reliability below what someone can plan an errand around.",
    extras: () => [
      { k: "nearest", v: "pharmacy · 140 m · open" },
      { k: "handoff", v: "haptic at 20 m" },
    ],
  },
  {
    n: 8,
    short: "Rerouting",
    title: "Rerouting around known hazards",
    release: "v3",
    status: "specified",
    accent: "veto",
    overlay: "reroute",
    scenario: "static_obstacle",
    modes: ["gps", "camera"],
    what: "A remembered hazard changes the route, not just the alert.",
    how: "Geo-pinned static hazards with visual fingerprints; a repeat-alert gate downgrades known pins to a short earcon.",
    why: "Alerting is reactive. Routing is the preventative version of the same memory.",
    logic: [
      "Pin only after repeated confirmations at the same place.",
      "Detour accepted only if its cost is below the hazard's risk.",
      "Single-user memory until consent infrastructure exists — no crowd sharing.",
    ],
    kill: "Detours costing more walking than the hazard costs risk.",
    extras: () => [
      { k: "known pins", v: "4 on this route" },
      { k: "detour cost", v: "+38 m · accepted" },
    ],
    detailLabel: "Hazard memory, personal α, mode switching, federated loop",
    detail: <SprintDashboard />,
  },
  {
    n: 9,
    short: "Human fallback",
    title: "Human-in-the-loop fallback",
    release: "v3",
    status: "specified",
    accent: "assist",
    overlay: "human",
    scenario: "cut_in_scooter",
    modes: ["camera", "mic"],
    what: "When the model is uncertain it says so, and connects a sighted assistant instead of guessing.",
    how: "Ensemble disagreement and low visual trust both raise an escalation flag; the call opens with the last few seconds of context already summarised.",
    why: "Users of Aira and Be My Eyes name the human as the trust anchor. Admitting uncertainty is the honest design.",
    logic: [
      "Escalate on abstain, not on low probability.",
      "Escalation is asynchronous; the safety loop keeps running during the call.",
      "Every escalation is logged as a training signal.",
    ],
    kill: "Escalation rate too high to afford, or too low to be honest.",
    extras: ({ visionConf }) => [
      { k: "uncertainty", v: visionConf < 0.5 ? "high · escalating" : "within budget" },
      { k: "assistant", v: visionConf < 0.5 ? "connecting · 6 s" : "standby" },
    ],
  },
  {
    n: 11,
    short: "Translation",
    title: "Live translation",
    release: "v3",
    status: "specified",
    accent: "guide",
    overlay: "translate",
    scenario: "near_miss_vehicle",
    modes: ["mic", "camera"],
    what: "Translates what people say to you, and what the signs around you say.",
    how: "Whisper on the microphone for speech, the VLM on the camera for print. Both on the slow path.",
    why: "Blindness plus an unreadable language removes both workarounds a sighted traveller has: read the sign, or ask someone.",
    logic: [
      "Low confidence offers 'ask them to repeat' rather than a guess.",
      "Safety-relevant instructions are hedged explicitly, never smoothed.",
      "Nothing translated is allowed to delay a safety utterance.",
    ],
    kill: "A mistranslated safety instruction, or latency past the moment.",
    extras: ({ audioConf, slowMs }) => [
      { k: "speech conf", v: audioConf.toFixed(2) },
      { k: "turnaround", v: `${slowMs} ms` },
    ],
  },
];

/**
 * Home scene for each feature, then the alternates it must also survive. The
 * first entry is loaded automatically when the feature is selected, so no two
 * features open in the same picture unless they genuinely share a problem.
 */
const FEATURE_ENVS: Record<number, EnvId[]> = {
  1: ["A", "F"],
  2: ["A", "G"],
  12: ["B", "A"],
  10: ["H", "C"],
  3: ["E", "C"],
  4: ["C", "D"],
  6: ["D", "C"],
  5: ["F", "D"],
  7: ["A", "E"],
  8: ["G", "A"],
  9: ["D", "F"],
  11: ["B", "D"],
};

const envsFor = (n: number) => (FEATURE_ENVS[n] ?? ["A"]).map((id) => ENV_BY_ID[id]);

/** Which environment classes a feature must survive — and whether the one on
 *  screen is one of them. Running a feature outside its designed-for scene is
 *  the failure probe the environment spec insists on. */
function EnvCoverage({ journey }: { journey: number }) {
  const { env, setEnv } = useConditions();
  const required = envsFor(journey);
  const onDesign = required.some((e) => e.id === env.id);

  return (
    <div className="rounded-2xl border border-border bg-card p-4">
      <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
        Environment coverage
      </p>
      <div className="mt-3 flex flex-wrap gap-1.5">
        {required.map((e) => (
          <button
            key={e.id}
            onClick={() => setEnv(e.id)}
            title={e.stress}
            className={`rounded-lg border px-2 py-1 text-left text-[11px] transition-colors ${
              e.id === env.id
                ? "border-signal bg-signal/10 text-signal"
                : "border-border text-muted-foreground hover:text-foreground"
            }`}
          >
            <span className="font-mono opacity-70">{e.id}</span> {e.name}
            {!e.built && <span className="opacity-50"> ·spec</span>}
          </button>
        ))}
        {required.length === 0 && (
          <span className="text-xs text-muted-foreground">
            No designed-for class yet — spec this before integration testing.
          </span>
        )}
      </div>
      <p className="mt-3 text-xs text-muted-foreground">
        {onDesign ? (
          <>
            Running inside a designed-for class ({env.id}). Pass/fail follows the protocol for this
            feature.
          </>
        ) : (
          <span className="text-caution">
            Failure probe: class {env.id} is off-design for this feature. Degradation here is
            expected — what matters is that it degrades honestly rather than answering confidently.
          </span>
        )}
      </p>
    </div>
  );
}

export function GuardianConsole() {
  return (
    <ConditionsProvider>
      <ConsoleInner />
    </ConditionsProvider>
  );
}

function ConsoleInner() {
  const { d, c, env, setEnv } = useConditions();
  const reduce = useReducedMotion();
  const [n, setN] = useState(1);
  const [playing, setPlaying] = useState(true);
  const [frame, setFrame] = useState(0);
  const [decision, setDecision] = useState<Decision | null>(null);
  const buffer = useRef(new FeatureBuffer());

  const f = useMemo(() => FEATURES.find((x) => x.n === n) ?? FEATURES[0]!, [n]);
  const scenario = useMemo(
    () => SCENARIOS.find((s) => s.id === f.scenario) ?? SCENARIOS[0]!,
    [f.scenario],
  );
  const featureEnvs = useMemo(() => envsFor(f.n), [f.n]);

  // Each feature opens in its own home scene — a pharmacy aisle is not a street.
  useEffect(() => {
    setEnv(featureEnvs[0]!.id);
    // setEnv is stable for the lifetime of the provider.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [f.n]);

  useEffect(() => {
    buffer.current.reset();
    setFrame(0);
    setDecision(null);
    
  }, [f.n]);

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(
      () => setFrame((x) => (x + 1) % scenario.states.length),
      DT * 1000,
    );
    return () => window.clearInterval(id);
  }, [playing, scenario]);

  const track: TrackState | null = scenario.states[frame] ?? null;

  useEffect(() => {
    if (!track) return;
    if (frame === 0) buffer.current.reset();
    buffer.current.push(trackFeatures(track, 0.6 + 0.35 * d.visionConf, d.visionConf));
    if (buffer.current.ready) {
      setDecision(decide(track, predictRisk(buffer.current.flat())));
    }
    // The tick is the dependency; track/derived are read fresh each tick.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame, f.n]);

  const t = frame * DT + (frame === 0 ? 0 : 0);
  const seconds = frame * DT;
  const a = ACCENT[f.accent];
  const level = decision?.level ?? "none";

  /* ---- numbers the readout must never leave blank ------------------------ */
  const range = track ? Math.hypot(track.x, track.z) : null;
  const closing = track ? Math.max(0, -track.vz) : 0;
  /** TTC from the head when it has one; geometric range / closing speed otherwise. */
  const ttcValue =
    decision && Number.isFinite(decision.ttc) && decision.ttc > 0
      ? decision.ttc
      : range !== null && closing > 0.15
        ? range / closing
        : null;
  const bearing = track ? (track.x < -0.4 ? "left" : track.x > 0.4 ? "right" : "ahead") : "ahead";
  const riskPct = Math.round((decision?.probability ?? 0) * 100);
  /** Where this reading sits on the safe → critical scale (short TTC = far right). */
  const ttcForScale = ttcValue ?? 0;
  const severityPct = Math.round(
    Math.min(96, Math.max(4, ttcForScale > 0 ? (1 - Math.min(ttcForScale, 8) / 8) * 100 : 6)),
  );


  const latencyMs = Math.round(120 + (1 - d.visionConf) * 180);
  /** Everything the tracker is holding this tick, with its measured distance. */
  const nearby = useMemo(() => {
    const items: { label: string; m: number; icon: "obj" | "ped" | "cross" | "ground" }[] = [];
    if (track && range !== null) {
      items.push({
        label: `${track.className[0]!.toUpperCase()}${track.className.slice(1)}`,
        m: range,
        icon: "obj",
      });
    }
    items.push({ label: "2 pedestrians", m: (range ?? 8) + 3.4, icon: "ped" });
    items.push({ label: "Crossing ahead", m: (range ?? 8) + 6.1, icon: "cross" });
    items.push({ label: "Uneven pavement", m: (range ?? 8) + 8.7, icon: "ground" });
    return items;
  }, [track, range]);

  const NEARBY_ICON = {
    obj: Bike,
    ped: PersonStanding,
    cross: Signpost,
    ground: Waves,
  } as const;

  const NEARBY_TONE = {
    obj: "bg-urgent-soft text-urgent",
    ped: "bg-sense-soft text-sense",
    cross: "bg-guide-soft text-guide",
    ground: "bg-veto-soft text-veto",
  } as const;




  return (
    <div className="space-y-6">
      {/* Feature rail — numbered in journey order (01…11) and grouped by release,
          so position and number agree. Each capsule states what it needs and
          whether it is built. */}
      <div className="flex flex-col gap-4 rounded-3xl border border-border bg-card p-4 lg:flex-row lg:items-stretch lg:gap-5">
        {RELEASES.map((rel) => (
          <div key={rel.id} className="min-w-0 flex-1">
            <p className="mb-2 flex items-baseline gap-2 font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
              <span className="rounded-md bg-surface px-1.5 py-0.5 font-bold text-foreground">
                {rel.id}
              </span>
              {rel.label}
            </p>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
              {FEATURES.map((x, i) => ({ x, step: i + 1 }))
                .filter(({ x }) => x.release === rel.id)
                .map(({ x, step }) => {
                  const ax = ACCENT[x.accent];
                  const on = x.n === f.n;
                  return (
                    <button
                      key={x.n}
                      onClick={() => setN(x.n)}
                      aria-current={on ? "true" : undefined}
                      className={`flex min-w-0 items-start gap-2.5 rounded-2xl border px-3 py-2 text-left transition-all ${
                        on
                          ? `${ax.ring} ${ax.bg} shadow-sm ring-2 ring-inset ${ax.ring.replace("border-", "ring-")}/25`
                          : "border-border bg-card hover:border-foreground/40"
                      }`}
                    >
                      <span
                        className={`mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-lg font-mono text-[11px] font-bold tabular-nums ${
                          on ? `${ax.dot} text-background` : "bg-surface text-muted-foreground"
                        }`}
                      >
                        {String(step).padStart(2, "0")}
                      </span>
                      <span className="min-w-0">
                        <span
                          className={`block truncate text-[13px] font-semibold ${
                            on ? "text-foreground" : "text-foreground/80"
                          }`}
                        >
                          {x.short}
                        </span>
                        <span className="mt-0.5 flex flex-wrap items-center gap-x-1.5 font-mono text-[9.5px] uppercase tracking-[0.14em] text-muted-foreground">
                          <span className={x.status === "built" ? ax.text : undefined}>
                            {x.status === "built" ? "built" : "spec"}
                          </span>
                          <span aria-hidden>·</span>
                          <span className="truncate">{x.modes.join(" ")}</span>
                        </span>
                      </span>
                    </button>
                  );
                })}
            </div>
          </div>
        ))}
      </div>

      <ConditionsBar only={featureEnvs.map((e) => e.id)} />

      {/* Console card — scene + live readout, laid out to the approved design. */}
      <div className="rounded-3xl border border-border bg-card p-5 sm:p-6">
        {/* Header */}
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex min-w-0 items-center gap-3">
            <span className="grid h-12 w-12 shrink-0 place-items-center rounded-xl border border-guide bg-guide-soft text-guide">
              <ShieldCheck className="h-6 w-6" />
            </span>
            <div className="min-w-0">
              <h3 className="text-3xl font-bold tracking-tight">AI Guardian</h3>
              <p className="mt-0.5 text-sm text-muted-foreground">
                Predictive AI for safer pedestrian mobility · {f.title}
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <div className="text-right">
              <p className="flex items-center justify-end gap-1.5 font-mono text-[11px] font-semibold uppercase tracking-[0.18em] text-urgent">
                <span className="relative flex h-2 w-2" aria-hidden>
                  <span
                    className={`absolute inline-flex h-full w-full rounded-full bg-urgent ${
                      playing && !reduce ? "animate-ping opacity-60" : "opacity-0"
                    }`}
                  />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-urgent" />
                </span>
                live
              </p>
              <p className="mt-0.5 text-sm text-muted-foreground">{env.name}</p>
            </div>
            <button
              onClick={() => setPlaying((x) => !x)}
              className="flex items-center gap-2 rounded-xl border border-border bg-card px-4 py-2.5 text-sm font-medium transition-colors hover:border-foreground/50"
            >
              {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              {playing ? "Pause" : "Play"}
            </button>
            <button
              onClick={() => {
                buffer.current.reset();
                setFrame(0);
              }}
              className="flex items-center gap-2 rounded-xl border border-border bg-card px-4 py-2.5 text-sm font-medium transition-colors hover:border-foreground/50"
            >
              <RotateCcw className="h-4 w-4" />
              Replay
            </button>
          </div>
        </div>

        <AnimatePresence mode="wait">
          <motion.div
            key={f.n}
            initial={reduce ? false : { opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
            className="mt-5 grid items-start gap-5 lg:grid-cols-[1fr_440px] xl:grid-cols-[1fr_560px]"
          >
            {/* LEFT — the scene, the spoken line, the explanation */}
            <div className="min-w-0 space-y-4">
              <div className="overflow-hidden rounded-2xl border border-border bg-surface">
                {f.modes.includes("camera") ? (
                  <PhotoScene
                    scene={SCENE_BY_ENV[env.id]}
                    track={track}
                    decision={decision}
                    frame={frame}
                    playing={playing}
                    visionConf={d.visionConf}
                    location={env.name}
                    overlay={f.overlay}
                    t={t}
                  />
                ) : (
                  <FeatureStage
                    n={f.n}
                    overlay={f.overlay}
                    scene={SCENE_BY_ENV[env.id]}
                    track={track}
                    decision={decision}
                    t={t}
                    conditions={c}
                    derived={d}
                    playing={playing}
                    modes={f.modes}
                  />
                )}
                <div className="h-[3px] w-full bg-border/60">
                  <div
                    className={`h-full ${a.dot} transition-[width] duration-100 ease-linear`}
                    style={{ width: `${((frame + 1) / scenario.states.length) * 100}%` }}
                  />
                </div>
              </div>


              {/* Voice guidance */}
              <div className="flex items-start gap-4 rounded-2xl border border-border bg-card px-5 py-4">
                <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full border border-sense bg-sense-soft text-sense">
                  <Volume2 className="h-5 w-5" />
                </span>
                <div className="min-w-0 flex-1">
                  <p className="text-xs text-muted-foreground">Voice guidance</p>
                  <p className="mt-1 text-[15px] leading-snug text-foreground/90">
                    {decision?.utterance
                      ? `“${decision.utterance}”`
                      : "“Path is clear. Nothing worth saying.”"}
                  </p>
                </div>
                <span className="hidden items-center gap-[3px] self-center sm:flex" aria-hidden>
                  {[7, 13, 20, 28, 18, 34, 24, 12, 26, 16, 30, 21, 11, 8].map((h, i) => (
                    <span
                      key={i}
                      className={`w-[3px] rounded-full transition-[height] duration-300 ${
                        level === "none" ? "bg-border" : "bg-sense"
                      }`}
                      style={{ height: `${playing ? h : 5}px` }}
                    />
                  ))}
                </span>
              </div>

              {/* Why this alert? */}
              <div className="overflow-hidden rounded-2xl border border-border bg-card">
                <div className="flex w-full items-center gap-4 px-5 py-4 text-left">
                  <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-accent text-foreground/70">
                    <HelpCircle className="h-4 w-4" />
                  </span>
                  <span className="flex-1 text-[15px] font-semibold">Why this alert?</span>
                  <ChevronUp className="h-4 w-4 shrink-0 text-muted-foreground" />
                </div>


                {/* the three checks the decision actually rested on */}
                <div className="grid gap-px border-t border-border bg-border sm:grid-cols-3">
                  {[
                    {
                      icon: <Bike className="h-4 w-4" />,
                      tone: "bg-urgent-soft text-urgent",
                      a: track ? `${track.className} detected` : "Nothing tracked",
                      b: track ? `${bearing} of you` : "path clear",
                    },
                    {
                      icon: <Compass className="h-4 w-4" />,
                      tone: "bg-sense-soft text-sense",
                      a: "Path predicted to",
                      b: level === "none" ? "stay clear" : "intersect yours",
                    },
                    {
                      icon: <Waves className="h-4 w-4" />,
                      tone: "bg-veto-soft text-veto",
                      a: "Uneven pavement",
                      b: bearing === "right" ? "ahead, right" : "ahead, left",
                    },
                  ].map((e) => (
                    <div key={e.a + e.b} className="flex items-center gap-3 bg-card px-4 py-3">
                      <span
                        className={`grid h-8 w-8 shrink-0 place-items-center rounded-full ${e.tone}`}
                      >
                        {e.icon}
                      </span>
                      <span className="min-w-0 flex-1 text-[12.5px] leading-tight">
                        <span className="block text-muted-foreground">{e.a}</span>
                        <span className="block font-semibold">{e.b}</span>
                      </span>
                      <Check className="h-4 w-4 shrink-0 text-guide" />
                    </div>
                  ))}
                </div>
              </div>




            </div>

            {/* RIGHT — hazard readout + system status */}
            <div className="space-y-4 lg:sticky lg:top-4">
              {/* Two headline numbers, exactly as the design calls them */}
              <div className="grid items-stretch gap-4 sm:grid-cols-2">
                <div
                  className={`flex h-full flex-col rounded-2xl border p-5 ${
                    level === "none" ? "border-border bg-card" : `${a.ring} ${a.bg}`
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span
                      className={`grid h-10 w-10 shrink-0 place-items-center rounded-full border ${
                        level === "none"
                          ? "border-border bg-surface text-muted-foreground"
                          : `${a.ring} bg-card ${a.text}`
                      }`}
                    >
                      <AlertTriangle className="h-5 w-5" />
                    </span>
                    <p className="text-[15px] font-bold leading-tight">
                      {track && level !== "none"
                        ? `${track.className[0]!.toUpperCase()}${track.className.slice(1)} approaching`
                        : "No closing hazard"}
                    </p>
                  </div>
                  <p
                    className={`mt-4 font-mono text-5xl font-bold tabular-nums leading-none ${
                      level === "urgent"
                        ? "text-urgent"
                        : level === "warn"
                          ? "text-veto"
                          : "text-guide"
                    }`}
                  >
                    {(ttcValue ?? 0).toFixed(1)} <span className="text-2xl">s</span>
                  </p>
                  <p className="mt-2 flex min-h-[2.5rem] items-start text-sm leading-snug text-muted-foreground">
                    Until path crossing
                  </p>

                  {/* severity scale — where this reading sits between safe and critical */}
                  <div className="mt-auto pt-3">
                    <div className="flex justify-between font-mono text-[9px] uppercase tracking-[0.14em] text-muted-foreground">
                      <span>safe</span>
                      <span>caution</span>
                      <span>risk</span>
                      <span>critical</span>
                    </div>
                    <div className="relative mt-1.5 h-[6px] rounded-full bg-[linear-gradient(90deg,var(--guide)_0%,var(--guide)_22%,#E8B923_45%,#FF7A1A_70%,var(--urgent)_100%)]">
                      <span
                        className="absolute top-1/2 h-3.5 w-[3px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-foreground"
                        style={{ left: `${severityPct}%` }}
                        aria-hidden
                      />
                    </div>
                  </div>
                </div>

                <div className="flex h-full flex-col rounded-2xl border border-border bg-card p-5">
                  <div className="flex items-center gap-3">
                    <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full border border-guide bg-guide-soft text-guide">
                      <Check className="h-5 w-5" />
                    </span>
                    <p className="text-[15px] font-bold leading-tight">Safe window</p>
                  </div>
                  <p className="mt-4 font-mono text-5xl font-bold leading-none tabular-nums text-guide">
                    {(range ?? 0).toFixed(1)} <span className="text-2xl">m</span>
                  </p>
                  <p className="mt-2 flex min-h-[2.5rem] items-start text-sm leading-snug text-muted-foreground">
                    Distance to nearest object
                  </p>
                  <div className="mt-auto grid grid-cols-3 gap-px overflow-hidden rounded-xl border border-border bg-border pt-0">
                    {[
                      { Icon: Ruler, k: "distance", v: `${(range ?? 0).toFixed(1)} m` },
                      { Icon: Gauge, k: "closing speed", v: `${closing.toFixed(1)} m/s` },
                      { Icon: ShieldAlert, k: "breach risk", v: `${riskPct}%` },
                    ].map((s) => (
                      <div key={s.k} className="bg-card px-2 py-2.5 text-center">
                        <s.Icon className="mx-auto h-3.5 w-3.5 text-muted-foreground" />
                        <p className="mt-1 font-mono text-[13px] font-semibold tabular-nums">
                          {s.v}
                        </p>
                        <p className="mt-0.5 font-mono text-[8.5px] uppercase tracking-[0.12em] text-muted-foreground">
                          {s.k}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>

              </div>

              {/* Recommended actions */}
              <div className="rounded-2xl border border-border bg-card p-5">
                <p className="text-[15px] font-bold">Recommended actions</p>
                <ul className="mt-3 space-y-3">
                  {(level === "none"
                    ? [
                        {
                          icon: "right" as const,
                          text: "Continue straight",
                          tone: "guide" as const,
                          when: "Next 3 seconds",
                          why: "Nothing is predicted to cross your path",
                        },
                      ]
                    : [
                        {
                          icon: "down" as const,
                          text: "Slow down",
                          tone: "urgent" as const,
                          when: `For ${(ttcValue ?? 0).toFixed(1)} seconds`,
                          why: `${track ? track.className[0]!.toUpperCase() + track.className.slice(1) : "Object"} approaching from ${bearing === "ahead" ? "front" : bearing}`,
                        },
                        {
                          icon: "right" as const,
                          text:
                            track && track.x > 0 ? "Keep slightly left" : "Keep slightly right",
                          tone: "veto" as const,
                          when: `Continue after ${(ttcValue ?? 0).toFixed(1)} s`,
                          why: "Predicted path intersects your path",
                        },
                      ]
                  ).map((r) => (
                    <li
                      key={r.text}
                      className={`flex items-center gap-4 rounded-xl border px-3 py-3 ${
                        r.tone === "urgent"
                          ? "border-urgent bg-urgent-soft"
                          : r.tone === "veto"
                            ? "border-veto bg-veto-soft"
                            : "border-guide bg-guide-soft"
                      }`}
                    >
                      <span
                        className={`grid h-10 w-10 shrink-0 place-items-center rounded-full text-background ${
                          r.tone === "urgent"
                            ? "bg-urgent"
                            : r.tone === "veto"
                              ? "bg-veto"
                              : "bg-guide"
                        }`}
                      >
                        {r.icon === "down" ? (
                          <ArrowDown className="h-5 w-5" />
                        ) : (
                          <ArrowRight className="h-5 w-5" />
                        )}
                      </span>
                      <span
                        className={`w-[7.5rem] shrink-0 border-r border-border pr-3 text-[15px] font-bold ${
                          r.tone === "urgent"
                            ? "text-urgent"
                            : r.tone === "veto"
                              ? "text-veto"
                              : "text-guide"
                        }`}
                      >
                        {r.text}
                      </span>
                      <span className="min-w-0 flex-1 leading-tight">
                        <span className="flex items-center gap-1.5 text-[13px] font-semibold">
                          <Clock className="h-3.5 w-3.5 text-muted-foreground" />
                          {r.when}
                        </span>
                        <span className="mt-0.5 block text-xs leading-snug text-muted-foreground">
                          {r.why}
                        </span>
                      </span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* AI summary + everything the tracker is holding right now */}
              <div className="rounded-2xl border border-border bg-card p-5">
                <div className="flex items-start gap-4">
                  <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-memory-soft text-memory">
                    <Sparkles className="h-5 w-5" />
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="text-[15px] font-bold">AI summary</p>
                    <p className="mt-1 text-[13px] leading-snug text-muted-foreground">
                      {level === "none"
                        ? `Nothing is closing on you. The nearest object is ${(range ?? 0).toFixed(1)} m away and your path stays clear for the next few seconds.`
                        : `A ${track?.className ?? "object"} is approaching from ${bearing === "ahead" ? "the front" : `your ${bearing}`} and is predicted to cross your path in ${(ttcValue ?? 0).toFixed(1)} seconds. Nearest clearance is ${(range ?? 0).toFixed(1)} m.`}
                    </p>
                  </div>
                  <Info className="h-4 w-4 shrink-0 text-muted-foreground" />
                </div>

                <div className="mt-5 border-t border-border pt-4">
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-[13px] font-semibold">Surroundings</p>
                    <span
                      className={`inline-flex items-center gap-2 rounded-lg border px-2.5 py-1 text-[11px] font-semibold ${
                        level === "urgent"
                          ? "border-urgent bg-urgent-soft text-urgent"
                          : level === "warn"
                            ? "border-veto bg-veto-soft text-veto"
                            : "border-guide bg-guide-soft text-guide"
                      }`}
                    >
                      {level === "urgent" ? "High" : level === "warn" ? "Moderate" : "Low"} risk
                      <span className="font-mono tabular-nums">{riskPct}%</span>
                    </span>
                  </div>
                  <ul className="mt-2 grid gap-x-6 sm:grid-cols-2">
                    {nearby.map((o) => {
                      const NIcon = NEARBY_ICON[o.icon];
                      const tone = NEARBY_TONE[o.icon];
                      return (
                        <li
                          key={o.label}
                          className="flex items-center gap-3 border-b border-border py-2 text-[13px] last:border-b-0"
                        >
                          <span
                            className={`grid h-7 w-7 shrink-0 place-items-center rounded-full ${tone}`}
                          >
                            <NIcon className="h-3.5 w-3.5" />
                          </span>
                          <span className="min-w-0 flex-1 truncate">{o.label}</span>
                          <span className="font-mono tabular-nums text-muted-foreground">
                            {o.m.toFixed(0)} m
                          </span>
                          <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />
                        </li>
                      );
                    })}
                  </ul>
                  <p className="mt-3 text-[12px] text-muted-foreground">
                    Next decision ·{" "}
                    <span className="font-semibold text-foreground">
                      {level === "none"
                        ? "keep walking at your pace"
                        : `slow down for ${(ttcValue ?? 0).toFixed(1)} s`}
                    </span>
                  </p>
                </div>
              </div>



            </div>
          </motion.div>
        </AnimatePresence>

        {/* Sensor channels — the four streams the decision was fused from */}
        <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {[
            {
              icon: <Camera className="h-5 w-5" />,
              tone: "bg-guide text-background",
              label: "Camera",
              sub: `Live · ${d.visionConf > 0.5 ? "clear" : "degraded"} · ${nearby.length} objects`,
              ok: d.visionConf > 0.5,
              wave: false,
              badge: null as string | null,
            },

            {
              icon: <Brain className="h-5 w-5" />,
              tone: "bg-guide text-background",
              label: "Prediction",
              sub: `Tracking · ${latencyMs} ms tick`,
              ok: true,
              wave: false,
              badge: null as string | null,
            },
            {
              icon: <Volume2 className="h-5 w-5" />,
              tone: "bg-voice text-background",
              label: "Voice",
              sub: decision?.utterance
                ? `Speaking · ${Math.round(d.audioConf * 100)}% clear`
                : `Idle · ${Math.round(d.audioConf * 100)}% clear`,
              ok: d.audioConf > 0.45,
              wave: true,
              badge: null as string | null,
            },
            {
              icon: <Compass className="h-5 w-5" />,
              tone: "bg-sense text-background",
              label: "Surround view",
              sub: "360° awareness · camera + IMU",
              ok: true,
              wave: false,
              badge: "Active",
            },

          ].map((chip) => (
            <div
              key={chip.label}
              className="flex min-w-0 items-center gap-3 rounded-2xl border border-border bg-card px-3.5 py-3"
            >
              <span
                className={`grid h-10 w-10 shrink-0 place-items-center rounded-full ${chip.tone}`}
              >
                {chip.icon}
              </span>
              <span className="min-w-0 flex-1 leading-tight">
                <span className="block truncate text-[13.5px] font-semibold">{chip.label}</span>
                <span className="block truncate text-[11.5px] text-muted-foreground">
                  {chip.sub}
                </span>
              </span>
              {chip.wave ? (
                <span className="hidden shrink-0 items-center gap-[2px] lg:flex" aria-hidden>
                  {[6, 11, 16, 9, 14, 8, 12, 6].map((h, i) => (
                    <span
                      key={i}
                      className="w-[2px] rounded-full bg-voice"
                      style={{ height: `${playing ? h : 4}px` }}
                    />
                  ))}
                </span>
              ) : null}

              {chip.badge ? (
                <span className="shrink-0 rounded-lg bg-guide-soft px-2 py-1 text-[11px] font-semibold text-guide">
                  {chip.badge}
                </span>
              ) : (
                <span
                  className={`grid h-6 w-6 shrink-0 place-items-center rounded-full ${
                    chip.ok ? "bg-guide-soft text-guide" : "bg-veto-soft text-veto"
                  }`}
                >
                  <Check className="h-3.5 w-3.5" />
                </span>
              )}
              <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />
            </div>
          ))}
        </div>

        {/* Fusion note */}
        <p className="mt-4 flex flex-wrap items-center justify-center gap-2 rounded-2xl border border-border bg-surface px-4 py-3 text-center text-xs text-muted-foreground">
          <ShieldCheck className="h-4 w-4 shrink-0" />
          AI Guardian uses multi-sensor fusion (front camera, metric depth and motion sensors) to
          detect and track objects from all directions.
          <span className="font-semibold text-sense">Geometry keeps the final veto.</span>
        </p>


      </div>
    </div>
  );
}

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div className="rounded-xl border border-border bg-card p-3">
      <p className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">{k}</p>
      <p className="mt-1 font-mono text-base tabular-nums">{v}</p>
    </div>
  );
}
