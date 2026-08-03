/**
 * One stage per feature — deliberately *not* one shared picture.
 *
 * A crossing is road geometry, a fall is an emergency panel, object memory is a
 * room you remember, reading is a document. Rendering all of them as the same
 * camera view is what makes assistive demos feel like a slideshow, so each
 * feature here gets its own environment, story and interaction model.
 *
 * What stays constant across every stage is the grammar, not the layout:
 * sensor chips, an audio/haptic timeline, a confidence + urgency readout, and
 * the standing "you decide" label. Those live in the shared frame at the bottom
 * of this file.
 */

import type { Conditions, Derived } from "@/components/guardian/SceneConditions";
import { StreetScene } from "@/components/guardian/StreetScene";
import type { SceneOverlay } from "@/components/guardian/StreetScene";
import type { SceneKind } from "@/lib/environments";
import type { Decision, TrackState } from "@/lib/riskHead";

export interface StageProps {
  n: number;
  overlay: SceneOverlay;
  scene: SceneKind;
  track: TrackState | null;
  decision: Decision | null;
  t: number;
  conditions: Conditions;
  derived: Derived;
  playing: boolean;
  modes: ("camera" | "mic" | "imu" | "gps")[];
}

/* ------------------------------------------------------------------ shared */

const MODE_LABEL: Record<string, string> = {
  camera: "Camera",
  mic: "Microphone",
  imu: "IMU",
  gps: "GPS",
};

function SensorChips({ modes, derived }: { modes: string[]; derived: Derived }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {(["camera", "imu", "gps", "mic"] as const).map((m) => {
        const on = modes.includes(m);
        const weak =
          on &&
          ((m === "camera" && derived.visionConf < 0.5) ||
            (m === "mic" && derived.audioConf < 0.5));
        return (
          <span
            key={m}
            className={`rounded-full border px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] ${
              !on
                ? "border-border text-muted-foreground/45"
                : weak
                  ? "border-veto bg-veto-soft text-veto"
                  : "border-sense bg-sense-soft text-sense"
            }`}
          >
            {MODE_LABEL[m]}
            {weak ? " · weak" : ""}
          </span>
        );
      })}
    </div>
  );
}

/**
 * Audio / haptic channel: a quiet inline stepper, not a row of boxes.
 */

function AlertTimeline({ steps, active }: { steps: string[]; active: number }) {
  return (
    <ol className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
      {steps.map((s, i) => {
        const done = i < active;
        const on = i === active;
        return (
          <li key={s} className="flex min-w-0 items-center gap-2">
            {i > 0 && (
              <span aria-hidden className="text-muted-foreground/40">
                →
              </span>
            )}
            <span
              className={`flex items-center gap-1.5 font-mono text-[10.5px] uppercase tracking-[0.08em] ${
                on ? "text-urgent" : done ? "text-guide" : "text-muted-foreground/50"
              }`}
            >
              <span
                aria-hidden
                className={`h-1.5 w-1.5 rounded-full ${
                  on ? "bg-urgent" : done ? "bg-guide" : "bg-border"
                }`}
              />
              {s}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

function StageShell({ children, caption }: { children: React.ReactNode; caption: string }) {
  return (
    <div className="space-y-3 bg-gradient-to-b from-surface to-surface-2/60 p-4">
      {children}
      <p className="text-[12.5px] leading-relaxed text-muted-foreground">{caption}</p>
    </div>
  );
}

/**
 * The camera stages get a viewfinder treatment: corner crop marks, a live
 * capture strip and a soft vignette, so the render reads as a wearable feed
 * rather than a floating illustration. It also stretches to the column height,
 * which is what kills the dead space under the canvas.
 */
function SceneFrame({
  children,
  label,
  tone = "sense",
}: {
  children: React.ReactNode;
  label: string;
  tone?: "sense" | "urgent";
}) {
  const ring = tone === "urgent" ? "border-urgent/50" : "border-sense/40";
  const dot = tone === "urgent" ? "bg-urgent" : "bg-sense";
  return (
    <div
      className={`relative flex min-h-[380px] flex-col overflow-hidden rounded-2xl border ${ring} bg-[oklch(0.16_0.03_260)] shadow-[0_28px_60px_-42px_oklch(0.2_0.06_260/0.8)]`}
    >
      <div className="relative flex-1 overflow-hidden">
        {children}
        {/* vignette */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 bg-[radial-gradient(120%_90%_at_50%_35%,transparent_45%,oklch(0_0_0/0.38)_100%)]"
        />
        {/* crop marks */}
        <div aria-hidden className="pointer-events-none absolute inset-2.5">
          {[
            "left-0 top-0 border-l-2 border-t-2",
            "right-0 top-0 border-r-2 border-t-2",
            "left-0 bottom-0 border-b-2 border-l-2",
            "right-0 bottom-0 border-b-2 border-r-2",
          ].map((c) => (
            <span key={c} className={`absolute h-4 w-4 rounded-[2px] border-white/45 ${c}`} />
          ))}
        </div>
      </div>
      <div className="flex items-center justify-between gap-2 border-t border-white/10 bg-black/35 px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.16em] text-white/70">
        <span className="flex items-center gap-1.5">
          <span className={`h-1.5 w-1.5 animate-pulse rounded-full ${dot}`} />
          {label}
        </span>
        <span className="tabular-nums text-white/45">1080p · 30 fps · rec</span>
      </div>
    </div>
  );
}

function Spoken({ text, tone = "guide" }: { text: string; tone?: "guide" | "urgent" | "sense" }) {
  const cls = { guide: "border-guide", urgent: "border-urgent", sense: "border-sense" }[tone];
  return (
    <p className={`border-l-2 pl-3.5 text-[15px] font-medium leading-snug ${cls}`}>“{text}”</p>
  );
}

/* --------------------------------------------------------------- 01 predict */

function PredictiveStage(p: StageProps) {
  const phase = (p.t % 5.5) / 5.5;
  const active = phase < 0.3 ? 0 : phase < 0.6 ? 1 : 2;
  return (
    <StageShell caption="Busy pavement beside a construction hoarding, late afternoon. A delivery bicycle comes out from behind the barrier while the walker approaches broken paving.">
      <div className="grid items-stretch gap-3 lg:grid-cols-[minmax(0,1fr)_220px]">
        <SceneFrame label="wearable camera · forward" tone="urgent">
          <StreetScene
            track={p.track}
            decision={p.decision}
            overlay={p.overlay}
            scene={p.scene}
            t={p.t}
            conditions={p.conditions}
            derived={p.derived}
            playing={p.playing}
          />
        </SceneFrame>
        <ol className="flex flex-col justify-center gap-0.5">
          {[
            { k: "Object detected", d: "bicycle · behind barrier" },
            { k: "Path intersecting", d: "curve crosses cone" },
            { k: "Alert issued", d: "2.4 s to contact" },
          ].map((s, i) => {
            const on = i <= active;
            return (
              <li
                key={s.k}
                className={`flex items-start gap-2.5 border-l-2 py-2.5 pl-3 transition-colors ${
                  on ? "border-urgent" : "border-border"
                }`}
              >
                <span
                  aria-hidden
                  className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${on ? "bg-urgent" : "bg-border"}`}
                />
                <span className="min-w-0">
                  <span
                    className={`block text-sm font-semibold leading-tight ${on ? "" : "text-muted-foreground"}`}
                  >
                    {s.k}
                  </span>
                  <span className="block font-mono text-[11px] text-muted-foreground">{s.d}</span>
                </span>
              </li>
            );
          })}
          <li className="mt-1 border-l-2 border-veto py-2.5 pl-3">
            <span className="block text-sm font-semibold leading-tight text-veto">Hazard zone</span>
            <span className="block font-mono text-[11px] text-muted-foreground">
              uneven paving · 4 m ahead-left
            </span>
          </li>
        </ol>
      </div>
      <Spoken
        tone="urgent"
        text="An approaching bicycle may cross your path in 2.4 seconds. Slow down and stay slightly right."
      />
    </StageShell>
  );
}

/* -------------------------------------------------------------- 02 crossing */

function CrossingStage(p: StageProps) {
  const walk = p.t % 5.5 > 1.2;
  return (
    <StageShell caption="Signal-controlled junction outside a metro station at night. Two vehicles are turning left across the crossing while the pedestrian signal is green.">
      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_260px]">
        <svg
          viewBox="0 0 400 300"
          className="w-full rounded-2xl border border-border bg-card"
          role="img"
          aria-label="Top-down junction map"
        >
          <rect width="400" height="300" fill="oklch(0.18 0.03 260)" />
          {/* road geometry */}
          <rect x="0" y="110" width="400" height="80" fill="oklch(0.28 0.02 260)" />
          <rect x="160" y="0" width="80" height="300" fill="oklch(0.28 0.02 260)" />
          <g stroke="oklch(0.75 0.02 260)" strokeDasharray="12 12" strokeWidth="2">
            <line x1="0" y1="150" x2="400" y2="150" />
            <line x1="200" y1="0" x2="200" y2="300" />
          </g>
          {/* crossing stripes */}
          {Array.from({ length: 7 }).map((_, i) => (
            <rect
              key={i}
              x={166 + i * 10}
              y={196}
              width="6"
              height="46"
              fill={walk ? "oklch(0.85 0.16 150)" : "oklch(0.8 0.02 260)"}
              opacity={walk ? 0.9 : 0.45}
            />
          ))}
          {/* turning vehicles */}
          <g fill="oklch(0.72 0.17 55)">
            <rect x={90 + ((p.t * 22) % 90)} y="122" width="26" height="14" rx="3" />
            <rect x={40 + ((p.t * 18) % 90)} y="160" width="26" height="14" rx="3" />
          </g>
          <path
            d="M120 129 C 180 129 196 160 196 210"
            fill="none"
            stroke="oklch(0.72 0.17 55)"
            strokeWidth="2"
            strokeDasharray="6 6"
          />
          {/* pedestrian */}
          <circle cx="196" cy="252" r="8" fill="oklch(0.72 0.15 250)" />
          <path
            d="M196 244 L196 200"
            stroke="oklch(0.72 0.15 250)"
            strokeWidth="2"
            strokeDasharray="5 5"
          />
          <text x="12" y="24" fill="oklch(0.85 0.02 260)" fontSize="11" fontFamily="monospace">
            metro junction · night · you at south kerb
          </text>
        </svg>
        <div className="space-y-2">
          <div
            className={`rounded-2xl border p-4 ${
              walk ? "border-veto bg-veto-soft" : "border-urgent bg-urgent-soft"
            }`}
          >
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              crossing status
            </p>
            <p className="mt-1 text-2xl font-bold leading-tight">
              {walk ? "Green — but hold" : "Do not cross"}
            </p>
            <p className="mt-1 text-xs text-foreground/75">
              {walk ? "2 vehicles turning across your line" : "Signal is red for pedestrians"}
            </p>
          </div>
          <dl className="rounded-xl border border-border bg-card px-3 py-2">
            {[
              ["signal", walk ? "green · advisory" : "red"],
              ["turning vehicles", "2 detected"],
              ["geometric veto", walk ? "active — holds" : "n/a"],
            ].map(([k, v]) => (
              <div
                key={k}
                className="flex items-baseline justify-between gap-2 border-b border-border/60 py-1.5 last:border-0"
              >
                <dt className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
                  {k}
                </dt>
                <dd className="font-mono text-xs">{v}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
      <Spoken
        tone="urgent"
        text="Pedestrian signal is green. Two vehicles are turning left. Wait briefly for the nearest vehicle to pass."
      />
    </StageShell>
  );
}

/* ------------------------------------------------------------------ 03 fall */

function FallStage(p: StageProps) {
  const secs = Math.max(0, 20 - (Math.floor(p.t * 2) % 21));
  const pct = secs / 20;
  const R = 54;
  return (
    <StageShell caption="Quiet residential path with a stepped kerb, just after sunset. The wearer slipped on the step down; the camera is not the sensor that matters here.">
      <div className="grid gap-3 md:grid-cols-[200px_minmax(0,1fr)]">
        <div className="grid place-items-center rounded-2xl border border-urgent bg-urgent-soft p-4">
          <svg
            viewBox="0 0 140 140"
            className="w-full max-w-[160px]"
            role="img"
            aria-label="Body-motion indicator and cancel countdown"
          >
            <circle
              cx="70"
              cy="70"
              r={R}
              fill="none"
              stroke="oklch(0.85 0.02 260)"
              strokeWidth="8"
              opacity="0.35"
            />
            <circle
              cx="70"
              cy="70"
              r={R}
              fill="none"
              stroke="oklch(0.62 0.21 25)"
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={2 * Math.PI * R}
              strokeDashoffset={2 * Math.PI * R * (1 - pct)}
              transform="rotate(-90 70 70)"
            />
            <text
              x="70"
              y="66"
              textAnchor="middle"
              fontSize="30"
              fontWeight="700"
              fill="currentColor"
            >
              {secs}
            </text>
            <text
              x="70"
              y="88"
              textAnchor="middle"
              fontSize="10"
              fontFamily="monospace"
              fill="currentColor"
              opacity="0.7"
            >
              SECONDS
            </text>
          </svg>
        </div>
        <div className="space-y-2">
          <div className="rounded-2xl border border-urgent bg-card p-4">
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-urgent">
              possible fall detected
            </p>
            <p className="mt-1 text-xl font-bold leading-tight">Are you safe?</p>
            <p className="mt-1 text-sm text-muted-foreground">
              Free-fall → impact 3.1 g → stillness 8 s. Nothing is sent until the countdown ends.
            </p>
          </div>
          <div className="grid gap-2 sm:grid-cols-3">
            {[
              { l: "I am okay", tone: "border-guide bg-guide-soft text-guide" },
              { l: "Call contact", tone: "border-urgent bg-urgent-soft text-urgent" },
              { l: "Share location", tone: "border-sense bg-sense-soft text-sense" },
            ].map((b) => (
              <button
                key={b.l}
                type="button"
                className={`rounded-xl border px-3 py-3 text-sm font-semibold transition-transform hover:scale-[1.02] ${b.tone}`}
              >
                {b.l}
              </button>
            ))}
          </div>
          <div className="rounded-xl border border-border bg-card px-3 py-2 font-mono text-[11px] text-muted-foreground">
            location snapshot · 25.2048 N, 55.2708 E · ±6 m · Al Wasl residential path
          </div>
        </div>
      </div>
      <Spoken
        tone="urgent"
        text="We detected a possible fall. Are you safe? Emergency assistance will be notified in 20 seconds unless cancelled."
      />
    </StageShell>
  );
}

/* --------------------------------------------------------------- 04 wayback */

function WayBackStage(p: StageProps) {
  const idx = Math.floor(p.t / 2) % 3;
  const legs = [
    { k: "Continue 40 metres", d: "pavement, café awning on your right", m: "40 m" },
    { k: "Turn left after the café entrance", d: "voice-tagged 22 minutes ago", m: "12 m" },
    { k: "Metro entrance ahead on your left", d: "step-free, handrail both sides", m: "60 m" },
  ];
  return (
    <StageShell caption="Leaving a familiar café in a shopping district, retracing this morning's walk back to the metro. Only the next three landmarks are ever spoken.">
      <div className="grid gap-3 md:grid-cols-[200px_minmax(0,1fr)]">
        <div className="rounded-2xl border border-guide bg-guide-soft p-4">
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-guide">retracing</p>
          <ol className="mt-3 space-y-0">
            {legs.map((l, i) => (
              <li key={l.k} className="flex gap-3">
                <span className="flex flex-col items-center">
                  <span
                    className={`mt-1 h-3 w-3 rounded-full ${i === idx ? "bg-guide" : "bg-border"}`}
                  />
                  {i < legs.length - 1 && <span className="h-12 w-px flex-1 bg-border" />}
                </span>
                <span className="pb-3">
                  <span className="block font-mono text-[11px] tabular-nums text-muted-foreground">
                    {l.m}
                  </span>
                  <span
                    className={`block text-xs ${i === idx ? "font-semibold" : "text-muted-foreground"}`}
                  >
                    {l.k.split(" ").slice(0, 3).join(" ")}
                  </span>
                </span>
              </li>
            ))}
          </ol>
        </div>
        <div className="space-y-2">
          {legs.map((l, i) => (
            <div
              key={l.k}
              className={`rounded-2xl border p-4 transition-colors ${
                i === idx ? "border-guide bg-card shadow-sm" : "border-border bg-card opacity-70"
              }`}
            >
              <p className="flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
                <span>{i === idx ? "now" : `then · ${i + 1}`}</span>
                <span className="tabular-nums">{l.m}</span>
              </p>
              <p className="mt-1 text-base font-semibold leading-snug">{l.k}</p>
              <p className="text-xs text-muted-foreground">{l.d}</p>
            </div>
          ))}
          <div className="rounded-xl border border-border bg-surface px-3 py-2 font-mono text-[11px] text-muted-foreground">
            drift ±{p.derived.visionConf < 0.5 ? "9.0" : "1.8"} m · haptic belt left/right pulses
          </div>
        </div>
      </div>
      <Spoken text="You are returning along a known route. Continue straight for 40 metres. The metro entrance is ahead on your left." />
    </StageShell>
  );
}

/* ---------------------------------------------------------------- 05 memory */

function ObjectMemoryStage(p: StageProps) {
  const pins = [
    { x: 32, y: 58, o: "Keys", w: "dining table", t: "18 min ago", c: 0.88 },
    { x: 68, y: 40, o: "Wallet", w: "hallway shelf", t: "2 h ago", c: 0.71 },
    { x: 78, y: 72, o: "Backpack", w: "sofa, left arm", t: "40 min ago", c: 0.64 },
    { x: 18, y: 30, o: "Medication", w: "kitchen counter", t: "yesterday", c: 0.52 },
  ];
  const active = Math.floor(p.t / 1.6) % pins.length;
  return (
    <StageShell caption="Home, searching for personal items. This is a memory archive, not a live camera feed — every pin is a thing the device saw earlier and where it saw it.">
      <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_240px]">
        <div className="relative overflow-hidden rounded-2xl border border-memory bg-memory-soft">
          <svg viewBox="0 0 400 260" className="w-full" role="img" aria-label="Room memory board">
            <rect width="400" height="260" fill="oklch(0.96 0.01 100)" />
            <rect
              x="16"
              y="16"
              width="368"
              height="228"
              fill="none"
              stroke="oklch(0.55 0.03 260)"
              strokeWidth="3"
            />
            {/* furniture */}
            <rect x="90" y="130" width="120" height="60" rx="6" fill="oklch(0.8 0.05 70)" />
            <text x="150" y="165" textAnchor="middle" fontSize="11" fontFamily="monospace">
              dining table
            </text>
            <rect x="270" y="160" width="100" height="50" rx="10" fill="oklch(0.78 0.06 250)" />
            <text x="320" y="190" textAnchor="middle" fontSize="11" fontFamily="monospace">
              sofa
            </text>
            <rect x="250" y="60" width="120" height="16" rx="4" fill="oklch(0.7 0.04 70)" />
            <text x="310" y="52" textAnchor="middle" fontSize="11" fontFamily="monospace">
              shelf
            </text>
            <rect x="30" y="50" width="120" height="34" rx="4" fill="oklch(0.86 0.02 200)" />
            <text x="90" y="72" textAnchor="middle" fontSize="11" fontFamily="monospace">
              counter
            </text>
          </svg>
          {pins.map((pin, i) => (
            <span
              key={pin.o}
              className={`absolute -translate-x-1/2 -translate-y-1/2 rounded-full border px-2 py-0.5 font-mono text-[10px] transition-all ${
                i === active
                  ? "scale-110 border-memory bg-card font-bold shadow-md"
                  : "border-border bg-card/80 text-muted-foreground"
              }`}
              style={{ left: `${pin.x}%`, top: `${pin.y}%` }}
            >
              {pin.o}
            </span>
          ))}
        </div>
        <div className="space-y-2">
          {pins.map((pin, i) => (
            <div
              key={pin.o}
              className={`rounded-xl border p-3 ${
                i === active ? "border-memory bg-card" : "border-border bg-card/70"
              }`}
            >
              <p className="flex items-baseline justify-between">
                <span className="text-sm font-semibold">{pin.o}</span>
                <span className="font-mono text-[11px] tabular-nums text-muted-foreground">
                  {pin.c.toFixed(2)}
                </span>
              </p>
              <p className="font-mono text-[11px] text-muted-foreground">
                {pin.w} · {pin.t}
              </p>
            </div>
          ))}
        </div>
      </div>
      <Spoken text="Your keys were last seen on the dining table 18 minutes ago, near a small black bag." />
    </StageShell>
  );
}

/* --------------------------------------------------------------- 06 reading */

function ReadingStage(p: StageProps) {
  const lines = [
    "BOOTS PHARMACY · dispensing label",
    "Take one tablet after breakfast.",
    "Do not exceed two tablets in 24 hours.",
    "Prescription refill date: 12 August.",
  ];
  const upTo = Math.floor(p.t / 1.2) % (lines.length + 1);
  const glare = p.derived.visionConf < 0.55;
  return (
    <StageShell caption="Pharmacy counter. The camera is pointed at a dispensing label — angled, plastic-wrapped, and glare-prone. The reading panel, not the scene, is the interface.">
      <div className="grid gap-3 md:grid-cols-[220px_minmax(0,1fr)]">
        <div className="relative overflow-hidden rounded-2xl border border-border bg-card">
          <svg
            viewBox="0 0 200 240"
            className="w-full"
            role="img"
            aria-label="Camera frame with detected text region"
          >
            <rect width="200" height="240" fill="oklch(0.22 0.02 260)" />
            <rect
              x="34"
              y="46"
              width="132"
              height="150"
              rx="6"
              fill="oklch(0.95 0.01 100)"
              transform="rotate(-4 100 120)"
            />
            {Array.from({ length: 7 }).map((_, i) => (
              <rect
                key={i}
                x="46"
                y={64 + i * 18}
                width={i % 3 === 0 ? 90 : 108}
                height="6"
                rx="3"
                fill="oklch(0.55 0.02 260)"
                transform="rotate(-4 100 120)"
              />
            ))}
            {glare && (
              <ellipse cx="130" cy="90" rx="46" ry="30" fill="oklch(1 0 0)" opacity="0.55" />
            )}
            <rect
              x="30"
              y="42"
              width="140"
              height="158"
              fill="none"
              stroke="oklch(0.72 0.17 150)"
              strokeWidth="2"
              strokeDasharray="8 6"
            />
            <text x="32" y="34" fontSize="10" fontFamily="monospace" fill="oklch(0.72 0.17 150)">
              text region locked
            </text>
          </svg>
          <p className="border-t border-border px-3 py-2 font-mono text-[10.5px] text-muted-foreground">
            {glare ? "glare · retake asked before reading digits" : "frame good · 4 lines detected"}
          </p>
        </div>
        <div className="space-y-2">
          <div className="rounded-2xl border border-memory bg-card p-4">
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-memory">
              reading · line {Math.min(upTo, lines.length)} of {lines.length}
            </p>
            <div className="mt-2 space-y-1.5">
              {lines.map((l, i) => (
                <p
                  key={l}
                  className={`text-lg font-semibold leading-snug tracking-tight ${
                    i < upTo ? "text-foreground" : "text-muted-foreground/40"
                  }`}
                >
                  {l}
                </p>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-4 gap-2">
            {["Read", "Repeat", "Summarise", "Save"].map((b) => (
              <button
                key={b}
                type="button"
                className="rounded-xl border border-border bg-card px-2 py-2.5 text-xs font-semibold transition-colors hover:border-foreground"
              >
                {b}
              </button>
            ))}
          </div>
        </div>
      </div>
      <Spoken text="Reading detected text: Take one tablet after breakfast. Prescription refill date: 12 August." />
    </StageShell>
  );
}

/* ----------------------------------------------------------------- 07 scene */

function SceneOnDemandStage(p: StageProps) {
  const zones = [
    {
      k: "Immediate · 0–2 m",
      tone: "border-sense bg-sense-soft",
      items: ["clear floor", "floor mat edge, half a metre"],
    },
    {
      k: "Middle · 2–8 m",
      tone: "border-assist bg-assist-soft",
      items: ["reception desk, 6 m ahead", "seating cluster, right", "luggage trolley, left"],
    },
    {
      k: "Far · 8 m +",
      tone: "border-border bg-card",
      items: ["lift lobby, far right", "glass entrance behind you"],
    },
  ];
  const lit = Math.floor(p.t / 1.5) % 3;
  return (
    <StageShell caption="Unfamiliar hotel lobby. You asked “what is in front of me?” — this is a summary, not an alarm, so nothing here is coloured as a hazard.">
      <div className="space-y-2">
        {zones.map((z, i) => (
          <div
            key={z.k}
            className={`rounded-2xl border p-3 transition-opacity ${z.tone} ${i === lit ? "opacity-100" : "opacity-75"}`}
          >
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              {z.k}
            </p>
            <div className="mt-2 flex flex-wrap gap-2">
              {z.items.map((it) => (
                <span
                  key={it}
                  className="rounded-lg border border-border bg-card px-2.5 py-1 text-xs"
                >
                  {it}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
      <Spoken
        tone="sense"
        text="In front of you is an open lobby. A reception desk is six metres ahead. Seating is on the right. The nearest clear path is slightly left."
      />
    </StageShell>
  );
}

/* ---------------------------------------------------------------- 08 places */

function PlacesStage(p: StageProps) {
  const places = [
    {
      n: "Life Pharmacy",
      d: "180 m ahead",
      s: "Open until 23:00",
      a: "Step-free entrance",
      w: "3 min",
    },
    {
      n: "Emirates NBD ATM",
      d: "240 m, right",
      s: "Open 24 h",
      a: "Two steps, handrail",
      w: "4 min",
    },
    { n: "Café Rider", d: "90 m behind", s: "Open now", a: "Level entry, wide door", w: "2 min" },
    { n: "Metro entrance", d: "320 m ahead", s: "Open", a: "Lift + tactile paving", w: "5 min" },
  ];
  const focus = Math.floor(p.t / 1.8) % places.length;
  return (
    <StageShell caption="A public neighbourhood block, choosing where to go rather than how to get there. Accessibility notes sit on the card, not three taps deep.">
      <div className="space-y-2">
        {places.map((pl, i) => (
          <div
            key={pl.n}
            className={`grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 rounded-2xl border p-4 transition-colors ${
              i === focus ? "border-memory bg-memory-soft" : "border-border bg-card"
            }`}
          >
            <div className="min-w-0">
              <p className="truncate text-base font-semibold">{pl.n}</p>
              <p className="font-mono text-[11px] text-muted-foreground">
                {pl.d} · {pl.s}
              </p>
              <p className="mt-1 inline-block rounded-md border border-guide bg-guide-soft px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.12em] text-guide">
                {pl.a}
              </p>
            </div>
            <span className="shrink-0 rounded-xl border border-border bg-surface px-3 py-2 text-center font-mono text-xs tabular-nums">
              {pl.w}
              <span className="block text-[9px] uppercase tracking-widest text-muted-foreground">
                walk
              </span>
            </span>
          </div>
        ))}
      </div>
      <Spoken text="Nearest pharmacy: 180 metres ahead, open now. Entrance has step-free access." />
    </StageShell>
  );
}

/* -------------------------------------------------------------- 09 reroute */

function RerouteStage(p: StageProps) {
  const dash = (p.t * 14) % 20;
  return (
    <StageShell caption="The planned pavement is closed by roadworks. Two routes, side by side, with the cost of the safer one stated up front.">
      <div className="grid gap-3 md:grid-cols-2">
        <div className="rounded-2xl border border-border bg-card p-3 opacity-80">
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            planned route · blocked
          </p>
          <svg viewBox="0 0 200 150" className="mt-2 w-full" role="img" aria-label="Blocked route">
            <rect width="200" height="150" rx="8" fill="oklch(0.93 0.005 260)" />
            <path
              d="M20 130 L20 70 L120 70 L120 20"
              fill="none"
              stroke="oklch(0.6 0.01 260)"
              strokeWidth="6"
              strokeLinecap="round"
            />
            <g transform="translate(120 62)">
              <circle r="14" fill="oklch(0.62 0.21 25)" />
              <text textAnchor="middle" y="5" fontSize="16" fill="white">
                !
              </text>
            </g>
            <text x="12" y="146" fontSize="10" fontFamily="monospace" fill="oklch(0.45 0.01 260)">
              construction closure · 60 m
            </text>
          </svg>
          <p className="mt-2 font-mono text-[11px] text-muted-foreground">
            14 min · blocked at 60 m
          </p>
        </div>
        <div className="rounded-2xl border border-guide bg-guide-soft p-3">
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-guide">
            suggested · covered walkway
          </p>
          <svg
            viewBox="0 0 200 150"
            className="mt-2 w-full"
            role="img"
            aria-label="Alternative route"
          >
            <rect width="200" height="150" rx="8" fill="oklch(0.97 0.01 160)" />
            <path
              d="M20 130 L20 100 L90 100 L90 40 L170 40 L170 20"
              fill="none"
              stroke="oklch(0.55 0.16 155)"
              strokeWidth="6"
              strokeLinecap="round"
              strokeDasharray="14 6"
              strokeDashoffset={-dash}
            />
            <circle cx="20" cy="130" r="6" fill="oklch(0.45 0.15 250)" />
            <circle cx="170" cy="20" r="6" fill="oklch(0.55 0.16 155)" />
            <text x="12" y="146" fontSize="9" fontFamily="monospace" fill="oklch(0.4 0.06 160)">
              covered · step-free
            </text>
          </svg>
          <p className="mt-2 font-mono text-[11px] text-foreground/80">
            16 min · +2 min · accepted
          </p>
        </div>
      </div>
      <Spoken text="Your planned path is blocked by construction. A safer route is available through the covered walkway. It adds 2 minutes." />
    </StageShell>
  );
}

/* ----------------------------------------------------------------- 10 human */

function HumanStage(p: StageProps) {
  return (
    <StageShell caption="An entrance the model cannot read: temporary event barriers, unclear signage, no obvious door. The visual here is consent, not geometry.">
      <div className="grid gap-3 md:grid-cols-[200px_minmax(0,1fr)]">
        <div className="overflow-hidden rounded-2xl border border-assist bg-card">
          <svg
            viewBox="0 0 200 150"
            className="w-full"
            role="img"
            aria-label="Blurred live preview of the ambiguous entrance"
          >
            <rect width="200" height="150" fill="oklch(0.72 0.02 250)" />
            <rect x="20" y="40" width="70" height="90" fill="oklch(0.6 0.03 250)" opacity="0.8" />
            <rect x="110" y="60" width="70" height="70" fill="oklch(0.66 0.04 90)" opacity="0.8" />
            <rect x="0" y="118" width="200" height="10" fill="oklch(0.8 0.14 90)" />
            <text x="10" y="20" fontSize="10" fontFamily="monospace" fill="oklch(0.2 0 0)">
              preview · not yet shared
            </text>
          </svg>
          <p className="border-t border-border px-3 py-2 font-mono text-[10.5px] text-muted-foreground">
            camera + location shared only after you agree
          </p>
        </div>
        <div className="space-y-2">
          <div className="rounded-2xl border border-assist bg-assist-soft p-4">
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-assist">
              uncertainty · ensemble disagreement
            </p>
            <p className="mt-1 text-sm text-foreground/85">
              Entrance layout unclear: two candidate doorways, temporary barrier, signage unreadable
              at this angle.
            </p>
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            {[
              { l: "Connect to trusted contact", s: "Fatima · usually answers in 20 s" },
              { l: "Connect to support assistant", s: "Trained volunteer · 24 h" },
            ].map((b) => (
              <button
                key={b.l}
                type="button"
                className="rounded-xl border border-assist bg-card px-3 py-3 text-left transition-colors hover:bg-assist-soft"
              >
                <span className="block text-sm font-semibold">{b.l}</span>
                <span className="block font-mono text-[10.5px] text-muted-foreground">{b.s}</span>
              </button>
            ))}
          </div>
          <div className="rounded-xl border border-border bg-surface px-3 py-2 font-mono text-[10.5px] text-muted-foreground">
            will be shared: live camera · approximate location · last 10 s summary. not shared:
            contacts, history, object memory.
          </div>
        </div>
      </div>
      <Spoken
        tone="sense"
        text="I am not confident about the path ahead because the entrance layout is unclear. Would you like assistance from a trusted person?"
      />
    </StageShell>
  );
}

/* ----------------------------------------------------------- 11 translation */

function TranslationStage(p: StageProps) {
  const low = p.derived.audioConf < 0.5;
  return (
    <StageShell caption="Airport concourse. An Arabic overhead sign, and a staff member speaking. Original and translation stay visually separate so nothing is silently smoothed over.">
      <div className="space-y-2">
        <div className="rounded-2xl border border-border bg-card p-4">
          <p className="flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            <span>detected · Arabic (ar)</span>
            <span>camera sign · OCR {Math.round(p.derived.visionConf * 100)}%</span>
          </p>
          <p dir="rtl" className="mt-2 text-2xl font-semibold leading-snug">
            مراقبة الجوازات والمغادرة
          </p>
        </div>
        <div className="rounded-2xl border border-guide bg-guide-soft p-4">
          <p className="flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.18em] text-guide">
            <span>translation · English</span>
            <span>{low ? "low confidence · hedged" : "confident"}</span>
          </p>
          <p className="mt-2 text-3xl font-bold leading-tight tracking-tight">
            Passport Control and Departures
          </p>
          <p className="mt-1 text-sm text-foreground/75">Direction: straight ahead.</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {["▶ Play audio", "Repeat", "Reply mode", low ? "Ask them to repeat" : "Slower"].map(
              (b) => (
                <button
                  key={b}
                  type="button"
                  className="rounded-xl border border-border bg-card px-3 py-2 text-xs font-semibold transition-colors hover:border-foreground"
                >
                  {b}
                </button>
              ),
            )}
          </div>
        </div>
        <div className="grid gap-2 sm:grid-cols-2">
          <div className="rounded-xl border border-border bg-surface px-3 py-2">
            <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
              heard (speech)
            </p>
            <p dir="rtl" className="text-sm">
              من فضلك، الطابور على اليمين
            </p>
          </div>
          <div className="rounded-xl border border-border bg-surface px-3 py-2">
            <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
              translated
            </p>
            <p className="text-sm">“Please, the queue is on the right.”</p>
          </div>
        </div>
      </div>
      <Spoken text="Arabic sign detected. Translation: ‘Passport Control and Departures.’ The direction is straight ahead." />
    </StageShell>
  );
}

/* ------------------------------------------------------------------ 12 edge */

function EdgeStage(p: StageProps) {
  const dist = 1.9 + 1.1 * Math.sin(p.t * 0.7);
  const urgent = dist < 1.2;
  return (
    <StageShell caption="Metro platform. There is no closing velocity here — the hazard is a distance to a drop, so the readout is metres, and it cannot be suppressed inside 1.2 m.">
      <div className="grid items-stretch gap-3 lg:grid-cols-[minmax(0,1fr)_200px]">
        <SceneFrame label="wearable camera · platform edge" tone="urgent">
          <StreetScene
            track={p.track}
            decision={p.decision}
            overlay={p.overlay}
            scene={p.scene}
            t={p.t}
            conditions={p.conditions}
            derived={p.derived}
            playing={p.playing}
          />
        </SceneFrame>
        <div className="flex flex-col gap-2">
          <div
            className={`rounded-2xl border p-4 ${urgent ? "border-urgent bg-urgent-soft" : "border-veto bg-veto-soft"}`}
          >
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              distance to drop
            </p>
            <p className="text-4xl font-bold tabular-nums leading-none">
              {dist.toFixed(1)}
              <span className="text-lg"> m</span>
            </p>
            <p className="mt-1 font-mono text-[11px]">
              {urgent ? "urgent · cannot be suppressed" : "advisory hold"}
            </p>
          </div>
          {[
            ["depth step", "confirmed · 0.9 m fall"],
            ["tactile strip", p.derived.visionConf > 0.5 ? "detected" : "not visible"],
            ["gait pitch", "stable"],
          ].map(([k, v]) => (
            <div
              key={k}
              className="flex items-baseline justify-between rounded-xl border border-border bg-card px-3 py-2"
            >
              <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
                {k}
              </span>
              <span className="font-mono text-[11px]">{v}</span>
            </div>
          ))}
        </div>
      </div>
      <Spoken
        tone="urgent"
        text="Platform edge one metre to your right. Step left and keep the tactile strip under your cane."
      />
    </StageShell>
  );
}

/* -------------------------------------------------------------- dispatcher */

const TIMELINES: Record<number, string[]> = {
  1: ["detect", "predict", "alert", "confirm"],
  2: ["read signal", "check turns", "hold", "release"],
  12: ["depth step", "tactile", "urgent hold"],
  10: ["impact", "stillness", "cancel window", "notify"],
  3: ["locate", "next leg", "haptic cue"],
  4: ["query", "match", "speak"],
  6: ["frame", "ocr", "read aloud"],
  5: ["ask", "answer", "speak"],
  7: ["search", "rank", "approach"],
  8: ["blocked", "compare", "reroute"],
  9: ["abstain", "consent", "connect"],
  11: ["capture", "translate", "speak"],
};

const STAGES: Record<number, (p: StageProps) => React.ReactElement> = {
  1: PredictiveStage,
  2: CrossingStage,
  12: EdgeStage,
  10: FallStage,
  3: WayBackStage,
  4: ObjectMemoryStage,
  6: ReadingStage,
  5: SceneOnDemandStage,
  7: PlacesStage,
  8: RerouteStage,
  9: HumanStage,
  11: TranslationStage,
};

export function FeatureStage(p: StageProps) {
  const Stage = STAGES[p.n] ?? PredictiveStage;
  const steps = TIMELINES[p.n] ?? ["sense", "decide", "speak"];
  const active = Math.floor((p.t / 1.4) % steps.length);

  return (
    <div>
      <Stage {...p} />

      {/* Shared grammar — one quiet strip, identical on every stage. */}
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-t border-border bg-card px-4 py-2.5">
        <AlertTimeline steps={steps} active={active} />
        <SensorChips modes={p.modes} derived={p.derived} />
      </div>
    </div>
  );
}
