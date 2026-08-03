/**
 * The continuous safety loop, and the contrast that justifies it.
 *
 * Each stage is an icon-led tile: sense → predict → fast-path TTC → geometric
 * veto → guide → confirm, with confirmation feeding the next tick. The veto tile
 * is the loudest on purpose — it is the only stage that can stop an action, and
 * no downstream signal can talk it out of it.
 */

import { Camera, Route, Timer, ShieldAlert, Volume2, RefreshCcw, ArrowRight } from "lucide-react";

const STAGES = [
  {
    k: "Sense",
    b: "Camera, depth, IMU and GPS at 10 Hz.",
    icon: Camera,
    tone: "border-sense bg-sense-soft text-sense",
    tag: "10 Hz",
  },
  {
    k: "Predict",
    b: "Track objects in 3D, forecast intent and closing speed.",
    icon: Route,
    tone: "border-sense bg-sense-soft text-sense",
    tag: "3D tracks",
  },
  {
    k: "Fast-path TTC",
    b: "Collision risk 1.5–3 s ahead. Hard deadline, no model waits.",
    icon: Timer,
    tone: "border-urgent bg-urgent-soft text-urgent",
    tag: "30 ms",
  },
  {
    k: "Geometric veto",
    b: "Absolute stop authority. Blocks anything the geometry calls unsafe.",
    icon: ShieldAlert,
    tone: "border-veto bg-veto-soft text-veto",
    tag: "absolute",
  },
  {
    k: "Guide",
    b: "One short spoken phrase, or a haptic pattern when the street is loud.",
    icon: Volume2,
    tone: "border-guide bg-guide-soft text-guide",
    tag: "1 phrase",
  },
  {
    k: "Confirm",
    b: "Outcome logged, calibration updated, fed back into the next pass.",
    icon: RefreshCcw,
    tone: "border-memory bg-memory-soft text-memory",
    tag: "↺ feedback",
  },
];

const REACTIVE = [
  "User asks for help",
  "System looks for info",
  "Basic description comes back",
  "Hazard noticed late",
  "User reacts manually",
];

const PREDICTIVE = [
  "Scene understood continuously",
  "Intent and closing speed estimated",
  "Geometry vetoes unsafe actions",
  "Warning lands before the hazard",
  "User keeps walking",
];

export function SafetyLoop() {
  return (
    <div className="space-y-12">
      <div>
        <p className="font-mono text-xs uppercase tracking-[0.25em] text-urgent">
          Continuous safety loop
        </p>

        <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-6 lg:items-stretch">
          {STAGES.map((s, i) => {
            const Icon = s.icon;
            return (
              <div key={s.k} className="relative flex h-full">
                <div className={`flex h-full w-full flex-col rounded-2xl border p-4 ${s.tone}`}>
                  <div className="flex items-start justify-between gap-2">
                    <span className="inline-flex h-9 w-9 items-center justify-center rounded-xl border border-current/30 bg-background/60">
                      <Icon size={17} strokeWidth={2.1} />
                    </span>
                    <span className="font-mono text-[10px] uppercase tracking-wider opacity-80">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                  </div>
                  <h3 className="mt-3 text-sm font-semibold text-foreground">{s.k}</h3>
                  <p className="mt-1.5 flex-1 text-xs leading-relaxed text-muted-foreground">{s.b}</p>
                  <p className="mt-3 inline-block w-fit rounded-full border border-current/30 px-2 py-0.5 font-mono text-[10px]">
                    {s.tag}
                  </p>
                </div>
                {i < STAGES.length - 1 && (
                  <ArrowRight
                    size={16}
                    className="pointer-events-none absolute -right-2.5 top-1/2 hidden -translate-y-1/2 text-muted-foreground lg:block"
                    aria-hidden
                  />
                )}
              </div>
            );
          })}
        </div>

        <p className="mt-3 font-mono text-[11px] text-muted-foreground">
          ↺ confirm feeds the next tick · every stage inside the 90 ms budget · the
          slow path can enrich a warning but never delay or cancel one
        </p>
      </div>


      <div className="grid gap-6 md:grid-cols-2">
        <div className="rounded-lg border border-border bg-surface p-6">
          <h3 className="text-sm font-semibold">Reactive description — how apps work today</h3>
          <ol className="mt-4 space-y-2">
            {REACTIVE.map((r, i) => (
              <li key={r} className="flex gap-3 text-sm text-muted-foreground">
                <span className="font-mono text-xs text-muted-foreground">{i + 1}</span>
                {r}
              </li>
            ))}
          </ol>
          <p className="mt-4 font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
            late · reactive · higher risk · higher effort
          </p>
        </div>
        <div className="rounded-lg border border-guide bg-guide-soft p-6">
          <h3 className="text-sm font-semibold">Predictive assistance — AI Guardian</h3>
          <ol className="mt-4 space-y-2">
            {PREDICTIVE.map((r, i) => (
              <li key={r} className="flex gap-3 text-sm text-muted-foreground">
                <span className="font-mono text-xs text-guide">{i + 1}</span>
                {r}
              </li>
            ))}
          </ol>
          <p className="mt-4 font-mono text-[11px] uppercase tracking-wider text-guide">
            early · proactive · lower risk · lower effort
          </p>
        </div>
      </div>
    </div>
  );
}
