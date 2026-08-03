/**
 * Shared "real world out there" conditions.
 *
 * Every feature simulation runs inside this context so the demo shows what the
 * device does on a dusty, crowded, loud pavement — not in a clean lab frame.
 * Two derived numbers drive every sim:
 *   visionConf — how much the camera can be trusted this second
 *   audioConf  — how much the microphone / speech channel can be trusted
 */
import { createContext, useContext, useMemo, useState } from "react";
import {
  ENVIRONMENTS,
  ENV_BY_ID,
  positioning,
  type EnvClass,
  type EnvId,
} from "@/lib/environments";


export interface Conditions {
  crowd: number; // 0..1  people density, occlusion, jostling
  ambientDb: number; // 35..95 traffic, buskers, station announcements
  light: number; // 0..1  1 = clean daylight, 0 = dusk / glare / tunnel
  particulate: number; // 0..1 dust, rain, spray on the lens
  motion: number; // 0..1 gait shake, head turn, cane sweep blur
}

export const PRESETS: { name: string; blurb: string; c: Conditions }[] = [
  {
    name: "Quiet street",
    blurb: "Residential pavement, mid-morning",
    c: { crowd: 0.1, ambientDb: 48, light: 0.9, particulate: 0.05, motion: 0.2 },
  },
  {
    name: "Busy market",
    blurb: "Crowd, chatter in three languages, stalls",
    c: { crowd: 0.85, ambientDb: 78, light: 0.7, particulate: 0.2, motion: 0.5 },
  },
  {
    name: "Rain at dusk",
    blurb: "Low light, spray on the lens, wet glare",
    c: { crowd: 0.35, ambientDb: 66, light: 0.25, particulate: 0.7, motion: 0.35 },
  },
  {
    name: "Metro platform",
    blurb: "Announcements, echo, moving crowd, edge nearby",
    c: { crowd: 0.7, ambientDb: 88, light: 0.55, particulate: 0.1, motion: 0.3 },
  },
  {
    name: "Dusty roadworks",
    blurb: "Jackhammer, dust cloud, diverted footway",
    c: { crowd: 0.4, ambientDb: 92, light: 0.6, particulate: 0.9, motion: 0.6 },
  },
];

export interface Derived {
  visionConf: number;
  audioConf: number;
  slowPathMs: number;
  stressors: string[];
}

export function derive(c: Conditions): Derived {
  const clamp = (x: number) => Math.max(0.05, Math.min(0.99, x));
  const visionConf = clamp(
    1 - 0.38 * c.particulate - 0.34 * (1 - c.light) - 0.22 * c.motion - 0.16 * c.crowd,
  );
  const audioConf = clamp(1 - (c.ambientDb - 45) / 58 - 0.22 * c.crowd);
  const slowPathMs = Math.round(
    700 + 520 * c.crowd + 420 * c.particulate + 300 * (1 - c.light),
  );

  const stressors: string[] = [];
  if (c.crowd > 0.6) stressors.push("crowd occlusion");
  if (c.ambientDb > 75) stressors.push(`${Math.round(c.ambientDb)} dB ambient`);
  if (c.light < 0.4) stressors.push("low light / glare");
  if (c.particulate > 0.5) stressors.push("dust or spray on lens");
  if (c.motion > 0.5) stressors.push("gait blur");
  return { visionConf, audioConf, slowPathMs, stressors };
}

const Ctx = createContext<{
  c: Conditions;
  set: (c: Conditions) => void;
  d: Derived;
  env: EnvClass;
  setEnv: (id: EnvId) => void;
} | null>(null);

export function ConditionsProvider({ children }: { children: React.ReactNode }) {
  const [envId, setEnvId] = useState<EnvId>("A");
  const env = ENV_BY_ID[envId];
  const [c, set] = useState<Conditions>(ENVIRONMENTS[0]!.c);
  const d = useMemo(() => derive(c), [c]);
  const setEnv = (id: EnvId) => {
    setEnvId(id);
    set(ENV_BY_ID[id].c);
  };
  return (
    <Ctx.Provider value={{ c, set, d, env, setEnv }}>{children}</Ctx.Provider>
  );
}


export function useConditions() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useConditions outside ConditionsProvider");
  return v;
}

/* ------------------------------------------------------------------ UI bar */

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  read,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  read: string;
}) {
  return (
    <label className="block">
      <span className="flex items-baseline justify-between text-xs text-muted-foreground">
        {label}
        <span className="font-mono text-[11px] text-foreground">{read}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1 w-full accent-[oklch(0.55_0.16_255)]"
      />
    </label>
  );
}

/**
 * One condition read as a person would say it: a name, a plain verdict, and a
 * bar. The 0..1 confidence stays available underneath for engineers, but the
 * headline is always words, never a decimal.
 */
function Meter({
  icon,
  name,
  value,
  good,
  mid,
  bad,
}: {
  icon: string;
  name: string;
  value: number;
  good: string;
  mid?: string;
  bad: string;
}) {
  const ok = value >= 0.65;
  const weak = value < 0.45;
  const tone = weak ? "--urgent" : ok ? "--guide" : "--veto";
  return (
    <div className="rounded-xl border border-border bg-background/60 p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="flex min-w-0 items-center gap-1.5 text-xs font-medium">
          <span aria-hidden className="shrink-0 opacity-70">
            {icon}
          </span>
          <span className="truncate">{name}</span>
        </span>
        <span
          className="shrink-0 font-mono text-[10px] tabular-nums text-muted-foreground"
          title="confidence 0–1"
        >
          {Math.round(value * 100)}%
        </span>
      </div>
      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-border">
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{ width: `${Math.round(value * 100)}%`, background: `var(${tone})` }}
        />
      </div>
      <p className="mt-1.5 text-[11px] leading-snug" style={{ color: `var(${tone})` }}>
        {weak ? bad : ok ? good : (mid ?? good)}
      </p>
    </div>
  );
}

/**
 * Conditions for the scene currently on screen. `only` restricts the class
 * chips to the environments this feature is actually validated in — the global
 * A–H strip made every feature look like it lived in the same place.
 */
export function ConditionsBar({ only }: { only?: EnvId[] } = {}) {
  const { c, set, d, env, setEnv } = useConditions();
  const [open, setOpen] = useState(false);
  const pos = positioning(env);
  const list = only?.length
    ? ENVIRONMENTS.filter((e) => only.includes(e.id))
    : ENVIRONMENTS;

  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-surface">
      {/* Where we are */}
      <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 border-b border-border bg-background/40 px-4 py-3 sm:flex sm:justify-between">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            Where we are
          </span>
          <div className="flex flex-wrap gap-1 rounded-full border border-border bg-surface p-1">
            {list.map((e) => (
              <button
                key={e.id}
                onClick={() => setEnv(e.id)}
                title={`${e.blurb} — ${e.stress}`}
                className={`rounded-full px-3 py-1 text-xs transition-colors ${
                  env.id === e.id
                    ? "bg-foreground text-background"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {e.name}
              </button>
            ))}
          </div>
        </div>
        <button
          onClick={() => setOpen((o) => !o)}
          className="shrink-0 rounded-full border border-border px-3 py-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
        >
          {open ? "Done" : "Change the weather"}
        </button>
      </div>

      <div className="p-4">
        <p className="text-sm leading-snug">
          {env.blurb}.{" "}
          <span className="text-muted-foreground">
            Hard parts here: {env.stress.toLowerCase()}.
          </span>
        </p>

        {/* Plain-language condition meters */}
        <div className="mt-3 grid gap-2 sm:grid-cols-3">
          <Meter
            icon="📷"
            name="Camera can see"
            value={d.visionConf}
            good="Clear enough to trust"
            mid="Usable, but softer than ideal"
            bad="Dust, glare or dark — leaning on other sensors"
          />
          <Meter
            icon="🎙"
            name="Voice can be heard"
            value={d.audioConf}
            good="Speech will get through"
            mid="Noisy — short lines plus a buzz"
            bad="Too loud — buzzes replace words"
          />
          <Meter
            icon="◎"
            name="Location is known"
            value={pos.tone === "ok" ? 0.9 : pos.tone === "warn" ? 0.55 : 0.25}
            good={pos.label}
            bad={pos.label}
          />
        </div>

        {d.stressors.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {d.stressors.map((s) => (
              <span
                key={s}
                className="rounded-full border border-veto/50 bg-veto-soft px-2.5 py-0.5 text-[11px] text-foreground"
              >
                {s}
              </span>
            ))}
          </div>
        )}

        <p className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-guide/40 bg-guide-soft px-2.5 py-0.5 text-guide">
            <span className="h-1.5 w-1.5 rounded-full bg-guide" />
            Safety warnings stay instant
          </span>
          Bad weather only slows the talking part
          <span className="font-mono opacity-70">
            (30 ms vs {(d.slowPathMs / 1000).toFixed(1)} s)
          </span>
        </p>

        {open && (
          <div className="mt-4 grid gap-4 rounded-xl border border-border bg-background/50 p-4 sm:grid-cols-2 lg:grid-cols-3">
            <Slider
              label="How crowded"
              value={c.crowd}
              min={0}
              max={1}
              step={0.05}
              read={`${Math.round(c.crowd * 100)}%`}
              onChange={(v) => set({ ...c, crowd: v })}
            />
            <Slider
              label="How loud"
              value={c.ambientDb}
              min={35}
              max={95}
              step={1}
              read={`${c.ambientDb} dB`}
              onChange={(v) => set({ ...c, ambientDb: v })}
            />
            <Slider
              label="How bright"
              value={c.light}
              min={0}
              max={1}
              step={0.05}
              read={c.light > 0.7 ? "daylight" : c.light > 0.4 ? "overcast" : "dusk/glare"}
              onChange={(v) => set({ ...c, light: v })}
            />
            <Slider
              label="Dust or rain on the lens"
              value={c.particulate}
              min={0}
              max={1}
              step={0.05}
              read={`${Math.round(c.particulate * 100)}%`}
              onChange={(v) => set({ ...c, particulate: v })}
            />
            <Slider
              label="How shaky the walk is"
              value={c.motion}
              min={0}
              max={1}
              step={0.05}
              read={`${Math.round(c.motion * 100)}%`}
              onChange={(v) => set({ ...c, motion: v })}
            />
          </div>
        )}
      </div>
    </div>
  );
}

/** Which sensors a feature actually leans on, shown next to its simulation. */
export function ModalityBadges({ modes }: { modes: ("camera" | "mic" | "imu" | "gps")[] }) {
  const label: Record<string, string> = {
    camera: "📷 capture",
    mic: "🎙 listen",
    imu: "〰 motion",
    gps: "◎ location",
  };
  return (
    <div className="flex flex-wrap gap-1.5">
      {modes.map((m) => (
        <span
          key={m}
          className="rounded border border-border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground"
        >
          {label[m]}
        </span>
      ))}
    </div>
  );
}
