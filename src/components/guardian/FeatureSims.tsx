/**
 * Compact, interactive simulations — one per product feature.
 *
 * Rule for everything in this file: show the *decision*, not the telemetry.
 * Every sim reads the shared environment (crowd, noise, dust, light, motion)
 * so what you see is what the device would do on a real pavement, not in a
 * clean lab frame.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { useConditions } from "@/components/guardian/SceneConditions";

/* ------------------------------------------------------------------ shell */

export function SimShell({
  hint,
  children,
}: {
  hint: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-border bg-surface-2 p-5">
      {children}
      <p className="mt-4 text-xs text-muted-foreground">{hint}</p>
    </div>
  );
}

/** Spoken output. Below ~0.45 mic/ambient trust, speech loses to the street. */
function Say({ lines }: { lines: string[] }) {
  const { d } = useConditions();
  const drowned = d.audioConf < 0.45;
  return (
    <>
      <ul className="mt-4 space-y-1">
        {lines.length === 0 && <li className="text-xs text-muted-foreground">—</li>}
        {lines.slice(-3).map((l, i) => (
          <li key={i} className="font-mono text-xs text-signal">
            “{l}”
          </li>
        ))}
      </ul>
      {lines.length > 0 && drowned && (
        <p className="mt-1 font-mono text-[11px] text-signal/80">
          ambient too loud for speech → shortened phrase + haptic pattern on the belt
        </p>
      )}
    </>
  );
}

function Btn({
  onClick,
  children,
  tone = "signal",
  disabled,
}: {
  onClick: () => void;
  children: React.ReactNode;
  tone?: "signal" | "ghost";
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={
        (tone === "signal"
          ? "rounded bg-signal px-3 py-1.5 text-sm font-medium text-signal-foreground"
          : "rounded border border-border px-3 py-1.5 text-sm text-muted-foreground hover:text-foreground") +
        (disabled ? " opacity-40" : "")
      }
    >
      {children}
    </button>
  );
}

/** Shared readout for anything that has to take a picture first. */
function CaptureReadout({ retakes }: { retakes: string[] }) {
  const { d } = useConditions();
  return (
    <p className="mt-3 font-mono text-[11px] text-muted-foreground">
      capture quality {d.visionConf.toFixed(2)} · {d.slowPathMs} ms
      {retakes.length > 0 && (
        <span className="ml-2 text-signal/80">retake: {retakes[0]}</span>
      )}
    </p>
  );
}

function retakeReasons(c: {
  particulate: number;
  light: number;
  motion: number;
  crowd: number;
}) {
  const r: string[] = [];
  if (c.particulate > 0.5) r.push("wipe the lens, spray is blurring the frame");
  if (c.light < 0.35) r.push("too dark — hold still, boosting exposure");
  if (c.motion > 0.55) r.push("hold steady for a second");
  if (c.crowd > 0.7) r.push("someone stepped in front — reframing");
  return r;
}

/* ------------------------------------------------- 2 · crossing + signals */

type CrossPhase = "dont" | "walk" | "flashing";

export function CrossingSim() {
  const { c, d } = useConditions();
  const [phase, setPhase] = useState<CrossPhase>("dont");
  const [turningCar, setTurningCar] = useState(false);
  const [lines, setLines] = useState<string[]>([]);

  // A crowded, dusty or low-light frame can hide the pedestrian signal head.
  const signalReadable = d.visionConf >= 0.5;
  const readPhase: CrossPhase = signalReadable ? phase : "dont";

  const verdict = turningCar
    ? { text: "HOLD — vehicle turning across your path", tone: "urgent" as const }
    : !signalReadable
      ? {
          text: "Signal not readable — defaulting to hold",
          tone: "warn" as const,
        }
      : readPhase === "walk"
        ? { text: "Walk signal · geometry clear", tone: "ok" as const }
        : readPhase === "flashing"
          ? { text: "Flashing — do not start crossing", tone: "warn" as const }
          : { text: "Don't walk", tone: "warn" as const };

  const ask = () => {
    setLines((l) => [
      ...l,
      turningCar
        ? "Stop. Car turning into the crossing."
        : !signalReadable
          ? "I can't see the signal head. Waiting at the kerb."
          : readPhase === "walk"
            ? "Walk signal on. Crossing clear, twelve metres."
            : "Don't walk. Wait at the kerb.",
    ]);
  };

  return (
    <SimShell hint="The signal reading is advisory. A geometric veto (turning vehicle) overrides a green walk sign — and an unreadable frame degrades to hold, never to go.">
      <div className="flex flex-wrap items-center gap-2">
        {(["dont", "flashing", "walk"] as CrossPhase[]).map((p) => (
          <button
            key={p}
            onClick={() => setPhase(p)}
            className={`rounded border px-3 py-1.5 text-xs ${
              p === phase
                ? "border-signal bg-signal/10 text-signal"
                : "border-border text-muted-foreground"
            }`}
          >
            {p === "dont" ? "Don't walk" : p === "walk" ? "Walk" : "Flashing"}
          </button>
        ))}
        <label className="ml-2 flex items-center gap-2 text-xs text-muted-foreground">
          <input
            type="checkbox"
            checked={turningCar}
            onChange={(e) => setTurningCar(e.target.checked)}
          />
          turning vehicle present
        </label>
      </div>

      <div
        className={`mt-4 rounded border p-4 ${
          verdict.tone === "urgent"
            ? "border-urgent/60 bg-urgent/10"
            : verdict.tone === "ok"
              ? "border-signal/50 bg-signal/5"
              : "border-border bg-surface"
        }`}
      >
        <p className="font-mono text-sm">{verdict.text}</p>
        <p className="mt-1 font-mono text-[11px] text-muted-foreground">
          true state “{phase}” · signal head {signalReadable ? "visible" : "occluded"}
          {c.crowd > 0.6 ? " by crowd" : c.particulate > 0.5 ? " by spray" : ""}
        </p>
      </div>

      <div className="mt-4">
        <Btn onClick={ask}>&ldquo;Can I cross?&rdquo;</Btn>
      </div>
      <Say lines={lines} />
    </SimShell>
  );
}

/* --------------------------------------------------- 3 · breadcrumb trail */

const CRUMBS = ["Front door", "Lift lobby", "Street corner", "Bus stop"];

export function BreadcrumbSim() {
  const { c, d } = useConditions();
  const [dropped, setDropped] = useState<string[]>([]);
  const [retraceIdx, setRetraceIdx] = useState<number | null>(null);

  // Visual landmark re-recognition degrades; dead reckoning takes over.
  const visualFix = d.visionConf > 0.55;
  const driftM = Math.round(2 + 14 * (1 - d.visionConf) + 6 * c.crowd);

  const lines =
    retraceIdx === null
      ? []
      : visualFix
        ? [`Turn toward ${dropped[retraceIdx]}. Twenty metres.`]
        : [`Heading back toward ${dropped[retraceIdx]}. Roughly twenty metres — landmark not confirmed yet.`];

  return (
    <SimShell hint="Voice-tagged waypoints retraced in reverse. When the crowd hides the landmark, the trail falls back to step-count dead reckoning and says so instead of pretending to be certain.">
      <div className="flex flex-wrap gap-2">
        {CRUMBS.map((cr) => (
          <button
            key={cr}
            onClick={() => setDropped((dl) => (dl.includes(cr) ? dl : [...dl, cr]))}
            disabled={dropped.includes(cr)}
            className="rounded border border-border px-3 py-1.5 text-xs text-muted-foreground disabled:opacity-40"
          >
            drop “{cr}”
          </button>
        ))}
      </div>

      <ol className="mt-4 space-y-px">
        {dropped.length === 0 && (
          <li className="rounded bg-surface p-3 text-xs text-muted-foreground">
            No waypoints yet.
          </li>
        )}
        {dropped.map((cr, i) => (
          <li
            key={cr}
            className={`flex items-center justify-between rounded px-3 py-2 text-sm ${
              retraceIdx === i ? "bg-signal/10 text-signal" : "bg-surface"
            }`}
          >
            <span>{cr}</span>
            <span className="font-mono text-xs text-muted-foreground">
              {retraceIdx === i ? "◀ guiding" : `#${i + 1}`}
            </span>
          </li>
        ))}
      </ol>

      <div className="mt-4 flex gap-2">
        <Btn
          onClick={() =>
            setRetraceIdx(dropped.length ? dropped.length - 1 : null)
          }
        >
          &ldquo;Take me back&rdquo;
        </Btn>
        <Btn
          tone="ghost"
          onClick={() =>
            setRetraceIdx((r) => (r === null || r === 0 ? r : r - 1))
          }
        >
          Next leg
        </Btn>
        <Btn
          tone="ghost"
          onClick={() => {
            setDropped([]);
            setRetraceIdx(null);
          }}
        >
          Reset
        </Btn>
      </div>
      <Say lines={lines} />
      <p className="mt-2 font-mono text-[11px] text-muted-foreground">
        {visualFix ? "visual landmark fix" : "dead reckoning only"} · position drift ±{driftM} m
      </p>
    </SimShell>
  );
}

/* ------------------------------------------------ 4 · indoor object memory */

const ROOM = [
  { id: "keys", label: "Keys", x: 18, y: 30 },
  { id: "cane", label: "Cane", x: 74, y: 22 },
  { id: "phone", label: "Phone", x: 52, y: 72 },
];

export function ObjectMemorySim() {
  const { d } = useConditions();
  const [stored, setStored] = useState<string[]>([]);
  const [asked, setAsked] = useState<string | null>(null);
  const target = ROOM.find((o) => o.id === asked) ?? null;

  const bearing = target
    ? Math.round((Math.atan2(target.x - 50, 92 - target.y) * 180) / Math.PI)
    : 0;
  // Indoors is kinder than the street, but a dim room still costs recall.
  const recall = Math.min(0.98, 0.55 + 0.45 * d.visionConf);
  const sure = recall > 0.8;

  return (
    <SimShell hint="One spatial fingerprint per object, stored on device. Recall is a directional haptic pulse plus a distance — and a hedge (“last seen”) whenever the stored frame was captured in poor light.">
      <div className="relative h-44 overflow-hidden rounded border border-border bg-surface">
        {ROOM.map((o) => (
          <button
            key={o.id}
            onClick={() => setStored((s) => (s.includes(o.id) ? s : [...s, o.id]))}
            style={{ left: `${o.x}%`, top: `${o.y}%` }}
            className={`absolute -translate-x-1/2 -translate-y-1/2 rounded border px-2 py-1 text-xs ${
              asked === o.id
                ? "border-signal bg-signal/20 text-signal"
                : stored.includes(o.id)
                  ? "border-signal/50 text-signal"
                  : "border-border text-muted-foreground"
            }`}
          >
            {o.label}
            {stored.includes(o.id) ? " ✓" : ""}
          </button>
        ))}
        <span className="absolute bottom-2 left-1/2 -translate-x-1/2 font-mono text-xs text-muted-foreground">
          you ▲
        </span>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        {ROOM.map((o) => (
          <Btn
            key={o.id}
            tone={stored.includes(o.id) ? "signal" : "ghost"}
            onClick={() => stored.includes(o.id) && setAsked(o.id)}
          >
            “Where are my {o.label.toLowerCase()}?”
          </Btn>
        ))}
      </div>

      <Say
        lines={
          target
            ? [
                `${target.label} ${sure ? "is" : "was last seen"} ${bearing > 8 ? "to your right" : bearing < -8 ? "to your left" : "straight ahead"}, about ${Math.round(3 + Math.abs(bearing) / 12)} metres.${sure ? "" : " I'm not certain."}`,
              ]
            : []
        }
      />
      <p className="mt-2 text-xs text-muted-foreground">
        Tap an object in the room to store it, then ask for it back. Recall confidence{" "}
        <span className="font-mono">{recall.toFixed(2)}</span>.
      </p>
    </SimShell>
  );
}

/* ---------------------------------------------- 5 · on-demand description */

const ASKS = [
  {
    q: "What's in front of me?",
    a: "A café front. Glass door on the left, two chairs on the pavement to your right.",
    degraded:
      "A shopfront ahead, maybe a café. Too much movement in front of me to be sure about the door.",
  },
  {
    q: "Read this label",
    a: "Ibuprofen 200 mg. Two tablets every four hours.",
    degraded: "I can see a medicine box but the print isn't sharp enough to read. Try again with the box closer.",
  },
  {
    q: "What colour is this shirt?",
    a: "Dark navy with a thin white stripe.",
    degraded: "Something dark — navy or black. The light here isn't good enough to call it.",
  },
];

export function SceneAskSim() {
  const { c, d } = useConditions();
  const [busy, setBusy] = useState(false);
  const [lines, setLines] = useState<string[]>([]);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);
  const retakes = retakeReasons(c);

  return (
    <SimShell hint="Takes a frame, answers on the slow path at 0.5–1 Hz. In a bad frame it hedges or asks for a retake — it never invents detail it could not see.">
      <div className="flex flex-wrap gap-2">
        {ASKS.map((a) => (
          <Btn
            key={a.q}
            tone="ghost"
            onClick={() => {
              setBusy(true);
              timer.current = setTimeout(() => {
                setBusy(false);
                setLines((l) => [...l, d.visionConf > 0.6 ? a.a : a.degraded]);
              }, Math.min(2200, d.slowPathMs));
            }}
          >
            {a.q}
          </Btn>
        ))}
      </div>
      <p className="mt-4 font-mono text-xs text-muted-foreground">
        {busy ? "shutter → slow path · thinking…" : "slow path · idle"}
        <span className="ml-3 text-signal">fast path unaffected · 0 ms</span>
      </p>
      <CaptureReadout retakes={retakes} />
      <Say lines={lines} />
    </SimShell>
  );
}

/* ---------------------------------------------- 6 · currency & documents */

type ReadItem = {
  k: string;
  kind: "currency" | "document";
  /** face tint for the rendered card */
  tint: string;
  /** big glyph shown on the rendered face */
  face: string;
  sub: string;
  /** fields OCR is trying to lift, in priority order */
  fields: { label: string; value: string }[];
  v: string;
  d: string;
  /** dominant failure mode for this surface */
  fail: string;
};

const READS: ReadItem[] = [
  {
    k: "£20 note",
    kind: "currency",
    tint: "oklch(0.62 0.13 320)",
    face: "£20",
    sub: "Bank of England · polymer",
    fields: [
      { label: "value", value: "20" },
      { label: "currency", value: "GBP" },
      { label: "orientation", value: "face up" },
    ],
    v: "Twenty pounds. Polymer note, facing up.",
    d: "A banknote — I can see it's polymer but the value window is glared out. Tilt it away from the light.",
    fail: "glare on the polymer window",
  },
  {
    k: "€50 note",
    kind: "currency",
    tint: "oklch(0.7 0.14 60)",
    face: "€50",
    sub: "Europa series · cotton",
    fields: [
      { label: "value", value: "50" },
      { label: "currency", value: "EUR" },
      { label: "orientation", value: "reverse" },
    ],
    v: "Fifty euro. Reverse side up — the value is the same either way.",
    d: "A euro note, but the corner numeral is cut off in frame. Move it back a little and I'll retry.",
    fail: "numeral cropped out of frame",
  },
  {
    k: "$1 / $20 stack",
    kind: "currency",
    tint: "oklch(0.68 0.09 150)",
    face: "$20",
    sub: "two notes, same size",
    fields: [
      { label: "top note", value: "20" },
      { label: "under note", value: "1" },
      { label: "currency", value: "USD" },
    ],
    v: "Top note twenty dollars, note underneath is a one. Two notes in hand.",
    d: "Two US notes overlapping — same size, so I won't guess which is on top. Separate them and show me one.",
    fail: "same-size notes overlapping",
  },
  {
    k: "₹500 note",
    kind: "currency",
    tint: "oklch(0.66 0.12 30)",
    face: "₹500",
    sub: "Mahatma Gandhi series",
    fields: [
      { label: "value", value: "500" },
      { label: "currency", value: "INR" },
      { label: "condition", value: "creased" },
    ],
    v: "Five hundred rupees. Note is creased across the middle but the value is clear.",
    d: "A rupee note, heavily creased through the numeral. I can't separate five hundred from fifty — flatten it.",
    fail: "crease running through the numeral",
  },
  {
    k: "Utility bill",
    kind: "document",
    tint: "oklch(0.95 0.01 250)",
    face: "BILL",
    sub: "A4 · printed",
    fields: [
      { label: "issuer", value: "Thames Water" },
      { label: "due date", value: "3 June" },
      { label: "amount due", value: "£42.10" },
    ],
    v: "Water bill dated 3 June. Amount due, forty-two pounds ten.",
    d: "A printed bill. The amount line is too blurred to read — hold it flatter and I'll retry.",
    fail: "motion blur on the amount line",
  },
  {
    k: "Prescription label",
    kind: "document",
    tint: "oklch(0.93 0.03 200)",
    face: "Rx",
    sub: "pharmacy label · curved",
    fields: [
      { label: "drug", value: "Amoxicillin 500mg" },
      { label: "dose", value: "1 capsule, 3× daily" },
      { label: "warning", value: "finish the course" },
    ],
    v: "Amoxicillin, five hundred milligrams. One capsule three times a day. Finish the course.",
    d: "A medicine label, but it curves away from the camera and the dose line is cut. I will not guess a dose — turn the box towards me.",
    fail: "label curves off the cylinder",
  },
  {
    k: "Passport page",
    kind: "document",
    tint: "oklch(0.9 0.02 260)",
    face: "MRZ",
    sub: "ID page · laminate",
    fields: [
      { label: "surname", value: "read locally" },
      { label: "expiry", value: "11 / 2029" },
      { label: "MRZ", value: "checksum ok" },
    ],
    v: "Passport ID page. Expires November two thousand twenty-nine. Nothing from this page leaves the device.",
    d: "Passport page under laminate glare — the MRZ checksum fails, so I'll say nothing rather than misread an ID.",
    fail: "laminate glare breaks the MRZ",
  },
  {
    k: "Restaurant receipt",
    kind: "document",
    tint: "oklch(0.96 0.01 90)",
    face: "£",
    sub: "thermal · faded",
    fields: [
      { label: "total", value: "£18.40" },
      { label: "service", value: "not included" },
      { label: "card", value: "•••• 4417" },
    ],
    v: "Total eighteen pounds forty. Service is not included.",
    d: "Thermal receipt, faded print. I can see a total line but not the figure — try under brighter light.",
    fail: "faded thermal print",
  },
];

/** Rendered face of whatever is being held up, with the live failure overlay. */
function ItemFace({ item, ok }: { item: ReadItem; ok: boolean }) {
  const currency = item.kind === "currency";
  return (
    <svg viewBox="0 0 200 116" className="h-28 w-full max-w-[260px]" role="img" aria-label={item.k}>
      <defs>
        <linearGradient id="rd-glare" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="white" stopOpacity="0" />
          <stop offset="45%" stopColor="white" stopOpacity="0.75" />
          <stop offset="100%" stopColor="white" stopOpacity="0" />
        </linearGradient>
      </defs>
      <rect
        x="10"
        y={currency ? 24 : 8}
        width="180"
        height={currency ? 68 : 100}
        rx={currency ? 5 : 3}
        fill={item.tint}
        stroke="oklch(0.3 0.02 250 / 0.35)"
      />
      {currency ? (
        <>
          <rect x="20" y="34" width="44" height="48" rx="3" fill="white" opacity="0.28" />
          <text x="150" y="52" textAnchor="middle" fontSize="22" fontWeight="700" fill="white">
            {item.face}
          </text>
          <text x="150" y="70" textAnchor="middle" fontSize="7" fill="white" opacity="0.85">
            {item.sub}
          </text>
        </>
      ) : (
        <>
          <text x="24" y="28" fontSize="13" fontWeight="700" fill="oklch(0.3 0.03 250)">
            {item.face}
          </text>
          {[0, 1, 2, 3, 4].map((i) => (
            <rect
              key={i}
              x="24"
              y={40 + i * 11}
              width={i === 2 ? 120 : 150}
              height="5"
              rx="2.5"
              fill="oklch(0.3 0.03 250)"
              opacity={i === 2 ? 0.6 : 0.28}
            />
          ))}
        </>
      )}
      {!ok && (
        <g>
          <rect
            x="10"
            y={currency ? 24 : 8}
            width="180"
            height={currency ? 68 : 100}
            rx="4"
            fill="url(#rd-glare)"
          />
          <rect
            x="10"
            y={currency ? 24 : 8}
            width="180"
            height={currency ? 68 : 100}
            rx="4"
            fill="oklch(0.63 0.213 26 / 0.10)"
            stroke="oklch(0.63 0.213 26 / 0.6)"
            strokeDasharray="4 3"
          />
        </g>
      )}
    </svg>
  );
}

export function ReaderSim() {
  const { c, d } = useConditions();
  const [tab, setTab] = useState<"currency" | "document">("currency");
  const [sel, setSel] = useState<string | null>(null);
  const list = READS.filter((r) => r.kind === tab);
  const item = READS.find((r) => r.k === sel) ?? null;
  const shown = item && item.kind === tab ? item : null;
  const ok = d.visionConf > 0.62;
  const retakes = retakeReasons(c);

  return (
    <SimShell hint="Pure capture feature: it lives or dies on frame quality. Glare on a polymer note, same-size dollar bills, a curved medicine label and laminate over a passport MRZ are the four failures that actually happen — all simulated rather than assumed away.">
      <div className="mb-3 inline-flex rounded border border-border p-0.5">
        {(["currency", "document"] as const).map((t) => (
          <button
            key={t}
            onClick={() => {
              setTab(t);
              setSel(null);
            }}
            className={
              "rounded px-3 py-1 text-xs font-medium capitalize " +
              (tab === t
                ? "bg-signal text-signal-foreground"
                : "text-muted-foreground hover:text-foreground")
            }
          >
            {t === "currency" ? "Currency" : "Documents"}
          </button>
        ))}
      </div>

      <div className="flex flex-wrap gap-2">
        {list.map((r) => (
          <Btn key={r.k} tone={r.k === sel ? "signal" : "ghost"} onClick={() => setSel(r.k)}>
            {r.k}
          </Btn>
        ))}
      </div>

      {shown && (
        <div className="mt-4 grid gap-4 sm:grid-cols-[auto_1fr] sm:items-start">
          <div className="rounded border border-border bg-surface p-3">
            <ItemFace item={shown} ok={ok} />
            <p className="mt-2 text-center font-mono text-[11px] text-muted-foreground">
              {ok ? "frame accepted" : `blocked · ${shown.fail}`}
            </p>
          </div>
          <ul className="space-y-1">
            {shown.fields.map((f, i) => {
              const lifted = ok || i === 0;
              return (
                <li
                  key={f.label}
                  className="flex items-center justify-between gap-3 rounded border border-border bg-surface px-3 py-1.5 font-mono text-[11px]"
                >
                  <span className="text-muted-foreground">{f.label}</span>
                  <span className={lifted ? "text-foreground" : "text-urgent"}>
                    {lifted ? f.value : "unreadable — not guessed"}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      <CaptureReadout retakes={retakes} />
      <Say lines={shown === null ? [] : [ok ? shown.v : shown.d]} />
    </SimShell>
  );
}


/* ------------------------------------------------------ 7 · places/errands */

const PLACES = [
  { q: "Nearest pharmacy", a: "Boots, 180 metres, open until 6pm.", door: "left" },
  { q: "Nearest ATM", a: "Cash machine, 60 metres, on this side of the street.", door: "ahead" },
  { q: "Is the café open?", a: "Yes — closes in forty minutes.", door: "right" },
];

export function PlacesSim() {
  const { c, d } = useConditions();
  const [i, setI] = useState<number | null>(null);
  const p = i === null ? null : PLACES[i]!;
  const gpsErr = Math.round(5 + 18 * c.crowd + 10 * (1 - d.visionConf));
  return (
    <SimShell hint="Places lookup ends at the block — GPS in a crowded street canyon is worth ±20 m at best. The breadcrumb haptic channel takes over for the last stretch and puts you at the door.">
      <div className="flex flex-wrap gap-2">
        {PLACES.map((x, idx) => (
          <Btn key={x.q} tone={idx === i ? "signal" : "ghost"} onClick={() => setI(idx)}>
            {x.q}
          </Btn>
        ))}
      </div>
      {p && (
        <div className="mt-4 rounded border border-border bg-surface p-4">
          <p className="text-sm">{p.a}</p>
          <p className="mt-2 font-mono text-xs text-signal">
            handoff → haptic guide · door {p.door}
          </p>
          <p className="mt-1 font-mono text-[11px] text-muted-foreground">
            GPS uncertainty ±{gpsErr} m · handoff at {gpsErr > 18 ? "30" : "20"} m
          </p>
        </div>
      )}
    </SimShell>
  );
}

/* ----------------------------------------------------- 8 · dynamic reroute */

export function RerouteSim() {
  const { c } = useConditions();
  const [avoid, setAvoid] = useState(true);
  const path = avoid
    ? "M 10 90 L 10 40 L 60 40 L 60 10 L 110 10"
    : "M 10 90 L 60 90 L 60 10 L 110 10";
  const detour = Math.round(40 + 60 * c.crowd);
  return (
    <SimShell hint="A hazard you met once changes the route, not just the alert. Crowd density is part of the cost model — a detour through a packed market is not a cheap detour.">
      <div className="flex items-center gap-4">
        <svg viewBox="0 0 120 100" className="h-40 w-full max-w-xs">
          <rect x="0" y="0" width="120" height="100" fill="transparent" />
          <path d={path} fill="none" stroke="oklch(0.7 0.12 220)" strokeWidth="3" />
          <circle cx="60" cy="90" r="7" fill="oklch(0.63 0.213 26 / 0.35)" />
          <text x="60" y="93" textAnchor="middle" fontSize="7" fill="oklch(0.63 0.213 26)">
            ⚠
          </text>
          <circle cx="10" cy="90" r="3" fill="oklch(0.79 0.155 78)" />
          <circle cx="110" cy="10" r="3" fill="oklch(0.79 0.155 78)" />
        </svg>
        <div>
          <Btn tone="ghost" onClick={() => setAvoid((a) => !a)}>
            {avoid ? "Show route without memory" : "Re-plan around hazard"}
          </Btn>
          <p className="mt-3 font-mono text-xs text-muted-foreground">
            {avoid
              ? `detour +${detour} m · 1 known hazard avoided`
              : "shortest path · walks the hazard"}
          </p>
        </div>
      </div>
    </SimShell>
  );
}

/* ------------------------------------------------ 9 · human-in-the-loop */

export function HumanLoopSim() {
  const { d } = useConditions();
  const [conf, setConf] = useState(0.82);
  // Bad frames drag the effective confidence down — that is the whole point.
  const eff = Math.min(conf, 0.35 + 0.75 * d.visionConf);
  const escalate = eff < 0.6;
  const lines = useMemo(
    () =>
      escalate
        ? ["I'm not sure about this one. Connecting you to a sighted assistant."]
        : ["Sign reads: residents' parking, permit holders only."],
    [escalate],
  );
  return (
    <SimShell hint="Escalation is triggered by the model's own uncertainty — including uncertainty caused by dust, glare or a crowd — not by the user noticing it was wrong.">
      <label className="block text-xs text-muted-foreground">
        Model confidence on a clean frame:{" "}
        <span className="font-mono text-signal">{conf.toFixed(2)}</span>
      </label>
      <input
        type="range"
        min={0.2}
        max={0.99}
        step={0.01}
        value={conf}
        onChange={(e) => setConf(Number(e.target.value))}
        className="mt-2 w-full"
      />
      <p className="mt-1 font-mono text-[11px] text-muted-foreground">
        after this environment: {eff.toFixed(2)}
      </p>
      <div
        className={`mt-4 rounded border p-4 text-sm ${
          escalate ? "border-signal/60 bg-signal/5" : "border-border bg-surface"
        }`}
      >
        {escalate ? "Escalating to a human assistant (≈12 s to connect)" : "Answered on device"}
      </div>
      <Say lines={lines} />
    </SimShell>
  );
}

/* ------------------------------------------------- 11 · live translation */

const TRANSLATION_SCENES = [
  {
    id: "speech",
    label: "Someone speaks to you",
    src: "¿Necesita ayuda para cruzar?",
    lang: "Spanish · speech",
    out: "Do you need help crossing?",
    partial: "Do you need help … crossing? (partly masked)",
    modality: "mic" as const,
  },
  {
    id: "announce",
    label: "Station announcement",
    src: "Le train pour Lyon partira du quai 4.",
    lang: "French · PA system",
    out: "The train to Lyon will leave from platform 4.",
    partial: "The train to Lyon … platform four, I think. Echo on the PA.",
    modality: "mic" as const,
  },
  {
    id: "sign",
    label: "Read that sign",
    src: "AUSGANG / NOTAUSGANG RECHTS",
    lang: "German · signage",
    out: "Exit. Emergency exit on the right.",
    partial: "Something like “exit” — the sign is too dim to read fully.",
    modality: "camera" as const,
  },
  {
    id: "menu",
    label: "Read this menu",
    src: "Zuppa del giorno — minestrone, 8 €",
    lang: "Italian · print",
    out: "Soup of the day, minestrone, eight euros.",
    partial: "Soup of the day … the price line is blurred.",
    modality: "camera" as const,
  },
];

export function TranslationSim() {
  const { c, d } = useConditions();
  const [sel, setSel] = useState<string>("speech");
  const scene = TRANSLATION_SCENES.find((s) => s.id === sel)!;
  const trust = scene.modality === "mic" ? d.audioConf : d.visionConf;
  const clean = trust > 0.55;

  // In a crowd the mic hears several voices; the device has to pick one.
  const voices = 1 + Math.round(4 * c.crowd);
  const latency = Math.round(
    (scene.modality === "mic" ? 450 : 700) + 400 * (1 - trust),
  );

  return (
    <SimShell hint="Two inputs, one feature: the microphone for speech and announcements, the camera for signs and menus. Both run on the slow path — a translation is never allowed to delay a hazard warning.">
      <div className="flex flex-wrap gap-2">
        {TRANSLATION_SCENES.map((s) => (
          <Btn key={s.id} tone={s.id === sel ? "signal" : "ghost"} onClick={() => setSel(s.id)}>
            {s.label}
          </Btn>
        ))}
      </div>

      <div className="mt-4 rounded border border-border bg-surface p-4">
        <p className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
          heard / seen · {scene.lang}
        </p>
        <p className="mt-1 text-sm">{scene.src}</p>
        <p className="mt-3 font-mono text-[11px] text-muted-foreground">
          {scene.modality === "mic"
            ? `${voices} overlapping voice${voices > 1 ? "s" : ""} · speaker separation ${d.audioConf > 0.5 ? "locked on nearest" : "uncertain"}`
            : `frame quality ${d.visionConf.toFixed(2)} · ${c.light < 0.4 ? "low light" : "OCR ok"}`}{" "}
          · {latency} ms
        </p>
      </div>

      <Say lines={[clean ? scene.out : scene.partial]} />
      {!clean && (
        <p className="mt-1 font-mono text-[11px] text-signal/80">
          low confidence → offers “ask them to repeat” or a human assistant instead of guessing
        </p>
      )}
    </SimShell>
  );
}
