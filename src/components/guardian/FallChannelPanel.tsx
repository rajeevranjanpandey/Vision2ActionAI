import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import fallData from "@/data/fall_detection.json";
import {
  FallEscalation,
  type EscalationSnapshot,
  type GpsFix,
} from "@/lib/fallEscalation";

interface TraceSample {
  t: number;
  svm: number;
}

interface TraceEvent {
  tS: number;
  confidence: number;
  peakG: number;
  freefallMs: number;
  tiltDeg: number;
  postImpactStdG: number;
  summary: string;
}

interface Trace {
  label: string;
  title: string;
  isFall: boolean;
  onsetS: number;
  fired: boolean;
  event: TraceEvent | null;
  samples: TraceSample[];
}

const D = fallData as unknown as {
  provenance: string;
  episodesPerClass: number;
  config: {
    sampleRateHz: number;
    freefallG: number;
    impactG: number;
    postureChangeDeg: number;
    stillnessWindowMs: number;
    stillnessStdG: number;
    countdownS: number;
  };
  report: {
    sensitivity: number;
    specificity: number;
    falsePositives: number;
    falseNegatives: number;
    nFalls: number;
    nAdls: number;
    meanConfidence: number;
    medianDetectLatencyMs: number;
    falsePositivesPerWeek: number;
  };
  perClass: { label: string; n: number; fireRate: number }[];
  sweep: {
    impact_g: number;
    sensitivity: number;
    specificity: number;
    false_positives_per_week: number;
  }[];
  traces: Trace[];
};

const CONTACT = { name: "Asha (sister)", phoneE164: "+447700900123" };
const FIX: GpsFix = { latitude: 51.5007, longitude: -0.1246, accuracyM: 12, ageS: 4 };
const PLAYBACK_HZ = 25;
const COUNTDOWN_SPEEDUP = 6; // 30 s cancel window replayed in 5 s

const WIDTH = 720;
const HEIGHT = 220;

function prettyLabel(label: string): string {
  return label.replace(/_/g, " ");
}

export function FallChannelPanel() {
  const [traceIndex, setTraceIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [cursor, setCursor] = useState(0);
  const [snapshot, setSnapshot] = useState<EscalationSnapshot | null>(null);

  const trace = D.traces[traceIndex]!;
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const escRef = useRef<FallEscalation | null>(null);
  const clockRef = useRef(0);

  const maxSvm = useMemo(
    () => Math.max(4.2, ...trace.samples.map((s) => s.svm)),
    [trace],
  );

  const reset = useCallback(() => {
    setPlaying(false);
    setCursor(0);
    clockRef.current = 0;
    escRef.current = new FallEscalation(CONTACT, {
      countdownS: D.config.countdownS,
      countdownPromptS: 5,
    });
    setSnapshot(escRef.current.snapshot(0));
  }, []);

  useEffect(() => {
    reset();
  }, [traceIndex, reset]);

  // Replay loop: advance the trace, then drive the escalation machine once the
  // classifier has confirmed. The countdown runs on a sped-up clock so the whole
  // 30 s cancel window is watchable.
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setCursor((c) => {
        const next = c + 1;
        const esc = escRef.current;
        const samples = trace.samples;
        if (!esc) return next;
        const tNow = samples[Math.min(next, samples.length - 1)]!.t;

        if (trace.event && tNow >= trace.event.tS + 1.5) {
          esc.onFall(
            {
              peakG: trace.event.peakG,
              freefallMs: trace.event.freefallMs,
              tiltDeg: trace.event.tiltDeg,
              confidence: trace.event.confidence,
              summary: trace.event.summary,
            },
            clockRef.current,
          );
        }
        if (next >= samples.length - 1) {
          clockRef.current += COUNTDOWN_SPEEDUP / PLAYBACK_HZ;
        } else {
          clockRef.current += 1 / PLAYBACK_HZ;
        }
        esc.tick(clockRef.current, FIX);
        setSnapshot(esc.snapshot(clockRef.current));

        const snap = esc.snapshot(clockRef.current);
        if (next >= samples.length - 1 && snap.state !== "countdown") {
          setPlaying(false);
          return samples.length - 1;
        }
        return Math.min(next, samples.length - 1);
      });
    }, 1000 / PLAYBACK_HZ);
    return () => window.clearInterval(id);
  }, [playing, trace]);

  // Canvas: the accelerometer magnitude trace with the three decision thresholds.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = WIDTH * dpr;
    canvas.height = HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, WIDTH, HEIGHT);

    const samples = trace.samples;
    const tMax = samples[samples.length - 1]!.t;
    const x = (t: number) => (t / tMax) * (WIDTH - 44) + 36;
    const y = (g: number) => HEIGHT - 26 - (g / maxSvm) * (HEIGHT - 46);

    ctx.strokeStyle = "rgba(148,163,184,0.16)";
    ctx.fillStyle = "rgba(148,163,184,0.75)";
    ctx.font = "10px ui-monospace, monospace";
    for (let g = 0; g <= Math.ceil(maxSvm); g += 1) {
      ctx.beginPath();
      ctx.moveTo(36, y(g));
      ctx.lineTo(WIDTH - 8, y(g));
      ctx.stroke();
      ctx.fillText(`${g}g`, 6, y(g) + 3);
    }

    // Impact and free-fall thresholds — the two lines the template tests against.
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(248,113,113,0.7)";
    ctx.beginPath();
    ctx.moveTo(36, y(D.config.impactG));
    ctx.lineTo(WIDTH - 8, y(D.config.impactG));
    ctx.stroke();
    ctx.strokeStyle = "rgba(56,189,248,0.7)";
    ctx.beginPath();
    ctx.moveTo(36, y(D.config.freefallG));
    ctx.lineTo(WIDTH - 8, y(D.config.freefallG));
    ctx.stroke();
    ctx.setLineDash([]);

    if (trace.event) {
      const x0 = x(trace.event.tS);
      const x1 = x(trace.event.tS + D.config.stillnessWindowMs / 1000);
      ctx.fillStyle = "rgba(250,204,21,0.10)";
      ctx.fillRect(x0, 8, x1 - x0, HEIGHT - 34);
    }

    ctx.lineWidth = 1.6;
    ctx.strokeStyle = "rgba(226,232,240,0.9)";
    ctx.beginPath();
    samples.forEach((s, i) => {
      const px = x(s.t);
      const py = y(s.svm);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();

    const head = samples[Math.min(cursor, samples.length - 1)]!;
    ctx.strokeStyle = "rgba(250,204,21,0.9)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x(head.t), 8);
    ctx.lineTo(x(head.t), HEIGHT - 26);
    ctx.stroke();

    ctx.fillStyle = "rgba(148,163,184,0.75)";
    ctx.fillText(`${head.t.toFixed(2)} s`, x(head.t) + 4, 18);
  }, [trace, cursor, maxSvm]);

  const state = snapshot?.state ?? "idle";
  const countdown = snapshot?.remainingS ?? 0;

  return (
    <div className="space-y-8">
      <div className="grid gap-6 lg:grid-cols-[3fr_2fr]">
        {/* Replay */}
        <div className="rounded-lg border border-border bg-surface p-6">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <h3 className="text-lg font-semibold">IMU replay · 50 Hz</h3>
            <span className="font-mono text-xs text-muted-foreground">
              async channel · 0 ms on the fast path
            </span>
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            {D.traces.map((t, i) => (
              <button
                key={t.label}
                onClick={() => setTraceIndex(i)}
                className={`rounded border px-3 py-1.5 text-xs transition-colors ${
                  i === traceIndex
                    ? "border-signal bg-signal/10 text-signal"
                    : "border-border bg-surface-2 text-muted-foreground hover:text-foreground"
                }`}
              >
                {t.title}
                <span className="ml-2 font-mono opacity-70">
                  {t.isFall ? "fall" : "ADL"}
                </span>
              </button>
            ))}
          </div>

          <div className="mt-4 overflow-hidden rounded border border-border bg-surface-2">
            <canvas
              ref={canvasRef}
              style={{ width: "100%", height: HEIGHT }}
              aria-label={`Accelerometer magnitude trace for ${trace.title}`}
            />
          </div>

          <div className="mt-4 flex flex-wrap items-center gap-3">
            <button
              onClick={() => setPlaying((p) => !p)}
              className="rounded bg-signal px-4 py-2 text-sm font-medium text-signal-foreground"
            >
              {playing ? "Pause" : "Replay episode"}
            </button>
            <button
              onClick={reset}
              className="rounded border border-border px-4 py-2 text-sm text-muted-foreground hover:text-foreground"
            >
              Reset
            </button>
            <p className="text-xs text-muted-foreground">
              Red line: 2.4 g impact gate. Blue line: 0.6 g free-fall gate. Amber band:
              the 1.5 s post-impact stillness test.
            </p>
          </div>
        </div>

        {/* Escalation */}
        <div
          className={`rounded-lg border p-6 ${
            state === "countdown"
              ? "border-urgent/60 bg-urgent/5"
              : "border-border bg-surface"
          }`}
        >
          <div className="flex items-baseline justify-between">
            <h3 className="text-lg font-semibold">Trusted-contact escalation</h3>
            <span className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
              {state}
            </span>
          </div>

          {trace.event ? (
            <dl className="mt-5 grid grid-cols-2 gap-px overflow-hidden rounded border border-border bg-border">
              <Cellet label="peak" value={`${trace.event.peakG.toFixed(1)} g`} />
              <Cellet label="free-fall" value={`${Math.round(trace.event.freefallMs)} ms`} />
              <Cellet label="posture Δ" value={`${Math.round(trace.event.tiltDeg)}°`} />
              <Cellet
                label="confidence"
                value={trace.event.confidence.toFixed(2)}
              />
            </dl>
          ) : (
            <p className="mt-5 rounded border border-border bg-surface-2 p-4 text-sm text-muted-foreground">
              Classifier stayed silent on this episode. An impact alone does not escalate:
              the wearer stayed upright, or kept moving, so no message is composed.
            </p>
          )}

          {state === "countdown" && (
            <div className="mt-5 rounded border border-urgent/50 bg-surface-2 p-4">
              <p className="font-mono text-3xl text-urgent">
                {countdown.toFixed(0)}s
              </p>
              <p className="mt-1 text-sm text-muted-foreground">
                until {CONTACT.name} is contacted
              </p>
              <button
                onClick={() => {
                  const esc = escRef.current;
                  if (!esc) return;
                  esc.cancel(clockRef.current);
                  setSnapshot(esc.snapshot(clockRef.current));
                }}
                className="mt-3 w-full rounded bg-urgent px-4 py-2 text-sm font-medium text-white"
              >
                &ldquo;Cancel&rdquo; — I&apos;m fine
              </button>
            </div>
          )}

          {snapshot?.message && (
            <div className="mt-5 rounded border border-border bg-surface-2 p-4">
              <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
                SMS to {CONTACT.phoneE164}
              </p>
              <p className="mt-2 text-sm leading-relaxed">{snapshot.message}</p>
            </div>
          )}

          {state === "cancelled" && (
            <p className="mt-5 rounded border border-border bg-surface-2 p-4 text-sm text-muted-foreground">
              Cancelled. Nothing was sent — and the episode is retained locally as a
              labelled negative with its features attached, which is exactly the payload
              the federated near-miss loop wants.
            </p>
          )}

          <div className="mt-5">
            <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
              Spoken
            </p>
            <ul className="mt-2 space-y-1">
              {(snapshot?.spoken ?? []).slice(-4).map((line, i) => (
                <li key={i} className="font-mono text-xs text-signal">
                  “{line}”
                </li>
              ))}
              {(snapshot?.spoken.length ?? 0) === 0 && (
                <li className="text-xs text-muted-foreground">—</li>
              )}
            </ul>
          </div>
        </div>
      </div>

      {/* Measured */}
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-lg border border-border bg-surface p-6">
          <h3 className="text-lg font-semibold">Fire rate by activity</h3>
          <p className="mt-2 text-sm text-muted-foreground">
            Sensitivity is the easy half. The four activities on the right are the ones a
            bare 2 g threshold gets wrong.
          </p>
          <div className="mt-6 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={D.perClass.map((c) => ({
                  ...c,
                  name: prettyLabel(c.label),
                  pct: c.fireRate * 100,
                }))}
                margin={{ top: 4, right: 12, bottom: 60, left: 0 }}
              >
                <CartesianGrid stroke="rgba(148,163,184,0.14)" vertical={false} />
                <XAxis
                  dataKey="name"
                  angle={-40}
                  textAnchor="end"
                  interval={0}
                  tick={{ fontSize: 11, fill: "rgba(148,163,184,0.9)" }}
                />
                <YAxis
                  unit="%"
                  domain={[0, 100]}
                  tick={{ fontSize: 11, fill: "rgba(148,163,184,0.9)" }}
                />
                <Tooltip
                  contentStyle={{
                    background: "oklch(0.245 0.016 265)",
                    border: "1px solid rgba(148,163,184,0.25)",
                    fontSize: 12,
                  }}
                  formatter={(v: number) => [`${v.toFixed(0)}%`, "fire rate"]}
                />
                <Bar dataKey="pct" radius={[3, 3, 0, 0]}>
                  {D.perClass.map((c) => (
                    <Cell
                      key={c.label}
                      fill={
                        c.label.startsWith("fall")
                          ? "oklch(0.79 0.155 78)"
                          : "oklch(0.63 0.213 26)"
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-lg border border-border bg-surface p-6">
          <h3 className="text-lg font-semibold">Impact-threshold sweep</h3>
          <p className="mt-2 text-sm text-muted-foreground">
            The operating point is picked as the highest specificity at sensitivity ≥ 0.95
            — not the best F1. A missed fall leaves the wearer with the phone they already
            had; a false alarm spends their family&apos;s trust.
          </p>
          <div className="mt-6 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={D.sweep.map((s) => ({
                  g: s.impact_g,
                  sensitivity: s.sensitivity * 100,
                  specificity: s.specificity * 100,
                }))}
                margin={{ top: 4, right: 12, bottom: 24, left: 0 }}
              >
                <CartesianGrid stroke="rgba(148,163,184,0.14)" />
                <XAxis
                  dataKey="g"
                  unit="g"
                  tick={{ fontSize: 11, fill: "rgba(148,163,184,0.9)" }}
                />
                <YAxis
                  unit="%"
                  domain={[0, 105]}
                  tick={{ fontSize: 11, fill: "rgba(148,163,184,0.9)" }}
                />
                <Tooltip
                  contentStyle={{
                    background: "oklch(0.245 0.016 265)",
                    border: "1px solid rgba(148,163,184,0.25)",
                    fontSize: 12,
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="sensitivity"
                  stroke="oklch(0.79 0.155 78)"
                  dot={false}
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="specificity"
                  stroke="oklch(0.7 0.12 220)"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-border bg-surface p-6">
        <div className="grid gap-px overflow-hidden rounded border border-border bg-border sm:grid-cols-4">
          <Stat
            label="Sensitivity"
            value={`${(D.report.sensitivity * 100).toFixed(1)}%`}
            note={`${D.report.nFalls} simulated falls`}
          />
          <Stat
            label="Specificity"
            value={`${(D.report.specificity * 100).toFixed(1)}%`}
            note={`${D.report.nAdls} confusable activities`}
          />
          <Stat
            label="False alerts / week"
            value={D.report.falsePositivesPerWeek.toFixed(1)}
            note="projected at 120 fall-like events/day"
          />
          <Stat
            label="Decision latency"
            value={`${(D.report.medianDetectLatencyMs / 1000).toFixed(1)} s`}
            note="dominated by the stillness window"
          />
        </div>
        <p className="mt-4 text-sm text-muted-foreground">
          <span className="text-foreground">Read these honestly.</span>{" "}
          {D.provenance}. The episode classes are separable by construction, so the
          perfect scores measure that the three-phase template <em>encodes the intended
          decision structure</em> — not that it will hold on worn sensors. The number that
          matters, false alerts per week, only becomes real after a field deployment; this
          harness exists so that regression is visible the moment real traces arrive.
        </p>
      </div>
    </div>
  );
}

function Stat({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="bg-surface-2 p-5">
      <p className="font-mono text-2xl text-signal">{value}</p>
      <p className="mt-2 text-sm font-medium">{label}</p>
      <p className="mt-1 text-xs text-muted-foreground">{note}</p>
    </div>
  );
}

function Cellet({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-surface-2 p-3">
      <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      <p className="mt-1 font-mono text-lg text-signal">{value}</p>
    </div>
  );
}
