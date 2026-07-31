import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  DT,
  SCENARIOS,
  type Scenario,
} from "@/lib/scenarios";
import {
  FeatureBuffer,
  HEAD_MEMBERS,
  HEAD_TEMPERATURE,
  HEAD_THRESHOLD,
  RISK,
  coneHalfWidthAt,
  decide,
  decideBaseline,
  predictRisk,
  trackFeatures,
  type AlertLevel,
  type Decision,
} from "@/lib/riskHead";

const WIDTH = 720;
const HEIGHT = 460;
const RANGE_M = 12;

const LEVEL_LABEL: Record<AlertLevel, string> = {
  none: "silent",
  info: "info",
  warn: "warn",
  urgent: "urgent",
};

function cssVar(name: string, fallback: string) {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/** Top-down ego frame: origin bottom-centre, +z up the canvas, +x right. */
function project(x: number, z: number) {
  const scale = (HEIGHT - 60) / RANGE_M;
  return {
    px: WIDTH / 2 + x * scale,
    py: HEIGHT - 40 - z * scale,
    scale,
  };
}

export function LiveDecisionDemo() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [scenarioId, setScenarioId] = useState(SCENARIOS[0]!.id);
  const [playing, setPlaying] = useState(true);
  const [frame, setFrame] = useState(0);
  const bufferRef = useRef(new FeatureBuffer());
  const [decision, setDecision] = useState<Decision | null>(null);
  const [baseline, setBaseline] = useState<AlertLevel>("none");
  const [firstAlert, setFirstAlert] = useState<{
    fused: number | null;
    baseline: number | null;
  }>({ fused: null, baseline: null });

  const scenario: Scenario = useMemo(
    () => SCENARIOS.find((s) => s.id === scenarioId) ?? SCENARIOS[0]!,
    [scenarioId],
  );

  const reset = useCallback(() => {
    bufferRef.current.reset();
    setFrame(0);
    setDecision(null);
    setBaseline("none");
    setFirstAlert({ fused: null, baseline: null });
  }, []);

  useEffect(() => {
    reset();
  }, [scenarioId, reset]);

  // 10 Hz tick, the same rate as the on-device fast path.
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setFrame((f) => (f + 1) % scenario.states.length);
    }, DT * 1000);
    return () => window.clearInterval(id);
  }, [playing, scenario]);

  useEffect(() => {
    if (frame === 0) {
      bufferRef.current.reset();
      setFirstAlert({ fused: null, baseline: null });
    }
    const state = scenario.states[frame];
    if (!state) return;

    bufferRef.current.push(trackFeatures(state));
    const prediction = predictRisk(bufferRef.current.flat());
    const next = decide(state, prediction);
    const base = decideBaseline(state);
    setDecision(next);
    setBaseline(base);
    setFirstAlert((prev) => ({
      fused: prev.fused ?? (next.level !== "none" ? frame : null),
      baseline: prev.baseline ?? (base !== "none" ? frame : null),
    }));
  }, [frame, scenario]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = WIDTH * dpr;
    canvas.height = HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const surface = cssVar("--surface-2", "#14161d");
    const border = cssVar("--border", "#2a2f3a");
    const signal = cssVar("--signal", "#f0a500");
    const urgent = cssVar("--urgent", "#e5484d");
    const safe = cssVar("--safe", "#3dd68c");
    const muted = cssVar("--muted-foreground", "#8b90a0");

    ctx.fillStyle = surface;
    ctx.fillRect(0, 0, WIDTH, HEIGHT);

    // Range rings every 2 m.
    ctx.strokeStyle = border;
    ctx.lineWidth = 1;
    ctx.font = "10px ui-monospace, monospace";
    for (let r = 2; r <= RANGE_M; r += 2) {
      const { py } = project(0, r);
      ctx.beginPath();
      ctx.moveTo(0, py);
      ctx.lineTo(WIDTH, py);
      ctx.stroke();
      ctx.fillStyle = muted;
      ctx.fillText(`${r} m`, 8, py - 4);
    }

    // Risk corridor: widens with range because heading uncertainty integrates.
    const near = project(-coneHalfWidthAt(0), 0);
    const far = project(-coneHalfWidthAt(RANGE_M), RANGE_M);
    const nearR = project(coneHalfWidthAt(0), 0);
    const farR = project(coneHalfWidthAt(RANGE_M), RANGE_M);
    ctx.beginPath();
    ctx.moveTo(near.px, near.py);
    ctx.lineTo(far.px, far.py);
    ctx.lineTo(farR.px, farR.py);
    ctx.lineTo(nearR.px, nearR.py);
    ctx.closePath();
    ctx.fillStyle = "color-mix(in oklab, " + signal + " 8%, transparent)";
    ctx.fill();
    ctx.strokeStyle = "color-mix(in oklab, " + signal + " 35%, transparent)";
    ctx.stroke();

    // Ego cylinder.
    const ego = project(0, 0);
    ctx.beginPath();
    ctx.arc(ego.px, ego.py, RISK.egoRadiusM * ego.scale, 0, Math.PI * 2);
    ctx.strokeStyle = "color-mix(in oklab, " + urgent + " 55%, transparent)";
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(ego.px, ego.py, 5, 0, Math.PI * 2);
    ctx.fillStyle = cssVar("--foreground", "#f2f3f5");
    ctx.fill();

    const state = scenario.states[frame];
    if (!state) return;

    // Constant-velocity forecast over the 3 s horizon.
    ctx.beginPath();
    for (let t = 0; t <= RISK.horizonS; t += 0.1) {
      const p = project(state.x + state.vx * t, state.z + state.vz * t);
      if (t === 0) ctx.moveTo(p.px, p.py);
      else ctx.lineTo(p.px, p.py);
    }
    ctx.strokeStyle = "color-mix(in oklab, " + muted + " 70%, transparent)";
    ctx.setLineDash([3, 5]);
    ctx.stroke();
    ctx.setLineDash([]);

    // 1-sigma growth of the forecast, drawn because a point prediction at 3 s is a lie.
    for (const t of [1, 2, 3]) {
      const p = project(state.x + state.vx * t, state.z + state.vz * t);
      ctx.beginPath();
      ctx.arc(p.px, p.py, 0.35 * t * p.scale, 0, Math.PI * 2);
      ctx.strokeStyle = "color-mix(in oklab, " + muted + " 25%, transparent)";
      ctx.stroke();
    }

    // The tracked object.
    const p = project(state.x, state.z);
    const level = decision?.level ?? "none";
    const colour = level === "urgent" ? urgent : level === "warn" ? signal : safe;
    const halfW = Math.max(state.widthM * 0.5 * p.scale, 6);
    ctx.fillStyle = "color-mix(in oklab, " + colour + " 22%, transparent)";
    ctx.strokeStyle = colour;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.rect(p.px - halfW, p.py - halfW, halfW * 2, halfW * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = colour;
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText(
      `${state.className}  ${Math.hypot(state.x, state.z).toFixed(1)} m`,
      p.px + halfW + 6,
      p.py + 4,
    );
  }, [frame, decision, scenario]);

  const ttcText =
    decision && Number.isFinite(decision.ttc) ? `${decision.ttc.toFixed(2)} s` : "—";
  const leadFused =
    firstAlert.fused !== null ? (frame - firstAlert.fused) * DT : null;
  const leadBaseline =
    firstAlert.baseline !== null ? (frame - firstAlert.baseline) * DT : null;

  return (
    <div className="rounded-lg border border-border bg-surface p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h3 className="text-lg font-semibold">Decision layer, running live</h3>
          <p className="mt-1 max-w-xl text-sm text-muted-foreground">
            The trained ensemble ({HEAD_MEMBERS} members, T={HEAD_TEMPERATURE.toFixed(2)},
            conformal threshold {HEAD_THRESHOLD.toFixed(2)}) evaluated in your browser at
            10 Hz over the same scenario physics used for training. Nothing here is
            pre-rendered.
          </p>
        </div>
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          className="rounded border border-border bg-surface-2 px-4 py-2 font-mono text-xs uppercase tracking-widest transition-colors hover:border-signal hover:text-signal"
        >
          {playing ? "pause" : "play"}
        </button>
      </div>

      <div className="mt-5 flex flex-wrap gap-2">
        {SCENARIOS.map((s) => (
          <button
            key={s.id}
            type="button"
            onClick={() => setScenarioId(s.id)}
            className={`rounded border px-3 py-1.5 text-xs transition-colors ${
              s.id === scenarioId
                ? "border-signal bg-signal/10 text-signal"
                : "border-border bg-surface-2 text-muted-foreground hover:text-foreground"
            }`}
          >
            {s.label}
          </button>
        ))}
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1fr_280px]">
        <div className="overflow-hidden rounded border border-border">
          <canvas
            ref={canvasRef}
            style={{ width: "100%", height: "auto", display: "block" }}
            aria-label={`Top-down replay of the ${scenario.label} scenario with the risk corridor and forecast`}
          />
        </div>

        <div className="space-y-3">
          <Readout label="Alert" value={LEVEL_LABEL[decision?.level ?? "none"]}
                   tone={decision?.level ?? "none"} />
          <Readout label="Time to contact" value={ttcText} />
          <Readout
            label="P(breach ≤ 2 s)"
            value={decision ? decision.probability.toFixed(3) : "—"}
          />
          <Readout
            label="Ensemble spread"
            value={decision ? `±${decision.epistemicStd.toFixed(3)}` : "—"}
          />
          <Readout label="Arbitration" value={decision?.source ?? "—"} mono />
          <div className="rounded border border-border bg-surface-2 p-3">
            <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              Utterance
            </p>
            <p className="mt-1 min-h-[1.5rem] text-sm text-signal">
              {decision?.utterance || "—"}
            </p>
          </div>
        </div>
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <div className="rounded border border-border bg-surface-2 p-4">
          <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            Constant-velocity baseline
          </p>
          <p className="mt-2 text-sm">
            <span className={baseline === "none" ? "text-muted-foreground" : "text-urgent"}>
              {LEVEL_LABEL[baseline]}
            </span>
            {leadBaseline !== null && (
              <span className="ml-2 font-mono text-xs text-muted-foreground">
                firing for {leadBaseline.toFixed(1)} s
              </span>
            )}
          </p>
        </div>
        <div className="rounded border border-border bg-surface-2 p-4">
          <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            Geometry + head (shipping)
          </p>
          <p className="mt-2 text-sm">
            <span
              className={
                (decision?.level ?? "none") === "none" ? "text-muted-foreground" : "text-signal"
              }
            >
              {LEVEL_LABEL[decision?.level ?? "none"]}
            </span>
            {leadFused !== null && (
              <span className="ml-2 font-mono text-xs text-muted-foreground">
                firing for {leadFused.toFixed(1)} s
              </span>
            )}
          </p>
        </div>
      </div>

      <p className="mt-5 border-l-2 border-signal pl-4 text-sm leading-relaxed text-muted-foreground">
        {scenario.note}
      </p>
    </div>
  );
}

function Readout({
  label,
  value,
  tone,
  mono,
}: {
  label: string;
  value: string;
  tone?: AlertLevel;
  mono?: boolean;
}) {
  const colour =
    tone === "urgent"
      ? "text-urgent"
      : tone === "warn"
        ? "text-signal"
        : tone === "info"
          ? "text-foreground"
          : "text-muted-foreground";
  return (
    <div className="rounded border border-border bg-surface-2 p-3">
      <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        {label}
      </p>
      <p className={`mt-1 text-lg ${mono ? "font-mono text-sm" : "font-mono"} ${tone ? colour : ""}`}>
        {value}
      </p>
    </div>
  );
}
