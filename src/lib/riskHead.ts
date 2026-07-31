/**
 * Browser port of the safety kernel and the learned risk head.
 *
 * Every constant, feature index, and threshold here mirrors the Python source of
 * truth (`guardian/guardian/risk/ttc.py` and `guardian/guardian/train/features.py`).
 * The weights are the real trained ensemble exported by
 * `scripts/export_web_head.py` — the demo is running the model, not a re-creation of it.
 */

import headWeights from "@/data/risk_head.json";

export const RISK = {
  coneHalfWidthM: 0.55,
  coneWidenPerM: 0.08,
  coneRangeM: 12.0,
  egoRadiusM: 1.0,
  horizonS: 3.0,
  ttcWarnS: 3.0,
  ttcUrgentS: 1.5,
} as const;

const V_SCALE = 3.0;
const W_SCALE = 2.0;
export const WINDOW = 8;
export const FEATURE_DIM = 14;

const DYNAMIC_CLASSES: Record<string, number> = {
  person: 0.7,
  bicycle: 1.0,
  motorcycle: 1.0,
  car: 1.0,
  bus: 1.0,
  truck: 1.0,
  dog: 0.8,
  scooter: 1.0,
};

export interface TrackState {
  x: number;
  z: number;
  vx: number;
  vz: number;
  widthM: number;
  className: string;
}

/* ------------------------------------------------------------------ geometry */

export function coneHalfWidthAt(z: number): number {
  return RISK.coneHalfWidthM + RISK.coneWidenPerM * Math.max(z, 0);
}

export function closingSpeed(t: TrackState): number {
  const r = Math.hypot(t.x, t.z);
  if (r < 1e-6) return 0;
  return -(t.x * t.vx + t.z * t.vz) / r;
}

/** Solves |p + v t| = R for the ego cylinder radius. Infinity when it never breaches. */
export function timeToCollision(t: TrackState): number {
  const radius = RISK.egoRadiusM + 0.5 * t.widthM;
  const a = t.vx * t.vx + t.vz * t.vz;
  const b = 2 * (t.x * t.vx + t.z * t.vz);
  const c = t.x * t.x + t.z * t.z - radius * radius;
  if (c <= 0) return 0;
  if (a < 1e-9) return Infinity;
  const disc = b * b - 4 * a * c;
  if (disc < 0) return Infinity;
  const sq = Math.sqrt(disc);
  const roots = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)].filter((v) => v >= 0);
  if (!roots.length) return Infinity;
  const ttc = Math.min(...roots);
  return ttc <= RISK.horizonS ? ttc : Infinity;
}

export function closestApproach(t: TrackState): { distance: number; time: number } {
  const v2 = t.vx * t.vx + t.vz * t.vz;
  if (v2 < 1e-9) return { distance: Math.hypot(t.x, t.z), time: 0 };
  const tc = Math.min(Math.max(-(t.x * t.vx + t.z * t.vz) / v2, 0), RISK.horizonS);
  return { distance: Math.hypot(t.x + t.vx * tc, t.z + t.vz * tc), time: tc };
}

export function severityOf(ttc: number, distance: number): number {
  const timeTerm = Number.isFinite(ttc)
    ? clamp(1 - ttc / RISK.horizonS, 0, 1)
    : 0;
  const distTerm = clamp(1 - distance / RISK.coneRangeM, 0, 1);
  return clamp(0.75 * timeTerm + 0.25 * distTerm, 0, 1);
}

function clamp(v: number, lo: number, hi: number) {
  return Math.min(Math.max(v, lo), hi);
}

/* ------------------------------------------------------------------ features */

export function trackFeatures(
  t: TrackState,
  detScore = 0.85,
  depthConf = 0.88,
): Float64Array {
  const ttc = timeToCollision(t);
  const cpa = closestApproach(t);
  const half = coneHalfWidthAt(t.z) + 0.5 * t.widthM;
  const rng = RISK.coneRangeM;

  const f = Float64Array.from([
    t.z / rng,
    t.x / rng,
    Math.hypot(t.x, t.z) / rng,
    t.vz / V_SCALE,
    t.vx / V_SCALE,
    closingSpeed(t) / V_SCALE,
    Number.isFinite(ttc) ? 1 / (ttc + 0.5) : 0,
    cpa.distance / rng,
    cpa.time / RISK.horizonS,
    (half - Math.abs(t.x)) / Math.max(half, 1e-6),
    t.widthM / W_SCALE,
    detScore,
    depthConf,
    DYNAMIC_CLASSES[t.className] ?? 0.2,
  ]);
  for (let i = 0; i < f.length; i++) {
    const v = f[i] as number;
    f[i] = Number.isFinite(v) ? clamp(v, -4, 4) : 0;
  }
  return f;
}

/** Ring buffer with first-observation padding, matching `features.FeatureBuffer`. */
export class FeatureBuffer {
  private buf = new Float64Array(WINDOW * FEATURE_DIM);
  private n = 0;

  push(f: Float64Array) {
    if (this.n === 0) {
      for (let t = 0; t < WINDOW; t++) this.buf.set(f, t * FEATURE_DIM);
    } else {
      this.buf.copyWithin(0, FEATURE_DIM);
      this.buf.set(f, (WINDOW - 1) * FEATURE_DIM);
    }
    this.n++;
  }

  get ready() {
    return this.n >= 2;
  }

  flat(): Float64Array {
    return this.buf;
  }

  reset() {
    this.n = 0;
    this.buf.fill(0);
  }
}

/* ---------------------------------------------------------------- learned head */

interface RawMember {
  W: number[][][];
  b: number[][];
}

const HEAD = headWeights as unknown as {
  temperature: number;
  threshold: number;
  window: number;
  members: RawMember[];
};

/** Row-major flat layout, prepared once at module load: 3 matmuls per member per tick. */
interface FlatLayer {
  W: Float64Array;
  b: Float64Array;
  inDim: number;
  outDim: number;
}

function flatten(W: number[][], b: number[]): FlatLayer {
  const inDim = W.length;
  const outDim = b.length;
  const flat = new Float64Array(inDim * outDim);
  for (let i = 0; i < inDim; i++) {
    const row = W[i] ?? [];
    for (let j = 0; j < outDim; j++) flat[i * outDim + j] = row[j] ?? 0;
  }
  return { W: flat, b: Float64Array.from(b), inDim, outDim };
}

const MEMBERS: FlatLayer[][] = HEAD.members.map((m) =>
  [0, 1, 2].map((k) => flatten(m.W[k] ?? [], m.b[k] ?? [])),
);

function dense(layer: FlatLayer, x: Float64Array, relu: boolean): Float64Array {
  const out = new Float64Array(layer.outDim);
  out.set(layer.b);
  for (let i = 0; i < layer.inDim; i++) {
    const xi = x[i] ?? 0;
    if (xi === 0) continue;
    const base = i * layer.outDim;
    for (let j = 0; j < layer.outDim; j++) {
      out[j] = (out[j] as number) + xi * (layer.W[base + j] as number);
    }
  }
  if (relu) {
    for (let j = 0; j < out.length; j++) {
      const v = out[j] as number;
      out[j] = v > 0 ? v : 0;
    }
  }
  return out;
}

function forward(layers: FlatLayer[], x: Float64Array): number {
  const l0 = layers[0];
  const l1 = layers[1];
  const l2 = layers[2];
  if (!l0 || !l1 || !l2) return 0;
  const h2 = dense(l1, dense(l0, x, true), true);
  return sigmoid(dense(l2, h2, false)[0] ?? 0);
}

function sigmoid(z: number) {
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

/** Temperature scaling, applied to the ensemble mean exactly as the Python runtime does. */
function applyTemperature(p: number, temperature: number) {
  const eps = 1e-6;
  const clamped = clamp(p, eps, 1 - eps);
  const logit = Math.log(clamped / (1 - clamped));
  return 1 / (1 + Math.exp(-logit / Math.max(temperature, 1e-6)));
}

export interface HeadPrediction {
  probability: number;
  epistemicStd: number;
  fired: boolean;
  trusted: boolean;
  upperBound: number;
}

export const HEAD_THRESHOLD = HEAD.threshold;
export const HEAD_TEMPERATURE = HEAD.temperature;
export const HEAD_MEMBERS = HEAD.members.length;

export function predictRisk(window: Float64Array, abstainStd = 0.18): HeadPrediction {
  const per = MEMBERS.map((layers) => forward(layers, window));
  const mean = per.reduce((a, b) => a + b, 0) / per.length;
  const variance =
    per.reduce((a, b) => a + (b - mean) * (b - mean), 0) / per.length;
  const std = Math.sqrt(variance);
  const probability = applyTemperature(mean, HEAD.temperature);
  return {
    probability,
    epistemicStd: std,
    fired: probability >= HEAD.threshold,
    trusted: std <= abstainStd,
    upperBound: clamp(probability + 2 * std, 0, 1),
  };
}

/* ----------------------------------------------------------------- arbitration */

export type AlertLevel = "none" | "info" | "warn" | "urgent";
export type Source = "geometry" | "head-escalate" | "head-suppress" | "head-abstain";

export interface Decision {
  level: AlertLevel;
  severity: number;
  source: Source;
  ttc: number;
  probability: number;
  epistemicStd: number;
  utterance: string;
}

/**
 * The shipping rule. Geometry owns imminent contact; the head may escalate freely and
 * may suppress only when it is confident *and* its ensemble agrees.
 */
export function decide(t: TrackState, prediction: HeadPrediction): Decision {
  const ttc = timeToCollision(t);
  const cpa = closestApproach(t);
  const base = severityOf(ttc, cpa.distance);
  const inCone = Math.abs(t.x) <= coneHalfWidthAt(t.z) + 0.5 * t.widthM;

  let severity = base;
  let source: Source = "geometry";

  if (ttc <= RISK.ttcUrgentS) {
    source = "geometry";
  } else if (!prediction.trusted) {
    source = "head-abstain";
  } else if (prediction.probability >= 0.75) {
    severity = clamp(Math.max(base, 0.5 * base + 0.5 * prediction.probability), 0, 1);
    source = "head-escalate";
  } else if (prediction.upperBound <= 0.15) {
    severity = base * 0.35;
    source = "head-suppress";
  }

  let level: AlertLevel = "none";
  if (ttc <= RISK.ttcUrgentS && inCone) level = "urgent";
  else if (severity >= 0.55 && Number.isFinite(ttc)) level = "warn";
  else if (severity >= 0.3 && inCone) level = "info";

  const side = t.x < -0.4 ? "left" : t.x > 0.4 ? "right" : "ahead";
  const utterance =
    level === "urgent"
      ? `Stop. ${t.className} ${side}.`
      : level === "warn"
        ? `${t.className} closing ${side}, ${t.z.toFixed(0)} metres.`
        : level === "info"
          ? `${t.className} ${side}.`
          : "";

  return {
    level,
    severity,
    source,
    ttc,
    probability: prediction.probability,
    epistemicStd: prediction.epistemicStd,
    utterance,
  };
}

/** Baseline arm for the side-by-side comparison: constant-velocity TTC only. */
export function decideBaseline(t: TrackState): AlertLevel {
  const ttc = timeToCollision(t);
  const inCone = Math.abs(t.x) <= coneHalfWidthAt(t.z) + 0.5 * t.widthM;
  if (ttc <= RISK.ttcUrgentS && inCone) return "urgent";
  if (ttc <= 2.0) return "warn";
  return "none";
}
