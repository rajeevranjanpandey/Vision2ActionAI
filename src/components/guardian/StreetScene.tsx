/**
 * The street, from the wearable's point of view.
 *
 * One perspective canvas shared by every feature: pavement, tactile paving, kerb,
 * road, the walker with a white cane in the foreground, and whatever the current
 * feature needs drawn on top of it. Everything it renders comes from live state —
 * the tracked actor, the fast-path decision, and the current street conditions —
 * so degrading the conditions visibly degrades the picture.
 */

import { useEffect, useRef } from "react";
import type { Conditions, Derived } from "@/components/guardian/SceneConditions";
import type { SceneKind } from "@/lib/environments";
import type { AlertLevel, Decision, TrackState } from "@/lib/riskHead";
import { coneHalfWidthAt } from "@/lib/riskHead";

export type SceneOverlay =
  | "predict"
  | "crossing"
  | "edge"
  | "breadcrumb"
  | "object"
  | "scene"
  | "reader"
  | "places"
  | "reroute"
  | "human"
  | "fall"
  | "translate";

export interface SceneProps {
  track: TrackState | null;
  decision: Decision | null;
  overlay: SceneOverlay;
  /** Which environment class is being rendered — street, aisle, platform… */
  scene: SceneKind;
  /** Elapsed simulated seconds; drives gait, sweep and blinking. */
  t: number;
  conditions: Conditions;
  derived: Derived;
  playing: boolean;
}


const W = 960;
const H = 640;
const HORIZON = H * 0.42;
const FOCAL = 520;
const CAM_H = 1.45;
const CX = W * 0.5;

function v(name: string, fallback: string) {
  if (typeof window === "undefined") return fallback;
  const val = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return val || fallback;
}

/** Ground point (x metres right, z metres ahead) → canvas pixel. */
function ground(x: number, z: number) {
  const zc = Math.max(z, 0.35);
  return { px: CX + (FOCAL * x) / zc, py: HORIZON + (FOCAL * CAM_H) / zc, s: FOCAL / zc };
}

const LEVEL_KEY: Record<AlertLevel, string> = {
  none: "--guide",
  info: "--sense",
  warn: "--veto",
  urgent: "--urgent",
};

export function StreetScene(props: SceneProps) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const state = useRef(props);
  state.current = props;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);

    let raf = 0;
    const loop = () => {
      draw(ctx, state.current);
      raf = window.requestAnimationFrame(loop);
    };
    raf = window.requestAnimationFrame(loop);
    return () => window.cancelAnimationFrame(raf);
  }, []);

  return (
    <canvas
      ref={ref}
      style={{ width: "100%", height: "100%", objectFit: "contain" }}
      className="block h-full w-full rounded-xl"
      role="img"
      aria-label="Simulated street view from the wearable, with the walker, hazards and the current alert state"
    />
  );
}

/* --------------------------------------------------------------------- render */

function draw(ctx: CanvasRenderingContext2D, p: SceneProps) {
  const { t, conditions: c, derived: d } = p;
  const dim = 0.35 + 0.65 * c.light;

  ctx.clearRect(0, 0, W, H);
  drawBackdrop(ctx, p, dim);
  drawCrowd(ctx, c.crowd, t, dim);

  if (p.overlay === "crossing" || p.overlay === "reroute") drawCrossing(ctx, dim);
  if (p.overlay === "breadcrumb") drawBreadcrumbs(ctx, t);
  if (p.overlay === "places" || p.overlay === "reroute") drawPins(ctx, p.overlay);
  if (p.overlay === "object") drawObjectPin(ctx, t);

  drawCone(ctx, p.decision);
  if (p.track) drawActor(ctx, p.track, p.decision);

  drawWalker(ctx, t, p.overlay, p.playing);
  drawWeather(ctx, c, t);
  drawHud(ctx, p, d);
  drawVignette(ctx);
}

/** The scene geometry itself changes with the environment class: a pharmacy
 *  aisle, a platform edge and a plaza are different problems, so they are
 *  different pictures — not one street with different sliders. */
function drawBackdrop(ctx: CanvasRenderingContext2D, p: SceneProps, dim: number) {
  const { t } = p;
  switch (p.scene) {
    case "room":
      drawInterior(ctx, dim, t, "room");
      break;
    case "retail":
      drawInterior(ctx, dim, t, "retail");
      break;
    case "transit":
      drawPlatform(ctx, dim, t);
      break;
    case "plaza":
      drawSky(ctx, dim);
      drawPlaza(ctx, dim, t);
      break;
    case "threshold":
      drawSky(ctx, dim);
      drawBuildings(ctx, dim);
      drawPavement(ctx, dim, t);
      drawShopfront(ctx, dim);
      break;
    case "worksite":
      drawSky(ctx, dim);
      drawBuildings(ctx, dim);
      drawRoad(ctx, dim, t);
      drawPavement(ctx, dim, t);
      drawWorksite(ctx, dim, t);
      break;
    case "rig":
      drawRig(ctx, t);
      break;
    default:
      drawSky(ctx, dim);
      drawBuildings(ctx, dim);
      drawRoad(ctx, dim, t);
      drawPavement(ctx, dim, t);
  }
}

/** Shared floor for the indoor classes: no sky, no kerb, and a ceiling that
 *  boxes the depth model in. */
function drawInterior(
  ctx: CanvasRenderingContext2D,
  dim: number,
  t: number,
  kind: "room" | "retail",
) {
  const warm = kind === "room" ? 70 : 250;
  // Back wall.
  const back = ctx.createLinearGradient(0, 0, 0, HORIZON + 120);
  back.addColorStop(0, `oklch(${0.5 + 0.28 * dim} 0.02 ${warm})`);
  back.addColorStop(1, `oklch(${0.62 + 0.24 * dim} 0.015 ${warm})`);
  ctx.fillStyle = back;
  ctx.fillRect(0, 0, W, H);

  // Ceiling plane, converging to the same vanishing point as the floor.
  const cw = ground(-4.4, 26);
  const ce = ground(4.4, 26);
  ctx.fillStyle = `oklch(${0.42 + 0.3 * dim} 0.012 ${warm})`;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(W, 0);
  ctx.lineTo(ce.px, HORIZON - 40);
  ctx.lineTo(cw.px, HORIZON - 40);
  ctx.closePath();
  ctx.fill();

  // Ceiling strip lights: the only light direction indoors.
  for (let z = 4; z < 26; z += 5) {
    const a = ground(-1.2, z);
    const b = ground(1.2, z);
    const y = HORIZON - 40 - (700 / z) * 0.28;
    ctx.fillStyle = `oklch(0.97 0.03 95 / ${0.28 + 0.4 * dim})`;
    ctx.fillRect(a.px, y, b.px - a.px, Math.max(2, 60 / z));
  }

  // Floor.
  const fFar = ground(-14, 30);
  const fFarR = ground(14, 30);
  const fNearR = ground(14, 0.6);
  const fNear = ground(-14, 0.6);
  ctx.fillStyle =
    kind === "room"
      ? `oklch(${0.56 + 0.16 * dim} 0.04 62)`
      : `oklch(${0.8 + 0.1 * dim} 0.006 250)`;
  ctx.beginPath();
  ctx.moveTo(fFar.px, fFar.py);
  ctx.lineTo(fFarR.px, fFarR.py);
  ctx.lineTo(fNearR.px, fNearR.py);
  ctx.lineTo(fNear.px, fNear.py);
  ctx.closePath();
  ctx.fill();

  // Floor seams / board lines, drifting with the walk.
  ctx.strokeStyle = "oklch(0.35 0 0 / 0.16)";
  ctx.lineWidth = 1;
  for (let i = 0; i < 20; i++) {
    const z = ((i * 1.6 + ((t * 1.1) % 1.6)) % 32) + 0.7;
    const a = ground(-14, z);
    const b = ground(14, z);
    ctx.beginPath();
    ctx.moveTo(a.px, a.py);
    ctx.lineTo(b.px, b.py);
    ctx.stroke();
  }

  if (kind === "retail") drawAisle(ctx, dim);
  else drawRoomFurniture(ctx, dim);

  // Glossy floor reflection — the material that breaks monocular depth.
  const gl = ctx.createLinearGradient(0, HORIZON, 0, H);
  gl.addColorStop(0, "oklch(1 0 0 / 0.14)");
  gl.addColorStop(1, "oklch(1 0 0 / 0)");
  ctx.fillStyle = gl;
  ctx.fillRect(0, HORIZON, W, H - HORIZON);
}

/** Shelving both sides: obstacle, description target and OCR surface at once. */
function drawAisle(ctx: CanvasRenderingContext2D, dim: number) {
  [-1, 1].forEach((side) => {
    for (let z = 22; z > 2; z -= 2.6) {
      const near = ground(side * 2.6, z);
      const far = ground(side * 2.6, z + 2.4);
      const h = 1900 / z;
      ctx.fillStyle = `oklch(${0.52 + 0.2 * dim} 0.02 250)`;
      ctx.beginPath();
      ctx.moveTo(near.px, near.py);
      ctx.lineTo(far.px, far.py);
      ctx.lineTo(far.px, far.py - h * 0.62);
      ctx.lineTo(near.px, near.py - h);
      ctx.closePath();
      ctx.fill();
      // Product bands + price labels.
      for (let s = 1; s <= 3; s++) {
        ctx.fillStyle = `oklch(${0.72 + 0.14 * dim} 0.11 ${(z * 37 + s * 90) % 360})`;
        ctx.fillRect(
          Math.min(near.px, far.px),
          near.py - (h * s) / 3.4,
          Math.abs(far.px - near.px),
          Math.max(2, h * 0.06),
        );
      }
    }
  });
}

/** A lived-in room: the clutter Feature 04 has to remember something inside. */
function drawRoomFurniture(ctx: CanvasRenderingContext2D, dim: number) {
  const box = (x: number, z: number, wM: number, hM: number, hue: number) => {
    const g = ground(x, z);
    const s = g.s * 0.0032 * 170;
    ctx.fillStyle = `oklch(${0.46 + 0.2 * dim} 0.06 ${hue})`;
    ctx.fillRect(g.px - (s * wM) / 2, g.py - s * hM, s * wM, s * hM);
    ctx.fillStyle = "oklch(0.2 0 0 / 0.12)";
    ctx.fillRect(g.px - (s * wM) / 2, g.py - 2, s * wM, 3);
  };
  box(-2.6, 7.5, 1.6, 0.9, 62); // counter
  box(2.7, 9.5, 1.4, 1.7, 40); // cabinet
  box(-1.4, 5.2, 0.7, 0.5, 150); // side table
  box(2.0, 5.6, 0.8, 0.45, 22); // chair
  // Doorway on the back wall.
  const d0 = ground(-0.4, 24);
  ctx.fillStyle = `oklch(${0.3 + 0.16 * dim} 0.02 60)`;
  ctx.fillRect(d0.px - 22, d0.py - 96, 44, 96);
}

/** Platform edge: hazard onset is distance to the drop, not TTC. */
function drawPlatform(ctx: CanvasRenderingContext2D, dim: number, t: number) {
  ctx.fillStyle = `oklch(${0.28 + 0.16 * dim} 0.02 260)`;
  ctx.fillRect(0, 0, W, H);

  // Vaulted ceiling arcs.
  ctx.strokeStyle = `oklch(${0.42 + 0.2 * dim} 0.02 260)`;
  ctx.lineWidth = 2;
  for (let i = 1; i <= 6; i++) {
    ctx.beginPath();
    ctx.ellipse(CX, HORIZON - 30, W * 0.5 - i * 22, 210 - i * 26, 0, Math.PI, 0);
    ctx.stroke();
  }

  // Platform slab (walkable side) and the track void beyond the edge.
  const eFar = ground(1.6, 40);
  const eNear = ground(1.6, 0.6);
  const lFar = ground(-16, 40);
  const lNear = ground(-16, 0.6);
  ctx.fillStyle = `oklch(${0.72 + 0.1 * dim} 0.006 250)`;
  ctx.beginPath();
  ctx.moveTo(lFar.px, lFar.py);
  ctx.lineTo(eFar.px, eFar.py);
  ctx.lineTo(eNear.px, eNear.py);
  ctx.lineTo(lNear.px, lNear.py);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = "oklch(0.18 0.01 260)";
  ctx.beginPath();
  ctx.moveTo(eFar.px, eFar.py);
  ctx.lineTo(W, eFar.py);
  ctx.lineTo(W, H);
  ctx.lineTo(eNear.px, eNear.py);
  ctx.closePath();
  ctx.fill();

  // Rails in the void.
  [4.2, 5.6].forEach((x) => {
    const a = ground(x, 40);
    const b = ground(x, 0.6);
    ctx.strokeStyle = "oklch(0.72 0.02 250 / 0.8)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(a.px, a.py);
    ctx.lineTo(b.px, b.py);
    ctx.stroke();
  });

  // Tactile warning strip along the edge — the cue the cane finds.
  ctx.fillStyle = v("--veto", "#d8a63c");
  for (let z = 1; z < 30; z += 0.5) {
    for (let x = 0.7; x < 1.5; x += 0.25) {
      const g = ground(x, z);
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      ctx.arc(g.px, g.py, Math.max(0.8, g.s * 0.006), 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;

  // Yellow edge line + an approaching train's headlights.
  const gl = ctx.createLinearGradient(0, HORIZON - 20, 0, HORIZON + 40);
  const arrive = (t % 16) / 16;
  gl.addColorStop(0, `oklch(0.95 0.16 95 / ${arrive > 0.5 ? (arrive - 0.5) * 1.6 : 0})`);
  gl.addColorStop(1, "oklch(0.95 0.16 95 / 0)");
  ctx.fillStyle = gl;
  ctx.fillRect(CX, HORIZON - 30, W - CX, 90);
}

/** Open plaza: nothing for the risk cone to key on. */
function drawPlaza(ctx: CanvasRenderingContext2D, dim: number, t: number) {
  const far = ground(-40, 60);
  const farR = ground(40, 60);
  const nearR = ground(40, 0.6);
  const near = ground(-40, 0.6);
  ctx.fillStyle = `oklch(${0.74 + 0.1 * dim} 0.012 95)`;
  ctx.beginPath();
  ctx.moveTo(far.px, far.py);
  ctx.lineTo(farR.px, farR.py);
  ctx.lineTo(nearR.px, nearR.py);
  ctx.lineTo(near.px, near.py);
  ctx.closePath();
  ctx.fill();

  // Grass band across the middle distance — a surface transition, not an edge.
  const g1 = ground(-40, 21);
  const g2 = ground(40, 21);
  const g3 = ground(40, 15);
  const g4 = ground(-40, 15);
  ctx.fillStyle = `oklch(${0.6 + 0.12 * dim} 0.12 148)`;
  ctx.beginPath();
  ctx.moveTo(g1.px, g1.py);
  ctx.lineTo(g2.px, g2.py);
  ctx.lineTo(g3.px, g3.py);
  ctx.lineTo(g4.px, g4.py);
  ctx.closePath();
  ctx.fill();

  // Radial paving joints from the vanishing point: open geometry, no corridor.
  ctx.strokeStyle = "oklch(0.45 0 0 / 0.14)";
  for (let x = -18; x <= 18; x += 3) {
    const a = ground(x, 60);
    const b = ground(x, 0.6);
    ctx.beginPath();
    ctx.moveTo(a.px, a.py);
    ctx.lineTo(b.px, b.py);
    ctx.stroke();
  }

  // Trees and benches scattered, plus a loose dog trotting across.
  [[-7, 24], [6.5, 30], [-12, 34]].forEach(([x, z]) => {
    const g = ground(x!, z!);
    const h = g.s * 0.0085;
    ctx.strokeStyle = "oklch(0.38 0.05 60)";
    ctx.lineWidth = Math.max(2, h * 0.1);
    ctx.beginPath();
    ctx.moveTo(g.px, g.py);
    ctx.lineTo(g.px, g.py - h * 1.4);
    ctx.stroke();
    ctx.fillStyle = "oklch(0.55 0.13 150)";
    ctx.beginPath();
    ctx.arc(g.px, g.py - h * 1.8, h * 0.85, 0, Math.PI * 2);
    ctx.fill();
  });

  [[-4.2, 12], [4.6, 16]].forEach(([x, z]) => {
    const g = ground(x!, z!);
    const s = g.s * 0.0032 * 170;
    ctx.fillStyle = `oklch(${0.5 + 0.16 * dim} 0.07 62)`;
    ctx.fillRect(g.px - s * 0.9, g.py - s * 0.45, s * 1.8, s * 0.16);
  });

  const dogZ = 9 + ((t * 1.6) % 12);
  const dog = ground(-6 + ((t * 1.1) % 11), dogZ);
  const ds = dog.s * 0.0032 * 90;
  ctx.fillStyle = "oklch(0.4 0.06 55)";
  ctx.beginPath();
  ctx.roundRect(dog.px - ds * 0.5, dog.py - ds * 0.5, ds, ds * 0.4, ds * 0.2);
  ctx.fill();
}

/** Bus stop → doorway: where positioning quality changes mid-stride. */
function drawShopfront(ctx: CanvasRenderingContext2D, dim: number) {
  const base = ground(-4.6, 11);
  const s = base.s * 0.0032 * 170;
  // Glass frontage — false depth returns, phantom detections.
  ctx.fillStyle = `oklch(${0.66 + 0.16 * dim} 0.05 220 / 0.75)`;
  ctx.fillRect(base.px - s * 2.4, base.py - s * 2.5, s * 3.4, s * 2.5);
  ctx.strokeStyle = "oklch(0.35 0.02 250)";
  ctx.lineWidth = 3;
  ctx.strokeRect(base.px - s * 2.4, base.py - s * 2.5, s * 3.4, s * 2.5);
  // The doorway itself, plus the threshold line the sim has to detect.
  ctx.fillStyle = "oklch(0.28 0.02 250)";
  ctx.fillRect(base.px - s * 0.4, base.py - s * 1.9, s * 1.0, s * 1.9);
  ctx.strokeStyle = v("--guide", "#3f8f63");
  ctx.setLineDash([7, 6]);
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(base.px - s * 0.4, base.py);
  ctx.lineTo(base.px + s * 0.6, base.py);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.font = "700 13px ui-monospace, monospace";
  ctx.fillStyle = v("--guide", "#3f8f63");
  ctx.fillText("threshold · GPS drops here", base.px - s * 2.3, base.py - s * 2.7);
}

/** Sidewalk closure: fencing, cones, a signed detour. */
function drawWorksite(ctx: CanvasRenderingContext2D, dim: number, t: number) {
  for (let z = 7; z < 17; z += 1.4) {
    const g = ground(-0.4, z);
    const h = g.s * 0.0032 * 110;
    ctx.strokeStyle = v("--veto", "#d8a63c");
    ctx.lineWidth = Math.max(1.5, h * 0.06);
    ctx.beginPath();
    ctx.moveTo(g.px, g.py);
    ctx.lineTo(g.px, g.py - h);
    ctx.stroke();
    ctx.globalAlpha = 0.6;
    ctx.beginPath();
    ctx.moveTo(g.px, g.py - h * 0.75);
    ctx.lineTo(g.px + 40, g.py - h * 0.7);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  [[-2.4, 6.4], [-1.6, 9.2], [-2.8, 12.4]].forEach(([x, z]) => {
    const g = ground(x!, z!);
    const h = g.s * 0.0032 * 70;
    ctx.fillStyle = "oklch(0.62 0.2 42)";
    ctx.beginPath();
    ctx.moveTo(g.px - h * 0.35, g.py);
    ctx.lineTo(g.px + h * 0.35, g.py);
    ctx.lineTo(g.px, g.py - h);
    ctx.closePath();
    ctx.fill();
  });
  // Dust plume from the works, pulsing with the jackhammer.
  const plume = ground(0.6, 13);
  const r = 60 + 12 * Math.sin(t * 5);
  const dg = ctx.createRadialGradient(plume.px, plume.py - 40, 4, plume.px, plume.py - 40, r);
  dg.addColorStop(0, `oklch(0.85 0.02 80 / ${0.35 * dim})`);
  dg.addColorStop(1, "oklch(0.85 0.02 80 / 0)");
  ctx.fillStyle = dg;
  ctx.fillRect(plume.px - r, plume.py - 40 - r, r * 2, r * 2);
}

/** Not a camera scene at all: the IMU physics rig, drawn as a lab volume. */
function drawRig(ctx: CanvasRenderingContext2D, t: number) {
  ctx.fillStyle = "oklch(0.18 0.02 260)";
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "oklch(0.42 0.05 250 / 0.55)";
  ctx.lineWidth = 1;
  for (let x = -14; x <= 14; x += 1) {
    const a = ground(x, 34);
    const b = ground(x, 0.6);
    ctx.beginPath();
    ctx.moveTo(a.px, a.py);
    ctx.lineTo(b.px, b.py);
    ctx.stroke();
  }
  for (let z = 1; z < 34; z += 1.2) {
    const a = ground(-14, z);
    const b = ground(14, z);
    ctx.beginPath();
    ctx.moveTo(a.px, a.py);
    ctx.lineTo(b.px, b.py);
    ctx.stroke();
  }
  // Accelerometer trace running across the volume.
  ctx.strokeStyle = v("--urgent", "#c0392b");
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let x = 0; x < W; x++) {
    const u = x / W;
    const spike = Math.exp(-Math.pow((u - ((t * 0.08) % 1)) * 22, 2));
    const y = 150 + Math.sin(u * 40 + t * 4) * 8 - spike * 90;
    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.font = "600 12px ui-monospace, monospace";
  ctx.fillStyle = "oklch(0.85 0 0 / 0.7)";
  ctx.fillText("physics rig · ragdoll IMU trace, no camera", 24, 40);
}


function drawSky(ctx: CanvasRenderingContext2D, dim: number) {
  const g = ctx.createLinearGradient(0, 0, 0, HORIZON + 60);
  g.addColorStop(0, `oklch(${0.44 + 0.3 * dim} 0.14 258)`);
  g.addColorStop(0.55, `oklch(${0.72 + 0.18 * dim} 0.09 240)`);
  g.addColorStop(1, `oklch(${0.9 + 0.06 * dim} 0.05 70)`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, HORIZON + 60);

  // Low sun behind the street: gives the scene a direction of light.
  const sun = ctx.createRadialGradient(W * 0.68, HORIZON - 40, 6, W * 0.68, HORIZON - 40, 240);
  sun.addColorStop(0, `oklch(0.98 0.12 85 / ${0.55 * dim + 0.15})`);
  sun.addColorStop(1, "oklch(0.98 0.12 85 / 0)");
  ctx.fillStyle = sun;
  ctx.fillRect(0, 0, W, HORIZON + 60);

  // Cloud bands.
  ctx.fillStyle = `oklch(1 0 0 / ${0.1 + 0.12 * dim})`;
  [[120, 60, 190, 16], [520, 40, 260, 12], [760, 96, 210, 14]].forEach(([x, y, w, h]) => {
    ctx.beginPath();
    ctx.ellipse(x!, y!, w! / 2, h!, 0, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawBuildings(ctx: CanvasRenderingContext2D, dim: number) {
  // Two silhouette layers so the skyline has depth rather than flat blocks.
  const far: [number, number, number][] = [
    [-40, 120, 150],
    [130, 170, 120],
    [270, 96, 180],
    [470, 140, 150],
    [640, 110, 130],
    [790, 165, 190],
  ];
  far.forEach(([x, h, w], i) => {
    const lg = ctx.createLinearGradient(0, HORIZON - h, 0, HORIZON);
    lg.addColorStop(0, `oklch(${0.5 + 0.16 * dim} 0.05 ${240 + i * 12})`);
    lg.addColorStop(1, `oklch(${0.66 + 0.16 * dim} 0.03 ${240 + i * 12})`);
    ctx.fillStyle = lg;
    ctx.fillRect(x, HORIZON - h, w, h);
    // Window grid — the detail that makes it read as a city, not a bar chart.
    ctx.fillStyle = `oklch(0.98 0.07 85 / ${0.5 - 0.35 * dim})`;
    for (let wy = HORIZON - h + 14; wy < HORIZON - 12; wy += 18) {
      for (let wx = x + 10; wx < x + w - 10; wx += 16) {
        if ((wx * 7 + wy * 13) % 5 < 2) ctx.fillRect(wx, wy, 6, 9);
      }
    }
  });

  // Kerbside furniture: trees and a lamp post, placed in world space.
  [[-8.5, 26], [-6.2, 17], [8.5, 24]].forEach(([x, z]) => {
    const g = ground(x!, z!);
    const h = g.s * 0.0075;
    ctx.strokeStyle = "oklch(0.4 0.05 60)";
    ctx.lineWidth = Math.max(2, h * 0.09);
    ctx.beginPath();
    ctx.moveTo(g.px, g.py);
    ctx.lineTo(g.px, g.py - h * 1.5);
    ctx.stroke();
    const foliage = ctx.createRadialGradient(g.px, g.py - h * 1.9, 2, g.px, g.py - h * 1.9, h * 0.9);
    foliage.addColorStop(0, "oklch(0.68 0.14 148)");
    foliage.addColorStop(1, "oklch(0.44 0.11 152)");
    ctx.fillStyle = foliage;
    ctx.beginPath();
    ctx.arc(g.px, g.py - h * 1.9, h * 0.85, 0, Math.PI * 2);
    ctx.fill();
  });

  const lamp = ground(1.7, 13);
  ctx.strokeStyle = "oklch(0.42 0.02 250)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(lamp.px, lamp.py);
  ctx.lineTo(lamp.px, lamp.py - 130);
  ctx.quadraticCurveTo(lamp.px, lamp.py - 150, lamp.px + 28, lamp.py - 150);
  ctx.stroke();
  ctx.fillStyle = `oklch(0.95 0.13 88 / ${1 - dim * 0.6})`;
  ctx.beginPath();
  ctx.ellipse(lamp.px + 30, lamp.py - 148, 9, 5, 0, 0, Math.PI * 2);
  ctx.fill();
}

function drawRoad(ctx: CanvasRenderingContext2D, dim: number, t: number) {
  // Road occupies x > 2.2 m; pavement is left of the kerb.
  const far = ground(2.2, 60);
  const near = ground(2.2, 0.6);
  const farEdge = ground(30, 60);
  const nearEdge = ground(30, 0.6);
  ctx.fillStyle = `oklch(${0.34 + 0.14 * dim} 0.012 250)`;
  ctx.beginPath();
  ctx.moveTo(far.px, far.py);
  ctx.lineTo(farEdge.px, farEdge.py);
  ctx.lineTo(nearEdge.px, nearEdge.py);
  ctx.lineTo(near.px, near.py);
  ctx.closePath();
  ctx.fill();

  // Lane dashes running toward the walker.
  ctx.fillStyle = "oklch(0.95 0 0 / 0.5)";
  for (let i = 0; i < 14; i++) {
    const z = ((i * 4 + ((t * 2.2) % 4)) % 56) + 1.2;
    const a = ground(5.2, z);
    const b = ground(5.2, z + 1.6);
    ctx.beginPath();
    ctx.moveTo(a.px - 3 * (a.s / FOCAL) * 40, a.py);
    ctx.lineTo(a.px + 3 * (a.s / FOCAL) * 40, a.py);
    ctx.lineTo(b.px + 3 * (b.s / FOCAL) * 40, b.py);
    ctx.lineTo(b.px - 3 * (b.s / FOCAL) * 40, b.py);
    ctx.closePath();
    ctx.fill();
  }
}

function drawPavement(ctx: CanvasRenderingContext2D, dim: number, t: number) {
  const far = ground(-30, 60);
  const farK = ground(2.2, 60);
  const nearK = ground(2.2, 0.6);
  const near = ground(-30, 0.6);
  ctx.fillStyle = `oklch(${0.78 + 0.1 * dim} 0.006 90)`;
  ctx.beginPath();
  ctx.moveTo(far.px, far.py);
  ctx.lineTo(farK.px, farK.py);
  ctx.lineTo(nearK.px, nearK.py);
  ctx.lineTo(near.px, near.py);
  ctx.closePath();
  ctx.fill();

  // Kerb line.
  ctx.strokeStyle = "oklch(0.55 0 0 / 0.5)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(farK.px, farK.py);
  ctx.lineTo(nearK.px, nearK.py);
  ctx.stroke();

  // Tactile paving strip beside the kerb — the surface cue the cane finds.
  ctx.fillStyle = v("--veto", "#d8a63c");
  for (let z = 1; z < 26; z += 0.55) {
    for (let x = 1.1; x < 2.1; x += 0.3) {
      const g = ground(x, z);
      const r = Math.max(0.7, g.s * 0.006);
      ctx.globalAlpha = 0.55;
      ctx.beginPath();
      ctx.arc(g.px, g.py, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;

  // Slab seams drifting toward the viewer: this is what makes the walk read as motion.
  ctx.strokeStyle = "oklch(0.5 0 0 / 0.16)";
  ctx.lineWidth = 1;
  for (let i = 0; i < 22; i++) {
    const z = ((i * 2 + ((t * 1.3) % 2)) % 44) + 0.8;
    const a = ground(-30, z);
    const b = ground(2.2, z);
    ctx.beginPath();
    ctx.moveTo(a.px, a.py);
    ctx.lineTo(b.px, b.py);
    ctx.stroke();
  }
}

function drawCrowd(ctx: CanvasRenderingContext2D, crowd: number, t: number, dim: number) {
  const n = Math.round(crowd * 9);
  for (let i = 0; i < n; i++) {
    const z = 6 + ((i * 3.7 + ((t * 0.6 + i) % 12)) % 22);
    const x = -3.4 + ((i * 1.31) % 4.6);
    const g = ground(x, z);
    const h = g.s * 0.0034 * 170;
    ctx.globalAlpha = 0.28 + 0.25 * dim;
    ctx.fillStyle = i % 2 ? v("--assist", "#7a4fd0") : v("--memory", "#3f7fa0");
    ctx.fillRect(g.px - h * 0.13, g.py - h, h * 0.26, h);
    ctx.beginPath();
    ctx.arc(g.px, g.py - h - h * 0.1, h * 0.11, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
  }
}

function drawCrossing(ctx: CanvasRenderingContext2D, dim: number) {
  ctx.fillStyle = `oklch(0.97 0 0 / ${0.5 + 0.3 * dim})`;
  for (let i = 0; i < 6; i++) {
    const x0 = 2.6 + i * 1.15;
    const a = ground(x0, 9.4);
    const b = ground(x0 + 0.6, 9.4);
    const cc = ground(x0 + 0.6, 12.6);
    const dd = ground(x0, 12.6);
    ctx.beginPath();
    ctx.moveTo(a.px, a.py);
    ctx.lineTo(b.px, b.py);
    ctx.lineTo(cc.px, cc.py);
    ctx.lineTo(dd.px, dd.py);
    ctx.closePath();
    ctx.fill();
  }
}

function drawBreadcrumbs(ctx: CanvasRenderingContext2D, t: number) {
  for (let i = 0; i < 6; i++) {
    const z = 3 + i * 3.2;
    const g = ground(-0.6 + Math.sin(i) * 0.5, z);
    const pulse = (t * 1.4 - i * 0.35) % 6 < 0.5 ? 1 : 0.45;
    ctx.globalAlpha = pulse;
    ctx.fillStyle = v("--guide", "#3f8f63");
    ctx.beginPath();
    ctx.arc(g.px, g.py, Math.max(3, g.s * 0.012), 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
  }
}

function drawPins(ctx: CanvasRenderingContext2D, overlay: SceneOverlay) {
  const pins: [number, number, string][] =
    overlay === "places"
      ? [
          [-4.2, 14, "pharmacy"],
          [3.6, 20, "ATM"],
        ]
      : [
          [1.4, 11, "roadworks"],
          [-3.2, 15, "detour"],
        ];
  pins.forEach(([x, z, label]) => {
    const g = ground(x, z);
    ctx.fillStyle = overlay === "places" ? v("--memory", "#3f7fa0") : v("--veto", "#d8a63c");
    ctx.beginPath();
    ctx.arc(g.px, g.py - 26, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(g.px - 6, g.py - 20);
    ctx.lineTo(g.px + 6, g.py - 20);
    ctx.lineTo(g.px, g.py - 6);
    ctx.closePath();
    ctx.fill();
    ctx.font = "600 12px ui-monospace, monospace";
    ctx.fillStyle = "oklch(0.2 0 0)";
    ctx.fillText(label, g.px + 14, g.py - 22);
  });
}

function drawObjectPin(ctx: CanvasRenderingContext2D, t: number) {
  const g = ground(-1.9, 5.4);
  const pulse = 0.6 + 0.4 * Math.sin(t * 3);
  ctx.strokeStyle = v("--assist", "#7a4fd0");
  ctx.lineWidth = 2.5;
  ctx.globalAlpha = pulse;
  ctx.strokeRect(g.px - 26, g.py - 44, 52, 44);
  ctx.globalAlpha = 1;
  ctx.font = "600 12px ui-monospace, monospace";
  ctx.fillStyle = v("--assist", "#7a4fd0");
  ctx.fillText("keys · on the shelf", g.px - 26, g.py - 52);
}

function drawCone(ctx: CanvasRenderingContext2D, decision: Decision | null) {
  const level = decision?.level ?? "none";
  const key = LEVEL_KEY[level];
  ctx.fillStyle = v(key, "#3b6fd0");
  ctx.globalAlpha = level === "none" ? 0.1 : 0.18;
  ctx.beginPath();
  const near = ground(-coneHalfWidthAt(0.8), 0.8);
  ctx.moveTo(near.px, near.py);
  for (let z = 0.8; z <= 12; z += 0.4) {
    const g = ground(coneHalfWidthAt(z) * -1, z);
    ctx.lineTo(g.px, g.py);
  }
  for (let z = 12; z >= 0.8; z -= 0.4) {
    const g = ground(coneHalfWidthAt(z), z);
    ctx.lineTo(g.px, g.py);
  }
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 1;
}

function drawActor(
  ctx: CanvasRenderingContext2D,
  tr: TrackState,
  decision: Decision | null,
) {
  if (tr.z > 40 || tr.z < 0.2) return;
  const g = ground(tr.x, tr.z);
  const px = g.s * 0.0032;
  const h = px * (tr.className === "car" ? 150 : tr.className === "bollard" ? 110 : 175);
  const w = px * 170 * tr.widthM;
  const level = decision?.level ?? "none";
  const col = v(LEVEL_KEY[level], "#3b6fd0");

  ctx.fillStyle = col;
  ctx.globalAlpha = 0.9;
  if (tr.className === "car") {
    ctx.beginPath();
    ctx.roundRect(g.px - w / 2, g.py - h, w, h, 6);
    ctx.fill();
    ctx.fillStyle = "oklch(0.98 0 0 / 0.7)";
    ctx.fillRect(g.px - w * 0.3, g.py - h * 0.85, w * 0.6, h * 0.3);
  } else if (tr.className === "person") {
    ctx.beginPath();
    ctx.roundRect(g.px - w * 0.28, g.py - h * 0.8, w * 0.56, h * 0.8, 4);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(g.px, g.py - h * 0.92, h * 0.11, 0, Math.PI * 2);
    ctx.fill();
  } else {
    ctx.beginPath();
    ctx.roundRect(g.px - w / 2, g.py - h, w, h, 5);
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  // Tracking box + label: what the fast path is actually holding on to.
  ctx.strokeStyle = col;
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 4]);
  ctx.strokeRect(g.px - w / 2 - 5, g.py - h - 8, w + 10, h + 12);
  ctx.setLineDash([]);
  ctx.font = "600 12px ui-monospace, monospace";
  ctx.fillStyle = col;
  ctx.fillText(
    `${tr.className} · ${tr.z.toFixed(1)} m`,
    g.px - w / 2 - 5,
    g.py - h - 14,
  );

  // Forecast tail: where constant velocity puts it over the next 2 s.
  ctx.strokeStyle = col;
  ctx.globalAlpha = 0.6;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let k = 0; k <= 20; k++) {
    const dt = k * 0.1;
    const q = ground(tr.x + tr.vx * dt, Math.max(tr.z + tr.vz * dt, 0.4));
    if (k === 0) ctx.moveTo(q.px, q.py);
    else ctx.lineTo(q.px, q.py);
  }
  ctx.stroke();
  ctx.globalAlpha = 1;
}

/**
 * The walker, drawn in a bold inked-vector style: near-black ink masses, a white
 * collar and shirt for contrast, hatching instead of soft shading. Same skeleton
 * and gait as before, but it reads as an illustration rather than a mannequin.
 */
function drawWalker(
  ctx: CanvasRenderingContext2D,
  t: number,
  overlay: SceneOverlay,
  playing: boolean,
) {
  const fallen = overlay === "fall" && (t % 12) > 7;
  const gait = playing && !fallen ? Math.sin(t * 4.6) : 0;
  const bob = playing && !fallen ? Math.abs(Math.cos(t * 4.6)) * 4 : 0;
  const baseX = W * 0.24;
  const baseY = H - 34 - bob;

  const ink = "oklch(0.16 0.02 260)";
  const inkSoft = "oklch(0.30 0.03 260)";
  const paper = "oklch(0.97 0.005 260)";
  const skin = "oklch(0.88 0.03 62)";

  ctx.save();
  ctx.fillStyle = "oklch(0.25 0.02 250 / 0.28)";
  ctx.beginPath();
  ctx.ellipse(baseX, H - 30, 40, 9, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.translate(baseX, baseY);
  if (fallen) {
    ctx.translate(26, 46);
    ctx.rotate(-Math.PI / 2.15);
  }
  ctx.scale(1.22, 1.22);

  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  /** Short ink hatch marks — the shading language of the reference. */
  const hatch = (x: number, y: number, n: number, len: number, gap: number, ang: number) => {
    ctx.strokeStyle = ink;
    ctx.lineWidth = 1.1;
    for (let i = 0; i < n; i++) {
      const oy = y + i * gap;
      ctx.beginPath();
      ctx.moveTo(x, oy);
      ctx.lineTo(x + Math.cos(ang) * len, oy + Math.sin(ang) * len);
      ctx.stroke();
    }
  };

  /** One leg: hip → knee → ankle, inked with a heavier outline. */
  const leg = (phase: number, colour: string) => {
    const kneeX = phase * 13;
    const ankleX = phase * 20;
    const lift = Math.max(0, phase) * 5;
    ctx.strokeStyle = colour;
    ctx.lineWidth = 16;
    ctx.beginPath();
    ctx.moveTo(0, -78);
    ctx.lineTo(kneeX, -42);
    ctx.stroke();
    ctx.lineWidth = 12;
    ctx.beginPath();
    ctx.moveTo(kneeX, -42);
    ctx.lineTo(ankleX, -8 - lift);
    ctx.stroke();
    ctx.fillStyle = ink;
    ctx.beginPath();
    ctx.roundRect(ankleX - 7, -10 - lift, 22, 10, [5, 6, 3, 3]);
    ctx.fill();
  };

  // Far leg, lighter ink so the near leg sits in front.
  leg(-gait, inkSoft);

  // Coat skirt over the hips.
  ctx.fillStyle = ink;
  ctx.beginPath();
  ctx.moveTo(-19, -98);
  ctx.lineTo(19, -98);
  ctx.quadraticCurveTo(22, -80, 17, -68);
  ctx.lineTo(-17, -68);
  ctx.quadraticCurveTo(-22, -80, -19, -98);
  ctx.closePath();
  ctx.fill();

  // Far arm swinging behind.
  ctx.strokeStyle = inkSoft;
  ctx.lineWidth = 11;
  ctx.beginPath();
  ctx.moveTo(-9, -138);
  ctx.quadraticCurveTo(-19 - gait * 8, -120, -16 - gait * 14, -100);
  ctx.stroke();

  // Satchel strap bag behind the shoulder.
  ctx.fillStyle = inkSoft;
  ctx.beginPath();
  ctx.roundRect(-34, -140, 18, 48, 7);
  ctx.fill();
  ctx.strokeStyle = ink;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Coat body: broad shoulders, cinched waist, flared hem.
  ctx.fillStyle = ink;
  ctx.beginPath();
  ctx.moveTo(-16, -150);
  ctx.quadraticCurveTo(0, -159, 16, -150);
  ctx.quadraticCurveTo(23, -140, 20, -122);
  ctx.quadraticCurveTo(18, -110, 20, -94);
  ctx.lineTo(-20, -94);
  ctx.quadraticCurveTo(-18, -110, -20, -122);
  ctx.quadraticCurveTo(-23, -140, -16, -150);
  ctx.closePath();
  ctx.fill();

  // White shirt wedge and popped collar — the reference's signature contrast.
  ctx.fillStyle = paper;
  ctx.beginPath();
  ctx.moveTo(1, -152);
  ctx.lineTo(11, -146);
  ctx.lineTo(7, -112);
  ctx.lineTo(-1, -114);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = ink;
  ctx.lineWidth = 1.6;
  ctx.stroke();

  ctx.fillStyle = paper;
  ctx.beginPath();
  ctx.moveTo(-3, -154);
  ctx.lineTo(6, -150);
  ctx.lineTo(-2, -136);
  ctx.lineTo(-11, -144);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Hatching down the coat's shadow side.
  hatch(-17, -142, 6, 7, 6, 0.5);

  // Neck.
  ctx.fillStyle = "oklch(0.74 0.04 58)";
  ctx.beginPath();
  ctx.roundRect(-2, -160, 11, 14, 4);
  ctx.fill();

  // Head: inked oval with a defined jaw.
  ctx.fillStyle = skin;
  ctx.strokeStyle = ink;
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.moveTo(-11, -173);
  ctx.quadraticCurveTo(-11, -190, 4, -190);
  ctx.quadraticCurveTo(17, -190, 17, -173);
  ctx.quadraticCurveTo(17, -161, 8, -156);
  ctx.quadraticCurveTo(-2, -152, -8, -161);
  ctx.quadraticCurveTo(-11, -165, -11, -173);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Cheek hatching.
  hatch(2, -166, 3, 5, 3.2, 0.9);

  // Swept ink hair with a lifted quiff.
  ctx.fillStyle = ink;
  ctx.beginPath();
  ctx.moveTo(-12, -175);
  ctx.quadraticCurveTo(-15, -196, 2, -197);
  ctx.quadraticCurveTo(14, -198, 19, -186);
  ctx.quadraticCurveTo(21, -178, 17, -176);
  ctx.quadraticCurveTo(15, -186, 4, -185);
  ctx.quadraticCurveTo(-6, -184, -12, -175);
  ctx.closePath();
  ctx.fill();

  // Ear + bone-conduction earpiece.
  ctx.fillStyle = skin;
  ctx.strokeStyle = ink;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.ellipse(-8, -170, 3.2, 4.6, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "oklch(0.66 0.17 152)";
  ctx.beginPath();
  ctx.arc(-10, -170, 3, 0, Math.PI * 2);
  ctx.fill();

  // Round inked spectacles, as in the reference.
  ctx.strokeStyle = ink;
  ctx.lineWidth = 2;
  ctx.fillStyle = "oklch(0.20 0.02 260 / 0.85)";
  ctx.beginPath();
  ctx.arc(4, -175, 5.4, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(15, -176, 4.6, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(9.4, -175.5);
  ctx.lineTo(10.4, -176);
  ctx.moveTo(-1.4, -175);
  ctx.lineTo(-8, -172);
  ctx.stroke();

  // Mouth line.
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(6, -161);
  ctx.quadraticCurveTo(10, -160, 13, -162);
  ctx.stroke();

  // Near leg, in front.
  leg(gait, ink);

  // The wearable on the chest strap, with a live pulse.
  ctx.fillStyle = ink;
  ctx.strokeStyle = paper;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.roundRect(-14, -134, 15, 11, 3);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = `oklch(0.72 0.2 258 / ${0.5 + 0.5 * Math.abs(Math.sin(t * 3))})`;
  ctx.beginPath();
  ctx.arc(-6.5, -128.5, 2.6, 0, Math.PI * 2);
  ctx.fill();

  // Near arm reaching down to the cane grip.
  const sweep = Math.sin(t * 2.3) * 0.42;
  ctx.strokeStyle = ink;
  ctx.lineWidth = 12;
  ctx.beginPath();
  ctx.moveTo(13, -142);
  ctx.quadraticCurveTo(23, -124, 27, -100);
  ctx.stroke();
  ctx.fillStyle = skin;
  ctx.strokeStyle = ink;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.ellipse(28, -98, 5.8, 6.8, 0.3, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.save();
  ctx.translate(28, -98);
  ctx.rotate(sweep);
  ctx.strokeStyle = "oklch(0.98 0 0 / 0.22)";
  ctx.lineWidth = 10;
  ctx.beginPath();
  ctx.arc(0, 0, 120, 0.72, 1.16);
  ctx.stroke();
  const grad = ctx.createLinearGradient(0, 0, 78, 100);
  grad.addColorStop(0, "oklch(0.99 0 0)");
  grad.addColorStop(0.75, "oklch(0.99 0 0)");
  grad.addColorStop(1, "oklch(0.55 0.21 27)");
  ctx.strokeStyle = grad;
  ctx.lineWidth = 5.5;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(78, 100);
  ctx.stroke();
  ctx.restore();
  ctx.restore();

  if (fallen) {
    ctx.fillStyle = "oklch(0.55 0.21 27)";
    ctx.font = "700 14px ui-monospace, monospace";
    ctx.fillText("impact + stillness · cancel window open", baseX - 40, baseY - 130);
  }
}


function drawWeather(ctx: CanvasRenderingContext2D, c: Conditions, t: number) {
  if (c.light < 0.75) {
    ctx.fillStyle = `oklch(0.2 0.03 265 / ${(0.75 - c.light) * 0.55})`;
    ctx.fillRect(0, 0, W, H);
  }
  const n = Math.round(c.particulate * 160);
  ctx.fillStyle = "oklch(0.98 0 0 / 0.35)";
  for (let i = 0; i < n; i++) {
    const x = (i * 137.5 + t * 90 * (0.4 + (i % 5) / 5)) % W;
    const y = (i * 61.7 + t * 240 * (0.5 + (i % 3) / 4)) % H;
    ctx.fillRect(x, y, 1.6, c.particulate > 0.5 ? 7 : 2);
  }
  if (c.motion > 0.35) {
    ctx.fillStyle = `oklch(0.6 0 0 / ${(c.motion - 0.35) * 0.18})`;
    ctx.fillRect(0, 0, W, H);
  }
}

/**
 * On-canvas overlay. Deliberately thin: the frame around the canvas already
 * draws crop marks and a vignette, and the right-hand column already carries
 * the numbers. Anything drawn here has to earn its place by being something a
 * non-technical viewer reads at a glance — the hazard, the spoken line, and
 * nothing else.
 */
function drawHud(ctx: CanvasRenderingContext2D, p: SceneProps, d: Derived) {
  const level = p.decision?.level ?? "none";


  // Signal head, only where a crossing is being read.
  if (p.overlay === "crossing") {
    const walk = d.visionConf > 0.45 && (p.t % 14) < 7;
    const g = ground(2.9, 10.5);
    ctx.fillStyle = "oklch(0.24 0 0)";
    ctx.fillRect(g.px - 16, g.py - 150, 32, 52);
    ctx.fillStyle = walk
      ? v("--guide", "#3f8f63")
      : d.visionConf < 0.45
        ? v("--muted-foreground", "#777")
        : v("--urgent", "#c0392b");
    ctx.beginPath();
    ctx.arc(g.px, g.py - 124, 11, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = "700 12px ui-monospace, monospace";
    ctx.fillStyle = "oklch(0.2 0 0)";
    ctx.fillText(
      d.visionConf < 0.45 ? "head occluded" : walk ? "walk" : "don't walk",
      g.px + 22,
      g.py - 120,
    );
  }

  // Drop-off band: platform edge, kerb or top step. Distance, not TTC.
  if (p.overlay === "edge") {
    const dist = 1.9 + 1.1 * Math.sin(p.t * 0.7);
    const near = ground(0.7, 1.0);
    const far = ground(1.5, 22);
    ctx.strokeStyle = dist < 1.2 ? v("--urgent", "#c0392b") : v("--veto", "#d8a63c");
    ctx.lineWidth = 4;
    ctx.setLineDash([12, 8]);
    ctx.beginPath();
    ctx.moveTo(near.px, near.py);
    ctx.lineTo(far.px, far.py);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = "700 14px ui-monospace, monospace";
    ctx.fillStyle = dist < 1.2 ? v("--urgent", "#c0392b") : v("--veto", "#d8a63c");
    ctx.fillText(`drop-off ${dist.toFixed(1)} m to your right`, near.px + 18, near.py - 26);
  }

  // Spoken line. One caption bar, measured with the font it is drawn in, so it
  // can never run past the edge of the frame.
  if (level !== "none" && p.decision?.utterance) {
    const col = v(LEVEL_KEY[level], "#3b6fd0");
    ctx.fillStyle = col;
    ctx.globalAlpha = level === "urgent" ? 0.1 + 0.05 * Math.sin(p.t * 6) : 0.06;
    ctx.fillRect(0, 0, W, H);
    ctx.globalAlpha = 1;
    caption(ctx, p.decision.utterance, col);
  }

  // At most one advisory bubble, always in the same place, always clamped.
  const advisory: Partial<Record<string, string>> = {
    translate: "“Careful, step”",
    reader: "20 euro note",
    scene: "Bench at 2 o'clock, 4 m",
    human: "Connecting a sighted assistant",
  };
  const line = p.overlay ? advisory[p.overlay] : undefined;
  if (line) bubble(ctx, W / 2, 62, line);
}

/** Bottom caption bar: fixed height, text clipped to the frame width. */
function caption(ctx: CanvasRenderingContext2D, text: string, col: string) {
  ctx.font = "700 19px ui-sans-serif, system-ui";
  const pad = 18;
  const maxW = W - 96;
  const w = Math.min(ctx.measureText(text).width + pad * 2, maxW);
  ctx.fillStyle = col;
  ctx.beginPath();
  ctx.roundRect(48, H - 82, w, 46, 12);
  ctx.fill();
  ctx.save();
  ctx.beginPath();
  ctx.rect(48, H - 82, w, 46);
  ctx.clip();
  ctx.fillStyle = "oklch(1 0 0)";
  ctx.fillText(text, 48 + pad, H - 52);
  ctx.restore();
}

function bubble(ctx: CanvasRenderingContext2D, x: number, y: number, text: string) {
  ctx.font = "600 15px ui-sans-serif, system-ui";
  const w = Math.min(ctx.measureText(text).width + 28, W - 80);
  const left = Math.max(40, Math.min(x - w / 2, W - 40 - w));
  ctx.fillStyle = v("--assist", "#7a4fd0");
  ctx.globalAlpha = 0.94;
  ctx.beginPath();
  ctx.roundRect(left, y - 20, w, 38, 12);
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.fillStyle = "oklch(1 0 0)";
  ctx.fillText(text, left + 14, y + 4);
}

function drawVignette(ctx: CanvasRenderingContext2D) {
  const g = ctx.createRadialGradient(W / 2, H * 0.55, H * 0.35, W / 2, H * 0.55, H * 0.95);
  g.addColorStop(0, "oklch(0 0 0 / 0)");
  g.addColorStop(1, "oklch(0.18 0.03 260 / 0.34)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
}
