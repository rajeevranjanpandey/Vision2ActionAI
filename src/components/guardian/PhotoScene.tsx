/**
 * PhotoScene — the camera plate.
 *
 * This replaces the vector cartoon with what the device actually works on: a still
 * frame captured from the head-worn camera, with the perception stack's output drawn
 * back onto it — detection box, class + score, metric range from the depth head, the
 * forecast track, and the walker's own corridor.
 *
 * Everything drawn here is projected from the same ego-frame state (x, z, vx, vz) the
 * risk head reads, using a pinhole model, so the overlay and the readout can never
 * disagree.
 */

import { useMemo, useState } from "react";
import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  Bike,
  Camera,
  Car,
  PersonStanding,
  TriangleAlert,
  X,
} from "lucide-react";

import { motion, useReducedMotion } from "motion/react";

import type { Decision, TrackState } from "@/lib/riskHead";
import type { SceneKind } from "@/lib/environments";

import plateStreet from "@/assets/plate-street.jpg";
import plateTransit from "@/assets/plate-transit.jpg";
import plateRoom from "@/assets/plate-room.jpg";
import plateRetail from "@/assets/plate-retail.jpg";
import plateThreshold from "@/assets/plate-threshold.jpg";
import platePlaza from "@/assets/plate-plaza.jpg";
import plateWorksite from "@/assets/plate-worksite.jpg";
import plateRig from "@/assets/plate-rig.jpg";
import plateFall from "@/assets/plate-fall.jpg";
import plateSign from "@/assets/plate-sign.jpg";
import plateShop from "@/assets/plate-shop.jpg";
import platePhone from "@/assets/plate-phone.jpg";
import plateSignage from "@/assets/plate-signage.jpg";
import plateCurb from "@/assets/plate-curb.jpg";

import { SceneAnnotation, type SceneOverlayKind } from "./SceneAnnotations";

/** Some capabilities look at a specific kind of thing, not at the ambient scene. */
const PLATE_BY_OVERLAY: Partial<Record<SceneOverlayKind, string>> = {
  edge: plateCurb,
  fall: plateFall,
  reader: plateSign,
  places: plateShop,
  human: platePhone,
  translate: plateSignage,
};

const PLATE: Record<SceneKind, string> = {
  street: plateStreet,
  transit: plateTransit,
  room: plateRoom,
  retail: plateRetail,
  threshold: plateThreshold,
  plaza: platePlaza,
  worksite: plateWorksite,
  rig: plateRig,
};

/* ------------------------------------------------------------------ projection */

const HORIZON = 0.47; // normalised image row of the horizon for these plates
const FX = 0.62; // focal length / image width
const FY = 0.86; // focal length / image height
const CAM_H = 1.5; // camera height above ground, metres

/** Ego-frame metres → normalised image coordinates (0..1). */
function project(x: number, z: number) {
  const zz = Math.max(z, 0.6);
  return {
    u: clamp(0.5 + (x / zz) * FX, 0.03, 0.97),
    v: clamp(HORIZON + (CAM_H * FY) / zz, 0.05, 0.98),
  };
}

function clamp(v: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, v));
}

function ClassIcon({ name, className }: { name: string; className?: string }) {
  const n = name.toLowerCase();
  if (n.includes("cycl") || n.includes("bike") || n.includes("bicycle"))
    return <Bike className={className} />;
  if (n.includes("car") || n.includes("veh") || n.includes("bus"))
    return <Car className={className} />;
  if (n.includes("person") || n.includes("ped")) return <PersonStanding className={className} />;
  return <TriangleAlert className={className} />;
}

export interface PhotoSceneProps {
  scene: SceneKind;
  track: TrackState | null;
  decision: Decision | null;
  frame: number;
  playing: boolean;
  /** 0..1 — how much the camera can be trusted under the current conditions. */
  visionConf: number;
  location: string;
  /** Which capability is being demonstrated — decides the overlay vocabulary. */
  overlay: SceneOverlayKind;
  /** 0..1 progress through the scripted scenario. */
  t: number;
}

export function PhotoScene({
  scene,
  track,
  decision,
  frame,
  playing,
  visionConf,
  location,
  overlay,
  t,
}: PhotoSceneProps) {
  const reduce = useReducedMotion();
  const plate = PLATE_BY_OVERLAY[overlay] ?? PLATE[scene] ?? plateStreet;
  /** Only the motion-hazard capabilities draw corridor, track box and forecast. */
  const isPredict = overlay === "predict";
  /** null = follow the hazard automatically; otherwise the viewer picked a camera. */
  const [camPick, setCamPick] = useState<"front" | "rear" | "left" | "right" | null>(null);


  const level = decision?.level ?? "none";
  const hazard = level !== "none";
  const stroke = level === "urgent" ? "#E5484D" : level === "warn" ? "#FF7A1A" : "#12A150";
  /** The predicted track always reads as the hazard channel, never as the safe green. */
  const forecastStroke = "#E5484D";


  /**
   * Forecast rollout, snapped onto the marked crossing and truncated at the
   * walker's corridor. Road users obey the crossing: the rollout is pulled onto
   * the zebra band (CROSSWALK_Z) instead of cutting diagonally across the
   * carriageway, and it stops the moment it enters the ±0.75 m ego corridor so
   * the impact reticle sits on the real conflict point.
   */
  const { forecast, conflict } = useMemo(() => {
    if (!track) return { forecast: [] as { u: number; v: number }[], conflict: null };
    /** Depth of the painted zebra band in the plate, in metres. */
    const CROSSWALK_Z = 6.2;
    const pts: { u: number; v: number }[] = [];
    let hit: { u: number; v: number } | null = null;
    for (let i = 0; i <= 12; i++) {
      const dt = i * 0.2;
      const s = i / 12;
      const x = track.x + track.vx * dt;
      const free = track.z + track.vz * dt;
      // ease onto the crossing band, then travel along it rather than into the road
      const z = free * (1 - s) + CROSSWALK_Z * s;
      if (z < 0.6) break;
      const p = project(x, z);
      pts.push(p);
      if (Math.abs(x) <= 0.75) {
        hit = p;
        break;
      }
    }
    return { forecast: pts, conflict: hit };
  }, [track]);


  const box = useMemo(() => {
    if (!track) return null;
    const zz = Math.max(track.z, 0.6);
    const base = project(track.x, track.z);
    const h = clamp((1.7 * FY) / zz, 0.05, 0.85);
    const w = clamp((Math.max(track.widthM, 0.5) * FX) / zz, 0.03, 0.6);
    return { left: base.u - w / 2, top: base.v - h, w, h, base };
  }, [track]);

  /** Ego corridor (±0.75 m) back-projected out to 16 m, in viewBox units. */
  const corridor = useMemo(() => {
    const far = 16;
    const l = project(-0.75, far);
    const r = project(0.75, far);
    return [
      "M 42 65",
      `L ${(l.u * 100).toFixed(2)} ${(l.v * 65).toFixed(2)}`,
      `L ${(r.u * 100).toFixed(2)} ${(r.v * 65).toFixed(2)}`,
      "L 58 65 Z",
    ].join(" ");
  }, []);


  const range = track ? Math.hypot(track.x, track.z) : null;
  const closing = track ? Math.max(0, -track.vz) : 0;
  /** Head TTC when it has one, geometric range / closing otherwise — never blank. */
  const ttcValue =
    decision && Number.isFinite(decision.ttc) && decision.ttc > 0
      ? decision.ttc
      : range !== null && closing > 0.15
        ? range / closing
        : 0;
  const bearing = track ? (track.x < -0.4 ? "left" : track.x > 0.4 ? "right" : "ahead") : "ahead";
  /** Camera the fusion stack is showing: the hazard side unless the viewer switched. */
  const hazardCam = bearing === "ahead" ? "front" : bearing;
  const activeCam = camPick ?? hazardCam;
  const onHazardCam = activeCam === hazardCam;
  /** Overlay geometry only exists on the camera that actually sees the track. */
  const geometric = isPredict && onHazardCam;
  const score = clamp(0.55 + 0.42 * visionConf, 0.4, 0.98);



  /**
   * The walker's own path, drawn forward from the feet. Under a hazard it bends
   * away from the side the track is coming from, so the frame shows both where
   * the person is and how they are avoiding the conflict.
   */
  const ownPath = useMemo(() => {
    const away = track ? (track.x >= 0 ? -1 : 1) : 1;
    const shift = hazard ? (level === "urgent" ? 0.7 : 0.45) * away : 0;
    const pts: { u: number; v: number }[] = [];
    for (let i = 0; i <= 10; i++) {
      const z = 1.2 + i * 0.9;
      const s = i / 10;
      const x = -0.15 + 0.35 * s ** 2 + shift * s ** 1.5;
      pts.push(project(x, z));
    }
    return pts;
  }, [track, hazard, level]);


  const confidencePct = Math.round(
    clamp(Math.max(score, decision?.probability ?? 0) * 100, 40, 98),
  );
  const ring = 2 * Math.PI * 26;

  return (
    <figure className="relative m-0 aspect-[1280/832] w-full overflow-hidden bg-black">
      <img
        src={plate}
        alt={`Camera frame from ${location}`}
        width={1280}
        height={832}
        className="absolute inset-0 h-full w-full object-cover"
      />
      {/* sensor tone: slight cool grade so overlays read as instrument, not sticker */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(120%_90%_at_50%_40%,transparent_45%,rgba(6,12,22,0.45)_100%)]" />

      {/* --- overlay geometry ------------------------------------------------ */}
      <svg
        viewBox="0 0 100 65"
        preserveAspectRatio="none"
        className="pointer-events-none absolute inset-0 h-full w-full"
        aria-hidden
      >
        {geometric && (
          <>
            {/* the walker's own corridor, back-projected from the ego cylinder */}
            <path
              d={corridor}
              fill="rgba(20,180,120,0.14)"
              stroke="rgba(20,220,140,0.55)"
              strokeWidth={0.35}
              strokeDasharray="2 1.6"
            />

            {/* the walker's own path, drawn forward from the feet */}
            <polyline
              points={ownPath.map((p) => `${p.u * 100},${p.v * 65}`).join(" ")}
              fill="none"
              stroke="#12A150"
              strokeWidth={0.7}
              strokeLinecap="round"
            />
            <circle cx={ownPath[0]!.u * 100} cy={ownPath[0]!.v * 65} r={0.8} fill="#12A150" />
            <polygon
              points={(() => {
                const a = ownPath[ownPath.length - 2]!;
                const b = ownPath[ownPath.length - 1]!;
                const dx = (b.u - a.u) * 100;
                const dy = (b.v - a.v) * 65;
                const len = Math.hypot(dx, dy) || 1;
                const ux = dx / len;
                const uy = dy / len;
                const tipX = b.u * 100;
                const tipY = b.v * 65;
                return `${tipX},${tipY} ${tipX - ux * 2.4 - uy * 1.1},${tipY - uy * 2.4 + ux * 1.1} ${tipX - ux * 2.4 + uy * 1.1},${tipY - uy * 2.4 - ux * 1.1}`;
              })()}
              fill="#12A150"
            />

            {forecast.length > 1 && (
              <>
                {/* white casing under the dashes so red never fights the plate */}
                <polyline
                  points={forecast.map((p) => `${p.u * 100},${p.v * 65}`).join(" ")}
                  fill="none"
                  stroke="rgba(255,255,255,0.95)"
                  strokeWidth={1.5}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <polyline
                  points={forecast.map((p) => `${p.u * 100},${p.v * 65}`).join(" ")}
                  fill="none"
                  stroke={forecastStroke}
                  strokeWidth={0.8}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeDasharray="2.2 1.8"
                  className={playing && !reduce ? "animate-[dash_24s_linear_infinite]" : undefined}
                />
                {/* impact reticle only where the track actually enters the corridor */}
                {conflict && (
                  <>
                    <circle
                      cx={conflict.u * 100}
                      cy={conflict.v * 65}
                      r={3}
                      fill="rgba(229,72,77,0.25)"
                      stroke="rgba(255,255,255,0.95)"
                      strokeWidth={0.5}
                    />
                    <circle
                      cx={conflict.u * 100}
                      cy={conflict.v * 65}
                      r={2}
                      fill="none"
                      stroke={forecastStroke}
                      strokeWidth={0.6}
                    />
                    <circle cx={conflict.u * 100} cy={conflict.v * 65} r={0.9} fill={forecastStroke} />
                  </>
                )}
              </>
            )}


            {box && (
              <rect
                x={box.left * 100}
                y={box.top * 65}
                width={box.w * 100}
                height={box.h * 65}
                fill="none"
                stroke={stroke}
                strokeWidth={0.45}
              />
            )}
          </>
        )}
      </svg>

      {/* class icon + white anchor line + dot — one overlay so they glide together */}
      {geometric && box && track && (
        <div
          className="pointer-events-none absolute"
          style={{
            left: `${clamp(box.left + box.w / 2, 0.04, 0.96) * 100}%`,
            top: `${clamp(box.top + box.h / 2, 0.08, 0.92) * 100}%`,
            transition: "left 0.3s linear, top 0.3s linear",
          }}
        >
          {/* leader terminates at the exact centre of the tracked rectangle */}
          <div className="absolute bottom-0 left-0 h-8 w-[2px] -translate-x-1/2 bg-white/95" />
          {/* anchor dot marks the rectangle centre */}
          <div className="absolute left-0 top-0 h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white/95" />
          {/* circular class icon moves with the centre anchor as one overlay */}
          <motion.span
            key={track.className}
            initial={reduce ? false : { scale: 0.85, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="absolute bottom-8 left-0 grid h-11 w-11 -translate-x-1/2 place-items-center rounded-full border-2 bg-black/55 backdrop-blur-sm"
            style={{ borderColor: hazard ? stroke : "rgba(255,255,255,0.9)" }}
          >
            <ClassIcon name={track.className} className="h-5 w-5 text-white" />
          </motion.span>
        </div>
      )}


      {/* predicted-path label pinned to the forecast */}
      {geometric && forecast.length > 2 && (
        <div
          className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2 whitespace-nowrap rounded-lg px-2.5 py-1.5 text-center leading-tight text-white shadow-sm"
          style={{
            left: `${clamp(forecast[Math.floor(forecast.length / 2)]!.u, 0.12, 0.86) * 100}%`,
            top: `${clamp(forecast[Math.floor(forecast.length / 2)]!.v - 0.06, 0.1, 0.8) * 100}%`,
            backgroundColor: forecastStroke,
          }}
        >
          <span className="block text-[12px] font-semibold">Predicted path</span>
          <span className="block font-mono text-[10px] tabular-nums opacity-90">
            {ttcValue.toFixed(1)} s to cross
          </span>

        </div>
      )}

      {/* the walker's own path label */}
      {geometric && (
        <div className="pointer-events-none absolute bottom-[14%] left-[30%] whitespace-nowrap rounded-lg bg-[#12A150] px-2.5 py-1.5 text-[12px] font-semibold text-white shadow-sm">
          Your path
        </div>
      )}

      {/* camera selector — defaults to the hazard side, viewer can switch feeds */}
      {isPredict && (
        <motion.div
          layout
          transition={{ type: "spring", stiffness: 320, damping: 30 }}
          className="absolute inset-x-0 z-20 flex flex-col items-center gap-1.5"
          style={{ bottom: camPick === null ? 12 : 58 }}
        >
          {!onHazardCam && (
            <span className="pointer-events-none rounded-full bg-black/60 px-3 py-1 text-[11px] font-semibold text-white backdrop-blur-sm">
              {activeCam.toUpperCase()} camera · no closing hazard on this feed
            </span>
          )}
          <div
            role="group"
            aria-label="Camera direction"
            className="flex items-center gap-1 rounded-full bg-black/45 p-1 backdrop-blur-sm"
          >
            {([
              { k: "front", label: "FRONT", Icon: ArrowUp },
              { k: "rear", label: "REAR", Icon: ArrowUp },
              { k: "left", label: "LEFT", Icon: ArrowDown },
              { k: "right", label: "RIGHT", Icon: ArrowRight },
            ] as const).map(({ k, label, Icon }) => {
              const active = k === activeCam;
              return (
                <button
                  key={k}
                  type="button"
                  aria-pressed={active}
                  onClick={() => setCamPick(k)}
                  className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[11px] font-semibold tracking-[0.08em] transition-colors ${
                    active
                      ? "bg-black text-white"
                      : "text-white/70 hover:bg-black/40 hover:text-white"
                  } ${k === hazardCam && !active ? "ring-1 ring-[#E5484D]" : ""}`}
                >
                  {k === "left" ? (
                    <ArrowLeft className="h-3.5 w-3.5" />
                  ) : (
                    <Icon className={`h-3.5 w-3.5 ${k === "rear" ? "rotate-180" : ""}`} />
                  )}
                  {label}
                </button>
              );
            })}
            {camPick !== null && (
              <button
                type="button"
                aria-label="Close manual camera view and follow the hazard automatically"
                title="Back to auto (follow hazard)"
                onClick={() => setCamPick(null)}
                className="ml-0.5 flex h-7 w-7 items-center justify-center rounded-full text-white/80 transition-colors hover:bg-black/50 hover:text-white"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        </motion.div>
      )}



      {/* confidence donut, bottom-right */}
      {geometric && (
        <div className="pointer-events-none absolute bottom-4 right-4 grid h-[76px] w-[76px] place-items-center rounded-full bg-white/95 shadow-sm">
          <svg viewBox="0 0 60 60" className="absolute inset-0 h-full w-full -rotate-90">
            <circle cx="30" cy="30" r="26" fill="none" stroke="rgba(0,0,0,0.08)" strokeWidth="4" />
            <circle
              cx="30"
              cy="30"
              r="26"
              fill="none"
              stroke="#12A150"
              strokeWidth="4"
              strokeLinecap="round"
              strokeDasharray={`${(confidencePct / 100) * ring} ${ring}`}
            />
          </svg>
          <div className="relative text-center leading-none">
            <p className="font-mono text-[18px] font-bold tabular-nums text-black">
              {confidencePct}%
            </p>
            <p className="mt-0.5 text-[7px] uppercase tracking-[0.1em] text-black/55">
              confidence
            </p>
          </div>
        </div>
      )}

      {/* --- capability-specific annotation layer ---------------------------- */}
      <SceneAnnotation
        kind={overlay}
        utterance={decision?.utterance ?? ""}
        t={t}
        ttc={decision && Number.isFinite(decision.ttc) ? decision.ttc : 0}
        confidence={visionConf}
      />

      {/* --- capture chip (top-left) ------------------------------------------ */}
      <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start justify-between gap-3 p-3">
        <span className="flex items-center gap-2 rounded-full bg-black/65 px-3 py-1.5 text-[12px] font-medium text-white backdrop-blur-sm">
          <span
            className="h-2 w-2 rounded-full"
            style={{ backgroundColor: visionConf > 0.5 ? "#12A150" : "#FF7A1A" }}
          />
          {visionConf > 0.5 ? "Camera clear" : "Camera degraded"}
        </span>
        <span className="rounded-full bg-black/45 px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.16em] text-white/80 backdrop-blur-sm">
          frame {String(frame).padStart(3, "0")} · 10 Hz
        </span>
      </div>

      {/* --- numeric readout (bottom) — only where TTC is not the story ------- */}
      {!geometric && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 grid grid-cols-2 gap-px bg-white/10 sm:grid-cols-4">
          {[
            { k: "region conf", v: `${Math.round(score * 100)}%` },
            { k: "vision conf", v: `${Math.round(visionConf * 100)}%` },
            { k: "answer", v: score > 0.7 ? "spoken" : "hedged" },
            { k: "latency", v: `${Math.round(600 + (1 - visionConf) * 800)} ms` },
          ].map((s) => (
            <div key={s.k} className="bg-black/65 px-3 py-2 backdrop-blur-sm">
              <p className="font-mono text-[9px] uppercase tracking-[0.16em] text-white/55">
                {s.k}
              </p>
              <p className="mt-0.5 font-mono text-sm font-semibold tabular-nums text-white">
                {s.v}
              </p>
            </div>
          ))}
        </div>
      )}

      {visionConf < 0.4 && (
        <div className="pointer-events-none absolute inset-0 grid place-items-center bg-black/45">
          <p className="rounded-xl border border-white/40 px-4 py-2 font-mono text-xs uppercase tracking-[0.2em] text-white">
            camera degraded · geometry only
          </p>
        </div>
      )}

      <style>{`@keyframes dash { to { stroke-dashoffset: -4; } }`}</style>
    </figure>
  );
}

