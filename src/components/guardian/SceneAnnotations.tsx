/**
 * SceneAnnotations — the per-capability overlay vocabulary.
 *
 * Each of the twelve capabilities draws a *different* thing on the camera plate,
 * because each one is answering a different question. Geometry-driven hazards get
 * boxes, corridors and forecasts; reading and translation get text regions and a
 * spoken result; memory and routing get anchors and paths.
 *
 * Everything here is presentation only — the numbers arrive as props.
 */

import {
  Check,
  Footprints,
  Home,
  PersonStanding,
  PhoneCall,
  Volume2,
  X,
} from "lucide-react";
import { motion } from "motion/react";

export type SceneOverlayKind =
  | "predict"
  | "crossing"
  | "edge"
  | "fall"
  | "breadcrumb"
  | "object"
  | "reader"
  | "scene"
  | "places"
  | "reroute"
  | "human"
  | "translate";

/* ------------------------------------------------------------------ primitives */

/** Frosted speech card — what the earpiece is saying, anchored in the frame. */
export function SpeechCard({
  text,
  className,
  wide,
}: {
  text: string;
  className?: string;
  wide?: boolean;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      className={`pointer-events-none absolute flex items-start gap-2 rounded-xl bg-white/95 px-3 py-2 shadow-lg backdrop-blur-sm ${
        wide ? "max-w-[46%]" : "max-w-[38%]"
      } ${className ?? ""}`}
    >
      <Volume2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-black/70" />
      <p className="text-[12px] font-medium leading-snug text-black">{text}</p>
    </motion.div>
  );
}

function Tag({
  text,
  tone = "neutral",
  className,
}: {
  text: string;
  tone?: "neutral" | "alert" | "ok" | "info";
  className?: string;
}) {
  const bg =
    tone === "alert"
      ? "bg-[#E5484D]"
      : tone === "ok"
        ? "bg-[#12A150]"
        : tone === "info"
          ? "bg-[#1F6FEB]"
          : "bg-black/75";
  return (
    <span
      className={`pointer-events-none absolute rounded-md px-2 py-1 font-mono text-[10px] font-semibold uppercase tracking-[0.12em] text-white ${bg} ${
        className ?? ""
      }`}
    >
      {text}
    </span>
  );
}

/** A region-of-interest rectangle in percentage coordinates. */
function Region({
  x,
  y,
  w,
  h,
  color,
  label,
  dashed,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  color: string;
  label?: string;
  dashed?: boolean;
}) {
  return (
    <div
      className="pointer-events-none absolute"
      style={{ left: `${x}%`, top: `${y}%`, width: `${w}%`, height: `${h}%` }}
    >
      <div
        className="h-full w-full rounded-[3px]"
        style={{
          border: `2px ${dashed ? "dashed" : "solid"} ${color}`,
          boxShadow: `0 0 0 1px rgba(0,0,0,0.25)`,
        }}
      />
      {label && (
        <span
          className="absolute -top-1 left-0 -translate-y-full whitespace-nowrap rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-[0.12em] text-white"
          style={{ background: color }}
        >
          {label}
        </span>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------- overlays */

export interface AnnotationProps {
  kind: SceneOverlayKind;
  /** Current spoken line for this tick. */
  utterance: string;
  /** 0..1 progress through the scripted scenario. */
  t: number;
  ttc: number;
  confidence: number;
}

export function SceneAnnotation({ kind, utterance, t, ttc, confidence }: AnnotationProps) {
  switch (kind) {
    case "crossing":
      return (
        <>
          <Region x={40} y={9} w={11} h={26} color="#12A150" label="walk signal 0.93" />
          <Tag
            text={`walk · ${Math.max(3, Math.round(14 - t * 11))} s left`}
            tone="ok"
            className="left-3 top-1/2"
          />
          <SpeechCard
            text={`Walk signal on, ${Math.max(3, Math.round(14 - t * 11))} seconds. Kerb ramp straight ahead.`}
            className="bottom-[22%] right-3"
          />
        </>
      );

    case "edge":
      return (
        <>
          <svg
            viewBox="0 0 100 65"
            preserveAspectRatio="none"
            className="pointer-events-none absolute inset-0 h-full w-full"
            aria-hidden
          >
            <polyline
              points="17,3 36,26 55,52 65,65"
              fill="none"
              stroke="#FF7A1A"
              strokeWidth={0.7}
              strokeLinecap="round"
            />
            <polyline
              points="24,3 43,26 62,52 72,65"
              fill="none"
              stroke="#FF7A1A"
              strokeWidth={0.6}
              strokeDasharray="2 1.5"
              opacity={0.8}
            />
          </svg>
          <Tag text="drop 14 cm · 1.2 m ahead" tone="alert" className="left-3 top-14" />
          <SpeechCard text="Step down, fourteen centimetres, one metre ahead." className="bottom-[22%] left-3" />
        </>
      );

    case "fall":
      return (
        <>
          <Region x={44} y={26} w={24} h={52} color="#E5484D" label="posture: down 0.91" />
          <div className="pointer-events-none absolute right-3 top-3 flex items-center gap-2 rounded-xl bg-[#E5484D] px-3 py-2 text-white shadow-lg">
            <PersonStanding className="h-4 w-4" />
            <span className="font-mono text-[11px] font-semibold uppercase tracking-[0.14em]">
              fall detected
            </span>
          </div>
          <SpeechCard
            text={`Are you alright? Contacting your trusted contact in ${Math.max(0, Math.round(20 - t * 20))} seconds.`}
            className="bottom-[22%] right-3"
            wide
          />
        </>
      );

    case "breadcrumb":
      return (
        <>
          <svg
            viewBox="0 0 100 65"
            preserveAspectRatio="none"
            className="pointer-events-none absolute inset-0 h-full w-full"
            aria-hidden
          >
            <path
              d="M 48 64 C 44 52, 58 46, 62 39 S 70 30, 78 26"
              fill="none"
              stroke="#1F6FEB"
              strokeWidth={0.8}
              strokeDasharray="2 2"
              strokeLinecap="round"
            />
          </svg>
          <span className="pointer-events-none absolute right-[18%] top-[32%] grid h-9 w-9 place-items-center rounded-full bg-[#1F6FEB] text-white shadow-lg">
            <Home className="h-4 w-4" />
          </span>
          <Tag text="retrace · 62 m to start" tone="info" className="left-3 top-14" />
          <SpeechCard text="Turn around. Your starting point is sixty metres back, bearing right." className="bottom-[22%] left-3" wide />
        </>
      );

    case "object":
      return (
        <>
          <Region x={34} y={38} w={16} h={38} color="#1F6FEB" label="bin · seen 11 min ago" />
          <SpeechCard text="Remembered: waste bin, half a metre to your left." className="bottom-[24%] right-3" />
          <Tag text="landmark match 0.88" tone="info" className="left-3 top-14" />
        </>
      );

    case "reader":
      return (
        <>
          <Region x={13} y={30} w={70} h={31} color="#F5A623" label="text region · ocr" dashed />
          <SpeechCard text="West thirty-fourth street. Numbers one hundred to one seventy-two." className="bottom-[10%] left-1/2 -translate-x-1/2" wide />
          <Tag text="read on request · 0.4 s" className="left-3 top-14" />
        </>
      );

    case "scene":
      return (
        <>
          <SpeechCard
            text="There is a crosswalk ahead. A bicycle is approaching from your left."
            className="right-3 top-[24%]"
            wide
          />
          <Tag text="scene query · slow path 780 ms" className="left-3 top-14" />
        </>
      );

    case "places":
      return (
        <>
          <Region x={24} y={20} w={40} h={44} color="#12A150" label="coffee shop · entrance" />
          <SpeechCard text="Coffee shop, twenty metres ahead on your right. Door is level." className="bottom-[18%] left-4" wide />
        </>
      );

    case "reroute":
      return (
        <>
          <svg
            viewBox="0 0 100 65"
            preserveAspectRatio="none"
            className="pointer-events-none absolute inset-0 h-full w-full"
            aria-hidden
          >
            <path d="M 46 64 L 44 34" stroke="#E5484D" strokeWidth={0.9} fill="none" strokeDasharray="2 2" />
            <path
              d="M 50 64 C 56 50, 66 44, 76 36"
              stroke="#12A150"
              strokeWidth={0.9}
              fill="none"
              strokeDasharray="2 1.6"
            />
          </svg>
          <span className="pointer-events-none absolute left-[41%] top-[46%] grid h-7 w-7 place-items-center rounded-full bg-[#E5484D] text-white shadow">
            <X className="h-4 w-4" />
          </span>
          <span className="pointer-events-none absolute left-[73%] top-[50%] grid h-7 w-7 place-items-center rounded-full bg-[#12A150] text-white shadow">
            <Check className="h-4 w-4" />
          </span>
          <SpeechCard text="Pavement closed ahead. Rerouting: cross here, then continue right." className="bottom-[20%] left-3" wide />
        </>
      );

    case "human":
      return (
        <>
          <div className="pointer-events-none absolute left-1/2 top-[16%] flex -translate-x-1/2 items-center gap-2 rounded-xl bg-black/80 px-3 py-2 text-white shadow-lg">
            <PhoneCall className="h-4 w-4 text-[#12A150]" />
            <span className="font-mono text-[11px] uppercase tracking-[0.14em]">
              live assistant · connected
            </span>
          </div>
          <SpeechCard text="Confidence low. Handing the camera to a human assistant." className="bottom-[18%] left-1/2 -translate-x-1/2" wide />
          <Tag text={`model confidence ${(confidence * 100).toFixed(0)}%`} tone="alert" className="left-3 top-14" />
        </>
      );

    case "translate":
      return (
        <>
          <Region x={10} y={13} w={66} h={68} color="#F5A623" label="signage · zh → en" dashed />
          <SpeechCard text="Subway entrance. Go straight fifty metres, then right." className="bottom-[8%] right-3" />
          <Tag text="translate · slow path" className="left-3 top-14" />
        </>
      );

    case "predict":
    default:
      /* The predictive path stays silent unless there is a live hazard to name. */
      if (!utterance || !(ttc > 0 && ttc < 6)) return null;
      return (
        <>
          <div className="pointer-events-none absolute bottom-[22%] left-4 flex items-center gap-2 rounded-xl bg-[#E5484D] px-3 py-2 text-white shadow-lg">
            <Footprints className="h-4 w-4" />
            <span className="text-[12px] font-semibold">
              {utterance || `Hazard approaching in ${ttc.toFixed(1)} s`}
            </span>
          </div>
        </>
      );
  }
}
