/**
 * The eight environment classes from ENVIRONMENT_SPEC.md, as runnable simulation
 * presets.
 *
 * The point of this file is the rule at the end of the spec: never validate a
 * feature only inside the environment it was designed around. Each class carries
 * its own sensor problem — GPS denial, clutter, lighting extremes, crowd
 * occlusion — so any feature simulation can be re-run somewhere hostile to it.
 */

import type { Conditions } from "@/components/guardian/SceneConditions";

export type EnvId = "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H";

export interface EnvClass {
  id: EnvId;
  name: string;
  blurb: string;
  /** Sensor problem this class exists to stress. */
  stress: string;
  indoor: boolean;
  /** 0..1 positioning quality. Indoor classes force dead reckoning. */
  gps: number;
  /** Scenario variants that must be run inside this class. */
  variants: string[];
  /** Feature numbers (journey order 01–11) this class is the proving ground for. */
  features: number[];
  /** Build order rank from Part 4 of the spec. 1 = first. */
  buildRank: number;
  built: boolean;
  c: Conditions;
}

export const ENVIRONMENTS: EnvClass[] = [
  {
    id: "A",
    name: "Street & intersection",
    blurb: "Signalised 4-way, T-junction, midblock, unmarked crossing",
    stress: "Glare, wet reflective lines, cambered road breaking ground-plane RANSAC",
    indoor: false,
    gps: 0.85,
    variants: [
      "4-way signalised",
      "T-junction",
      "unmarked midblock",
      "dusk sun into camera",
      "night + headlight glare",
      "wet lines / puddle false depth",
      "unprotected turn branch",
    ],
    features: [1, 2, 6, 7, 8],
    buildRank: 1,
    built: true,
    c: { crowd: 0.35, ambientDb: 68, light: 0.75, particulate: 0.15, motion: 0.35 },
  },
  {
    id: "H",
    name: "Fall / impact physics rig",
    blurb: "Not a camera scene — ragdoll IMU traces plus a false-positive library",
    stress: "Separating a genuine fall from sitting down hard, jostling and stair descent",
    indoor: false,
    gps: 0.6,
    variants: [
      "pavement fall",
      "grass fall",
      "stair fall",
      "wet-floor fall",
      "sit down abruptly",
      "device dropped on table",
      "jogging",
      "curb step-off near-fall",
      "no-connectivity dispatch",
    ],
    features: [10],
    buildRank: 2,
    built: true,
    c: { crowd: 0.2, ambientDb: 55, light: 0.7, particulate: 0.1, motion: 0.9 },
  },
  {
    id: "C",
    name: "Indoor residential",
    blurb: "Kitchen, hallway, bedroom — where keys and bags actually land",
    stress: "Full GPS denial: pure IMU dead reckoning, drift grows with time",
    indoor: true,
    gps: 0.05,
    variants: [
      "sparse clutter",
      "high clutter + similar distractors",
      "window daylight dynamic range",
      "lamp-only low light",
      "drift over 10 min",
    ],
    features: [3, 4, 2],
    buildRank: 3,
    built: false,
    c: { crowd: 0.05, ambientDb: 42, light: 0.5, particulate: 0.05, motion: 0.3 },
  },
  {
    id: "E",
    name: "Indoor / outdoor transition",
    blurb: "Bus stop → café entrance → store interior as one continuous walk",
    stress: "Positioning quality changes at the threshold — breadcrumb nav's worst case",
    indoor: false,
    gps: 0.4,
    variants: [
      "GPS → no-GPS → GPS",
      "threshold detection",
      "wrong-turn injection mid-route",
      "multi-waypoint chained retrace",
    ],
    features: [2, 6],
    buildRank: 4,
    built: false,
    c: { crowd: 0.45, ambientDb: 64, light: 0.6, particulate: 0.1, motion: 0.35 },
  },
  {
    id: "D",
    name: "Retail & public buildings",
    blurb: "Grocery aisle, pharmacy counter, restaurant, lobby signage",
    stress: "Label glare, angled presentation, shelving clutter, ambient noise floor",
    indoor: true,
    gps: 0.1,
    variants: [
      "grocery aisle + cart actors",
      "pharmacy counter (dosage digits)",
      "restaurant menu + noise floor",
      "lobby signage / forms",
      "concurrent hazard during query",
    ],
    features: [4, 5, 6, 8, 9],
    buildRank: 5,
    built: false,
    c: { crowd: 0.5, ambientDb: 62, light: 0.65, particulate: 0.05, motion: 0.3 },
  },
  {
    id: "F",
    name: "Park / open plaza",
    blurb: "Unstructured space with no path edges or corridor geometry",
    stress: "Risk cone with nothing to key on; off-leash dogs as unpredictable actors",
    indoor: false,
    gps: 0.7,
    variants: [
      "empty plaza",
      "event-density crowd",
      "grass/gravel transition",
      "off-leash dog actor",
      "crosswalk logic run with no crosswalk (failure probe)",
    ],
    features: [1, 4, 7, 9],
    buildRank: 6,
    built: false,
    c: { crowd: 0.3, ambientDb: 58, light: 0.85, particulate: 0.1, motion: 0.3 },
  },
  {
    id: "G",
    name: "Construction / obstruction",
    blurb: "Sidewalk closure, temporary fencing, marked detour",
    stress: "Stale hazard pin vs live perception — plus a hazard-resolved scene state",
    indoor: false,
    gps: 0.75,
    variants: [
      "closure present",
      "hazard resolved (fencing removed)",
      "alternate route available",
      "no alternate — dead end",
    ],
    features: [8, 1],
    buildRank: 7,
    built: false,
    c: { crowd: 0.4, ambientDb: 90, light: 0.6, particulate: 0.85, motion: 0.5 },
  },
  {
    id: "B",
    name: "Transit interiors",
    blurb: "Platform edge, bus boarding, escalator and stairs",
    stress: "Hazard onset is distance to the edge, not TTC; PA noise floor buries alerts",
    indoor: true,
    gps: 0.08,
    variants: [
      "platform edge + tactile strip",
      "train arrival / departure",
      "bus door-closing timing",
      "escalator step-edge depth discontinuity",
      "crowd-density sweep",
    ],
    features: [1, 4, 10],
    buildRank: 8,
    built: false,
    c: { crowd: 0.75, ambientDb: 86, light: 0.55, particulate: 0.08, motion: 0.35 },
  },
];

export const ENV_BY_ID = Object.fromEntries(
  ENVIRONMENTS.map((e) => [e.id, e]),
) as Record<EnvId, EnvClass>;

/**
 * What the simulation canvas actually draws for each class. A retail aisle must
 * not be rendered as a street: the whole point of the catalog is that the scene
 * geometry, not just the noise sliders, changes with the class.
 */
export type SceneKind =
  | "street"
  | "transit"
  | "room"
  | "retail"
  | "threshold"
  | "plaza"
  | "worksite"
  | "rig";

export const SCENE_BY_ENV: Record<EnvId, SceneKind> = {
  A: "street",
  B: "transit",
  C: "room",
  D: "retail",
  E: "threshold",
  F: "plaza",
  G: "worksite",
  H: "rig",
};


/** Environments a given feature (journey number 01–11) must be validated in. */
export function environmentsForFeature(n: number): EnvClass[] {
  return ENVIRONMENTS.filter((e) => e.features.includes(n));
}

/** Positioning readout used by breadcrumb / object-memory sims. */
export function positioning(env: EnvClass): {
  label: string;
  tone: "ok" | "warn" | "bad";
} {
  if (env.gps >= 0.6) return { label: `GPS ${(env.gps * 100) | 0}% · fix held`, tone: "ok" };
  if (env.gps >= 0.3)
    return { label: `GPS ${(env.gps * 100) | 0}% · multipath, fix drifting`, tone: "warn" };
  return { label: "GPS denied · IMU dead reckoning only", tone: "bad" };
}
