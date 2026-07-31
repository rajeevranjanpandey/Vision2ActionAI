/**
 * Deterministic replays of the scenarios the risk head is trained on.
 *
 * Same physics as `guardian/guardian/train/simulate.py`, with the measurement noise
 * turned off so the demo shows the decision boundary rather than a different noise draw
 * on every play. Ego motion is folded into the relative velocity, exactly as the
 * on-device tracker reports it after ego compensation.
 */

import type { TrackState } from "@/lib/riskHead";

export const DT = 0.1;

export interface Scenario {
  id: string;
  label: string;
  className: string;
  widthM: number;
  /** What a reviewer should watch for. */
  note: string;
  /** True if the trajectory really does reach the ego cylinder. */
  hazardous: boolean;
  states: TrackState[];
}

function integrate(
  x: number,
  z: number,
  vx: number,
  vz: number,
  steps: number,
  className: string,
  widthM: number,
  accel?: (t: number) => [number, number],
  stopWhenSlow = false,
): TrackState[] {
  const out: TrackState[] = [];
  for (let i = 0; i < steps; i++) {
    out.push({ x, z, vx, vz, widthM, className });
    const [ax, az] = accel ? accel(i * DT) : [0, 0];
    vx += ax * DT;
    vz += az * DT;
    if (stopWhenSlow && Math.hypot(vx, vz) < 0.12) {
      vx = 0;
      vz = 0;
    }
    x += vx * DT;
    z += vz * DT;
  }
  return out;
}

const STEPS = 55;

function breaches(states: TrackState[]): boolean {
  return states.some(
    (s) => Math.hypot(s.x, s.z) <= 1.0 + 0.5 * s.widthM,
  );
}

function make(
  id: string,
  label: string,
  note: string,
  states: TrackState[],
): Scenario {
  const first = states[0]!;
  return {
    id,
    label,
    note,
    className: first.className,
    widthM: first.widthM,
    hazardous: breaches(states),
    states,
  };
}

const crossingCyclist = make(
  "crossing_cyclist",
  "Crossing cyclist",
  "Constant velocity is sufficient here: the geometric kernel fires on its own and the head agrees. The head's job is to not get in the way.",
  integrate(-5.2, 9.5, 1.35, -2.45, STEPS, "bicycle", 0.6),
);

const yieldingPedestrian = make(
  "yielding_pedestrian",
  "Pedestrian who yields",
  "The hard case. For the first second this is indistinguishable from a collision course, so the constant-velocity baseline warns. The deceleration signature inside the 0.8 s window is the only evidence that it will stop — and it is the evidence the head learns.",
  integrate(
    3.1,
    6.4,
    -1.45,
    -0.72,
    STEPS,
    "person",
    0.5,
    (t) => (t >= 1.0 ? [1.45, 0.72] : [0, 0]),
    true,
  ),
);

const staticObstacle = make(
  "static_obstacle",
  "A-board in the corridor",
  "Zero closing speed of its own: only the walker's 1.3 m/s closes the gap. Systems that key on object motion miss this entirely.",
  integrate(0.15, 7.5, 0, -1.3, STEPS, "bollard", 0.7),
);

const nearMissVehicle = make(
  "near_miss_vehicle",
  "Vehicle passing wide",
  "Fast, loud, and irrelevant. This is the alarm-fatigue scenario: every false warning here costs trust that a real warning later needs.",
  integrate(6.4, 11.0, -0.12, -6.0, STEPS, "car", 1.8),
);

const cutInScooter = make(
  "cut_in_scooter",
  "Scooter cutting in",
  "Overtakes on one side, then turns in. The curvature only becomes visible mid-window, which is why the head sees a window and not a frame.",
  integrate(2.4, 10.0, 0, -3.0, STEPS, "scooter", 0.7, (t) =>
    t >= 0.9 ? [-1.5, 0] : [0, 0],
  ),

);

export const SCENARIOS: Scenario[] = [
  yieldingPedestrian,
  crossingCyclist,
  cutInScooter,
  staticObstacle,
  nearMissVehicle,
];
