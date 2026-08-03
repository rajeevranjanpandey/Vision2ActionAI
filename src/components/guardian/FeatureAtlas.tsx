import { useState } from "react";
import { AnimatePresence, LayoutGroup, motion, useReducedMotion } from "motion/react";
import { LiveDecisionDemo } from "@/components/guardian/LiveDecisionDemo";
import { ResultsDashboard } from "@/components/guardian/ResultsDashboard";
import { FallChannelPanel } from "@/components/guardian/FallChannelPanel";
import { SprintDashboard } from "@/components/guardian/SprintDashboard";
import {
  ConditionsBar,
  ConditionsProvider,
  ModalityBadges,
} from "@/components/guardian/SceneConditions";
import {
  BreadcrumbSim,
  CrossingSim,
  HumanLoopSim,
  ObjectMemorySim,
  PlacesSim,
  ReaderSim,
  RerouteSim,
  SceneAskSim,
  TranslationSim,
} from "@/components/guardian/FeatureSims";

type Release = "v1" | "v2" | "v3";
type Modality = "camera" | "mic" | "imu" | "gps";

interface Feature {
  n: number;
  title: string;
  release: Release;
  status: "built" | "specified";
  one: string;
  gap: string;
  kill: string;
  modes: Modality[];
  /** Bento weight: wide tiles carry the anchor features. */
  span: "sm" | "md" | "lg";
  sim: React.ReactNode;
  detail?: React.ReactNode;
  detailLabel?: string;
}

const RELEASE_LABEL: Record<Release, string> = {
  v1: "v1 · safety core",
  v2: "v2 · independence",
  v3: "v3 · world-connected",
};

const SPAN_CLASS: Record<Feature["span"], string> = {
  sm: "md:col-span-2",
  md: "md:col-span-3",
  lg: "md:col-span-4",
};

const FEATURES: Feature[] = [
  {
    n: 1,
    title: "Predictive hazard alerts",
    release: "v1",
    status: "built",
    one: "Forecasts a collision 1.5–3 s before contact instead of describing what is already there.",
    gap: "Every shipping app describes on demand. None runs a continuous safety loop.",
    kill: "Median lead time drops below 1.5 s, or false alarms exceed 2 / km.",
    modes: ["camera", "imu"],
    span: "lg",
    sim: <LiveDecisionDemo />,
    detailLabel: "Full calibration and conformal results",
    detail: <ResultsDashboard />,
  },
  {
    n: 2,
    title: "Traffic signal & crosswalk reading",
    release: "v1",
    status: "specified",
    one: "Reads walk / don't-walk state and crosswalk geometry — advisory only, with the geometric veto always on top.",
    gap: "Nothing on the market does this automatically and predictively.",
    kill: "A single case where a 'walk' reading suppresses a geometric hold.",
    modes: ["camera"],
    span: "sm",
    sim: <CrossingSim />,
  },
  {
    n: 10,
    title: "Fall detection + trusted contact",
    release: "v1",
    status: "built",
    one: "Three-phase IMU signature, spoken cancel window, then an SMS with GPS to a trusted contact.",
    gap: "A smartwatch feature that belongs on a device already worn all day.",
    kill: "More than ~1 false alert per week in the field.",
    modes: ["imu", "mic", "gps"],
    span: "sm",
    sim: <FallChannelPanel />,
  },
  {
    n: 3,
    title: "Find my way back",
    release: "v2",
    status: "specified",
    one: "Voice-tagged waypoints, retraced in reverse with the haptic channel that already exists.",
    gap: "Soundscape does spatial landmarks; nobody does a simple retrace.",
    kill: "Users stop dropping waypoints because tagging costs more than it returns.",
    modes: ["camera", "imu", "gps"],
    span: "md",
    sim: <BreadcrumbSim />,
  },
  {
    n: 4,
    title: "Where did I leave it",
    release: "v2",
    status: "specified",
    one: "Store a visual + spatial fingerprint of an object once; recall it later as a directional pulse.",
    gap: "No app has persistent object memory. Families name this as a daily friction point.",
    kill: "Recall accuracy below ~80 % in a lived-in room.",
    modes: ["camera"],
    span: "md",
    sim: <ObjectMemorySim />,
  },
  {
    n: 6,
    title: "Currency & document reading",
    release: "v2",
    status: "specified",
    one: "Mail, menus, labels, banknotes — read aloud through the VLM already in the pipeline.",
    gap: "Table stakes. Seeing AI's most-used feature.",
    kill: "Nothing kills it; it is a completeness requirement.",
    modes: ["camera"],
    span: "sm",
    sim: <ReaderSim />,
  },
  {
    n: 5,
    title: "Live scene description",
    release: "v2",
    status: "specified",
    one: "Tap-to-ask narration on the slow path: what's in front of me, read this, what colour is this.",
    gap: "Matches Be My AI, but shares the safety loop's world model.",
    kill: "Answer latency above ~3 s makes it unusable conversationally.",
    modes: ["camera", "mic"],
    span: "lg",
    sim: <SceneAskSim />,
  },
  {
    n: 7,
    title: "Places & errands",
    release: "v3",
    status: "specified",
    one: "Nearest pharmacy, opening hours, distance to the ATM — then a haptic handoff to the actual door.",
    gap: "Turns a safety device into an independence tool.",
    kill: "API reliability below what a person can plan an errand around.",
    modes: ["gps", "mic"],
    span: "md",
    sim: <PlacesSim />,
  },
  {
    n: 8,
    title: "Rerouting around known hazards",
    release: "v3",
    status: "specified",
    one: "A remembered hazard changes the route, not just the alert. Single-user until consent infrastructure exists.",
    gap: "Live alerting is reactive; routing is the preventative version of the same memory.",
    kill: "Detours that cost more walking than the hazard costs risk.",
    modes: ["gps", "camera"],
    span: "md",
    sim: <RerouteSim />,
    detailLabel: "Hazard memory, personal α, mode switching, federated loop",
    detail: <SprintDashboard />,
  },
  {
    n: 9,
    title: "Human-in-the-loop fallback",
    release: "v3",
    status: "specified",
    one: "When the model is uncertain, it says so and connects a sighted assistant instead of guessing.",
    gap: "Users of Aira and Be My Eyes name the human as their trust anchor.",
    kill: "Escalation rate high enough to be unaffordable, or low enough to be dishonest.",
    modes: ["camera", "mic"],
    span: "sm",
    sim: <HumanLoopSim />,
  },
  {
    n: 11,
    title: "Live translation",
    release: "v3",
    status: "specified",
    one: "Translates what people say to you and what signs and menus say around you — speech through the mic, print through the camera.",
    gap: "Blindness plus a language you don't read removes the two workarounds sighted travellers rely on: reading the sign, or asking someone.",
    kill: "Mistranslation of a safety-relevant instruction, or a delay long enough that the conversation has moved on.",
    modes: ["mic", "camera"],
    span: "lg",
    sim: <TranslationSim />,
  },
];

export function FeatureAtlas() {
  const [openN, setOpenN] = useState<number>(1);
  const reduce = useReducedMotion();

  return (
    <ConditionsProvider>
      <div className="space-y-14">
        <motion.div
          initial={reduce ? false : { opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          <p className="font-mono text-xs uppercase tracking-[0.25em]">
            Set the street first
          </p>
          <p className="mb-3 mt-2 max-w-2xl text-sm text-muted-foreground">
            Every simulation below runs inside these conditions. Crowds occlude signal
            heads, dust and dusk break the camera, and an 88 dB platform drowns speech —
            so the sims degrade the same way the device would.
          </p>
          <ConditionsBar />
        </motion.div>

        <LayoutGroup>
          {(["v1", "v2", "v3"] as Release[]).map((rel) => (
            <section key={rel} className="space-y-4">
              <motion.div
                initial={reduce ? false : { opacity: 0, x: -12 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: "-60px" }}
                transition={{ duration: 0.45 }}
                className="flex items-center gap-4"
              >
                <p className="font-mono text-xs uppercase tracking-[0.25em]">
                  {RELEASE_LABEL[rel]}
                </p>
                <span className="h-px flex-1 bg-border" />
              </motion.div>

              <div className="grid grid-cols-1 gap-4 md:auto-rows-auto md:grid-cols-6">
                {FEATURES.filter((f) => f.release === rel).map((f, i) => {
                  const open = openN === f.n;
                  return (
                    <motion.article
                      layout={!reduce}
                      key={f.n}
                      initial={reduce ? false : { opacity: 0, y: 24 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true, margin: "-60px" }}
                      transition={{
                        duration: 0.5,
                        delay: reduce ? 0 : i * 0.06,
                        ease: [0.16, 1, 0.3, 1],
                        layout: { duration: 0.4, ease: [0.16, 1, 0.3, 1] },
                      }}
                      className={`group relative overflow-hidden rounded-xl border bg-card ${
                        open
                          ? "md:col-span-6 border-foreground shadow-[0_18px_50px_-28px_oklch(0_0_0/0.55)]"
                          : `${SPAN_CLASS[f.span]} border-border hover:border-foreground/40`
                      }`}
                      whileHover={reduce || open ? {} : { y: -4 }}
                    >
                      <button
                        onClick={() => setOpenN(open ? -1 : f.n)}
                        aria-expanded={open}
                        className="relative flex w-full flex-col items-start gap-3 p-5 text-left"
                      >
                        <span className="flex w-full items-center gap-3">
                          <span className="font-mono text-xs tabular-nums text-muted-foreground">
                            {String(f.n).padStart(2, "0")}
                          </span>
                          <span
                            className={`rounded-full border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider ${
                              f.status === "built"
                                ? "border-foreground bg-foreground text-background"
                                : "border-border text-muted-foreground"
                            }`}
                          >
                            {f.status}
                          </span>
                          <motion.span
                            animate={{ rotate: open ? 45 : 0 }}
                            transition={{ duration: 0.25 }}
                            className="ml-auto grid h-6 w-6 shrink-0 place-items-center rounded-full border border-border font-mono text-xs"
                            aria-hidden
                          >
                            +
                          </motion.span>
                        </span>
                        <span className="block text-lg font-semibold leading-tight tracking-tight">
                          {f.title}
                        </span>
                        <span className="block text-sm leading-relaxed text-muted-foreground">
                          {f.one}
                        </span>
                        <span
                          className="absolute bottom-0 left-0 h-px w-full origin-left scale-x-0 bg-foreground transition-transform duration-500 group-hover:scale-x-100"
                          aria-hidden
                        />
                      </button>

                      <AnimatePresence initial={false}>
                        {open && (
                          <motion.div
                            key="body"
                            initial={reduce ? false : { opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: "auto" }}
                            exit={{ opacity: 0, height: 0 }}
                            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                            className="overflow-hidden border-t border-border"
                          >
                            <div className="p-5">
                              {/* Simulation always comes first. */}
                              <div className="flex flex-wrap items-center justify-between gap-2">
                                <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
                                  Simulation
                                </p>
                                <ModalityBadges modes={f.modes} />
                              </div>
                              <motion.div
                                initial={reduce ? false : { opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.4, delay: 0.1 }}
                                className="scanline relative mt-3 rounded-lg"
                              >
                                {f.sim}
                              </motion.div>

                              <dl className="mt-6 grid gap-4 sm:grid-cols-2">
                                <div className="rounded-lg bg-surface p-4">
                                  <dt className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
                                    Why it earns its place
                                  </dt>
                                  <dd className="mt-1 text-sm text-muted-foreground">
                                    {f.gap}
                                  </dd>
                                </div>
                                <div className="rounded-lg bg-surface p-4">
                                  <dt className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
                                    What would kill it
                                  </dt>
                                  <dd className="mt-1 text-sm text-muted-foreground">
                                    {f.kill}
                                  </dd>
                                </div>
                              </dl>

                              {f.detail && (
                                <details className="mt-6 rounded-lg border border-border bg-surface p-4">
                                  <summary className="cursor-pointer text-sm text-muted-foreground">
                                    {f.detailLabel ?? "Evaluation detail"}
                                  </summary>
                                  <div className="mt-6">{f.detail}</div>
                                </details>
                              )}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </motion.article>
                  );
                })}
              </div>
            </section>
          ))}
        </LayoutGroup>
      </div>
    </ConditionsProvider>
  );
}
