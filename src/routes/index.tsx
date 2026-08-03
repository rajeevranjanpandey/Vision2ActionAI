import { createFileRoute } from "@tanstack/react-router";
import { motion, useReducedMotion, useScroll, useSpring } from "motion/react";
import { GuardianConsole } from "@/components/guardian/GuardianConsole";
import { SafetyLoop } from "@/components/guardian/SafetyLoop";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "AI Guardian — Predictive Mobility Assistance for Blind Pedestrians" },
      {
        name: "description",
        content:
          "A wearable vision-language-action system for blind pedestrians: ten features across three releases, each with a runnable simulation, anchored on a predictive 10 Hz safety loop.",
      },
      {
        property: "og:title",
        content: "AI Guardian — Predictive Mobility Assistance for Blind Pedestrians",
      },
      {
        property: "og:description",
        content:
          "Ten features, three releases, one rule: nothing is allowed to slow the safety loop. Each feature ships with a simulation you can run.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

const CONTAINER = {
  hidden: {},
  show: { transition: { staggerChildren: 0.09, delayChildren: 0.05 } },
};

const RISE = {
  hidden: { opacity: 0, y: 22 },
  show: { opacity: 1, y: 0 },
};


function Index() {
  const reduce = useReducedMotion();
  const { scrollYProgress } = useScroll();
  const progress = useSpring(scrollYProgress, { stiffness: 90, damping: 24, mass: 0.3 });

  return (
    <main className="min-h-screen bg-background text-foreground">
      {/* Scroll progress rule */}
      <motion.div
        style={{ scaleX: progress }}
        className="fixed inset-x-0 top-0 z-50 h-[2px] origin-left bg-foreground"
        aria-hidden
      />

      {/* Hero */}
      <section className="grid-field grid-drift relative overflow-hidden border-b border-border">
        <motion.div
          variants={CONTAINER}
          initial={reduce ? false : "hidden"}
          animate="show"
          className="mx-auto grid max-w-[104rem] items-end gap-8 px-6 py-14 lg:grid-cols-[1.35fr_1fr] sm:py-16"
        >

          <div>
          <motion.h1
            variants={RISE}
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
            className="text-6xl font-bold leading-[0.92] tracking-[-0.045em] sm:text-8xl"
          >
            AI Guardian
          </motion.h1>

          <motion.p
            variants={RISE}
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
            className="mt-6 max-w-3xl text-xl font-semibold leading-snug text-foreground sm:text-2xl"
          >
            Predictive vision–language–action assistance for real-world mobility of visually
            impaired pedestrians.
          </motion.p>


          <motion.p
            variants={RISE}
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
            className="mt-6 inline-block rounded-full border border-veto bg-veto-soft px-4 py-2 text-sm"
          >
            <span className="font-semibold text-veto">Advisory assistance only.</span>{" "}
            <span className="text-muted-foreground">
              User judgment and the geometric safety veto remain absolute.
            </span>
          </motion.p>
          </div>

          <motion.dl
            variants={RISE}
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
            className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-2"
          >
            {[
              { v: "> 2.0 s", l: "median lead time", c: "border-urgent bg-urgent-soft text-urgent" },
              { v: "< 2 / km", l: "false alarms", c: "border-veto bg-veto-soft text-veto" },
              { v: "30 ms", l: "fast-path tick", c: "border-sense bg-sense-soft text-sense" },
              { v: "11", l: "features, one spine", c: "border-guide bg-guide-soft text-guide" },
            ].map((m) => (
              <div key={m.l} className={`rounded-2xl border p-4 ${m.c}`}>
                <dt className="font-mono text-2xl font-bold tabular-nums tracking-tight">
                  {m.v}
                </dt>
                <dd className="mt-1 text-xs uppercase tracking-wider text-muted-foreground">
                  {m.l}
                </dd>
              </div>
            ))}
          </motion.dl>
        </motion.div>
      </section>

      {/* Console */}
      <section id="features" className="border-b border-border">
        <div className="mx-auto max-w-[104rem] px-6 py-10">
          <GuardianConsole />
        </div>
      </section>

      {/* How it holds together */}
      <section className="border-b border-border bg-surface">
        <div className="mx-auto max-w-6xl px-6 py-14">
          <motion.h2
            initial={reduce ? false : { opacity: 0, y: 18 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-80px" }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
            className="text-4xl font-bold tracking-tight sm:text-5xl"
          >
            How it holds together
          </motion.h2>
          <div className="mt-10 grid gap-4 md:grid-cols-3">
            {[
              {
                t: "Fast path · 10 Hz",
                c: "border-urgent bg-urgent-soft",
                b: "Detect, metric depth, track, decide. Pure geometry, hard deadline, keeps an absolute veto over every other signal in the system.",
              },
              {
                t: "Slow path · 0.5–1 Hz",
                c: "border-assist bg-assist-soft",
                b: "Vision-language narration, signal reading, reading text aloud. Advisory only. It enriches a warning; it can never delay or cancel one.",
              },
              {
                t: "Async channels",
                c: "border-memory bg-memory-soft",
                b: "Fall detection, hazard memory, breadcrumbs, federated feedback. They write parameters the fast path reads, and never run inside its tick.",
              },
            ].map((c, i) => (
              <motion.div
                key={c.t}
                initial={reduce ? false : { opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-60px" }}
                transition={{ duration: 0.5, delay: i * 0.08, ease: [0.16, 1, 0.3, 1] }}
                whileHover={reduce ? {} : { y: -4 }}
                className={`rounded-2xl border p-6 ${c.c}`}
              >
                <h3 className="font-mono text-sm">{c.t}</h3>
                <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{c.b}</p>
              </motion.div>
            ))}
          </div>
          <div className="mt-14">
            <SafetyLoop />
          </div>
        </div>
      </section>

      {/* Constraints */}
      <section>
        <div className="mx-auto max-w-6xl px-6 py-12">
          <motion.h2
            initial={reduce ? false : { opacity: 0, y: 18 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-80px" }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
            className="text-3xl font-bold tracking-tight sm:text-4xl"
          >
            Rules we don't break
          </motion.h2>
          <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {[
              { t: "It adds, never replaces", b: "The white cane and the guide dog stay." },
              { t: "Geometry has the last word", b: "A “walk” sign can never override a turning car." },
              { t: "Blind users co-design it", b: "Not test subjects — authors of the behaviour." },
              { t: "Speed is measured honestly", b: "On the real device, hot, at p50/p95/p99." },
              { t: "Failures get published", b: "Including a near-miss the system missed." },
            ].map((r, i) => (
              <motion.div
                key={r.t}
                initial={reduce ? false : { opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-40px" }}
                transition={{ duration: 0.45, delay: i * 0.05 }}
                className="rounded-2xl border border-border bg-surface p-5"
              >
                <p className="font-mono text-[11px] tabular-nums text-muted-foreground">
                  {String(i + 1).padStart(2, "0")}
                </p>
                <h3 className="mt-1 text-base font-semibold">{r.t}</h3>
                <p className="mt-1 text-sm leading-snug text-muted-foreground">{r.b}</p>
              </motion.div>
            ))}
          </div>
          <p className="mt-8 font-mono text-xs text-muted-foreground">
            AI Guardian · research prototype · not a certified mobility aid
          </p>
        </div>
      </section>


    </main>
  );
}
