import { createFileRoute } from "@tanstack/react-router";
import { LiveDecisionDemo } from "@/components/guardian/LiveDecisionDemo";
import { ResultsDashboard } from "@/components/guardian/ResultsDashboard";
import { FallChannelPanel } from "@/components/guardian/FallChannelPanel";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "AI Guardian — Predictive Mobility Assistance for Blind Pedestrians" },
      {
        name: "description",
        content:
          "A wearable vision-language-action system that forecasts hazards 2-3 seconds before contact. Two-path architecture, metric monocular depth, and GuardianBench lead-time evaluation.",
      },
      {
        property: "og:title",
        content: "AI Guardian — Predictive Mobility Assistance for Blind Pedestrians",
      },
      {
        property: "og:description",
        content:
          "A wearable vision-language-action system that forecasts hazards 2-3 seconds before contact, evaluated on warning lead time rather than detection accuracy.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

const fastPath = [
  { stage: "Detect", detail: "RT-DETR-l, INT8 TensorRT", ms: 25 },
  { stage: "Depth", detail: "Depth Anything V2-S + ground-plane scale", ms: 30 },
  { stage: "Gate", detail: "Risk-corridor pre-filter", ms: 1 },
  { stage: "Segment", detail: "SAM 2 tiny, ≤4 gated prompts", ms: 25 },
  { stage: "Track", detail: "Ego-compensated Kalman, Hungarian match", ms: 5 },
  { stage: "Decide", detail: "TTC, forecast rollout, alert policy", ms: 2 },
];

const metrics = [
  { label: "Lead time (median)", value: "> 2.0 s", note: "before the 1 m ego cylinder" },
  { label: "Lead time (p10)", value: "> 1.0 s", note: "the tail is the safety number" },
  { label: "False alarms", value: "< 2 / km", note: "or the user stops wearing it" },
  { label: "Fast path p99", value: "< 100 ms", note: "on-device, thermally soaked" },
];

const modules = [
  {
    path: "perception/",
    blurb:
      "Detection, monocular depth, and sparse SAM 2 segmentation. Ground-plane RANSAC recovers metric scale from relative depth using known camera height and IMU pitch.",
  },
  {
    path: "tracking/",
    blurb:
      "Constant-velocity Kalman filters in the ego frame with explicit ego-motion compensation, associated by the Hungarian algorithm on 3D distance.",
  },
  {
    path: "risk/",
    blurb:
      "The predictive kernel: closing speed, time-to-collision against a widening risk cone, 3 s trajectory rollout, and alert arbitration with hysteresis and cooldown.",
  },
  {
    path: "language/",
    blurb:
      "Slow-path Qwen2.5-VL narration on a background thread at 0.5–1 Hz. Advisory only — it can never delay or veto a geometric warning.",
  },
  {
    path: "audio/",
    blurb:
      "Whisper intent capture, Piper speech, spatialised earcons, and three-channel directional haptics. Haptic fires first, speech last.",
  },
  {
    path: "bench/",
    blurb:
      "GuardianBench loader and the lead-time metrics: recall, false alarms per kilometre, and the late-alert rate that mAP cannot express.",
  },
];

function Index() {
  return (
    <main className="min-h-screen bg-background text-foreground">
      {/* Hero */}
      <section className="grid-field border-b border-border">
        <div className="mx-auto max-w-5xl px-6 py-24 sm:py-32">
          <p className="font-mono text-xs uppercase tracking-[0.25em] text-signal">
            Master&apos;s research project · assistive perception
          </p>
          <h1 className="mt-6 text-4xl font-bold leading-[1.05] sm:text-6xl">
            AI Guardian
          </h1>
          <p className="mt-4 max-w-2xl text-xl text-muted-foreground sm:text-2xl">
            A vision-language-action system for{" "}
            <span className="text-foreground">predictive</span> assistance in the
            real-world mobility of visually impaired pedestrians.
          </p>

          <div className="mt-10 max-w-2xl border-l-2 border-signal pl-5">
            <p className="text-base leading-relaxed text-muted-foreground">
              Existing assistive vision describes what is already there. A blind
              pedestrian does not need to be told about the cyclist at the moment of
              contact — they need to know 2.5 seconds earlier, while there is still time
              to stop. This system forecasts hazards instead of reporting them, and is
              evaluated on <span className="text-foreground">warning lead time</span>,
              not detection accuracy.
            </p>
          </div>

          <dl className="mt-12 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-border bg-border sm:grid-cols-4">
            {metrics.map((m) => (
              <div key={m.label} className="bg-surface p-5">
                <dd className="font-mono text-2xl font-medium text-signal">{m.value}</dd>
                <dt className="mt-2 text-sm font-medium">{m.label}</dt>
                <p className="mt-1 text-xs text-muted-foreground">{m.note}</p>
              </div>
            ))}
          </dl>
        </div>
      </section>

      {/* Architecture */}
      <section className="border-b border-border">
        <div className="mx-auto max-w-5xl px-6 py-20">
          <h2 className="text-3xl font-bold">Two-path architecture</h2>
          <p className="mt-3 max-w-2xl text-muted-foreground">
            Safety and semantics have incompatible latency budgets. Running them in one
            loop means either a sluggish warning or a shallow description. So they are
            separate processes that never block each other.
          </p>

          <div className="mt-10 grid gap-6 lg:grid-cols-[3fr_2fr]">
            <div className="rounded-lg border border-border bg-surface p-6">
              <div className="flex items-baseline justify-between">
                <h3 className="text-lg font-semibold">Fast path</h3>
                <span className="font-mono text-xs text-signal">10 Hz · hard deadline</span>
              </div>
              <p className="mt-2 text-sm text-muted-foreground">
                Pure geometry. Never waits on a language model.
              </p>
              <ul className="mt-6 space-y-px">
                {fastPath.map((s) => (
                  <li
                    key={s.stage}
                    className="flex items-center gap-4 rounded bg-surface-2 px-4 py-3"
                  >
                    <span className="w-20 shrink-0 font-mono text-sm font-medium">
                      {s.stage}
                    </span>
                    <span className="flex-1 text-sm text-muted-foreground">{s.detail}</span>
                    <span className="font-mono text-sm text-signal">{s.ms} ms</span>
                  </li>
                ))}
              </ul>
              <div className="mt-4 flex justify-between border-t border-border pt-4 font-mono text-sm">
                <span className="text-muted-foreground">budget</span>
                <span className="text-signal">&lt; 90 ms of the 100 ms tick</span>
              </div>
            </div>

            <div className="space-y-6">
              <div className="rounded-lg border border-border bg-surface p-6">
                <div className="flex items-baseline justify-between">
                  <h3 className="text-lg font-semibold">Slow path</h3>
                  <span className="font-mono text-xs text-muted-foreground">
                    0.5–1 Hz · best effort
                  </span>
                </div>
                <p className="mt-3 text-sm text-muted-foreground">
                  Qwen2.5-VL on a background thread supplies scene context and answers
                  spoken questions. Its output is advisory: it enriches a warning that
                  the geometry has already decided to give, and it can never delay one.
                </p>
              </div>

              <div className="rounded-lg border border-urgent/40 bg-urgent/5 p-6">
                <h3 className="text-lg font-semibold">Degraded mode</h3>
                <p className="mt-3 text-sm text-muted-foreground">
                  If the fast path overruns, or ground-plane confidence collapses so that
                  metres are no longer metres, the device says so out loud. A safety
                  device that fails silently is worse than no device, because the user has
                  already adapted their behaviour to trust it.
                </p>
                <p className="mt-4 font-mono text-sm text-urgent">
                  &ldquo;Guardian degraded. Rely on your cane.&rdquo;
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Live decision layer + measured results */}
      <section id="results" className="border-b border-border">
        <div className="mx-auto max-w-6xl space-y-12 px-6 py-20">
          <div>
            <h2 className="text-3xl font-bold">The decision layer, measured</h2>
            <p className="mt-3 max-w-3xl text-muted-foreground">
              A learned risk head sits on top of the geometric safety kernel: an 8-frame,
              14-feature window per track, a deep ensemble trained with focal loss,
              temperature scaling for calibration, and split-conformal thresholds for a
              distribution-free miss-rate budget. Geometry keeps the veto on imminent
              contact; the head is only allowed to quiet the device when it is both
              confident and internally in agreement.
            </p>
          </div>
          <LiveDecisionDemo />
          <ResultsDashboard />
        </div>
      </section>

      {/* Post-incident channel */}
      <section id="fall" className="border-b border-border">
        <div className="mx-auto max-w-6xl space-y-10 px-6 py-20">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.25em] text-signal">
              Async channel · sprint item 3
            </p>
            <h2 className="mt-4 text-3xl font-bold">When prevention fails</h2>
            <p className="mt-3 max-w-3xl text-muted-foreground">
              The fast path exists to stop a collision. This channel exists for the case
              where it did not. An IMU classifier — reusing the same inertial stream that
              already supplies ground-plane pitch, so the sensor cost is zero — watches
              for the three-phase signature of a fall: a free-fall dip, an impact peak,
              then a rotated gravity vector that stops moving. Confirmation opens a spoken
              cancel window before anyone is contacted, and the whole thing runs off the
              90 ms budget entirely.
            </p>
          </div>
          <FallChannelPanel />
        </div>
      </section>





      {/* Contributions */}
      <section className="border-b border-border bg-surface/40">
        <div className="mx-auto max-w-5xl px-6 py-20">
          <h2 className="text-3xl font-bold">What is actually new</h2>
          <div className="mt-10 grid gap-6 md:grid-cols-3">
            {[
              {
                n: "01",
                title: "Prediction, not description",
                body: "An anticipatory risk formulation over forecast trajectories in a metric ego frame, so the warning arrives while avoidance is still possible.",
              },
              {
                n: "02",
                title: "Metric scale from one camera",
                body: "Ground-plane RANSAC over relative depth, anchored by known camera height and IMU pitch, turns a scale-ambiguous depth map into metres — with a confidence signal that gates the whole system.",
              },
              {
                n: "03",
                title: "GuardianBench",
                body: "An egocentric pedestrian dataset labelled by O&M instructors with hazard onset: the frame at which a sighted guide would have intervened. That label is what makes lead time measurable at all.",
              },
            ].map((c) => (
              <div key={c.n} className="rounded-lg border border-border bg-surface p-6">
                <span className="font-mono text-sm text-signal">{c.n}</span>
                <h3 className="mt-3 text-lg font-semibold">{c.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{c.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Repository */}
      <section className="border-b border-border">
        <div className="mx-auto max-w-5xl px-6 py-20">
          <h2 className="text-3xl font-bold">The repository</h2>
          <p className="mt-3 text-muted-foreground">
            Full implementation under <code className="font-mono text-signal">guardian/</code>,
            with 49 tests over the safety-critical maths.
          </p>

          <div className="mt-10 grid gap-px overflow-hidden rounded-lg border border-border bg-border md:grid-cols-2">
            {modules.map((m) => (
              <div key={m.path} className="bg-surface p-6">
                <h3 className="font-mono text-sm text-signal">{m.path}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                  {m.blurb}
                </p>
              </div>
            ))}
          </div>

          <div className="mt-8 overflow-x-auto rounded-lg border border-border bg-surface-2 p-6">
            <pre className="font-mono text-sm leading-relaxed text-muted-foreground">
              <code>{`# run the wearable loop
python scripts/run_live.py --config configs/default.yaml --source 0

# score lead time and false alarms on GuardianBench
python scripts/eval.py --data data/guardianbench --split test

# the ablation table the paper needs
python scripts/eval.py --ablate

# per-stage p50/p95/p99 latency on the Jetson
python -m guardian.deploy.benchmark --frames 600`}</code>
            </pre>
          </div>
        </div>
      </section>

      {/* Ethics */}
      <section>
        <div className="mx-auto max-w-5xl px-6 py-20">
          <h2 className="text-3xl font-bold">Constraints that are not negotiable</h2>
          <ul className="mt-8 space-y-4">
            {[
              "The device is additive. It never replaces a white cane or guide dog, and the consent form says so plainly.",
              "Blind users are co-designers, not test subjects. The alert vocabulary is reviewed by the people who will hear it under traffic noise before it is frozen.",
              "Latency is reported from the Orin under thermal load, as p50/p95/p99. Desktop means are not evidence.",
              "Recorded routes contain bystander faces and a participant's daily movements. That data never enters version control.",
              "Failure cases are published, including at least one near-miss the system missed.",
            ].map((line) => (
              <li key={line} className="flex gap-4 border-l-2 border-border pl-5">
                <p className="text-muted-foreground">{line}</p>
              </li>
            ))}
          </ul>

          <p className="mt-14 border-t border-border pt-8 font-mono text-xs text-muted-foreground">
            AI Guardian · research prototype · not a certified mobility aid
          </p>
        </div>
      </section>
    </main>
  );
}
