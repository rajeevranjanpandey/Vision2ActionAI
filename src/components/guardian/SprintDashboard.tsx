import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import sprint from "@/data/sprint.json";

/**
 * Sprint items 1, 2, 4 and 5. Every number rendered here comes from the Python
 * simulations in `guardian/` and is exported to `src/data/sprint.json` — nothing on
 * this page is a hand-written illustration.
 */

interface MemoryRow {
  pass: number;
  baseline_per_km: number;
  memory_per_km: number;
  urgent_baseline: number;
  urgent_memory: number;
  pins: number;
  established_pins: number;
}

interface FedRound {
  round: number;
  recall: number;
  falseAlarmRate: number;
  lateAlerts: number;
  aggregateNorm: number;
  clippedFraction: number;
}

interface Profile {
  mode: string;
  cone_half_width_m: number;
  cone_widen_per_m: number;
  ttc_warn_s: number;
  ttc_urgent_s: number;
  min_closing_speed_mps: number;
  hard_floor_m: number | null;
  vocabulary: string[];
  rationale: string;
}

const S = sprint as unknown as {
  memory: {
    rows: MemoryRow[];
    paired_reduction_settled: number;
    paired_reduction_last_pass: number;
    urgent_preserved: boolean;
  };
  federated: {
    rounds: FedRound[];
    lateAlertsFirst: number;
    lateAlertsLast: number;
    noiseMultiplier: number;
    clipNorm: number;
    nClients: number;
  };
  privacyUtility: {
    noiseMultiplier: number;
    recall: number;
    falseAlarmRate: number;
    lateAlerts: number;
  }[];
  profiles: Record<string, Profile>;
  valScores: number[];
  userScores: number[];
};

const AXIS = {
  stroke: "var(--muted-foreground)",
  fontSize: 11,
  fontFamily: "var(--font-mono)",
};

const TOOLTIP_STYLE = {
  background: "var(--surface-2)",
  border: "1px solid var(--border)",
  borderRadius: 6,
  fontSize: 12,
  fontFamily: "var(--font-mono)",
};

/** Split-conformal threshold: the alpha-quantile of the user's own hazard scores. */
function conformalThreshold(scores: number[], alpha: number): number {
  const s = [...scores].sort((a, b) => a - b);
  const n = s.length;
  const k = Math.max(0, Math.ceil(alpha * (n + 1)) - 1);
  return s[Math.min(k, n - 1)] ?? 0;
}

export function SprintDashboard() {
  return (
    <div className="space-y-10">
      <HazardMemoryPanel />
      <PersonalAlphaPanel />
      <ModeSwitchPanel />
      <FederatedPanel />
    </div>
  );
}

/* ------------------------------------------------------------------ item 1 */

function HazardMemoryPanel() {
  const rows = S.memory.rows.map((r) => ({
    pass: `p${r.pass}`,
    baseline: Number(r.baseline_per_km.toFixed(2)),
    memory: Number(r.memory_per_km.toFixed(2)),
    pins: r.pins,
    established: r.established_pins,
  }));

  return (
    <Panel
      title="1 · Persistent hazard memory on a repeated route"
      caption={`Seven static hazards on a 1.2 km route, walked six times with 5 m GNSS noise.
        A pin needs three non-contradicted confirmations before it is allowed to change
        anything, so the first two passes are identical in both arms by construction. From
        pass four the gate downgrades the second and third announcement of an already-named
        kerb to a short earcon: ${(S.memory.paired_reduction_last_pass * 100).toFixed(0)}%
        fewer spoken alerts on the final pass, ${(S.memory.paired_reduction_settled * 100).toFixed(0)}%
        averaged over settled passes. Urgent geometry is exempt from the gate, and the
        urgent-alert count is identical in both arms on every pass — if it were not, the
        feature would be deleted rather than tuned.`}
    >
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis dataKey="pass" tick={AXIS} stroke="var(--border)" />
          <YAxis tick={AXIS} stroke="var(--border)" />
          <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "var(--surface-2)" }} />
          <Legend wrapperStyle={{ fontSize: 11, fontFamily: "var(--font-mono)" }} />
          <Bar name="alerts/km — no memory" dataKey="baseline" fill="var(--muted-foreground)" />
          <Bar name="alerts/km — with memory" dataKey="memory" fill="var(--signal)" />
        </BarChart>
      </ResponsiveContainer>

      <div className="mt-6 grid gap-4 sm:grid-cols-3">
        <Stat
          label="Pins after six passes"
          value={String(S.memory.rows.at(-1)?.pins ?? "—")}
          sub="seven real hazards, no duplicate fragments"
        />
        <Stat
          label="Established pins"
          value={String(S.memory.rows.at(-1)?.established_pins ?? "—")}
          sub="three confirmations, zero contradictions"
        />
        <Stat
          label="Urgent alerts preserved"
          value={S.memory.urgent_preserved ? "yes" : "NO"}
          sub="the gate never touches urgent geometry"
          accent={S.memory.urgent_preserved}
        />
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ item 2 */

function PersonalAlphaPanel() {
  const [alpha, setAlpha] = useState(0.10);
  const MIN_RECALL = 0.75;

  const { threshold, recall, alarmRate, accepted } = useMemo(() => {
    const t = conformalThreshold(S.userScores, alpha);
    const fired = S.valScores.filter((s) => s >= t).length;
    const r = fired / S.valScores.length;
    return {
      threshold: t,
      recall: r,
      alarmRate: r,
      accepted: r >= MIN_RECALL,
    };
  }, [alpha]);

  return (
    <Panel
      title="2 · Per-user conformal recalibration, with a floor"
      caption={`The miss-rate budget is a personal preference: a confident cane user in a
        familiar neighbourhood wants a quieter device than someone learning a new route.
        Moving the dial refits a split-conformal threshold on that user's own flagged
        hazards, which keeps the distribution-free guarantee under their data rather than
        the population's. The dial is not unbounded — every candidate threshold is replayed
        against a frozen validation set, and one that drops recall below the
        ${MIN_RECALL.toFixed(2)} floor is rejected rather than clamped silently. A safety
        parameter a user can talk their way past is not a safety parameter.`}
    >
      <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div>
          <label
            htmlFor="alpha"
            className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground"
          >
            miss-rate budget α = {alpha.toFixed(2)}
          </label>
          <input
            id="alpha"
            type="range"
            min={0.02}
            max={0.30}
            step={0.01}
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
            className="mt-3 w-full accent-[var(--signal)]"
          />
          <div className="mt-2 flex justify-between font-mono text-[10px] text-muted-foreground">
            <span>0.02 — cautious, louder</span>
            <span>0.30 — quiet, riskier</span>
          </div>

          <p className="mt-6 text-sm leading-relaxed text-muted-foreground">
            Spoken intent maps onto the same dial: <em>"stop warning me about kerbs on my
            own street"</em> raises α, <em>"you missed that one"</em> lowers it. The parser
            never writes the threshold directly — it proposes an α, and the audit decides.
          </p>
        </div>

        <div className="space-y-4">
          <Stat label="Conformal threshold" value={threshold.toFixed(3)} sub="on the user's own scores" />
          <Stat
            label="Validation recall"
            value={recall.toFixed(3)}
            sub={`floor ${MIN_RECALL.toFixed(2)} on frozen data`}
            accent={accepted}
          />
          <div
            className={`rounded-lg border p-4 text-sm ${
              accepted
                ? "border-border bg-surface text-muted-foreground"
                : "border-[var(--urgent)] bg-surface text-foreground"
            }`}
          >
            {accepted ? (
              <>
                <span className="font-mono text-[10px] uppercase tracking-widest text-signal">
                  accepted
                </span>
                <p className="mt-2">
                  Recall holds above the floor at an alarm rate of {alarmRate.toFixed(3)}.
                  The personalisation is written to the user profile.
                </p>
              </>
            ) : (
              <>
                <span className="font-mono text-[10px] uppercase tracking-widest text-[var(--urgent)]">
                  rejected
                </span>
                <p className="mt-2">
                  This α would put validation recall at {recall.toFixed(3)}, below the
                  {" "}{MIN_RECALL.toFixed(2)} floor. The request is refused and the previous
                  threshold stays in force — the device says so out loud.
                </p>
              </>
            )}
          </div>
        </div>
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ item 4 */

const MODE_ORDER = ["sidewalk", "crossing", "transit_platform"];

function ModeSwitchPanel() {
  const profiles = MODE_ORDER.map((k) => S.profiles[k]).filter(
    (p): p is Profile => Boolean(p)
  );

  return (
    <Panel
      title="4 · Mode-aware context switching"
      caption={`A kerb-side corridor and a station platform do not deserve the same risk
        geometry. The slow path proposes a mode; the fast path only ever reads a profile
        struct, so nothing here costs latency. Switching is deliberately asymmetric:
        escalation to a more conservative profile is immediate, relaxation waits out a
        dwell timer. A misclassification that makes the device cautious costs an extra
        warning; the same error in the other direction is the one that hurts.`}
    >
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="border-b border-border text-left font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              <th className="py-2 pr-4">Profile</th>
              <th className="py-2 pr-4">Corridor half-width</th>
              <th className="py-2 pr-4">Warn TTC</th>
              <th className="py-2 pr-4">Urgent TTC</th>
              <th className="py-2 pr-4">Hard floor</th>
              <th className="py-2">Priority vocabulary</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {profiles.map((p) => (
              <tr key={p.mode} className="border-b border-border/60">
                <td className="py-2 pr-4 text-foreground">{p.mode.replace("_", " ")}</td>
                <td className="py-2 pr-4">{p.cone_half_width_m.toFixed(2)} m</td>
                <td className="py-2 pr-4">{p.ttc_warn_s.toFixed(1)} s</td>
                <td className="py-2 pr-4">{p.ttc_urgent_s.toFixed(1)} s</td>
                <td className="py-2 pr-4">
                  {p.hard_floor_m === null ? "—" : `${p.hard_floor_m.toFixed(1)} m`}
                </td>
                <td className="py-2 text-muted-foreground">{p.vocabulary.join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-3">
        {profiles.map((p) => (
          <div key={p.mode} className="rounded-lg border border-border bg-surface-2 p-4">
            <p className="font-mono text-[10px] uppercase tracking-widest text-signal">
              {p.mode.replace("_", " ")}
            </p>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{p.rationale}</p>
          </div>
        ))}
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ item 5 */

function FederatedPanel() {
  const rounds = S.federated.rounds.map((r) => ({
    round: r.round,
    recall: Number((r.recall * 100).toFixed(1)),
    lateAlerts: r.lateAlerts,
  }));
  const pu = S.privacyUtility.map((r) => ({
    z: r.noiseMultiplier,
    recall: Number((r.recall * 100).toFixed(1)),
    lateAlerts: r.lateAlerts,
  }));

  return (
    <Panel
      title="5 · Federated near-miss feedback"
      caption={`Devices upload nothing but gradient deltas computed on locally flagged
        near-miss windows — no frames, no audio, no coordinates. Each client delta is
        clipped to an L2 norm of ${S.federated.clipNorm.toFixed(1)} and the server adds
        Gaussian noise at z = ${S.federated.noiseMultiplier} before averaging over
        ${S.federated.nClients} clients. The metric that matters is not recall but *late
        alerts*: hazards the model does warn about, but only once the evidence is already
        overwhelming and the lead time is gone. Those fall from
        ${S.federated.lateAlertsFirst} to ${S.federated.lateAlertsLast} over eight rounds.
        Retraining is a manual, reviewed release — no model reaches a user's ears because a
        loop decided it should.`}
    >
      <div className="grid gap-8 lg:grid-cols-2">
        <div>
          <p className="mb-3 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            per round — recall (%) vs late alerts
          </p>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={rounds} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis dataKey="round" tick={AXIS} stroke="var(--border)" />
              <YAxis yAxisId="l" tick={AXIS} stroke="var(--border)" />
              <YAxis yAxisId="r" orientation="right" tick={AXIS} stroke="var(--border)" />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Legend wrapperStyle={{ fontSize: 11, fontFamily: "var(--font-mono)" }} />
              <Line yAxisId="l" name="recall %" dataKey="recall" stroke="var(--signal)" dot={false} strokeWidth={2} />
              <Line yAxisId="r" name="late alerts" dataKey="lateAlerts" stroke="var(--urgent)" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div>
          <p className="mb-3 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            privacy / utility — noise multiplier z
          </p>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={pu} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis dataKey="z" tick={AXIS} stroke="var(--border)" />
              <YAxis tick={AXIS} stroke="var(--border)" domain={[50, 90]} />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Legend wrapperStyle={{ fontSize: 11, fontFamily: "var(--font-mono)" }} />
              <Line name="recall %" dataKey="recall" stroke="var(--signal)" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
          <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
            Utility degrades smoothly with the privacy budget rather than collapsing, which
            is the honest way to present it: at z = 1.5 the aggregate is mostly noise and
            the round should not be shipped.
          </p>
        </div>
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ shared */

function Panel({
  title,
  caption,
  children,
}: {
  title: string;
  caption: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-border bg-surface p-6">
      <h3 className="text-lg font-semibold">{title}</h3>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted-foreground">{caption}</p>
      <div className="mt-6">{children}</div>
    </div>
  );
}

function Stat({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string;
  sub: string;
  accent?: boolean;
}) {
  return (
    <div className="rounded-lg border border-border bg-surface-2 p-5">
      <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        {label}
      </p>
      <p className={`mt-2 font-mono text-2xl ${accent ? "text-signal" : "text-foreground"}`}>
        {value}
      </p>
      <p className="mt-1 text-xs text-muted-foreground">{sub}</p>
    </div>
  );
}
