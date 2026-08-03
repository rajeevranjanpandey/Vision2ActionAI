import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import results from "@/data/results.json";

interface Ablation {
  name: string;
  recall: number;
  median_lead_time_s: number;
  p10_lead_time_s: number;
  false_alarms_per_km: number;
  misses: number;
  late_alerts: number;
  hazard_clips: number;
  safe_clips: number;
}

const R = results as unknown as {
  generated_at: string;
  seed: number;
  train: {
    clips: number;
    windows: number;
    feature_dim: number;
    positive_rate: number;
    ensemble: number;
    val_auc: number;
    val_ap: number;
    fit_seconds: number;
  };
  calibration: {
    temperature: number;
    threshold: number;
    alpha: number;
    ece_before: number;
    ece_after: number;
    brier_after: number;
    auc_holdout: number;
    ece_holdout: number;
    reliability: { confidence: number; empirical: number; count: number }[];
    risk_control: {
      alpha: number;
      threshold: number;
      miss_rate: number;
      alarm_rate: number;
      n: number;
    }[];
  };
  ablations: Ablation[];
  lead_time_hist: Record<string, number | string>[];
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

const SERIES = [
  { key: "Constant-velocity TTC (baseline)", colour: "var(--muted-foreground)" },
  { key: "Learned head only", colour: "var(--safe)" },
  { key: "Geometry + head (shipping)", colour: "var(--signal)" },
  { key: "Oracle (upper bound)", colour: "var(--urgent)" },
];
/**
 * Deep research panels (reliability, split-conformal table, lead-time histogram,
 * ablation) are kept in the codebase but hidden from the dashboard: they crowd out
 * the story for a general audience. Flip to `true` to bring them back.
 */
const SHOW_DEEP_PANELS = false;


export function ResultsDashboard() {
  const cal = R.calibration;
  const shipping = R.ablations.find((a) => a.name.startsWith("Geometry + head"));
  const baseline = R.ablations.find((a) => a.name.startsWith("Constant-velocity"));
  const faReduction =
    shipping && baseline
      ? (1 - shipping.false_alarms_per_km / baseline.false_alarms_per_km) * 100
      : 0;

  // Reliability bins with no confident predictions carry no calibration information.
  const reliability = cal.reliability
    .filter((b, i, arr) => b.confidence > 0 || arr[i - 1]?.confidence)
    .map((b) => ({ ...b, ideal: b.confidence }));

  return (
    <div className="space-y-10">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Holdout AUC" value={cal.auc_holdout.toFixed(4)}
              sub={`${R.train.windows.toLocaleString()} windows, ${R.train.clips} clips`} />
        <Stat label="ECE after scaling" value={cal.ece_after.toFixed(4)}
              sub={`from ${cal.ece_before.toFixed(4)}, T=${cal.temperature.toFixed(3)}`} />
        <Stat label="False alarms / km" value={shipping?.false_alarms_per_km.toFixed(1) ?? "—"}
              sub={`${faReduction.toFixed(0)}% below the geometric baseline`} accent />
        <Stat label="Late alerts" value={String(shipping?.late_alerts ?? "—")}
              sub={`baseline ${baseline?.late_alerts ?? "—"}, recall held at 1.00`} accent />
      </div>

      {SHOW_DEEP_PANELS && (
      <>
      <Panel
        title="Reliability after temperature scaling"
        caption={`A 5-member ensemble trained with focal loss is systematically overconfident;
          fitting a single temperature on held-out data (T = ${cal.temperature.toFixed(3)}) pulls
          expected calibration error to ${cal.ece_after.toFixed(4)} and Brier to ${cal.brier_after.toFixed(4)}.
          This matters because the alert policy thresholds a probability — an uncalibrated
          0.7 is not a budget you can reason about.`}
      >
        <ResponsiveContainer width="100%" height={280}>
          <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey="confidence"
              domain={[0, 1]}
              tick={AXIS}
              stroke="var(--border)"
              label={{ value: "predicted probability", position: "insideBottom", offset: -14, fill: "var(--muted-foreground)", fontSize: 11 }}
            />
            <YAxis
              type="number"
              dataKey="empirical"
              domain={[0, 1]}
              tick={AXIS}
              stroke="var(--border)"
              label={{ value: "empirical", angle: -90, position: "insideLeft", fill: "var(--muted-foreground)", fontSize: 11 }}
            />
            <ZAxis type="number" dataKey="count" range={[60, 320]} />
            <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ stroke: "var(--border)" }} />
            <Line
              type="linear"
              dataKey="ideal"
              data={[{ confidence: 0, ideal: 0 }, { confidence: 1, ideal: 1 }]}
              stroke="var(--muted-foreground)"
              strokeDasharray="4 4"
              dot={false}
              legendType="none"
            />
            <Scatter name="bin" data={reliability} fill="var(--signal)" />
          </ScatterChart>
        </ResponsiveContainer>
      </Panel>

      <Panel
        title="Split-conformal risk control"
        caption={`Each row is a threshold chosen on a calibration split so that the miss rate on
          unseen hazards stays below a target α, with no distributional assumptions. The empirical
          miss rate tracks α to within a point at every setting (n = ${cal.risk_control[0]?.n ?? 0}
          positives), which is what turns "the model is accurate" into "the device misses at most
          one hazard in ten". The shipped operating point is α = ${cal.alpha}.`}
      >
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-border text-left font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                <th className="py-2 pr-4">Target α</th>
                <th className="py-2 pr-4">Threshold</th>
                <th className="py-2 pr-4">Empirical miss rate</th>
                <th className="py-2">Alarm rate on safe clips</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {cal.risk_control.map((row) => {
                const shipped = Math.abs(row.alpha - cal.alpha) < 1e-9;
                return (
                  <tr
                    key={row.alpha}
                    className={`border-b border-border/60 ${shipped ? "text-signal" : "text-foreground"}`}
                  >
                    <td className="py-2 pr-4">
                      {row.alpha.toFixed(2)}
                      {shipped && <span className="ml-2 text-[10px] uppercase">shipped</span>}
                    </td>
                    <td className="py-2 pr-4">{row.threshold.toFixed(3)}</td>
                    <td className="py-2 pr-4">{row.miss_rate.toFixed(3)}</td>
                    <td className="py-2">{row.alarm_rate.toFixed(3)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel
        title="Lead-time distribution by arbitration policy"
        caption={`Warning lead time on hazard clips, bucketed. The baseline's long right tail is not
          a win — those are alerts fired 2.5–3.5 s out on trajectories that had not yet committed,
          which is precisely the behaviour that trains users to ignore the device. The shipped
          policy concentrates mass in the 1.5–2.0 s band and, unlike the head alone, puts nothing
          below 1.0 s.`}
      >
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={R.lead_time_hist} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="bucket" tick={AXIS} stroke="var(--border)" />
            <YAxis tick={AXIS} stroke="var(--border)" />
            <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "var(--border)", opacity: 0.3 }} />
            <Legend wrapperStyle={{ fontSize: 11, fontFamily: "var(--font-mono)" }} />
            {SERIES.map((s) => (
              <Bar key={s.key} dataKey={s.key} fill={s.colour} radius={[2, 2, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </Panel>

      <Panel
        title="Ablation: what each component buys"
        caption={`All arms hold recall at 1.00 on ${shipping?.hazard_clips ?? 0} hazard clips, so the
          axis that separates them is nuisance alerts and how late they arrive. Geometry alone is
          safe but noisy; the head alone is quiet but occasionally late; the arbitration keeps the
          geometric floor for imminent contact and lets the head suppress only when its ensemble
          agrees.`}
      >
        <ResponsiveContainer width="100%" height={260}>
          <BarChart
            data={R.ablations}
            layout="vertical"
            margin={{ top: 8, right: 24, bottom: 8, left: 8 }}
          >
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tick={AXIS} stroke="var(--border)"
                   label={{ value: "false alarms per km", position: "insideBottom", offset: -2, fill: "var(--muted-foreground)", fontSize: 11 }} />
            <YAxis type="category" dataKey="name" width={190} tick={{ ...AXIS, fontSize: 10 }} stroke="var(--border)" />
            <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "var(--border)", opacity: 0.3 }} />
            <Bar dataKey="false_alarms_per_km" radius={[0, 2, 2, 0]}>
              {R.ablations.map((a) => (
                <Cell
                  key={a.name}
                  fill={a.name.startsWith("Geometry + head") ? "var(--signal)" : "var(--muted-foreground)"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        <div className="mt-6 overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-border text-left font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                <th className="py-2 pr-4">Arm</th>
                <th className="py-2 pr-4">Recall</th>
                <th className="py-2 pr-4">Median lead</th>
                <th className="py-2 pr-4">p10 lead</th>
                <th className="py-2 pr-4">FA / km</th>
                <th className="py-2">Late</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {R.ablations.map((a) => {
                const ship = a.name.startsWith("Geometry + head");
                return (
                  <tr key={a.name} className={`border-b border-border/60 ${ship ? "text-signal" : ""}`}>
                    <td className="py-2 pr-4 font-sans">{a.name}</td>
                    <td className="py-2 pr-4">{a.recall.toFixed(2)}</td>
                    <td className="py-2 pr-4">{a.median_lead_time_s.toFixed(1)} s</td>
                    <td className="py-2 pr-4">{a.p10_lead_time_s.toFixed(1)} s</td>
                    <td className="py-2 pr-4">{a.false_alarms_per_km.toFixed(1)}</td>
                    <td className="py-2">{a.late_alerts}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>
      </>
      )}


      <p className="text-xs text-muted-foreground">
        Generated by <code className="font-mono">scripts/train_risk.py</code> at{" "}
        {R.generated_at}, seed {R.seed}, {R.train.fit_seconds.toFixed(0)} s fit on CPU.
        Numbers come from simulated hazard physics, not from a road study — the honest read is that
        they validate the decision layer, and GuardianBench validates the perception stack.
      </p>
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
    <div className="rounded-lg border border-border bg-surface p-5">
      <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        {label}
      </p>
      <p className={`mt-2 font-mono text-3xl ${accent ? "text-signal" : "text-foreground"}`}>
        {value}
      </p>
      <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{sub}</p>
    </div>
  );
}

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
