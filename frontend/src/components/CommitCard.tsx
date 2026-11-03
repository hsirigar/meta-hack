import type { Commitment } from "../api";
import { MODEL_COLORS, MODEL_LABELS, fmt } from "../api";

const OBS: Record<string, string> = { period_s: "period", terminal_velocity_mps: "terminal velocity", fall_time_2m_s: "2 m fall time", implied_length_m: "implied length" };
const UNIT: Record<string, string> = { period_s: "s", terminal_velocity_mps: "m/s", fall_time_2m_s: "s", implied_length_m: "m" };

export default function CommitCard({ c }: { c: Commitment }) {
  const [knob, value] = Object.entries(c.knob_setting)[0];
  return (
    <div className="card overflow-hidden border-accent/50">
      <div className="flex flex-wrap items-center gap-3 border-b border-line bg-accent/5 px-4 py-3">
        <span className="text-xl">🔒</span>
        <div>
          <div className="text-xs uppercase tracking-wide text-mute">Predictions committed before filming</div>
          <div className="mono text-2xl font-bold tracking-wider text-accent" title={c.sha256}>
            {c.sha256.slice(0, 12)}
          </div>
        </div>
        <div className="ml-auto text-right text-xs text-mute">
          <div>SHA-256 of canonical JSON</div>
          <div className="mono">{new Date(c.created_at).toUTCString()}</div>
        </div>
      </div>
      <div className="grid gap-3 p-4 md:grid-cols-2">
        {c.models.map((m) => (
          <div key={m.name} className="rounded-lg border border-line bg-panel2 p-3">
            <div className="mb-1 flex items-center gap-2 text-sm font-semibold">
              <span className="inline-block h-3 w-3 rounded-full" style={{ background: MODEL_COLORS[m.name] }} />
              {MODEL_LABELS[m.name]}
            </div>
            {Object.entries(m.predicted_observables)
              .filter(([k]) => OBS[k])
              .map(([k, v]) => (
                <div key={k} className="mono text-xs">
                  {OBS[k]} at {knob === "release_angle_deg" ? `${value}°` : `${value} filters`}: <span className="text-white">{fmt(v, 3)} {UNIT[k]}</span>
                </div>
              ))}
            <div className="mt-1 text-[11px] text-mute">{m.knob_transfer}</div>
          </div>
        ))}
      </div>
      <details className="border-t border-line px-4 py-2 text-xs text-mute">
        <summary className="cursor-pointer">Full hash and committed JSON</summary>
        <div className="mono mt-2 break-all text-[#c9cfdb]">{c.sha256}</div>
        <pre className="mono mt-2 max-h-56 overflow-auto whitespace-pre-wrap break-all rounded bg-black/40 p-2 text-[10px]">{c.canonical_json}</pre>
        <div className="mt-1">Success rule: {c.protocol.success_rule}</div>
      </details>
    </div>
  );
}
