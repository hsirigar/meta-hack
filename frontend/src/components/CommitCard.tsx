import type { Commitment } from "../api";
import { MODEL_COLORS, MODEL_LABELS, fmt } from "../api";

const OBS: Record<string, string> = { period_s: "Period", terminal_velocity_mps: "Terminal velocity", fall_time_2m_s: "2 m fall time", implied_length_m: "Implied length" };
const UNIT: Record<string, string> = { period_s: "s", terminal_velocity_mps: "m/s", fall_time_2m_s: "s", implied_length_m: "m" };

export default function CommitCard({ c }: { c: Commitment }) {
  const [knob, value] = Object.entries(c.knob_setting)[0];
  const at = knob === "release_angle_deg" ? `${value}° release` : `${value} filters`;
  const obsKeys = Array.from(new Set(c.models.flatMap((m) => Object.keys(m.predicted_observables).filter((k) => OBS[k]))));
  const ts = new Date(c.created_at);
  return (
    <div className="border-2 border-seal bg-sheet">
      <div className="flex flex-wrap items-end gap-x-8 gap-y-3 border-b border-seal/40 px-5 py-4">
        <div>
          <div className="label !text-seal">Sealed before filming</div>
          <div className="mono mt-1 text-[30px] font-medium leading-none tracking-[0.06em] text-seal" title={c.sha256}>
            {c.sha256.slice(0, 12)}
          </div>
        </div>
        <dl className="mono grid grid-cols-[auto_1fr] gap-x-3 text-[11px] text-ink2">
          <dt className="text-mute">algorithm</dt>
          <dd>SHA-256 of canonical JSON</dd>
          <dt className="text-mute">sealed at</dt>
          <dd>{ts.toISOString().replace("T", " ").slice(0, 19)} UTC</dd>
          <dt className="text-mute">for</dt>
          <dd>{at}</dd>
        </dl>
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-line text-left">
            <th className="label px-5 py-2 font-normal">Law</th>
            {obsKeys.map((k) => (
              <th key={k} className="label px-5 py-2 text-right font-normal">
                {OBS[k]} at {at}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {c.models.map((m) => (
            <tr key={m.name} className="border-b border-line">
              <td className="px-5 py-2.5">
                <span className="swatch mr-2" style={{ background: MODEL_COLORS[m.name] }} />
                {MODEL_LABELS[m.name]}
              </td>
              {obsKeys.map((k) => (
                <td key={k} className="mono px-5 py-2.5 text-right">
                  {m.predicted_observables[k] !== undefined ? `${fmt(m.predicted_observables[k], 3)} ${UNIT[k]}` : "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <details className="px-5 py-2.5 text-xs text-ink2">
        <summary className="cursor-pointer select-none">Full hash, protocol and sealed JSON</summary>
        <div className="mono mt-2 break-all text-ink">{c.sha256}</div>
        <div className="mt-2">Success rule: {c.protocol.success_rule}.</div>
        <pre className="mono mt-2 max-h-56 overflow-auto whitespace-pre-wrap break-all bg-inset p-2 text-[10px]">{c.canonical_json}</pre>
      </details>
    </div>
  );
}
