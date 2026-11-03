import { MODEL_COLORS, MODEL_LABELS, fmt } from "../api";

export default function VerdictCard({ verdict, hash }: { verdict: any; hash?: string }) {
  if (!verdict) return null;
  const refused = verdict.summary_facts?.refused;
  if (refused)
    return (
      <div className="card border-bad/60 bg-bad/5 p-5">
        <div className="text-2xl font-bold text-bad">INVALID: NOT SCORED</div>
        <div className="mt-1 text-sm text-[#c9cfdb]">{refused}. The commitment stays open; re-film with the camera fixed.</div>
        <ul className="mt-3 list-disc space-y-1 pl-5 text-sm">
          {(verdict.summary_facts.reasons || []).map((r: string) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      </div>
    );
  const pm = verdict.per_model || [];
  const unit = pm[0]?.observable === "period_s" ? "s" : "m/s";
  const obsName = pm[0]?.observable === "period_s" ? "Period" : "Late-fall speed";
  return (
    <div className="card overflow-hidden border-ok/50">
      <div className="flex flex-wrap items-center gap-3 bg-ok/10 px-5 py-4">
        <div>
          <div className="text-xs uppercase tracking-wide text-mute">Verdict</div>
          <div className="text-2xl font-bold" style={{ color: MODEL_COLORS[verdict.winner] }}>
            {MODEL_LABELS[verdict.winner] || verdict.winner} wins
          </div>
        </div>
        <div className="ml-auto text-right text-xs">
          <div className={verdict.hash_verified ? "text-ok" : "text-bad"}>{verdict.hash_verified ? "✓ commitment hash verified" : "✕ hash mismatch"}</div>
          {hash && <div className="mono text-mute">{hash.slice(0, 12)}</div>}
        </div>
      </div>
      <table className="w-full text-sm">
        <thead className="text-left text-xs uppercase tracking-wide text-mute">
          <tr className="border-b border-line">
            <th className="px-5 py-2">Committed model</th>
            <th className="px-5 py-2 text-right">Prediction error (RMSE)</th>
            <th className="px-5 py-2 text-right">{obsName} predicted</th>
            <th className="px-5 py-2 text-right">measured</th>
          </tr>
        </thead>
        <tbody>
          {pm.map((m: any) => (
            <tr key={m.model} className="border-b border-line/60">
              <td className="px-5 py-2 font-semibold">
                <span className="mr-2 inline-block h-3 w-3 rounded-full" style={{ background: MODEL_COLORS[m.model] }} />
                {MODEL_LABELS[m.model]}
              </td>
              <td className="mono px-5 py-2 text-right">{fmt(m.rmse_sigma, 1)}σ</td>
              <td className="mono px-5 py-2 text-right">
                {fmt(m.observable_predicted, 3)} {unit} <span className="text-mute">({m.observable_error_pct > 0 ? "+" : ""}{fmt(m.observable_error_pct, 1)}%)</span>
              </td>
              <td className="mono px-5 py-2 text-right">{fmt(m.observable_measured, 3)} {unit}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="px-5 py-2 text-xs text-mute">
        Parameters were frozen at commit time. Only the starting state was read from the first {verdict.summary_facts?.conditioning_window_s}s; everything after is pure prediction.
      </div>
    </div>
  );
}
