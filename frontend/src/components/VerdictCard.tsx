import { MODEL_COLORS, MODEL_LABELS, fmt } from "../api";

export default function VerdictCard({ verdict, hash }: { verdict: any; hash?: string }) {
  if (!verdict) return null;
  const f = verdict.summary_facts || {};
  if (f.refused)
    return (
      <div className="border-2 border-bad bg-sheet p-5">
        <div className="label !text-bad">Verdict</div>
        <div className="serif mt-1 text-[30px] font-medium leading-tight text-bad">Not scored. This recording can't be trusted.</div>
        <ul className="mt-3 space-y-1 text-sm">
          {(f.reasons || []).map((r: string) => (
            <li key={r} className="grid grid-cols-[1rem_1fr]">
              <span className="text-bad">–</span>
              {r}
            </li>
          ))}
        </ul>
        <div className="mt-3 text-[13px] text-ink2">The seal stays open. Re-film with the camera fixed and score again.</div>
      </div>
    );
  const pm = verdict.per_model || [];
  const unit = pm[0]?.observable === "period_s" ? "s" : "m/s";
  const obsName = pm[0]?.observable === "period_s" ? "Period" : "Late-fall speed";
  const bad = !!f.all_models_inadequate;
  const winner = MODEL_LABELS[verdict.winner] || verdict.winner;
  return (
    <div className={`border-2 bg-sheet ${bad ? "border-bad" : "border-ok"}`}>
      <div className="flex flex-wrap items-end gap-x-8 gap-y-2 px-5 py-4">
        <div className="max-w-3xl">
          <div className={`label ${bad ? "!text-bad" : "!text-ok"}`}>Verdict</div>
          {bad ? (
            <>
              <div className="serif mt-1 text-[30px] font-medium leading-tight">None of the candidate laws describes this system.</div>
              <div className="mt-1 text-sm text-ink2">
                Least wrong: <span style={{ color: MODEL_COLORS[verdict.winner] }}>{winner}</span>. {f.inadequacy_note}
              </div>
            </>
          ) : (
            <div className="serif mt-1 text-[30px] font-medium leading-tight">
              <span style={{ color: MODEL_COLORS[verdict.winner] }}>{winner}</span> predicted the new experiment.
            </div>
          )}
        </div>
        <div className="mono ml-auto text-right text-[11px]">
          <div className={verdict.hash_verified ? "text-ok" : "text-bad"}>{verdict.hash_verified ? "seal verified" : "seal mismatch"}</div>
          {hash && <div className="text-mute">{hash.slice(0, 12)}</div>}
        </div>
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-y border-line text-left">
            <th className="label px-5 py-2 font-normal">Sealed law</th>
            <th className="label px-5 py-2 text-right font-normal">Prediction error</th>
            <th className="label px-5 py-2 text-right font-normal">{obsName} predicted</th>
            <th className="label px-5 py-2 text-right font-normal">Measured</th>
          </tr>
        </thead>
        <tbody>
          {pm.map((m: any, i: number) => (
            <tr key={m.model} className="border-b border-line">
              <td className={`px-5 py-2.5 ${i === 0 ? "font-medium" : ""}`}>
                <span className="swatch mr-2" style={{ background: MODEL_COLORS[m.model] }} />
                {MODEL_LABELS[m.model]}
              </td>
              <td className="mono px-5 py-2.5 text-right">{fmt(m.rmse_sigma, 1)}σ</td>
              <td className="mono px-5 py-2.5 text-right">
                {fmt(m.observable_predicted, 3)} {unit}{" "}
                <span className="text-mute">
                  ({m.observable_error_pct > 0 ? "+" : ""}
                  {fmt(m.observable_error_pct, 1)}%)
                </span>
              </td>
              <td className="mono px-5 py-2.5 text-right">
                {fmt(m.observable_measured, 3)} {unit}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="px-5 py-2.5 text-xs text-ink2">
        Parameters were frozen when the seal was made. Only the starting state was read, from the first {f.conditioning_window_s} s; everything after is prediction.
      </div>
    </div>
  );
}
