import { MODEL_COLORS, MODEL_EQ, MODEL_LABELS, fmt } from "../api";

const PARAM_LABEL: Record<string, string> = { omega0: "ω₀", gamma: "γ", c: "c", theta0: "θ₀", dtheta0: "θ̇₀", v_t: "vₜ", y0: "y₀", v0: "v₀" };

export default function ComparisonTable({ run }: { run: any }) {
  const fits = run.fits || [];
  const cmp = run.comparison;
  if (!fits.length) return null;
  const noise = run.series ? (Object.values(run.series)[0] as any).noise_sigma : 1;
  const amb = cmp?.status === "ambiguous";
  const f = cmp?.explanation_facts || {};
  return (
    <div className="card overflow-hidden">
      {cmp && (
        <div className={`flex flex-wrap items-center gap-3 px-4 py-3 ${amb ? "bg-warn/10" : "bg-ok/10"}`}>
          <span className={`text-lg font-bold ${amb ? "text-warn" : "text-ok"}`}>{amb ? "INCONCLUSIVE" : `DECISIVE: ${MODEL_LABELS[cmp.best] || cmp.best}`}</span>
          <span className="text-sm text-[#c9cfdb]">
            {amb
              ? `The models differ by at most ${fmt(cmp.max_separation_sigma, 2)}σ of measurement noise on held-out data.`
              : `Models separate by up to ${fmt(cmp.max_separation_sigma, 1)}σ on held-out data.`}
          </span>
          <span className="mono ml-auto text-xs text-mute">ΔBIC {fmt(f.delta_bic, 1)} · RMSE ratio {fmt(f.holdout_rmse_ratio, 3)}</span>
        </div>
      )}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase tracking-wide text-mute">
            <tr className="border-b border-line">
              <th className="px-4 py-2">Model</th>
              <th className="px-4 py-2">Fitted parameters (± 1σ)</th>
              <th className="px-4 py-2 text-right">Train RMSE</th>
              <th className="px-4 py-2 text-right">Holdout RMSE</th>
              <th className="px-4 py-2 text-right">BIC</th>
              <th className="px-4 py-2 text-right">Residual whiteness p</th>
            </tr>
          </thead>
          <tbody>
            {fits.map((x: any) => (
              <tr key={x.model} className="border-b border-line/60 align-top">
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2 font-semibold">
                    <span className="inline-block h-3 w-3 rounded-full" style={{ background: MODEL_COLORS[x.model] }} />
                    {MODEL_LABELS[x.model] || x.model}
                  </div>
                  <div className="mono mt-1 text-xs text-mute">{MODEL_EQ[x.model]}</div>
                </td>
                <td className="mono px-4 py-3 text-xs">
                  {Object.entries(x.params).filter(([k]) => k !== "t_ref").map(([k, v]) => (
                    <div key={k}>
                      {PARAM_LABEL[k] || k} = {fmt(v as number, 4)} <span className="text-mute">± {fmt(x.param_std[k], 2)}</span>
                    </div>
                  ))}
                </td>
                <td className="mono px-4 py-3 text-right text-xs">{fmt(x.train_rmse / noise, 2)}σ</td>
                <td className="mono px-4 py-3 text-right text-xs font-semibold">{x.holdout_rmse != null ? `${fmt(x.holdout_rmse / noise, 2)}σ` : "—"}</td>
                <td className="mono px-4 py-3 text-right text-xs">{fmt(x.bic, 1)}</td>
                <td className="mono px-4 py-3 text-right text-xs">{fmt(x.residual_whiteness_p, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="px-4 py-2 text-xs text-mute">
        Errors are in units of measurement noise σ = {fmt(noise, 5)} {run.series?.theta ? "rad" : "m"}. Models were fitted on the training window only; holdout is pure prediction.
        {f.measured_length_m && <> Measured string length {fmt(f.measured_length_m, 3)} m.</>}
      </div>
    </div>
  );
}
