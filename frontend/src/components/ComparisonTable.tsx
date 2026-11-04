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
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 border-b border-line px-4 py-4">
          <span className="label">Result</span>
          <span className={`serif text-[24px] font-medium ${amb ? "text-warn" : "text-ok"}`}>{amb ? "Inconclusive." : `${MODEL_LABELS[cmp.best] || cmp.best} predicts better.`}</span>
          <span className="w-full text-sm text-ink2">
            {amb
              ? `On the held-out footage the laws differ by at most ${fmt(cmp.max_separation_sigma, 2)}σ, less than the measurement noise can resolve.`
              : `On the held-out footage the laws separate by up to ${fmt(cmp.max_separation_sigma, 1)}σ.`}
            {f.bic_ignored_autocorrelated_residuals && " BIC was not used: the residuals are autocorrelated, which makes it overstate the evidence."}
          </span>
          <span className="mono w-full text-[11px] text-mute">ΔBIC {fmt(f.delta_bic, 1)} · holdout RMSE ratio {fmt(f.holdout_rmse_ratio, 3)}</span>
        </div>
      )}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-left">
            <tr className="border-b border-line">
              <th className="label px-4 py-2 font-normal">Law</th>
              <th className="label px-4 py-2 font-normal">Fitted parameters ± 1σ</th>
              <th className="label px-4 py-2 text-right font-normal">Train</th>
              <th className="label px-4 py-2 text-right font-normal !text-ink">Holdout</th>
              <th className="label px-4 py-2 text-right font-normal">BIC</th>
              <th className="label px-4 py-2 text-right font-normal">Whiteness p</th>
            </tr>
          </thead>
          <tbody>
            {fits.map((x: any) => (
              <tr key={x.model} className="border-b border-line align-top last:border-b-0">
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2 font-medium">
                    <span className="swatch" style={{ background: MODEL_COLORS[x.model] }} />
                    {MODEL_LABELS[x.model] || x.model}
                  </div>
                  <div className="mono mt-1 text-[11px] text-ink2">{MODEL_EQ[x.model]}</div>
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
      <div className="hair px-4 py-2 text-xs text-ink2">
        Errors in units of measurement noise, σ = {fmt(noise, 5)} {run.series?.theta ? "rad" : "m"}. Models were fitted on the training window only; holdout is pure prediction.
        {f.measured_length_m && <> Measured string length {fmt(f.measured_length_m, 3)} m.</>}
      </div>
    </div>
  );
}
