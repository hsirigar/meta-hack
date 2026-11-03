import { MODEL_COLORS, MODEL_LABELS } from "../api";
import { Plot, baseLayout, plotConfig } from "./Plot";

export default function SeriesPlot({ run }: { run: any }) {
  const key = run.series ? Object.keys(run.series)[0] : null;
  if (!key) return null;
  const s = run.series[key];
  const isAngle = key === "theta";
  const k = isAngle ? 180 / Math.PI : 100;
  const unit = isAngle ? "θ (deg)" : "fall distance y (cm)";
  const t: number[] = [], y: number[] = [];
  s.t.forEach((tt: number, i: number) => {
    if (s.valid_mask[i]) {
      t.push(tt);
      y.push(s.y[i] * k);
    }
  });
  const traces: any[] = [{ x: t, y, mode: "markers", type: "scatter", name: "measured", marker: { size: 3, color: "#e8ebf2", opacity: 0.75 } }];
  const curves = run.curves || {};
  const ct: number[] = curves.t || [];
  const tMin = t[0] ?? 0, tMax = t[t.length - 1] ?? 1;
  const inRange = (i: number) => ct[i] >= tMin - 0.05 && ct[i] <= tMax + 0.05;
  for (const f of run.fits || []) {
    if (run.role === "reveal" || !f.band) continue;
    const c = MODEL_COLORS[f.model] || "#fff";
    traces.push({ x: f.band.t, y: f.band.hi.map((v: number) => v * k), mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" });
    traces.push({ x: f.band.t, y: f.band.lo.map((v: number) => v * k), mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: c + "22", name: `${MODEL_LABELS[f.model]} 95% band`, showlegend: false, hoverinfo: "skip" });
  }
  for (const m of Object.keys(curves).filter((x) => x !== "t")) {
    const idx = ct.map((_, i) => i).filter(inRange);
    traces.push({ x: idx.map((i) => ct[i]), y: idx.map((i) => curves[m][i] * k), mode: "lines", name: MODEL_LABELS[m] || m, line: { color: MODEL_COLORS[m] || "#fff", width: 2 } });
  }
  const split = run.split?.train_end_t;
  const shapes: any[] = [];
  const annotations: any[] = [];
  if (split !== undefined) {
    shapes.push({ type: "rect", xref: "x", yref: "paper", x0: tMin, x1: split, y0: 0, y1: 1, fillcolor: "#ffffff", opacity: 0.04, line: { width: 0 } });
    shapes.push({ type: "line", xref: "x", yref: "paper", x0: split, x1: split, y0: 0, y1: 1, line: { color: "#ffdc00", width: 2, dash: "dot" } });
    annotations.push({ x: split, y: 1, xref: "x", yref: "paper", text: run.role === "reveal" ? "pure prediction →" : "holdout (never seen) →", showarrow: false, xanchor: "left", yanchor: "top", font: { color: "#ffdc00", size: 11 } });
    annotations.push({ x: tMin, y: 1, xref: "x", yref: "paper", text: run.role === "reveal" ? "conditioning" : "training", showarrow: false, xanchor: "left", yanchor: "top", font: { color: "#8a93a6", size: 11 } });
  }
  return (
    <div className="card p-2">
      <Plot data={traces} layout={baseLayout({ height: 320, shapes, annotations, xaxis: { title: { text: "time (s)" }, gridcolor: "#262b37" }, yaxis: { title: { text: unit }, gridcolor: "#262b37", autorange: isAngle ? true : "reversed" } })} config={plotConfig} style={{ width: "100%" }} useResizeHandler />
    </div>
  );
}
