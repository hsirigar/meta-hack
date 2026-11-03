import type { Proposal } from "../api";
import { fmt } from "../api";
import { Plot, baseLayout, plotConfig } from "./Plot";

export default function DiscriminationCurve({ proposal, currentKnob }: { proposal: Proposal; currentKnob?: number }) {
  const isAngle = proposal.knob === "release_angle_deg";
  const xs = proposal.discrimination_curve.map((p) => p[0]);
  const ys = proposal.discrimination_curve.map((p) => Math.max(p[1], 1e-2));
  const shapes: any[] = [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 3, y1: 3, line: { color: "#8a93a6", dash: "dot", width: 1 } }];
  const ann: any[] = [
    { x: 0, xref: "paper", y: Math.log10(3), text: "3σ: distinguishable", showarrow: false, xanchor: "left", yanchor: "bottom", font: { size: 10, color: "#8a93a6" } },
    { x: proposal.value, y: Math.log10(Math.max(proposal.expected_separation_sigma, 1e-2)), text: `recommended: ${proposal.value}${isAngle ? "°" : " filters"}`, showarrow: true, arrowcolor: "#ffdc00", font: { color: "#ffdc00" }, ax: -60, ay: -30 },
  ];
  if (currentKnob !== undefined) {
    const cur = proposal.discrimination_curve.reduce((a, b) => (Math.abs(b[0] - currentKnob) < Math.abs(a[0] - currentKnob) ? b : a));
    ann.push({ x: cur[0], y: Math.log10(Math.max(cur[1], 1e-2)), text: `you filmed: ${currentKnob}${isAngle ? "°" : ""}`, showarrow: true, arrowcolor: "#8a93a6", font: { color: "#c9cfdb" }, ax: 40, ay: -40 });
  }
  return (
    <div className="card p-2">
      <Plot
        data={[
          { x: xs, y: ys, type: "scatter", mode: "lines+markers", line: { color: "#ffdc00", width: 2 }, marker: { size: 6 }, name: "expected separation", hovertemplate: "%{x}: %{y:.1f}σ<extra></extra>" },
        ]}
        layout={baseLayout({ height: 280, shapes, annotations: ann, showlegend: false, xaxis: { title: { text: isAngle ? "release angle (deg)" : "number of stacked filters" }, gridcolor: "#262b37", dtick: isAngle ? 10 : 1 }, yaxis: { title: { text: "model separation (σ, log)" }, type: "log", gridcolor: "#262b37" } })}
        config={plotConfig}
        style={{ width: "100%" }}
        useResizeHandler
      />
      <div className="px-3 pb-2 text-xs text-mute">
        Expected gap between the committed predictions under the reveal protocol (condition on 0.5 s, then predict), in units of measurement noise. Recommended: {fmt(proposal.expected_separation_sigma, 1)}σ.
      </div>
    </div>
  );
}
