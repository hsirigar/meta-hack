import type { Proposal } from "../api";
import { fmt } from "../api";
import { INK, MUTE, Plot, baseLayout, plotConfig } from "./Plot";

export default function DiscriminationCurve({ proposal, currentKnob }: { proposal: Proposal; currentKnob?: number }) {
  const isAngle = proposal.knob === "release_angle_deg";
  const xs = proposal.discrimination_curve.map((p) => p[0]);
  const ys = proposal.discrimination_curve.map((p) => Math.max(p[1], 1e-2));
  const shapes: any[] = [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 3, y1: 3, line: { color: MUTE, dash: "dot", width: 1 } }];
  const ann: any[] = [
    { x: 0, xref: "paper", y: Math.log10(3), text: "3σ: distinguishable", showarrow: false, xanchor: "left", yanchor: "bottom", font: { size: 10, color: MUTE } },
    { x: proposal.value, y: Math.log10(Math.max(proposal.expected_separation_sigma, 1e-2)), text: `recommended: ${proposal.value}${isAngle ? "°" : " filters"}`, showarrow: true, arrowcolor: INK, arrowwidth: 1, font: { color: INK }, bgcolor: "#fbfaf6", ax: -60, ay: -30 },
  ];
  if (currentKnob !== undefined) {
    const cur = proposal.discrimination_curve.reduce((a, b) => (Math.abs(b[0] - currentKnob) < Math.abs(a[0] - currentKnob) ? b : a));
    ann.push({ x: cur[0], y: Math.log10(Math.max(cur[1], 1e-2)), text: `filmed: ${currentKnob}${isAngle ? "°" : ""}`, showarrow: true, arrowcolor: MUTE, arrowwidth: 1, font: { color: MUTE }, ax: 40, ay: -40 });
  }
  return (
    <figure className="card px-2 pt-2">
      <Plot
        data={[
          { x: xs, y: ys, type: "scatter", mode: "lines+markers", line: { color: INK, width: 1.5 }, marker: { size: 5, color: INK }, name: "expected separation", hovertemplate: "%{x}: %{y:.1f}σ<extra></extra>" },
        ]}
        layout={baseLayout({ height: 280, shapes, annotations: ann, showlegend: false, xaxis: { title: { text: isAngle ? "release angle (deg)" : "number of stacked filters" }, dtick: isAngle ? 10 : 1 }, yaxis: { title: { text: "separation (σ, log)" }, type: "log" } })}
        config={plotConfig}
        style={{ width: "100%" }}
        useResizeHandler
      />
      <figcaption className="hair px-2 py-2 text-xs text-ink2">
        <span className="label mr-2 !text-ink">Fig.</span>How far apart the laws' predictions would be at each setting, in units of measurement noise, under the reveal protocol (read the
        starting state from 0.5 s, then predict). At the recommendation: <span className="mono">{fmt(proposal.expected_separation_sigma, 1)}σ</span>.
      </figcaption>
    </figure>
  );
}
