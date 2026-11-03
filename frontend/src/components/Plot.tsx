import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";

export const Plot = createPlotlyComponent(Plotly);

export const baseLayout = (extra: Record<string, any> = {}) => ({
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#c9cfdb", family: "Inter, sans-serif", size: 12 },
  margin: { l: 56, r: 16, t: 12, b: 44 },
  xaxis: { gridcolor: "#262b37", zerolinecolor: "#323848" },
  yaxis: { gridcolor: "#262b37", zerolinecolor: "#323848" },
  legend: { orientation: "h", y: 1.12, x: 0, bgcolor: "rgba(0,0,0,0)" },
  ...extra,
});

export const plotConfig = { displayModeBar: false, responsive: true };
