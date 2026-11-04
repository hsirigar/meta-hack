import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";

export const Plot = createPlotlyComponent(Plotly);

export const INK = "#1d1c19";
export const MUTE = "#8a8377";
export const GRID = "#e6e0d3";

const axis = { gridcolor: GRID, zerolinecolor: "#cfc7b6", linecolor: "#bdb4a2", showline: true, ticks: "outside", tickcolor: "#bdb4a2", ticklen: 4 };

export const baseLayout = (extra: Record<string, any> = {}) => ({
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#4b473f", family: "IBM Plex Mono, monospace", size: 11 },
  margin: { l: 56, r: 16, t: 40, b: 44 },
  legend: { orientation: "h", y: 1.02, yanchor: "bottom", x: 0, bgcolor: "rgba(0,0,0,0)", font: { family: "IBM Plex Sans, sans-serif", size: 12, color: INK } },
  hoverlabel: { bgcolor: "#fbfaf6", bordercolor: "#d8d1c2", font: { family: "IBM Plex Mono, monospace", color: INK } },
  ...extra,
  xaxis: { ...axis, ...(extra.xaxis || {}) },
  yaxis: { ...axis, ...(extra.yaxis || {}) },
});

export const plotConfig = { displayModeBar: false, responsive: true };
