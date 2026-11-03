import { useCallback, useEffect, useState } from "react";
import type { AgentEvent } from "../api";
import { get, stream } from "../api";
import ComparisonTable from "./ComparisonTable";
import GhostPlayer from "./GhostPlayer";
import LabNotebook from "./LabNotebook";
import SeriesPlot from "./SeriesPlot";
import ValidityPanel from "./ValidityPanel";
import VerdictCard from "./VerdictCard";

export default function RunPanel({ runId, live, hash, onChanged }: { runId: string; live?: AgentEvent[] | null; hash?: string; onChanged?: () => void }) {
  const [run, setRun] = useState<any>(null);
  const [events, setEvents] = useState<AgentEvent[] | null>(null);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(() => get<any>(`/api/runs/${runId}`).then(setRun).catch((e) => setErr(String(e))), [runId]);
  useEffect(() => {
    load();
  }, [load]);
  useEffect(() => {
    if (live && live.some((e) => e.type === "done")) load();
  }, [live, load]);

  const task = run?.role === "reveal" ? "score" : "analyze";
  const go = async (replay: boolean) => {
    setEvents([]);
    setRunning(true);
    setErr(null);
    try {
      await stream(`/api/runs/${runId}/${task}${replay ? "?replay=true" : ""}`, (e) => setEvents((prev) => [...(prev || []), e]));
    } catch (e) {
      setErr(String(e));
    } finally {
      setRunning(false);
      await load();
      onChanged?.();
    }
  };

  if (!run) return <div className="card p-6 text-mute">{err || "Loading run…"}</div>;
  const shown = live ?? events ?? run.traces?.[task] ?? [];
  const isLive = !!live && !live.some((e) => e.type === "done");
  const knobTxt = Object.entries(run.knobs || {}).map(([k, v]) => (k === "release_angle_deg" ? `${v}° release` : `${v} filter${v === 1 ? "" : "s"}`)).join(" · ");
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <span className={`chip ${run.role === "reveal" ? "border-accent/60 text-accent" : ""}`}>{run.role === "reveal" ? "reveal run" : "exploration run"}</span>
        <span className="text-sm text-[#c9cfdb]">{knobTxt}</span>
        <span className="mono text-xs text-mute">{run.id}</span>
        <div className="ml-auto flex gap-2">
          {shown.length > 0 && !running && !isLive && (
            <button className="btn btn-ghost" onClick={() => go(true)}>
              ↻ Replay notebook
            </button>
          )}
          {!running && !isLive && (
            <button className="btn btn-ghost" onClick={() => go(false)}>
              ▶ {run.role === "reveal" ? "Score against commitment" : "Analyze this run"} (live)
            </button>
          )}
        </div>
      </div>
      {err && <div className="text-sm text-bad">{err}</div>}
      <div className="grid gap-4 lg:grid-cols-5">
        <div className="space-y-4 lg:col-span-3">
          <GhostPlayer runId={run.id} hasOverlay={!!run.overlay_path} version={run.overlay_path ? String(shown.length) : undefined} />
          <SeriesPlot run={run} />
        </div>
        <div className="lg:col-span-2">
          <LabNotebook events={shown} running={running || isLive} />
        </div>
      </div>
      {run.role === "reveal" && <VerdictCard verdict={run.verdict} hash={hash} />}
      <div className="grid gap-4 lg:grid-cols-5">
        <div className="lg:col-span-3">{run.role !== "reveal" && <ComparisonTable run={run} />}</div>
        <div className="space-y-4 lg:col-span-2">
          <ValidityPanel validity={run.validity} />
          {run.discovery?.equation && (
            <div className="card p-4 text-sm">
              <div className="mb-1 font-semibold">Equation discovery (PySINDy, supporting evidence)</div>
              <div className="mono text-accent">{run.discovery.equation}</div>
              <div className="mt-1 text-xs text-mute">
                R² = {Number(run.discovery.r2).toFixed(4)} · {run.discovery.method}
                {run.discovery.caveat && <> · {run.discovery.caveat}</>}
              </div>
            </div>
          )}
          {run.scene && (
            <div className="card p-4 text-xs text-mute">
              <div className="mb-1 text-sm font-semibold text-white">Scene</div>
              Tracking <span className="text-white">“{run.scene.tracked_concepts.join(", ")}”</span> · reference{" "}
              <span className="text-white">{run.scene.reference_object.concept}</span> ({(run.scene.reference_object.known_size_m * 1000).toFixed(2)} mm {run.scene.reference_object.dimension})
              {run.calibration && <> · scale {(run.calibration.meters_per_pixel * 1000).toFixed(3)} mm/px</>}
              {run.calibration?.length_m && <> · string {run.calibration.length_m.toFixed(3)} m</>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
