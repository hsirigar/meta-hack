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

  if (!run) return <div className="py-6 text-mute">{err || "Loading run…"}</div>;
  const shown = live ?? events ?? run.traces?.[task] ?? [];
  const isLive = !!live && !live.some((e) => e.type === "done");
  const knobTxt = Object.entries(run.knobs || {}).map(([k, v]) => (k === "release_angle_deg" ? `${v}° release` : `${v} filter${v === 1 ? "" : "s"}`)).join(" · ");
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-2">
        <span className="label !text-ink">{run.role === "reveal" ? "Reveal run" : "Exploration run"}</span>
        {knobTxt && <span className="serif text-[17px] italic">{knobTxt}</span>}
        <span className="mono text-[11px] text-mute">{run.id}</span>
        <div className="ml-auto flex gap-2">
          {shown.length > 0 && !running && !isLive && (
            <button className="btn btn-ghost" onClick={() => go(true)}>
              Replay log
            </button>
          )}
          {!running && !isLive && (
            <button className="btn btn-ghost" onClick={() => go(false)}>
              {run.role === "reveal" ? "Re-score against seal" : "Re-run analysis"}
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
          <LabNotebook events={shown} running={running || isLive} title="Agent log" />
        </div>
      </div>
      {run.role === "reveal" && <VerdictCard verdict={run.verdict} hash={hash} />}
      <div className="grid gap-4 lg:grid-cols-5">
        <div className="lg:col-span-3">{run.role !== "reveal" && <ComparisonTable run={run} />}</div>
        <div className="space-y-4 lg:col-span-2">
          <ValidityPanel validity={run.validity} />
          {run.discovery?.equation && (
            <div className="card p-4 text-sm">
              <div className="label mb-2">Equation discovery · PySINDy · supporting evidence</div>
              <div className="mono text-[13px]">{run.discovery.equation}</div>
              <div className="mt-2 text-xs text-ink2">
                R² = {Number(run.discovery.r2).toFixed(4)} · {run.discovery.method}
                {run.discovery.caveat && <> · {run.discovery.caveat}</>}
              </div>
            </div>
          )}
          {run.scene && (
            <div className="card p-4 text-[13px]">
              <div className="label mb-2">Scene · read by Muse Spark</div>
              <dl className="grid grid-cols-[6.5rem_1fr] gap-y-1">
                <dt className="text-mute">Tracking</dt>
                <dd>“{run.scene.tracked_concepts.join(", ")}”</dd>
                <dt className="text-mute">Reference</dt>
                <dd>
                  {run.scene.reference_object.concept}, <span className="mono">{(run.scene.reference_object.known_size_m * 1000).toFixed(2)} mm</span> {run.scene.reference_object.dimension.replace("_", " ")}
                </dd>
                {run.calibration && (
                  <>
                    <dt className="text-mute">Scale</dt>
                    <dd className="mono">{(run.calibration.meters_per_pixel * 1000).toFixed(3)} mm/px</dd>
                  </>
                )}
                {run.calibration?.length_m && (
                  <>
                    <dt className="text-mute">Arc radius</dt>
                    <dd className="mono">{run.calibration.length_m.toFixed(3)} m</dd>
                  </>
                )}
              </dl>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
