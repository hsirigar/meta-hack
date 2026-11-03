import { useEffect, useRef, useState } from "react";
import type { AgentEvent } from "../api";

const TOOL_ICON: Record<string, string> = {
  describe_scene: "👁", track_objects: "◎", measure: "📏", check_validity: "🛡", list_models: "∑",
  fit_models: "∿", compare_models: "⚖", discover_equation: "ƒ", design_experiment: "🧪",
  commit_predictions: "🔒", evaluate_commitment: "✓", render_ghosts: "👻",
};

function ResultView({ result }: { result: any }) {
  const [open, setOpen] = useState(false);
  if (!result) return null;
  const summary = summarize(result);
  return (
    <div className="mt-1">
      <button className="text-xs text-mute hover:text-white" onClick={() => setOpen(!open)}>
        {open ? "▾" : "▸"} {summary}
      </button>
      {open && (
        <pre className="mono mt-1 max-h-72 overflow-auto rounded-lg bg-black/40 p-3 text-[11px] leading-relaxed text-[#b8c2d6]">
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
}

function summarize(r: any): string {
  if (r.error) return `error: ${r.error}`;
  if (r.status && r.checks) return `validity: ${r.status}`;
  if (r.status && r.max_separation_sigma !== undefined) return `${r.status} · max separation ${Number(r.max_separation_sigma).toFixed(2)}σ`;
  if (r.fits) return `${r.fits.length} models fitted on training window`;
  if (r.recommended_value !== undefined) return `recommend ${r.knob} = ${r.recommended_value}`;
  if (r.hash12) return `commitment ${r.hash12}`;
  if (r.winner !== undefined) return `winner: ${r.winner ?? "none"} · hash ${r.hash_verified ? "verified" : "FAILED"}`;
  if (r.overlay_url) return `overlay rendered (${(r.ghosts || []).length} ghosts)`;
  if (r.tracks) return `tracked ${Object.keys(r.tracks).length} objects`;
  if (r.equation) return r.equation;
  if (r.series) return `${r.series}(t): ${r.n_valid} samples, noise σ=${Number(r.noise_sigma).toExponential(2)}`;
  if (r.system_type) return `${r.system_type} · track "${(r.tracked_concepts || [])[0]}"`;
  if (r.models) return `${r.models.length} candidate models`;
  return "result";
}

export default function LabNotebook({ events, running, title = "Lab notebook" }: { events: AgentEvent[]; running?: boolean; title?: string }) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (running) end.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [events.length, running]);
  const status = events.find((e) => e.type === "status");
  const results = new Map(events.filter((e) => e.type === "tool_result").map((e) => [e.id, e]));
  return (
    <div className="card flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-line px-4 py-3">
        <div className="text-sm font-semibold">{title}</div>
        <div className="flex items-center gap-2 text-xs text-mute">
          {running && <span className="pulse inline-block h-2 w-2 rounded-full bg-accent" />}
          {status?.text}
          {events.some((e) => e.replay) && <span className="chip">replay</span>}
        </div>
      </div>
      <div className="flex-1 space-y-3 overflow-auto px-4 py-3 text-sm" style={{ maxHeight: 560 }}>
        {events.length === 0 && <div className="text-mute">No agent activity yet.</div>}
        {events.map((e, i) => {
          if (e.type === "thought") return <div key={i} className="italic text-[#aab3c5]">{e.text}</div>;
          if (e.type === "tool_call") {
            const res = results.get(e.id);
            return (
              <div key={i} className="rounded-lg border border-line bg-panel2 px-3 py-2">
                <div className="mono flex items-center gap-2 text-[12px]">
                  <span>{TOOL_ICON[e.name || ""] || "⚙"}</span>
                  <span className="font-semibold text-accent">{e.name}</span>
                  <span className="truncate text-mute">({Object.entries(e.args || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})</span>
                  {!res && running && <span className="pulse ml-auto text-mute">running…</span>}
                  {res && <span className={`ml-auto ${res.ok ? "text-ok" : "text-bad"}`}>{res.ok ? "ok" : "error"}</span>}
                </div>
                {res && <ResultView result={res.result} />}
              </div>
            );
          }
          if (e.type === "message")
            return (
              <div key={i} className="rounded-lg border border-accent/40 bg-accent/5 px-3 py-2 leading-relaxed text-white">
                {e.text}
              </div>
            );
          if (e.type === "error") return <div key={i} className="text-bad">⚠ {e.text}</div>;
          return null;
        })}
        <div ref={end} />
      </div>
    </div>
  );
}
