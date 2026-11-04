import { useEffect, useRef, useState } from "react";
import type { AgentEvent } from "../api";

function ResultView({ result }: { result: any }) {
  const [open, setOpen] = useState(false);
  if (!result) return null;
  const summary = summarize(result);
  return (
    <div className="mt-1">
      <button className="mono text-left text-[11px] text-ink2 hover:text-ink" onClick={() => setOpen(!open)}>
        <span className="text-mute">{open ? "−" : "+"}</span> {summary}
      </button>
      {open && (
        <pre className="mono mt-1 max-h-72 overflow-auto bg-inset p-3 text-[11px] leading-relaxed text-ink2">
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
  let step = 0;
  return (
    <div className="card flex h-full flex-col">
      <div className="flex items-baseline justify-between gap-3 border-b border-ink px-4 py-2.5">
        <div className="label !text-ink">{title}</div>
        <div className="mono flex items-center gap-2 text-[11px] text-mute">
          {running && <span className="pulse text-ink">● recording</span>}
          {status?.text?.replace(/^agent: /, "")}
          {events.some((e) => e.replay) && <span>· replay</span>}
        </div>
      </div>
      <div className="flex-1 space-y-3 overflow-auto px-4 py-3 text-sm" style={{ maxHeight: 560 }}>
        {events.length === 0 && <div className="text-mute">No entries yet.</div>}
        {events.map((e, i) => {
          if (e.type === "thought") return <p key={i} className="serif text-[15px] italic leading-snug text-ink2">{e.text}</p>;
          if (e.type === "tool_call") {
            const res = results.get(e.id);
            step += 1;
            return (
              <div key={i} className="grid grid-cols-[1.8rem_1fr] gap-x-1">
                <span className="mono pt-px text-[11px] text-mute">{String(step).padStart(2, "0")}</span>
                <div className="min-w-0">
                  <div className="mono flex items-baseline gap-2 text-[12px]">
                    <span className="font-semibold">{e.name}</span>
                    <span className="truncate text-mute">{Object.entries(e.args || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")}</span>
                    {!res && running && <span className="pulse ml-auto shrink-0 text-mute">running</span>}
                    {res && <span className={`ml-auto shrink-0 ${res.ok ? "text-mute" : "text-bad"}`}>{res.ok ? "ok" : "error"}</span>}
                  </div>
                  {res && <ResultView result={res.result} />}
                </div>
              </div>
            );
          }
          if (e.type === "message")
            return (
              <div key={i} className="serif border-l-2 border-ink py-0.5 pl-3 text-[16px] leading-snug text-ink">
                {e.text}
              </div>
            );
          if (e.type === "error") return <div key={i} className="text-bad">Error: {e.text}</div>;
          return null;
        })}
        <div ref={end} />
      </div>
    </div>
  );
}
