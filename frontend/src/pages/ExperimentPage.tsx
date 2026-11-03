import { useCallback, useEffect, useState } from "react";
import type { AgentEvent, Experiment } from "../api";
import { API, MODEL_COLORS, MODEL_EQ, MODEL_LABELS, get, stream } from "../api";
import CommitCard from "../components/CommitCard";
import DiscriminationCurve from "../components/DiscriminationCurve";
import LabNotebook from "../components/LabNotebook";
import RunPanel from "../components/RunPanel";
import UploadBox from "../components/UploadBox";

function Step({ n, title, sub, children }: { n: number; title: string; sub?: string; children: React.ReactNode }) {
  return (
    <section className="relative border-l border-line pb-12 pl-8">
      <div className="absolute -left-4 top-0 flex h-8 w-8 items-center justify-center rounded-full border border-line bg-panel text-sm font-bold text-accent">{n}</div>
      <h2 className="text-xl font-semibold">{title}</h2>
      {sub && <p className="mt-1 max-w-3xl text-sm text-mute">{sub}</p>}
      <div className="mt-4 space-y-4">{children}</div>
    </section>
  );
}

export default function ExperimentPage({ id }: { id: string }) {
  const [exp, setExp] = useState<Experiment | null>(null);
  const [nextEvents, setNextEvents] = useState<AgentEvent[] | null>(null);
  const [nextRunning, setNextRunning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [live, setLive] = useState<{ runId: string | null; events: AgentEvent[] } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(() => get<Experiment>(`/api/experiments/${id}`).then(setExp).catch((e) => setErr(String(e))), [id]);
  useEffect(() => {
    load();
  }, [load]);

  if (!exp) return <div className="text-mute">{err || "Loading…"}</div>;
  const knobName = exp.system_type === "pendulum" ? "release_angle_deg" : "n_filters";
  const explore = exp.runs.filter((r) => r.role === "exploration");
  const reveals = exp.runs.filter((r) => r.role === "reveal");
  const commitment = exp.commitments?.[exp.commitments.length - 1];
  const lastExplore = explore[explore.length - 1];
  const candidates = exp.system_type === "pendulum" ? ["pendulum_linear", "pendulum_nonlinear"] : ["drag_linear", "drag_quadratic"];

  const runNext = async (replay: boolean) => {
    setNextEvents([]);
    setNextRunning(true);
    try {
      await stream(`/api/experiments/${id}/next${replay ? "?replay=true" : ""}`, (e) => setNextEvents((p) => [...(p || []), e]));
    } catch (e) {
      setErr(String(e));
    } finally {
      setNextRunning(false);
      load();
    }
  };

  const uploadExplore = async (f: File, knobs: Record<string, number>) => {
    setBusy(true);
    setErr(null);
    try {
      const fd = new FormData();
      fd.append("video", f);
      fd.append("knobs", JSON.stringify(knobs));
      const r = await fetch(`${API}/api/experiments/${id}/runs`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(await r.text());
      const run = await r.json();
      await load();
      setLive({ runId: run.id, events: [] });
      await stream(`/api/runs/${run.id}/analyze`, (e) => setLive((p) => ({ runId: run.id, events: [...(p?.events || []), e] })));
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
      load();
    }
  };

  const uploadReveal = async (f: File, knobs: Record<string, number>) => {
    if (!commitment) return;
    setBusy(true);
    setErr(null);
    setLive({ runId: null, events: [] });
    try {
      const fd = new FormData();
      fd.append("video", f);
      fd.append("knobs", JSON.stringify(knobs));
      let rid: string | null = null;
      await stream(`/api/commits/${commitment.id}/reveal`, (e) => {
        if (!rid && e.type === "status") {
          rid = (e as any).ids?.run_id ?? null;
          load();
        }
        setLive((p) => ({ runId: rid, events: [...(p?.events || []), e] }));
      }, { body: fd });
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
      load();
    }
  };

  const nextShown = nextEvents ?? exp.next_trace ?? [];

  return (
    <div>
      <a href="#/" className="text-sm text-mute hover:text-white">
        ← all experiments
      </a>
      <div className="mb-8 mt-2">
        <h1 className="text-3xl font-bold">{exp.name}</h1>
        <p className="mt-1 text-[#aab3c5]">{exp.description}</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {candidates.map((m) => (
            <span key={m} className="chip">
              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: MODEL_COLORS[m] }} />
              {MODEL_LABELS[m]} <span className="mono text-mute">{MODEL_EQ[m]}</span>
            </span>
          ))}
        </div>
      </div>
      {err && <div className="mb-4 rounded-lg border border-bad/50 bg-bad/10 p-3 text-sm text-bad">{err}</div>}

      <Step n={1} title="Observe and test" sub="Each model is fitted on the first 60% of the clip and must predict the rest. The ghosts show its predictions over the real motion.">
        {explore.map((r) => (
          <RunPanel key={r.id} runId={r.id} live={live?.runId === r.id ? live.events : null} onChanged={load} />
        ))}
        {explore.length === 0 && <UploadBox label="Upload & analyze" knobName={knobName} onSubmit={uploadExplore} busy={busy} />}
      </Step>

      {lastExplore?.comparison && (
        <Step
          n={2}
          title="What should I film next?"
          sub="When the evidence can't decide, Ghostline searches over the knob you control for the experiment where the models disagree most, then commits every model's prediction before you film."
        >
          {!commitment && (
            <button className="btn btn-primary" disabled={nextRunning || lastExplore.comparison.status !== "ambiguous"} onClick={() => runNext(false)}>
              🧪 Design experiment & commit predictions
            </button>
          )}
          {lastExplore.comparison.status !== "ambiguous" && !commitment && <div className="text-sm text-mute">The first run was already decisive, so no follow-up experiment is needed.</div>}
          <div className="grid gap-4 lg:grid-cols-5">
            <div className="space-y-4 lg:col-span-3">
              {exp.proposal && <DiscriminationCurve proposal={exp.proposal} currentKnob={lastExplore.knobs[knobName]} />}
              {exp.proposal && (
                <div className="card p-4 text-sm">
                  <div className="font-semibold">Filming instructions</div>
                  <div className="mt-1 text-[#c9cfdb]">{exp.proposal.filming_instructions}</div>
                  <div className="mt-2 text-xs text-mute">{exp.proposal.feasibility_note}</div>
                </div>
              )}
            </div>
            <div className="lg:col-span-2">
              {nextShown.length > 0 && <LabNotebook events={nextShown} running={nextRunning} title="Lab notebook: design & commit" />}
              {nextShown.length > 0 && !nextRunning && (
                <button className="btn btn-ghost mt-2" onClick={() => runNext(true)}>
                  ↻ Replay notebook
                </button>
              )}
            </div>
          </div>
          {commitment && <CommitCard c={commitment} />}
        </Step>
      )}

      {commitment && (
        <Step n={3} title="Reveal" sub="Film the committed experiment. Ghostline verifies the hash, reads only the starting state from the first 0.5 s, and lets the frozen predictions run. If the recording can't be trusted, it refuses to score.">
          {reveals.map((r) => (
            <div key={r.id} className="rounded-2xl border border-line/60 p-4">
              <RunPanel runId={r.id} live={live?.runId === r.id ? live.events : null} hash={commitment.sha256} onChanged={load} />
            </div>
          ))}
          {live && live.runId === null && <LabNotebook events={live.events} running title="Lab notebook: processing upload" />}
          <div>
            <div className="mb-2 text-sm font-semibold">Score a new run against commitment {commitment.sha256.slice(0, 12)}</div>
            <UploadBox label="Upload & score" knobName={knobName} defaultKnob={Object.values(commitment.knob_setting)[0]} onSubmit={uploadReveal} busy={busy} />
          </div>
        </Step>
      )}
    </div>
  );
}
