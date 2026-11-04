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
    <section className="rule pb-14 pt-4">
      <div className="grid gap-x-8 gap-y-2 md:grid-cols-[9rem_1fr]">
        <div className="mono text-[13px] text-mute">§{n}</div>
        <div>
          <h2 className="serif text-[28px] font-medium leading-tight">{title}</h2>
          {sub && <p className="prose-serif mt-1 max-w-3xl !text-[16px]">{sub}</p>}
        </div>
      </div>
      <div className="mt-6 space-y-5">{children}</div>
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
      <a href="#/" className="label hover:text-ink">
        ← Experiments
      </a>
      <div className="mb-10 mt-3 grid gap-8 md:grid-cols-[1.3fr_1fr]">
        <div>
          <h1 className="serif text-[40px] font-medium leading-tight tracking-tight">{exp.name}</h1>
          <p className="prose-serif mt-2">{exp.description}</p>
        </div>
        <div className="self-end">
          <div className="label mb-2">Candidate laws</div>
          <ul className="rule">
            {(commitment ? commitment.models.map((x) => x.name) : candidates).map((m) => (
              <li key={m} className="hair flex items-baseline gap-3 py-2 text-sm">
                <span className="swatch shrink-0" style={{ background: MODEL_COLORS[m] }} />
                <span className="w-44 shrink-0">{MODEL_LABELS[m]}</span>
                <span className="mono text-[12px] text-ink2">{MODEL_EQ[m]}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
      {err && <div className="mb-4 border border-bad/50 bg-bad/5 p-3 text-sm text-bad">{err}</div>}

      <Step n={1} title="Observe and test" sub="Each law is fitted on the first 60% of the clip and must predict the rest, which it never saw. The ghosts are those predictions, drawn over the real motion.">
        {explore.map((r) => (
          <RunPanel key={r.id} runId={r.id} live={live?.runId === r.id ? live.events : null} onChanged={load} />
        ))}
        {explore.length === 0 && <UploadBox label="Upload & analyze" knobName={knobName} onSubmit={uploadExplore} busy={busy} />}
      </Step>

      {lastExplore?.comparison && (
        <Step
          n={2}
          title="What to film next"
          sub="When the evidence can't decide, Ghostline searches the setting you control for the experiment where the laws disagree most, then seals every law's prediction before you film."
        >
          {!commitment && (
            <button className="btn btn-primary" disabled={nextRunning || lastExplore.comparison.status !== "ambiguous"} onClick={() => runNext(false)}>
              Design the next experiment and seal predictions
            </button>
          )}
          {lastExplore.comparison.status !== "ambiguous" && !commitment && <div className="text-sm text-ink2">The first run was already decisive, so no follow-up experiment is needed.</div>}
          <div className="grid gap-4 lg:grid-cols-5">
            <div className="space-y-4 lg:col-span-3">
              {exp.proposal && <DiscriminationCurve proposal={exp.proposal} currentKnob={lastExplore.knobs[knobName]} />}
              {exp.proposal && (
                <div className="border-l-2 border-ink pl-4">
                  <div className="label">Instructions for the experimenter</div>
                  <div className="serif mt-1 text-[18px] leading-snug">{exp.proposal.filming_instructions}</div>
                  <div className="mt-2 text-[13px] leading-relaxed text-ink2">{exp.proposal.feasibility_note}</div>
                </div>
              )}
            </div>
            <div className="lg:col-span-2">
              {nextShown.length > 0 && <LabNotebook events={nextShown} running={nextRunning} title="Agent log: design and seal" />}
              {nextShown.length > 0 && !nextRunning && (
                <button className="btn btn-ghost mb-4 mt-3" onClick={() => runNext(true)}>
                  Replay log
                </button>
              )}
            </div>
          </div>
          {commitment && <CommitCard c={commitment} />}
        </Step>
      )}

      {commitment && (
        <Step n={3} title="Reveal" sub="The new recording is scored against the sealed predictions. Ghostline checks the seal, reads only the starting state from the first 0.5 s, and lets the frozen predictions run. If the recording can't be trusted, it refuses to score.">
          {reveals.map((r) => (
            <div key={r.id} className="pb-6">
              <RunPanel runId={r.id} live={live?.runId === r.id ? live.events : null} hash={commitment.sha256} onChanged={load} />
            </div>
          ))}
          {live && live.runId === null && <LabNotebook events={live.events} running title="Agent log: processing upload" />}
          <div>
            <div className="label mb-2">Score another recording against seal <span className="mono normal-case tracking-normal text-ink">{commitment.sha256.slice(0, 12)}</span></div>
            <UploadBox label="Upload & score" knobName={knobName} defaultKnob={Object.values(commitment.knob_setting)[0]} onSubmit={uploadReveal} busy={busy} />
          </div>
        </Step>
      )}
    </div>
  );
}
