import { useEffect, useState } from "react";
import type { Experiment } from "../api";
import { get, post } from "../api";

const STEP_LABEL = (e: Experiment) => {
  const reveal = e.runs.filter((r) => r.role === "reveal");
  const winner = reveal.find((r) => r.verdict?.winner)?.verdict?.winner;
  if (winner) return { text: `resolved: ${winner.replace("_", " ")}`, cls: "text-ok" };
  if (e.commit_ids.length) return { text: "predictions committed, awaiting reveal", cls: "text-accent" };
  const amb = e.runs.find((r) => r.comparison?.status === "ambiguous");
  if (amb) return { text: "inconclusive, design next experiment", cls: "text-warn" };
  return { text: `${e.runs.length} run(s)`, cls: "text-mute" };
};

export default function Home() {
  const [exps, setExps] = useState<Experiment[] | null>(null);
  const [name, setName] = useState("");
  const [sys, setSys] = useState("pendulum");
  const load = () => get<Experiment[]>("/api/experiments").then(setExps);
  useEffect(() => {
    load();
  }, []);
  const create = async () => {
    const e = await post<Experiment>("/api/experiments", { name: name || "New experiment", system_type: sys });
    location.hash = `#/exp/${e.id}`;
  };
  return (
    <div className="space-y-10">
      <section className="pt-6">
        <h1 className="text-4xl font-bold tracking-tight md:text-5xl">
          Film a physical system. <span className="text-accent">Watch the equations race reality.</span>
        </h1>
        <p className="mt-4 max-w-3xl text-lg text-[#aab3c5]">
          Ghostline tracks the motion, fits competing physical models on footage it is allowed to see, and draws each model's prediction as a ghost over the
          footage it wasn't. When the evidence can't decide, it designs the experiment that will and cryptographically commits its predictions before you film.
        </p>
        <div className="mt-5 flex flex-wrap gap-2 text-xs">
          {["Observe", "Fit on 60%", "Predict the rest", "Inconclusive?", "Design experiment", "Commit hash", "Film", "Verdict"].map((s, i) => (
            <span key={s} className="chip">
              <span className="text-accent">{i + 1}</span> {s}
            </span>
          ))}
        </div>
      </section>
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-mute">Experiments</h2>
        <div className="grid gap-4 md:grid-cols-2">
          {exps?.map((e) => {
            const s = STEP_LABEL(e);
            return (
              <a key={e.id} href={`#/exp/${e.id}`} className="card block p-5 transition hover:border-accent/60">
                <div className="flex items-center gap-2">
                  <span className="text-2xl">{e.system_type === "pendulum" ? "🕰" : "🪶"}</span>
                  <div className="text-lg font-semibold">{e.name}</div>
                </div>
                <div className="mt-1 text-sm text-[#aab3c5]">{e.description}</div>
                <div className={`mt-3 text-sm ${s.cls}`}>{s.text}</div>
                <div className="mt-1 text-xs text-mute">
                  {e.runs.length} runs · {e.commit_ids.length} commitments
                </div>
              </a>
            );
          })}
          {exps && exps.length === 0 && <div className="text-mute">No experiments yet.</div>}
        </div>
      </section>
      <section className="card flex flex-wrap items-end gap-3 p-5">
        <div className="w-full text-sm font-semibold">Start your own experiment</div>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Kitchen pendulum" className="rounded-md border border-line bg-panel2 px-3 py-2 text-sm" />
        <select value={sys} onChange={(e) => setSys(e.target.value)} className="rounded-md border border-line bg-panel2 px-3 py-2 text-sm">
          <option value="pendulum">Pendulum</option>
          <option value="falling_drag">Falling object with drag</option>
        </select>
        <button className="btn btn-primary" onClick={create}>
          Create
        </button>
      </section>
    </div>
  );
}
