import { useEffect, useState } from "react";
import type { Experiment } from "../api";
import { MODEL_LABELS, get, post } from "../api";

function status(e: Experiment): { text: string; tone: string } {
  const reveal = e.runs.filter((r) => r.role === "reveal");
  const scored = reveal.find((r) => r.verdict?.winner);
  if (scored?.verdict?.all_models_inadequate) return { text: "No candidate fits", tone: "text-bad" };
  if (scored) return { text: `${MODEL_LABELS[scored.verdict!.winner!] ?? scored.verdict!.winner} confirmed`, tone: "text-ok" };
  if (e.commit_ids.length) return { text: "Predictions sealed, awaiting reveal", tone: "text-seal" };
  if (e.runs.some((r) => r.comparison?.status === "ambiguous")) return { text: "Inconclusive", tone: "text-warn" };
  return { text: e.runs.length ? "Analysed" : "No runs yet", tone: "text-mute" };
}

const isReal = (e: Experiment) => /real/i.test(e.name) || /real phone footage/i.test(e.description);

const METHOD = [
  ["Observe", "Track the object in a phone video and convert pixels to calibrated measurements."],
  ["Test", "Fit competing models on the first 60% of the clip; each must predict the rest unseen."],
  ["Admit doubt", "If the predictions differ by less than the noise, say so: inconclusive."],
  ["Design", "Search the setting you control for the experiment that would separate them."],
  ["Seal", "Freeze every model's prediction and publish its SHA-256 before anything is filmed."],
  ["Reveal", "Score the new video against the sealed predictions, or refuse if the recording can't be trusted."],
];

export default function Home() {
  const [exps, setExps] = useState<Experiment[] | null>(null);
  const [name, setName] = useState("");
  const [sys, setSys] = useState("pendulum");
  useEffect(() => {
    get<Experiment[]>("/api/experiments").then(setExps);
  }, []);
  const create = async () => {
    const e = await post<Experiment>("/api/experiments", { name: name || "New experiment", system_type: sys });
    location.hash = `#/exp/${e.id}`;
  };
  return (
    <div className="space-y-14">
      <section className="grid gap-10 pt-2 md:grid-cols-[1.25fr_1fr]">
        <div>
          <h1 className="serif text-[40px] font-medium leading-[1.08] tracking-tight md:text-[52px]">
            Film a physical system.
            <br />
            <span className="italic">See which equation is wrong.</span>
          </h1>
          <p className="prose-serif mt-5 max-w-xl">
            Ghostline tracks the motion, fits competing laws on part of the footage, and draws each law's prediction over the rest as a translucent ghost. When the
            evidence can't decide, it designs the experiment that will, and seals its predictions before you film it.
          </p>
        </div>
        <ol className="self-end text-sm">
          {METHOD.map(([t, d], i) => (
            <li key={t} className="hair grid grid-cols-[2.2rem_1fr] gap-x-2 py-2.5 first:border-t-ink">
              <span className="mono text-mute">{String(i + 1).padStart(2, "0")}</span>
              <span>
                <span className="font-medium">{t}.</span> <span className="text-ink2">{d}</span>
              </span>
            </li>
          ))}
        </ol>
      </section>

      <section>
        <div className="mb-3 flex items-baseline justify-between">
          <h2 className="label">Experiments</h2>
          <span className="label">{exps?.length ?? "–"} on file</span>
        </div>
        <div className="rule overflow-x-auto">
          <table className="w-full min-w-[640px] text-left text-sm">
            <thead>
              <tr className="border-b border-line">
                <th className="label py-2 pr-4 font-normal">No.</th>
                <th className="label py-2 pr-4 font-normal">Experiment</th>
                <th className="label py-2 pr-4 font-normal">Footage</th>
                <th className="label py-2 pr-4 text-right font-normal">Runs</th>
                <th className="label py-2 font-normal">Status</th>
              </tr>
            </thead>
            <tbody>
              {exps?.map((e, i) => {
                const st = status(e);
                return (
                  <tr key={e.id} className="group border-b border-line align-top">
                    <td className="mono py-4 pr-4 text-mute">{String(i + 1).padStart(3, "0")}</td>
                    <td className="py-4 pr-4">
                      <a href={`#/exp/${e.id}`} className="serif text-[19px] font-medium underline decoration-line underline-offset-4 group-hover:decoration-ink">
                        {e.name}
                      </a>
                      <div className="mt-1 max-w-xl text-[13px] leading-snug text-ink2">{e.description}</div>
                    </td>
                    <td className="py-4 pr-4 text-[13px]">{isReal(e) ? <span className="font-medium">Real, iPhone</span> : <span className="text-ink2">Rendered, known truth</span>}</td>
                    <td className="mono py-4 pr-4 text-right">{e.runs.length}</td>
                    <td className={`py-4 text-[13px] font-medium ${st.tone}`}>{st.text}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {exps && exps.length === 0 && <div className="py-6 text-mute">No experiments yet.</div>}
        </div>
      </section>

      <section className="grid gap-6 md:grid-cols-[1fr_1.4fr]">
        <div>
          <h2 className="label">New experiment</h2>
          <p className="mt-2 text-sm text-ink2">Name a setup, then upload a video of it. Keep the camera still and a card or coin of known size in view.</p>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1">
            <span className="label">Name</span>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Kitchen pendulum" className="field w-56" />
          </label>
          <label className="flex flex-col gap-1">
            <span className="label">System</span>
            <select value={sys} onChange={(e) => setSys(e.target.value)} className="field">
              <option value="pendulum">Pendulum</option>
              <option value="falling_drag">Falling object with drag</option>
            </select>
          </label>
          <button className="btn btn-primary" onClick={create}>
            Create
          </button>
        </div>
      </section>
    </div>
  );
}
