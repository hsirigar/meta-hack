import { useEffect, useState } from "react";
import type { Health } from "./api";
import { get } from "./api";
import ExperimentPage from "./pages/ExperimentPage";
import Home from "./pages/Home";

function useHash() {
  const [h, setH] = useState(location.hash);
  useEffect(() => {
    const f = () => {
      setH(location.hash);
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", f);
    return () => window.removeEventListener("hashchange", f);
  }, []);
  return h;
}

export default function App() {
  const hash = useHash();
  const [health, setHealth] = useState<Health | null>(null);
  useEffect(() => {
    get<Health>("/api/health").then(setHealth).catch(() => setHealth(null));
  }, []);
  const m = hash.match(/^#\/exp\/([\w-]+)/);
  const seg = health?.segment_backend === "meta" ? "sam-3.1" : health?.segment_backend;
  const vision = health?.scene_backend?.match(/\(([^)]+)\)/)?.[1] ?? health?.scene_backend;
  return (
    <div className="min-h-screen">
      <header className="border-b border-ink">
        <div className="mx-auto flex max-w-6xl flex-wrap items-baseline gap-x-6 gap-y-1 px-4 pb-3 pt-5 md:px-6">
          <a href="#/" className="serif text-[26px] font-semibold leading-none tracking-tight">
            Ghostline
          </a>
          <span className="serif text-[15px] italic text-ink2">an instrument that runs the scientific method on video</span>
          {health && (
            <dl className="mono ml-auto flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-mute">
              <div><dt className="inline">agent </dt><dd className="inline text-ink">{health.agent_model}</dd></div>
              <div><dt className="inline">vision </dt><dd className="inline text-ink">{vision}</dd></div>
              <div><dt className="inline">tracking </dt><dd className="inline text-ink">{seg}</dd></div>
              {health.demo_mode && <div className="text-ink2">· offline replay</div>}
            </dl>
          )}
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-8 md:px-6">{m ? <ExperimentPage id={m[1]} /> : <Home />}</main>
      <footer className="mx-auto max-w-6xl px-4 md:px-6">
        <div className="hair flex flex-wrap justify-between gap-2 py-5 text-xs text-mute">
          <span>Ghostline · Meta Global AI Developer Hackathon, Nov 2026</span>
          <span>
            Muse Wonder (agent) · Muse Spark (scene) · SAM 3.1 (tracking) ·{" "}
            <a className="link" href="https://github.com/hsirigar/meta-hack">source</a>
          </span>
          {health?.muse_spark_open && <span className="w-full font-medium text-ink">Built with Muse Spark Open</span>}
        </div>
      </footer>
    </div>
  );
}
