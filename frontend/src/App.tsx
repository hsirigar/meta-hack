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
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 border-b border-line bg-ink/85 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center gap-3 px-4 py-3">
          <a href="#/" className="flex items-center gap-2 text-lg font-bold tracking-tight">
            <span className="text-2xl">👻</span> Ghostline
          </a>
          <span className="hidden text-sm text-mute md:inline">the scientific method, run by an agent; you are the hands</span>
          {health && (
            <div className="ml-auto hidden flex-wrap gap-2 text-xs lg:flex">
              <span className="chip" title="agent">🧠 {health.agent_model}</span>
              <span className="chip" title="scene understanding">👁 {health.scene_backend}</span>
              <span className="chip" title="segmentation">◎ {health.segment_backend === "meta" ? "SAM 3.1" : health.segment_backend}</span>
              {health.demo_mode && <span className="chip border-accent/60 text-accent">demo mode (offline)</span>}
            </div>
          )}
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-8">{m ? <ExperimentPage id={m[1]} /> : <Home />}</main>
      <footer className="border-t border-line py-6 text-center text-xs text-mute">
        Ghostline · Meta Global AI Developer Hackathon 2026 · Muse Wonder agent · Muse Spark scene understanding · SAM 3.1 tracking
        {health?.muse_spark_open && <div className="mt-1 font-semibold text-white">Built with Muse Spark Open</div>}
      </footer>
    </div>
  );
}
