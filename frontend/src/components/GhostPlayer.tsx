import { useState } from "react";
import { API } from "../api";

export default function GhostPlayer({ runId, hasOverlay, version }: { runId: string; hasOverlay: boolean; version?: string }) {
  const [raw, setRaw] = useState(false);
  const src = `${API}/api/runs/${runId}/${hasOverlay && !raw ? "overlay" : "video"}.mp4${version ? `?v=${version}` : ""}`;
  return (
    <div className="card overflow-hidden">
      <video key={src} src={src} className="max-h-[480px] w-full bg-black object-contain" controls autoPlay muted loop playsInline />
      <div className="flex items-center justify-between px-3 py-2 text-xs text-mute">
        <span>{hasOverlay && !raw ? "Ghost overlay: each colour is one model's prediction" : "Original video"}</span>
        {hasOverlay && (
          <button className="hover:text-white" onClick={() => setRaw(!raw)}>
            {raw ? "show ghosts" : "show original"}
          </button>
        )}
      </div>
    </div>
  );
}
