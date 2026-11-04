import { useState } from "react";
import { API } from "../api";

export default function GhostPlayer({ runId, hasOverlay, version }: { runId: string; hasOverlay: boolean; version?: string }) {
  const [raw, setRaw] = useState(false);
  const ghosts = hasOverlay && !raw;
  const src = `${API}/api/runs/${runId}/${ghosts ? "overlay" : "video"}.mp4${version ? `?v=${version}` : ""}`;
  return (
    <figure className="border border-ink bg-[#151412]">
      <video key={src} src={src} className="max-h-[480px] w-full object-contain" controls autoPlay muted loop playsInline />
      <figcaption className="flex items-center justify-between border-t border-ink bg-sheet px-3 py-2 text-xs">
        <span className="text-ink2">
          <span className="label mr-2 !text-ink">Fig.</span>
          {ghosts ? "Each translucent ghost is one law's prediction, drawn over the real recording." : "Original recording."}
        </span>
        {hasOverlay && (
          <button className="link text-xs" onClick={() => setRaw(!raw)}>
            {raw ? "show ghosts" : "show original"}
          </button>
        )}
      </figcaption>
    </figure>
  );
}
