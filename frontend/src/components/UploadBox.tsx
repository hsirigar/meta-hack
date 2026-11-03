import { useState } from "react";

export default function UploadBox({ label, knobName, defaultKnob, onSubmit, busy }: { label: string; knobName: string; defaultKnob?: number; onSubmit: (f: File, knobs: Record<string, number>) => void; busy?: boolean }) {
  const [file, setFile] = useState<File | null>(null);
  const [knob, setKnob] = useState<string>(defaultKnob !== undefined ? String(defaultKnob) : "");
  const knobLabel = knobName === "release_angle_deg" ? "Release angle (°)" : "Number of filters";
  return (
    <div className="card flex flex-wrap items-end gap-4 border-dashed p-4">
      <label className="flex flex-col text-xs text-mute">
        Video (mp4 / mov)
        <input type="file" accept="video/*" className="mt-1 text-sm text-white file:mr-3 file:rounded-md file:border-0 file:bg-panel2 file:px-3 file:py-1.5 file:text-white" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </label>
      <label className="flex flex-col text-xs text-mute">
        {knobLabel} (optional)
        <input value={knob} onChange={(e) => setKnob(e.target.value)} className="mono mt-1 w-32 rounded-md border border-line bg-panel2 px-2 py-1.5 text-sm text-white" />
      </label>
      <button className="btn btn-primary" disabled={!file || busy} onClick={() => file && onSubmit(file, knob ? { [knobName]: Number(knob) } : {})}>
        {busy ? "Working…" : label}
      </button>
    </div>
  );
}
