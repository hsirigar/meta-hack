import { useState } from "react";

export default function UploadBox({ label, knobName, defaultKnob, onSubmit, busy }: { label: string; knobName: string; defaultKnob?: number; onSubmit: (f: File, knobs: Record<string, number>) => void; busy?: boolean }) {
  const [file, setFile] = useState<File | null>(null);
  const [knob, setKnob] = useState<string>(defaultKnob !== undefined ? String(defaultKnob) : "");
  const knobLabel = knobName === "release_angle_deg" ? "Release angle (°)" : "Number of filters";
  return (
    <div className="flex flex-wrap items-end gap-4 border border-dashed border-ink/40 bg-sheet p-4">
      <label className="flex flex-col gap-1">
        <span className="label">Video (mp4 / mov)</span>
        <input type="file" accept="video/*" className="text-sm file:mr-3 file:border file:border-line file:bg-paper file:px-3 file:py-1.5 file:text-ink" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </label>
      <label className="flex flex-col gap-1">
        <span className="label">{knobLabel}, optional</span>
        <input value={knob} onChange={(e) => setKnob(e.target.value)} className="field mono w-32" />
      </label>
      <button className="btn btn-primary" disabled={!file || busy} onClick={() => file && onSubmit(file, knob ? { [knobName]: Number(knob) } : {})}>
        {busy ? "Working…" : label}
      </button>
    </div>
  );
}
