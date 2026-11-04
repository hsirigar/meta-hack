export default function ValidityPanel({ validity }: { validity: any }) {
  if (!validity) return null;
  const tone = { valid: "text-ok", warning: "text-warn", invalid: "text-bad" }[validity.status as string];
  const mark = (s: string) => (s === "pass" ? "pass" : s === "warning" ? "warn" : "FAIL");
  const mc = (s: string) => (s === "pass" ? "text-mute" : s === "warning" ? "text-warn" : "text-bad font-semibold");
  return (
    <div className={`card p-4 ${validity.status === "invalid" ? "!border-bad" : ""}`}>
      <div className="mb-3 flex items-baseline justify-between">
        <span className="label">Can this recording be trusted?</span>
        <span className={`serif text-[18px] font-medium ${tone}`}>{validity.status === "valid" ? "Yes" : validity.status === "warning" ? "Yes, with caveats" : "No"}</span>
      </div>
      <ul className="text-[13px]">
        {validity.checks.map((c: any) => (
          <li key={c.name} className="hair grid grid-cols-[2.6rem_1fr] gap-2 py-1.5">
            <span className={`mono text-[11px] uppercase ${mc(c.status)}`}>{mark(c.status)}</span>
            <span className={c.status === "pass" ? "text-ink2" : "text-ink"}>{c.reason}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
