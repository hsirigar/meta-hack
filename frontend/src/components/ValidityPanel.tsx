export default function ValidityPanel({ validity }: { validity: any }) {
  if (!validity) return null;
  const color = { valid: "text-ok", warning: "text-warn", invalid: "text-bad" }[validity.status as string];
  const icon = (s: string) => (s === "pass" ? "✓" : s === "warning" ? "!" : "✕");
  const ic = (s: string) => (s === "pass" ? "text-ok" : s === "warning" ? "text-warn" : "text-bad");
  return (
    <div className={`card p-4 ${validity.status === "invalid" ? "border-bad/60" : ""}`}>
      <div className="mb-2 flex items-center gap-2">
        <span className="text-sm font-semibold">Validity</span>
        <span className={`chip ${color}`}>{validity.status.toUpperCase()}</span>
      </div>
      <ul className="space-y-1 text-sm">
        {validity.checks.map((c: any) => (
          <li key={c.name} className="flex gap-2">
            <span className={`mono w-4 ${ic(c.status)}`}>{icon(c.status)}</span>
            <span className={c.status === "pass" ? "text-[#aab3c5]" : "text-white"}>{c.reason}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
