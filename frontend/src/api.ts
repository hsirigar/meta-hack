export const API = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

export type AgentEvent = {
  type: "status" | "thought" | "tool_call" | "tool_result" | "message" | "error" | "done";
  ts?: number;
  text?: string;
  name?: string;
  id?: string;
  args?: Record<string, unknown>;
  result?: any;
  ok?: boolean;
  replay?: boolean;
};

export type RunSummary = {
  id: string;
  role: "exploration" | "reveal";
  knobs: Record<string, number>;
  created_at: string;
  commitment_id: string | null;
  validity: "valid" | "warning" | "invalid" | null;
  comparison: { status: string; best: string | null; max_separation_sigma: number } | null;
  verdict: { winner: string | null; hash_verified: boolean } | null;
  has_overlay: boolean;
  trace_tasks: string[];
};

export type Commitment = {
  id: string;
  experiment_id: string;
  created_at: string;
  knob_setting: Record<string, number>;
  models: { name: string; params: Record<string, number>; param_std: Record<string, number>; knob_transfer: string; predicted_observables: Record<string, number> }[];
  protocol: Record<string, any>;
  canonical_json: string;
  sha256: string;
};

export type Proposal = {
  knob: string;
  value: number;
  expected_separation_sigma: number;
  expected_max_separation_sigma: number;
  discrimination_curve: [number, number][];
  feasibility_note: string;
  filming_instructions: string;
  models: string[];
};

export type Experiment = {
  id: string;
  name: string;
  system_type: "pendulum" | "falling_drag";
  description: string;
  run_ids: string[];
  commit_ids: string[];
  proposal: Proposal | null;
  runs: RunSummary[];
  commitments?: Commitment[];
  next_trace?: AgentEvent[] | null;
};

export type Health = {
  demo_mode: boolean;
  scene_backend: string;
  segment_backend: string;
  llm_backend: string;
  agent_model: string;
  muse_spark_open: boolean;
};

export async function get<T>(path: string): Promise<T> {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export async function post<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export async function upload<T>(path: string, file: File, knobs: Record<string, number>): Promise<T> {
  const fd = new FormData();
  fd.append("video", file);
  fd.append("knobs", JSON.stringify(knobs));
  const r = await fetch(API + path, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

/** POST that returns an SSE stream (fetch-based so we can POST multipart / JSON). */
export async function stream(
  path: string,
  onEvent: (e: AgentEvent) => void,
  init?: { body?: BodyInit },
): Promise<void> {
  const r = await fetch(API + path, { method: "POST", body: init?.body });
  if (!r.ok || !r.body) throw new Error(`${r.status} ${await r.text()}`);
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.search(/\r?\n\r?\n/)) >= 0) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx).replace(/^\r?\n\r?\n/, "");
      const data = chunk
        .split(/\r?\n/)
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trimStart())
        .join("\n");
      if (data) {
        try {
          onEvent(JSON.parse(data));
        } catch {
          /* ignore keep-alives */
        }
      }
    }
  }
}

export const MODEL_COLORS: Record<string, string> = {
  pendulum_linear: "#00beff",
  pendulum_nonlinear: "#ff3cc8",
  pendulum_nonlinear_quaddamp: "#ffe600",
  drag_linear: "#00beff",
  drag_quadratic: "#ff3cc8",
};

export const MODEL_LABELS: Record<string, string> = {
  pendulum_linear: "Linear (small-angle)",
  pendulum_nonlinear: "Nonlinear (sin θ)",
  pendulum_nonlinear_quaddamp: "Nonlinear + quadratic damping",
  drag_linear: "Linear drag",
  drag_quadratic: "Quadratic drag",
};

export const MODEL_EQ: Record<string, string> = {
  pendulum_linear: "θ'' + 2γθ' + ω₀²θ = 0",
  pendulum_nonlinear: "θ'' + 2γθ' + ω₀² sin θ = 0",
  pendulum_nonlinear_quaddamp: "θ'' + c·θ'|θ'| + ω₀² sin θ = 0",
  drag_linear: "v' = g(1 − v/vₜ)",
  drag_quadratic: "v' = g(1 − (v/vₜ)²)",
};

export const fmt = (x: number | null | undefined, d = 3) =>
  x === null || x === undefined || Number.isNaN(x) ? "—" : Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(2) : x.toFixed(d);
