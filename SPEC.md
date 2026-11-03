# Ghostline — Build Specification

> **Film a physical system. Get candidate equations. Watch their predictions appear as ghosts over reality — and when the evidence can't decide, Ghostline designs the experiment that will.**

Built for the **Meta Global AI Developer Hackathon** (Cerebral Valley), track: **Push the Frontier**.
Hard deadline: **Wednesday, Nov 4, 2026, 9:00 PM ET** (submission on the CV platform).

---

## 0. Instructions for Claude Code

- Read this entire file before writing code. It is the source of truth.
- Build in the **priority order in §15** (P0 → P1 → P2). A working, reliable P0 demo beats a broken P2.
- **Numbers come from code, never from the LLM.** The agent reasons, chooses, and explains; deterministic Python tools do all measurement, fitting, simulation, and scoring.
- **Leakage prevention is enforced in code, not in prompts** (see §8). Write the tests for it.
- Cache every external call (SAM, LLM) to disk keyed by input hash. The demo must replay without network calls.
- Where this spec says **VERIFY**, check the official Meta Model API docs (https://ai.developer.meta.com/docs/ / dev.meta.ai) before implementing, and isolate the call behind an interface so it can be swapped.
- Keep commits small and descriptive. The hackathon requires that judges can see what was built during the event.

---

## 1. The idea in one paragraph

A user films a simple physical system with a phone (a pendulum, a falling coffee filter). Ghostline understands the scene, tracks the moving object with SAM 3.1, converts pixels to calibrated physical measurements, and fits several competing dynamical models. It tests each model on footage it was **not** allowed to see during fitting and draws each model's prediction as a translucent "ghost" over the real video. When several models explain the data equally well, Ghostline says so, computes which new experiment would best separate them, and **cryptographically commits its predictions before the new experiment is filmed**. When the user films the new experiment, one ghost follows reality and the other drifts away. If the new recording is invalid (camera bumped, scale changed, tracking lost), Ghostline refuses to score it.

**Pitch line:** *An agent that runs the scientific method on its own; the human only has to be the hands.*

---

## 2. Hackathon constraints (must satisfy)

| Rule | How we satisfy it |
|---|---|
| Must use Meta models | Muse Wonder (agent), Muse Spark (scene understanding), SAM 3.1 (tracking) via Meta Model API; Muse Spark Open via Baseten (open-weights bonus); optional Muse Voice Transcribe |
| Public repo | Public GitHub repo from day one |
| New work only, clearly identified | Fresh repo created during the event; README section "Built during the hackathon" listing every component |
| 3-minute demo video | Script in §16 |
| Avoid anti-projects | No medical/biological framing. Not an education chatbot: it is a measurement-and-reasoning instrument |
| Muse Spark Open license | If used, display **"Built with Muse Spark Open"** in the UI footer and README |
| Secrets | `.env` only, never committed; `.env.example` committed |

Judging weights for prioritization: Impact 30%, Use of Meta tooling 25%, Technical depth 20%, Demo 25%. Bonus prizes in reach: Best Open-Weight Build, Best Agentic Use-Case, Most Viral.

---

## 3. The demo (north star — everything serves this)

**Scene A — Pendulum (the main story)**
1. Upload a video of a pendulum released at a **small angle (~8°)**.
2. Ghostline identifies the pendulum, pivot, and a reference object; tracks the bob; extracts θ(t).
3. It fits two models on the first 60% of the clip: **linear** (θ'' + 2γθ' + ω₀²θ = 0) and **nonlinear** (θ'' + 2γθ' + ω₀² sin θ = 0).
4. Both predict the held-out 40% nearly perfectly; both ghosts overlap the real bob. Ghostline reports: **"Inconclusive. At this amplitude the models differ by less than the measurement noise."**
5. It runs experiment design over the release angle and recommends **~70°**, reporting the expected separation in units of measurement noise.
6. It **commits** both models' predictions for the new experiment: canonical JSON → SHA-256 hash shown on screen with a timestamp.
7. Upload the 70° video. Ghostline verifies the commitment hash, conditions only on the first 0.5 s to get the initial state, and plays the rest with both ghosts. The **nonlinear ghost tracks the bob; the linear ghost drifts ahead** (its period is ~10% too short at 70°). Verdict card with errors and the winning model.

**Scene B — Coffee filters (shows it is not a textbook trick)**
1. One filter dropped from ~2 m. **Linear drag** and **quadratic drag** both fit (it is near terminal velocity almost immediately).
2. Ghostline recommends **stacking 4 filters**: linear drag predicts terminal velocity ×4, quadratic predicts ×2.
3. Commit → film → reveal. Quadratic wins.

**Scene C — Invalid experiment (trust)**
1. Upload a video where the camera is bumped mid-recording.
2. Ghostline detects reference-object drift, marks the run **INVALID**, explains why, and refuses to score the commitment.

---

## 4. Architecture

```
            ┌───────────── Frontend (React + TS) ─────────────┐
            │ Upload · Run view · Ghost player · Plots ·       │
            │ Lab notebook (agent trace) · Commit / Reveal      │
            └───────────────────────┬──────────────────────────┘
                                    │ REST + SSE (agent trace stream)
┌───────────────────────────────────▼──────────────────────────────────┐
│ Backend (FastAPI, Python 3.11)                                       │
│                                                                      │
│  ingest ──► scene (Muse Spark / Spark Open) ──► segment (SAM 3.1)    │
│     │                                              │                 │
│     ▼                                              ▼                 │
│  frame timestamps                   measure + calibrate + validity   │
│                                              │                       │
│                       ┌──────────────────────▼───────────────┐       │
│                       │ Agent (Muse Wonder, tool calling)    │       │
│                       │  tools: fit · compare · design ·     │       │
│                       │         commit · evaluate · render   │       │
│                       └──────────────────────┬───────────────┘       │
│                                              ▼                       │
│                         numerics (numpy/scipy/pysindy)               │
│                         render (OpenCV → H.264 mp4)                  │
│  storage: ./data/{experiments,runs,commits,cache}  (JSON + files)    │
└──────────────────────────────────────────────────────────────────────┘
```

**Stack**
- Backend: Python 3.11, FastAPI, Pydantic v2, numpy, scipy, opencv-python, pysindy, ffmpeg (system), `openai` SDK (pointed at Meta), httpx, pytest.
- Frontend: Vite + React + TypeScript + Tailwind; Plotly.js for plots; native `<video>` for rendered overlay videos.
- Storage: local filesystem JSON (no DB needed for the hackathon).
- Deploy: backend on Modal or Render; frontend on Vercel. Precomputed demo experiments must be viewable with zero uploads.

---

## 5. Meta model usage

All calls go through `backend/ghostline/clients/`. Each client has an interface plus a cached implementation.

**Meta Model API basics**
- Base URL: `https://api.meta.ai/v1`. Auth: `Authorization: Bearer $MODEL_API_KEY`.
- Supports OpenAI **Responses** (`/v1/responses`), **Chat Completions** (`/v1/chat/completions`), and Anthropic-style **Messages**. Use the OpenAI Python SDK with `base_url` set and `api_key=os.environ["MODEL_API_KEY"]` passed explicitly (the SDK does not read `MODEL_API_KEY` automatically).
- Model IDs come from env vars. **VERIFY** current IDs in the model list:
  - `AGENT_MODEL=muse-wonder-1.0`
  - `VISION_MODEL=muse-spark-1.1` (check whether a newer Spark ID exists)
  - `SAM_MODEL=sam-3.1`

| Role | Model | Notes |
|---|---|---|
| Agent / reasoning | Muse Wonder | Responses API with function tools. The agent loop lives in our code. |
| Scene understanding | Muse Spark (API) **or** Muse Spark Open (Baseten) | Keyframes → strict JSON (§7.2). Config flag `SCENE_BACKEND=meta|baseten`. Use Baseten for the open-weights bonus. |
| Tracking | SAM 3.1 | Text-prompted detection + segmentation + video tracking. |
| Narration (P2) | Muse Voice Transcribe | Timestamped spoken interventions ("releasing now"). |

**SAM 3.1 backends** (`SegmentationBackend` interface)
- `MetaApiSam`: SAM 3.1 on the Meta Model API (reported as served via the Responses API under `sam-3.1`). **VERIFY** the request format (video upload vs frames, prompt field, output: masks / RLE / boxes / track IDs / scores) and size limits; downscale to ≤720p and cap at 60 fps if needed.
- `LocalSam` (fallback, also counts as open weights): official SAM 3 repo + `facebook/sam3.1` checkpoint on a Modal GPU function.
- Output normalized to: `{track_id: [{frame_idx, t, mask_rle | polygon, bbox, score}]}`.

**Muse Spark Open on Baseten**
- Model string for the hackathon: `meta-models/muse-spark-open-1026` (the Baseten docs show a different string; use this one). OpenAI-compatible endpoint. **VERIFY** the base URL in the Baseten quickstart.

**Caching**: `cache/{sha256(model + request_json + file_hashes)}.json`. `DEMO_MODE=1` reads only from cache and never calls the network.

---

## 6. Data model (Pydantic)

```python
class Experiment(BaseModel):          # one physical setup, many runs
    id: str
    name: str                          # "Pendulum A"
    system_type: Literal["pendulum", "falling_drag", "spring_mass", "unknown"]
    run_ids: list[str]
    commit_ids: list[str]

class Run(BaseModel):                  # one video
    id: str
    experiment_id: str
    video_path: str
    video_sha256: str
    fps_nominal: float
    frame_times: list[float]           # real per-frame timestamps from ffprobe (phones are VFR)
    scene: SceneDescription | None
    tracks: dict[str, Track] | None
    calibration: Calibration | None
    series: dict[str, TimeSeries] | None   # e.g. "theta", "y"
    validity: ValidityReport | None
    split: Split | None                # train/holdout boundary, fixed at creation
    knobs: dict[str, float]            # experimental settings e.g. {"release_angle_deg": 8} or {"n_filters": 1}

class SceneDescription(BaseModel):
    system_type: str
    tracked_concepts: list[str]        # SAM prompts, e.g. ["pendulum bob"]
    reference_object: ReferenceObject  # e.g. {"concept": "US quarter", "known_size_m": 0.02426, "dimension": "diameter"}
    anchor_hint: str | None            # "pivot at top center"
    gravity_direction_hint: str | None
    notes: str
    confidence: float

class Calibration(BaseModel):
    meters_per_pixel: float
    meters_per_pixel_std: float
    method: Literal["reference_object", "ruler_points", "manual"]
    homography: list[list[float]] | None
    pivot_px: tuple[float, float] | None
    length_px: float | None

class TimeSeries(BaseModel):
    name: str
    unit: str
    t: list[float]
    y: list[float]
    y_smooth: list[float]
    noise_sigma: float                 # estimated from residuals of the smoother
    valid_mask: list[bool]

class Split(BaseModel):
    train_end_t: float                 # holdout = t > train_end_t; never changes after creation
    conditioning_window_s: float = 0.5

class ModelSpec(BaseModel):
    name: str                          # "pendulum_linear"
    latex: str
    params: list[ParamSpec]            # name, unit, bounds, init_guess_fn
    state_vars: list[str]
    knob_transfer: str                 # how params map to new knob settings (§9)

class FitResult(BaseModel):
    model: str
    run_id: str
    params: dict[str, float]
    param_std: dict[str, float]
    train_rmse: float
    holdout_rmse: float
    aic: float
    bic: float
    residual_whiteness_p: float        # Ljung-Box p-value

class Comparison(BaseModel):
    fits: list[FitResult]
    best: str | None
    status: Literal["decisive", "ambiguous"]
    max_separation_sigma: float        # max predicted-trajectory gap between top models / noise
    explanation_facts: dict            # numbers for the agent to cite

class ExperimentProposal(BaseModel):
    knob: str
    value: float
    expected_separation_sigma: float
    discrimination_curve: list[tuple[float, float]]   # knob value → separation
    feasibility_note: str

class Commitment(BaseModel):
    id: str
    experiment_id: str
    created_at: str                    # ISO 8601 UTC
    knob_setting: dict[str, float]
    models: list[CommittedModel]       # frozen params + prediction rule + predicted observables
    protocol: dict                     # conditioning window, metrics, success criterion
    canonical_json: str
    sha256: str

class Verdict(BaseModel):
    commitment_id: str
    run_id: str
    hash_verified: bool
    validity: ValidityReport
    per_model: list[dict]              # rmse, observable errors, sigma units
    winner: str | None
    summary_facts: dict
```

---

## 7. Pipeline modules

### 7.1 Ingest (`ingest.py`)
- Save the upload, compute SHA-256, run `ffprobe` for **per-frame presentation timestamps** (phones record variable frame rate; never assume `i / fps`).
- Normalize to H.264 mp4 at ≤720p with ffmpeg; extract keyframes (first frame, middle frame, last frame, plus 3 evenly spaced) for scene understanding.

### 7.2 Scene understanding (`scene.py`)
- Send keyframes to the vision model with a strict JSON schema (`SceneDescription`). Validate with Pydantic; retry up to 2 times with the validation error appended.
- Include a known-sizes table in the prompt (US quarter 24.26 mm, penny 19.05 mm, credit card 85.60 × 53.98 mm, A4/Letter paper) so the model picks a reference the user can actually place.
- The user can edit the scene description in the UI before tracking (human in the loop; also a demo safety valve).

### 7.3 Segmentation (`segment.py`)
- Prompt SAM 3.1 with `tracked_concepts + [reference_object.concept]`.
- Keep the highest-scoring consistent track per concept; drop frames with score < `SAM_MIN_SCORE` (default 0.5) into `valid_mask=False`.

### 7.4 Measurement and calibration (`measure.py`)
- Per frame: mask centroid (sub-pixel, from moments), area, bounding box, orientation.
- **Scale**: reference mask → equivalent diameter in pixels (or a minimum-area rectangle for cards/paper) → `meters_per_pixel`, with std across frames.
- **Pendulum**: fit a circle to the bob trajectory (algebraic Kåsa fit for initialization, then geometric least-squares refinement) → `pivot_px`, `length_px`. θ(t) = atan2(x − pₓ, y − p_y) with image y pointing down. Effective length L = length_px × meters_per_pixel.
- **Falling body**: vertical position y(t) in meters along the gravity direction (from the scene hint, or the principal axis of the trajectory).
- **Perspective**: assume motion in a plane parallel to the sensor (filming guide enforces this). P1: optional 4-point homography from ruler or paper corners clicked in the UI.
- **Smoothing**: Savitzky-Golay (window ≈ 0.1 s, polyorder 3) for display and derivatives. **Fitting uses raw measurements**; smoothing is not used for fitting. `noise_sigma` = robust std (MAD × 1.4826) of raw minus smooth.
- Outlier rejection: centroid jumps > 5 robust sigmas of frame-to-frame motion → invalid frame.

### 7.5 Validity checks (`validity.py`) — the "refuses to be fooled" feature
Produce `ValidityReport{status: valid|warning|invalid, checks: [...]}` with a clear human-readable reason per check:
- **Camera moved**: reference object centroid drift > 3% of frame width, or a sudden jump > 10 robust sigmas → invalid.
- **Scale changed**: reference apparent size changes > 5% (zoom or the object moved toward the camera) → invalid.
- **Tracking lost**: > 15% of holdout frames invalid → invalid; > 5% → warning.
- **Frame timing**: timestamp gaps > 3× the median interval → warning.
- **Pendulum geometry**: circle-fit residual too large (bob not on an arc) → warning.
- **Setup mismatch** (for reveal runs): estimated pendulum length differs > 5% from the committed experiment → invalid ("this looks like a different setup").

---

## 8. Leakage prevention (non-negotiable; must be tested)

1. `Split.train_end_t` is fixed when the run is created (default: first 60% of valid duration) and stored immutably.
2. `fit_models` receives **sliced arrays** (`t <= train_end_t`) only. The fitting module has no access to the `Run` object.
3. Holdout scoring happens in a separate function that receives frozen `FitResult.params`.
4. For commitments: the reveal run's parameters are **never refit**. Only the initial state (θ₀, θ̇₀ or y₀, v₀) is estimated from the first `conditioning_window_s` seconds; everything after that is pure prediction.
5. `evaluate_commitment` refuses to run if the stored `canonical_json` re-hashes to a different `sha256`.
6. **Tests**: (a) mutating holdout data does not change fit results; (b) a tampered commitment fails verification; (c) a commitment's predictions are byte-identical before and after the reveal run exists.

---

## 9. Model library (`models/`)

Each model implements `simulate(t, params, x0) -> y`, built on `scipy.integrate.solve_ivp` (RK45, rtol 1e-8), plus parameter bounds and initial guesses.

| Name | Equation | Fitted params | Knob transfer |
|---|---|---|---|
| `pendulum_linear` | θ'' + 2γθ' + ω₀²θ = 0 | ω₀, γ, θ₀, θ̇₀ | ω₀, γ fixed for the same pendulum; new θ₀ from conditioning |
| `pendulum_nonlinear` | θ'' + 2γθ' + ω₀² sin θ = 0 | ω₀, γ, θ₀, θ̇₀ | same |
| `pendulum_nonlinear_quaddamp` (P1) | θ'' + c·θ'\|θ'\| + ω₀² sin θ = 0 | ω₀, c, θ₀, θ̇₀ | same |
| `drag_linear` | v' = g(1 − v/v_t) | v_t, y₀, v₀ | v_t(k filters) = k · v_t(1) |
| `drag_quadratic` | v' = g(1 − (v/v_t)²) | v_t, y₀, v₀ | v_t(k) = √k · v_t(1) |
| `spring_mass` (P2) | x'' + 2γx' + ω²x = 0 | ω, γ, x₀, ẋ₀ | — |

Notes:
- Use g = 9.81 m/s². For pendulums, also report the implied L = g/ω₀² against the measured L as a sanity check the agent can cite.
- Pendulum models have **equal parameter counts**, so the comparison is fair.
- Sanity numbers: at 70° amplitude the true period is ≈ 10% longer than the linear prediction; at 8° the difference is ≈ 0.1%.

**Equation discovery (P1, `discover.py`)**: PySINDy on (θ, θ̇) with a library {θ, θ³, sin θ, θ̇, θ̇\|θ̇\|}, sparse regression (STLSQ). Show the discovered equation in the lab notebook as supporting evidence. The candidate-model comparison remains the primary mechanism.

---

## 10. Fitting, comparison, ambiguity (`fit.py`, `compare.py`)

- **Fit**: `scipy.optimize.least_squares` (trf, soft_l1 loss, bounds) on raw train data, with multi-start (5 starts). Parameter std from the Jacobian covariance scaled by residual variance; P1: residual bootstrap (100 resamples) for prediction bands.
- **Holdout**: simulate from the fitted initial state across the full timeline and score RMSE on `t > train_end_t`.
- **Metrics**: train RMSE, holdout RMSE, AIC, BIC, Ljung-Box p-value on residuals.
- **Ambiguity rule** (thresholds in config): `ambiguous` if
  `|ΔBIC| < 6` **and** `holdout_rmse_ratio < 1.2` **and** `max_separation_sigma < 3`,
  where `max_separation_sigma = max_t |ŷ_A(t) − ŷ_B(t)| / noise_sigma` over the holdout window. Otherwise `decisive`, with a named winner.

---

## 11. Experiment design (`design.py`)

- Knobs per system: pendulum `release_angle_deg ∈ [5, 80]`; drag `n_filters ∈ {1..6}`; (P2) pendulum length.
- For each candidate knob value: simulate each top model under its knob-transfer rule for a fixed horizon (pendulum: 6 periods; drag: full fall time from 2 m), compute
  `separation_sigma = sqrt(mean_t((ŷ_A − ŷ_B)²)) / noise_sigma`,
  and `max_separation_sigma`.
- Feasibility penalty: discourage extremes (angle > 75° risks a slack string; > 5 filters gets unstable).
- Return the best knob value plus the full discrimination curve, plotted in the UI ("models separate by 0.2σ at 8°, 18σ at 70°").
- The agent explains the recommendation in plain language and gives filming instructions.

---

## 12. Commit and reveal (`commit.py`)

**Commit**
- Content: experiment id, knob setting, for each model {name, frozen params with std, knob-transfer rule, predicted observables (pendulum: period at the target amplitude; drag: terminal velocity)}, protocol {conditioning window, metric, success rule: "winner = lowest RMSE after the conditioning window, provided the run is valid"}, `created_at`.
- Canonical JSON: sorted keys, floats rounded to 9 significant digits, no whitespace → SHA-256. Display the first 12 hex characters large and the full hash on hover. Store it immutably.

**Reveal**
1. Verify the hash.
2. Run ingest → scene (reuse the experiment's scene, prompts, and calibration method) → tracking → measurement → validity.
3. If invalid, return a Verdict with `winner=None` and the reasons. Do not score.
4. Estimate the initial state from the conditioning window only; simulate each committed model with frozen params; compute RMSE, observable errors (measured period vs predicted; measured v_t vs predicted), and errors in σ units.
5. Winner + Verdict.

---

## 13. The agent (`agent/`)

**Loop**: the Muse Wonder Responses API with function tools, run in our own loop (max 25 tool calls per task). Every step (thought summary, tool call, args, result digest) streams to the frontend over SSE as the **lab notebook**, so judges can see the agent work.

**System prompt principles** (`agent/system_prompt.md`):
- You are an experimental scientist. Your job is to find which model explains this system and to say honestly when the evidence is insufficient.
- Never state a number that did not come from a tool result. Cite tool outputs.
- Never claim a model is correct because it fits the training data; only holdout and committed-prediction results count as evidence.
- If validity is `invalid`, stop, explain why, and tell the user how to re-film.
- When results are ambiguous, call `design_experiment`, then `commit_predictions` **before** asking the user to film.
- Keep user-facing explanations short, concrete, and in plain language.

**Tools** (JSON-schema function definitions; all deterministic Python):

| Tool | Input | Output |
|---|---|---|
| `describe_scene` | run_id | SceneDescription |
| `track_objects` | run_id, concepts[] | track summary (coverage, mean score) |
| `measure` | run_id | series summary, calibration, noise_sigma |
| `check_validity` | run_id | ValidityReport |
| `list_models` | system_type | ModelSpec[] |
| `fit_models` | run_id, model_names[] | FitResult[] (train-only by construction) |
| `compare_models` | run_id | Comparison |
| `discover_equation` (P1) | run_id | SINDy equation + score |
| `design_experiment` | experiment_id, model_names[], knob | ExperimentProposal |
| `commit_predictions` | experiment_id, model_names[], knob_setting | Commitment (id, sha256) |
| `evaluate_commitment` | commitment_id, run_id | Verdict |
| `render_ghosts` | run_id, model_names[] or commitment_id | overlay video path |

**Agent tasks** exposed in the UI: "Analyze this run", "What should I film next?", "Score my new run against the commitment".

---

## 14. Rendering (`render.py`)

- OpenCV reads the original frames; for each model, compute the predicted pixel position over time:
  - Pendulum: (pₓ + L_px·sin θ̂, p_y + L_px·cos θ̂); draw the rod line + filled circle (bob radius from the mask) at 35–45% alpha, with a fading trail of the last 0.3 s.
  - Drag: horizontal position from the tracked mean x, vertical from ŷ(t) converted via calibration.
- Distinct color per model with a label; legend box with live running RMSE per model.
- Vertical marker and on-screen text at the train/holdout boundary ("PREDICTION STARTS HERE"); for reveals, mark the end of the conditioning window and show "COMMITTED <hash12>".
- Encode with ffmpeg `libx264 -preset veryfast -crf 20 -pix_fmt yuv420p`.
- Also export a side-by-side plot video (P2) for social posts.

---

## 15. Build plan and priorities

Today is Sunday Nov 1. Deadline Wed Nov 4, 9 PM ET.

**P0: must ship (the demo depends on it)**
1. Repo scaffold, `.env.example`, config, caching layer, `DEMO_MODE`.
2. Ingest with real frame timestamps.
3. **Synthetic video generator** (`scripts/make_synthetic.py`): renders a pendulum or falling disk with known parameters, background, a reference circle, and pixel noise. It is used for tests and lets the numerics be built before SAM is wired up.
4. SAM 3.1 client (Meta API) + normalization. Fallback: OpenCV color/blob tracker behind the same interface (synthetic videos only) so development is never blocked.
5. Measurement: centroid, reference calibration, pendulum circle fit, θ(t), drag y(t), noise estimate.
6. Models: `pendulum_linear`, `pendulum_nonlinear`, `drag_linear`, `drag_quadratic`; fit; holdout; compare; ambiguity rule.
7. Experiment design + discrimination curve.
8. Commit/reveal with hashing and verification.
9. Validity checks (camera moved, scale changed, tracking lost).
10. Ghost overlay renderer.
11. Agent loop with the tools above + SSE lab notebook.
12. Frontend: experiment list (precomputed demos), run view (overlay video, θ(t) or y(t) plot with model curves, comparison table, lab notebook), design view (discrimination curve), commit card, reveal/verdict card.
13. Unit tests: models recover known params from synthetic data; circle fit; calibration; hash determinism; leakage tests (§8).

**P1: strongly wanted**
- Scene understanding via Muse Spark Open on Baseten (open-weights bonus) with the Meta API as fallback.
- Residual-bootstrap prediction bands drawn as a ghost "halo".
- PySINDy equation discovery in the notebook.
- Homography calibration from clicked points.
- Evaluation harness (§17) with a results table for the README.
- Deployed URL.

**P2: only if ahead**
- Muse Voice Transcribe narration → interventions on the timeline.
- Quadratic-damping pendulum, spring–mass.
- Social-ready vertical export.

**Suggested schedule**
- **Sun (today)**: P0 items 1–6 working on synthetic videos; film all real demo videos tonight (§18).
- **Mon**: SAM on real videos, validity, design, commit/reveal, renderer, agent loop.
- **Tue**: frontend, end-to-end on real videos, caching for demo mode, deploy, eval harness, README.
- **Wed (by 6 PM)**: record the demo video, X post, submit. Leave 3 hours of buffer before the 9 PM deadline.

**Cut order if behind**: P2 → P1 → Scene C → Scene B. Scene A end-to-end is the minimum viable submission.

---

## 16. Demo video script (3:00)

| Time | Content |
|---|---|
| 0:00–0:15 | Hook: the 70° ghost split shot. "One of these ghosts is wrong. Ghostline knew which experiment would prove it." |
| 0:15–0:35 | Problem: measuring and modeling physical behavior from video is manual, slow, and tools rarely tell you when your evidence is insufficient. |
| 0:35–1:15 | Scene A part 1: upload 8° → tracking masks → θ(t) → two fits → both ghosts overlap → "Inconclusive" → discrimination curve → recommends 70° → commit hash. |
| 1:15–1:50 | Scene A part 2: upload 70° → hash verified → ghosts split → verdict card. |
| 1:50–2:15 | Scene B (fast): filters, 1 → 4, quadratic wins. |
| 2:15–2:30 | Scene C: bumped camera → INVALID, refuses to score. |
| 2:30–2:50 | Under the hood: architecture diagram; Muse Wonder agent + SAM 3.1 + Muse Spark Open; numerics in code; leakage guards; eval results. |
| 2:50–3:00 | Impact: any camera becomes an instrument that reasons like an experimentalist. Repo link. |

---

## 17. Evaluation harness (`scripts/eval.py`)

- **Synthetic**: ~50 generated videos with known model, parameters, and noise levels → report parameter recovery error, model-selection accuracy, and ambiguity-detection rate (does it say "ambiguous" at small amplitudes?).
- **Real**: every real clip filmed → holdout RMSE per model, selection results, and validity-check results (including deliberately bumped clips).
- Output `eval/results.md` and `eval/results.json`; the README embeds the table. These are the numbers for the resume line, so they must be real.

---

## 18. Filming guide (put in the README and use for demo footage)

- Phone on a tripod or taped to a stable surface, camera **perpendicular** to the plane of motion, 1–2 m away. 60 fps if available; lock exposure and focus.
- Plain, contrasting background; good, even light; bright-colored bob.
- Reference object (a quarter or credit card) taped flat **in the plane of motion**, visible in every frame.
- Pendulum: string 0.5–1.0 m, heavy compact bob. Record 8° and 70° releases (2–3 takes each), at least 8 swings per take.
- Coffee filters: drop from ~2 m in front of a wall; record 1 filter and 4 stacked (3 takes each).
- Bumped-camera clip: nudge the tripod halfway through one take.
- Keep spares of everything; cache the best takes for demo mode.

---

## 19. Repository layout

```
ghostline/
  README.md                  # pitch, demo GIF, architecture, eval table, built-during-hackathon, filming guide
  SPEC.md                    # this file
  .env.example               # MODEL_API_KEY, BASETEN_API_KEY, AGENT_MODEL, VISION_MODEL, SAM_MODEL, SCENE_BACKEND, DEMO_MODE
  backend/
    pyproject.toml
    ghostline/
      api.py                 # FastAPI routes
      config.py
      schemas.py
      storage.py
      ingest.py  scene.py  segment.py  measure.py  validity.py
      fit.py  compare.py  design.py  commit.py  discover.py  render.py
      models/  base.py  pendulum.py  drag.py
      clients/ meta.py  baseten.py  sam_meta.py  sam_local.py  cache.py  fallback_tracker.py
      agent/   loop.py  tools.py  system_prompt.md
    tests/
  frontend/
    src/  pages/  components/ (GhostPlayer, SeriesPlot, ComparisonTable, LabNotebook, CommitCard, VerdictCard, DiscriminationCurve)
  scripts/  make_synthetic.py  eval.py  precompute_demo.py
  data/     (gitignored except demo fixtures)
  eval/
```

**API routes**
- `POST /experiments` · `GET /experiments` · `GET /experiments/{id}`
- `POST /experiments/{id}/runs` (multipart video + knobs) · `GET /runs/{id}`
- `POST /runs/{id}/analyze` → SSE stream of agent steps
- `POST /experiments/{id}/design` · `POST /experiments/{id}/commit` · `GET /commits/{id}`
- `POST /commits/{id}/reveal` (multipart video) → SSE → Verdict
- `GET /runs/{id}/overlay.mp4`

---

## 20. README requirements

- Hero GIF of the ghost split, then a one-line pitch.
- How it works (diagram) and **which Meta models do what**.
- **Built during the hackathon**: an explicit list of every component, all created Oct 27–Nov 4.
- Eval results table (§17).
- Limitations, stated honestly: planar-motion assumption; a video cannot uniquely determine physical law; results are limited to the candidate model set; tracking depends on lighting and contrast.
- Filming guide and setup instructions.
- "Built with Muse Spark Open" if that backend is used.

---

## 21. Definition of done

- [ ] Scene A runs end-to-end from cached real videos in `DEMO_MODE` with no network.
- [ ] The commit hash is shown before the reveal and verified after it.
- [ ] A bumped-camera run is rejected with a clear reason.
- [ ] All leakage tests and model-recovery tests pass.
- [ ] The public repo has a README with the eval table and the built-during-hackathon section.
- [ ] A deployed URL works with precomputed demos.
- [ ] 3-minute video recorded; X post published and linked; submission on the CV platform before **Wed Nov 4, 9 PM ET**.
