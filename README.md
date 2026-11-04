# 👻 Ghostline

**Film a physical system. Get candidate equations. Watch their predictions appear as ghosts over reality. When the evidence can't decide, Ghostline designs the experiment that will.**

*An agent that runs the scientific method on its own; the human only has to be the hands.*

Built for the **Meta Global AI Developer Hackathon** (Cerebral Valley), track **Push the Frontier**.

**Live demo:** https://ghostline-nfsm.onrender.com (precomputed scenes, no upload needed; the free instance may take ~1 min to wake up)

<!-- HERO: ghost-split GIF -->

---

## What it does

1. **Observe.** Upload a phone video of a pendulum or a falling coffee filter. Ghostline identifies the moving object and a reference object of known size, tracks both, and converts pixels into a calibrated time series (θ(t) or y(t)) using real per-frame timestamps.
2. **Test honestly.** Competing models (linear vs. nonlinear pendulum; linear vs. quadratic drag) are fitted on the first 60% of the clip **only**. Each must predict the remaining 40%, and its prediction is drawn as a translucent *ghost* over the video.
3. **Admit uncertainty.** If the models differ by less than the measurement noise, Ghostline says **"Inconclusive"** rather than picking one.
4. **Design the next experiment.** It searches over the knob you control (release angle, number of stacked filters) for the setting where the models disagree most *under the actual reveal protocol*.
5. **Commit.** Every model's predictions for that experiment are frozen into canonical JSON and **SHA-256 hashed** before you film.
6. **Reveal.** Film the new experiment. Ghostline verifies the hash, reads only the starting state from the first 0.5 s, and lets the frozen predictions run. One ghost follows reality; the other drifts away.
7. **Refuse to be fooled.** If the camera was bumped, zoomed, or tracking was lost, the run is marked **INVALID** and the commitment is not scored.

## Demo scenes (precomputed, viewable offline)

| Scene | First run | Ghostline recommends | Result |
|---|---|---|---|
| A: pendulum | 8° release → *inconclusive* (<1σ apart) | 70° release | nonlinear wins; linear period is ~6% short |
| B: coffee filters | 1 filter → *inconclusive* | stack 4 filters (linear predicts vₜ×4, quadratic ×2) | quadratic drag wins |
| C: bumped camera | 70° reveal with a mid-clip bump | n/a | **INVALID**: reference jumped 32 px; not scored |
| D: **real phone footage**, a computer mouse on its cable | 3.9° swing → *inconclusive* (<1σ apart) | 20° (the most this rig allows) | Quadratic damping is least wrong, but **none of the candidates fits**: the best misses the period by 2% (328σ). Ghostline says so and names the cause: the mouse's outline changes size by 62% as it twists |

> Scenes A–C use rendered videos with known physics (`scripts/make_synthetic.py`), so the right answer is known. Scene D is real iPhone footage, processed end to end by Muse Spark, SAM 3.1 and Muse Wonder. A mouse hanging on its cable is not a point mass on a string: it is a rigid body that twists and pushes air. Ghostline's candidate models leave a 2% period error, and instead of declaring a winner it reports that no candidate describes the system.

> **Why no candidate fits (diagnosed):** frame timing is clean (regular 33.3 ms frames), so the error is physical. (1) The mouse twists on its cable and tilts toward the camera: SAM's box around it changes width from 152 to 335 px within a swing, so the box centre is not a fixed point on the body. The fit residuals oscillate at twice the swing frequency, which is the signature of this. (2) Even fitted to its own footage, the best point pendulum leaves 46σ residuals on the 20° take (9σ on the 4° take). (3) At equal amplitude the 20° take's period is 1–1.5% longer than the 4° take's, so the effective cable length shifted between takes. Ghostline now checks the tracked object's outline (`object_shape` in `validity.py`) and puts the likely cause into the verdict. The ID card used as the size reference is blurred for privacy, and only its ends are kept sharp.

## How it works

```
Upload ─► ingest (ffprobe PTS, H.264 ≤720p, keyframes)
       ─► scene understanding (Muse Spark / Muse Spark Open)  → what to track, reference size
       ─► segmentation + tracking (SAM 3.1)                   → masks per frame
       ─► measure: centroids, scale, pivot circle-fit, θ(t)/y(t), noise σ
       ─► validity checks (camera drift/jump, scale, tracking loss, timing, setup match)
       ─► Agent (Muse Wonder, tool calling) drives deterministic tools:
            fit · compare · discover · design · commit · evaluate · render
       ─► ghost overlay video + plots + lab notebook (SSE)
```

**Numbers come from code, never from the LLM.** The agent reasons, chooses tools and explains. All measurement, fitting, simulation and scoring is deterministic Python (numpy/scipy/pysindy/OpenCV), and the agent is instructed to cite only tool outputs.

### Which Meta models do what

| Role | Model | Where |
|---|---|---|
| Agent / reasoning (tool loop, lab notebook) | **Muse Wonder** (`muse-wonder-1.0`) via Meta Model API Responses | `backend/ghostline/agent/loop.py` |
| Scene understanding from keyframes | **Muse Spark** (`muse-spark-1.3`) via Meta Model API, or **Muse Spark Open** via Baseten | `backend/ghostline/scene.py` |
| Object segmentation + video tracking | **SAM 3.1** (`sam-3.1`) via Meta Model API (or open weights `facebook/sam3.1` on Modal) | `backend/ghostline/clients/sam_meta.py`, `sam_local.py` |

How the Meta models are used, as tested against the live API on Nov 4:

- **Muse Spark** looks at 6 keyframes plus the user's note ("the green disk is a US quarter") and returns a validated `SceneDescription`: which system it is, short segmentation prompts, and the reference size. It identifies the system correctly even with no hint. It cannot know the real size of a reference object from pixels alone, so the size comes from the user's note or the scene-edit form.
- **SAM 3.1** gets the whole uploaded clip (`input_video`) and returns per-frame boxes for every frame. It matches short concrete phrases ("orange ball") much better than descriptive ones ("orange pendulum bob"), so Ghostline probes a few short variants on one frame and then tracks with the first one that matches.
- **Muse Wonder** runs the tool loop: it decides when the evidence is inconclusive, designs the next experiment, commits before filming, and writes the lab notebook. It reports only numbers that came from tools.

Every external call is cached to disk (`cache/{sha256(model+request+file hashes)}.json`). With `DEMO_MODE=1` the app never touches the network. Without API keys, deterministic fallbacks run behind the same interfaces: a template scene, an OpenCV colour tracker, and a scripted agent policy. They are labelled as such in the UI.

### Leakage prevention (enforced in code, tested)

- The train/holdout split is fixed when a run is first measured and is immutable (frozen Pydantic model).
- `fit_models` receives **sliced training arrays only**. It never sees the run object.
- Holdout scoring is a separate function that receives frozen parameters.
- Reveal runs are **never refit** (`fit_run` raises). Only the initial state is estimated from the 0.5 s conditioning window.
- Commitments are write-once files. `evaluate_commitment` re-hashes the canonical JSON and also checks that the structured copy matches it; any tampering means the run is refused.
- Tests: `backend/tests/test_commit_leakage.py` covers mutating holdout data (fit unchanged), a tampered commitment (refused), and commitment bytes being identical before and after the reveal.

### Experiment design

For each candidate knob value, for each ordered pair (A = truth, B = candidate), Ghostline simulates what the camera would see if A were true. It conditions B on the first 0.5 s exactly as a reveal would, then measures B's drift in units of the measured noise σ. Among feasible settings that separate the models by ≥10σ, it chooses the one with the largest *physical* effect, which is the most robust to unmodelled systematics. That gives ~70° for the pendulum (the period is ~10% longer than the linear prediction) and 4 filters for drag.

## Evaluation

Rendered clips with known physics are run through the full pipeline (`scripts/eval.py`). Two tracking backends were used:

| Check | SAM 3.1 via Meta API (15 clips) | Colour tracker (50 clips) |
|---|---|---|
| Says *inconclusive* for small-angle pendulums (5–10°) | 3/3 | 10/10 |
| Picks nonlinear at 45–75° (nonlinear truth) | 3/3 | 10/10 |
| Picks linear when linear is the truth (control) | 1/1 | 5/5 |
| Says *inconclusive* for a single coffee filter | (none sampled) | 4/5 |
| Picks the true drag law for 3–4 stacked filters | 3/3 | 5/5 |
| Rejects bumped / zoomed recordings as INVALID | 5/5 | 15/15 |
| False INVALID on clean recordings (lower is better) | 0/10 | 0/35 |

| Parameter recovery, true model (median abs error / p90) | SAM 3.1 | Colour tracker |
|---|---|---|
| pendulum ω₀ | 0.05% / 0.07% | 0.05% / 0.08% |
| pendulum γ | 2.8% / 5.0% | 3.3% / 6.9% |
| drag terminal velocity | 2.7% / 3.2% | 2.4% / 3.2% |

Pipeline errors: 0 in both runs. The SAM run is smaller because each clip takes about 80 s of API time. Full results: `eval/results_sam.md`, `eval/results.md`.

These tables use synthetic clips, where the truth is known. Real footage is Scene D above.

## Run it

Requirements: Python 3.11, [uv](https://docs.astral.sh/uv/), ffmpeg, Node 20+.

```bash
cp .env.example .env              # add MODEL_API_KEY etc. (optional: fallbacks work without keys)
cd backend && uv sync && uv run pytest          # 22 tests

# demo data (synthetic, or real clips from data/raw/real/)
uv run python ../scripts/make_synthetic.py --all
uv run python ../scripts/precompute_demo.py

# serve: API + built frontend on :8000
(cd ../frontend && npm install && npm run build)
DATA_DIR=../data/demo DEMO_MODE=1 uv run uvicorn ghostline.api:app --port 8000
# or dev mode: `npm run dev` in frontend/ (proxies /api to :8000)

uv run python ../scripts/eval.py --n 50   # writes eval/results.md
```

Switch to the real Meta models in `.env`: `LLM_BACKEND=meta`, `SCENE_BACKEND=meta` (or `baseten`), `SEGMENT_BACKEND=meta`.

**Deploy:** `Dockerfile` builds one service (frontend + API + precomputed demos, `DEMO_MODE=1`). `render.yaml` deploys it on Render.

## Filming guide

- Phone on a tripod or taped to a stable surface, **perpendicular** to the plane of motion, 1–2 m away. Use 60 fps if available, and lock exposure and focus.
- Plain, contrasting background; even light. Use a **brightly coloured (orange) bob / filter**.
- Reference object taped flat **in the plane of motion**, visible in every frame. A quarter works; a credit card is more accurate. For the offline colour tracker, use a **green** sticker or card of known size.
- Pendulum: string 0.5–1.0 m, heavy compact bob. Record 8° and 70° releases (2–3 takes each), at least 8 swings per take.
- Coffee filters: drop from ~2 m in front of a wall. Record 1 filter and 4 stacked (3 takes each).
- Bumped-camera clip: nudge the tripod halfway through one take.
- Name the clips `pendulum_8.mp4`, `pendulum_70.mp4`, `pendulum_70_bumped.mp4`, `filters_1.mp4`, `filters_4.mp4` and put them in `data/raw/real/`.

## Limitations (stated honestly)

- Assumes planar motion parallel to the sensor (4-point homography calibration is available via `POST /api/runs/{id}/homography`).
- A video cannot uniquely determine a physical law. Results are limited to the candidate model set.
- Short pendulum arcs make the string length from the circle fit imprecise. Ghostline then uses g/ω₀² for setup matching and says so.
- Tracking depends on lighting and contrast. The colour-tracker fallback needs coloured objects.
- SAM 3.1's mask payload uses a compressed codec that is not publicly documented, so Ghostline measures from SAM's per-frame boxes (centre and size). That works well for the compact objects used here (bob, reference marker). Decoding the masks through Meta's `meta-sam-parser` would give sub-pixel centroids.
- Real footage (Scene D) is one rig filmed in one session. The reference card sat slightly behind the swing plane, so lengths measured from its scale are approximate. Angles and periods do not depend on scale.
- The candidate set is point-mass pendulums. Rigid, twisting bobs (like a mouse) fall outside it, and Ghostline reports that rather than forcing a winner.

## Built during the hackathon

Everything in this repository was created during the event (Oct 27 – Nov 4, 2026):

- `backend/ghostline/`: ingest, scene understanding, segmentation clients (SAM 3.1 API, local, colour fallback), measurement and calibration, validity checks, model library (pendulum linear/nonlinear/quadratic-damping, linear/quadratic drag), fitting with bootstrap bands, comparison and ambiguity rule, experiment design, commit/reveal with canonical hashing, PySINDy-based equation discovery, ghost renderer, agent loop and tools, FastAPI + SSE API
- `backend/tests/`: leakage, commitment integrity, model recovery, validity, and video end-to-end tests
- `frontend/`: React + TypeScript + Tailwind + Plotly UI (ghost player, series plot, comparison table, lab notebook, discrimination curve, commit card, verdict card)
- `scripts/`: synthetic video generator, demo precompute, evaluation harness
- `Dockerfile`, `render.yaml`
