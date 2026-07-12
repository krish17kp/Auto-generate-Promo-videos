# ROADMAP — Auto-Generate Promo Videos

> Companion docs: [MILESTONES.md](MILESTONES.md) (acceptance criteria per milestone) · [PRD.md](PRD.md) (F-numbers) · [ARCHITECTURE.md](ARCHITECTURE.md) (D-numbers)
>
> Phases map 1:1 to scope tiers: Phase 1 = **MVP**, Phase 2 = **Capstone Demo**, Phase 3 = **Advanced AI**, Phase 4 = **Production-Future**. Phase 0 is repair work with no product scope. Everything below is **(planned)** — nothing in this roadmap has been started unless its milestone in MILESTONES.md is checked.

## Phase 0 — Repair (make the repo honest and runnable)

The app currently 500s on every request and the pipeline script cannot run (current gaps, detailed in [ARCHITECTURE.md §1](ARCHITECTURE.md)). Nothing else matters until this is done. All items are small, mechanical, and deliberately scoped so a smaller model (Sonnet/Haiku) can execute them from this list:

1. **Fix URLConf:** `promoapp/urls.py` must route to the views that exist — `''→views.home`, `generate/→views.generate_video`; delete the dead `generate_promo` route (or stub the view). Site must load.
2. **Refactor `promo4.1.py` → `promoapp/pipeline/`** per the layout in [VIDEO_PIPELINE.md §1](VIDEO_PIPELINE.md): valid module names, add missing `typing`/`argparse` imports, de-indent `main`/`parse_args` out of the class, fix `args.no-effects` → `args.no_effects` (dest fix), port `moviepy.editor` imports to MoviePy 2.x API.
3. **Env-based config (D-5):** `SECRET_KEY`, `DEBUG`, `ALLOWED_HOSTS` from env via the already-installed `python-dotenv`; add `.env.example`; remove `db.sqlite3` from git and add to `.gitignore`; drop the duplicate unconditional media serving in `promo_project/urls.py`.
4. **Reconcile Python version:** bump `runtime.txt` to a 3.11+ version compatible with the pinned TF 2.20 / numpy 2.2 stack; pin `pandas` (used by the eval script, currently only transitive).
5. **Smoke checks:** `python manage.py check` passes; `python -m promoapp.pipeline.cli --help` runs; a first Django test asserting the two routes return 200.

## Phase 1 — MVP: real end-to-end promo (F-1…F-5)

- DB models `VideoUpload`, `PromoJob`, `Scene`, `PromoOutput` + migrations ([DATABASE_SCHEMA.md](DATABASE_SCHEMA.md)).
- Upload validation (extension/size/ffprobe) in the view; job created and run by the thread-based runner (`jobs.py`, decision D-1).
- Pipeline MVP path: ingest → scene detection → audio-energy + frame-difference scoring → top-3 chronological selection → MoviePy 2.x render, single 16:9 output ≤30 s.
- Result page plays the real output; polling endpoint for status; failures land as `failed` + message.
- Tests: pipeline unit tests on a bundled ≤30 s fixture video; view tests for upload validation and job lifecycle.

**Exit condition: an uploaded video returns a genuinely edited promo. The mock copy in `views.generate_video` is deleted.**

## Phase 2 — Capstone Demo: the show-off version (F-6…F-12)

Ordered so evaluation gates the polish:

1. **Hybrid scoring:** CLIP zero-shot visual signal, optical-flow motion, quality gates; fusion per [VIDEO_PIPELINE.md §5](VIDEO_PIPELINE.md); `SegmentScore` persistence.
2. **Eval integration:** score dumps into the existing `promo4.1.evaluate.py` methodology; ablation report (hybrid vs audio-only vs CLIP-only vs motion-only) with bootstrap CIs → `EVAL_REPORT.md`. **Gate:** hybrid must beat single-signal baselines on precision@5 before proceeding — if it doesn't, tune weights/prompts here, not later.
3. **Narrative assembly:** hook → build → climax → outro with diversity constraint ([VIDEO_PIPELINE.md §6](VIDEO_PIPELINE.md)).
4. **Aspect presets:** 9:16 and 1:1 (center-crop first, content-aware crop as polish); target-duration presets.
5. **Demo UX:** progress page with live stage/progress, explainability panel (per-signal scores per chosen segment), title/CTA overlay, upload-page preset controls.
6. **Portfolio packaging:** README results section with real eval numbers, sample input/output pair, screenshots.

**Exit condition: a 2-minute live demo — upload, watch staged progress, play a trailer-arc promo in 9:16, open the explainability panel, show the eval report.**

## Phase 3 — Advanced AI (F-13…F-16)

Each item ships only if it lifts the eval report (rule from [BRD.md §5](BRD.md)):

- faster-whisper transcription; transcript hook-phrase signal joins the fusion.
- Burned-in captions per aspect preset.
- Optional LLM hook-line picker (degradable to keyword heuristic; only metered external call in the system).
- Learned scoring head fine-tuned on `labels.csv` — explicitly gated on beating the zero-shot hybrid on a held-out split; if `labels.csv` is too small, this item is dropped and the doc says so.

## Phase 4 — Production-Future

Shift from single-threaded to a distributed, scalable job queue. Options (pick one per deployment model):

### Option A: Database-backed job queue (simplest, no external services)
- Store jobs in `PromoJob(status=queued)` in Postgres.
- Separate background worker process (e.g., `python manage.py run_worker`) polls the queue, picks up queued jobs, updates status as it processes.
- Frontend polls same API endpoint for status (zero change to UI).
- Scales by running multiple worker processes on separate dynos.
- **Pros:** zero external dependencies beyond Postgres (already in use), job durability (survives restarts), simple to debug (audit trail in DB).
- **Cons:** polling latency (~1–5 s), worker heartbeat required to detect crashes.
- **Cost:** Railway free-tier compatible.

### Option B: Celery + Redis (if high concurrency demands it later)
- Job queue via Redis message broker; Celery workers consume tasks.
- Scales to many concurrent jobs; lower polling latency.
- **Pros:** production-grade, handles thousands of jobs/minute.
- **Cons:** Redis cost ($5–15/month), more infrastructure to operate.
- **When to migrate:** when Option A's polling becomes a bottleneck (>100 concurrent jobs).

### Option C: Serverless (AWS Lambda, Google Cloud Functions)
- Offload pipeline to cloud functions, pay per invocation.
- Auto-scales, zero operational overhead.
- **When to use:** if traffic is bursty and you want zero idle costs.

**For now:** thread-based runner (Capstone Demo) is production-ready for single-server deployments on Railway free tier. Migrate to Option A (DB queue) when you need multi-worker durability, then Option B (Celery) if that's insufficient.

Additional items (all tiers):
- Postgres (already gated on `DATABASE_URL`); S3-compatible object storage for media (already gated on `AWS_STORAGE_BUCKET_NAME`).
- Auth + per-user job history; retention sweeps as scheduled jobs.
- Dockerfile + GitHub Actions CI (lint, tests, eval smoke run on the fixture video).
- Rate limiting, upload scanning, cost dashboards ([BRD.md §6](BRD.md)).

## Explicitly not on the roadmap

Timeline-editing UI, multi-video inputs, music generation, live processing, mobile apps ([PRD.md §7](PRD.md)).
