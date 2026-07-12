# ARCHITECTURE — Auto-Generate Promo Videos

> Companion docs: [PRD.md](PRD.md) · [VIDEO_PIPELINE.md](VIDEO_PIPELINE.md) · [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) · [ROADMAP.md](ROADMAP.md)
>
> Scope tiers: **MVP → Capstone Demo → Advanced AI → Production-Future**. Unbuilt = **(planned)**; defects = **(current gap)**.

## 1. Current state (as of this document)

```
Browser ──POST /generate/──► Django (promoapp.views.generate_video)
                                 │
                                 ├── save upload → media/input/<name>
                                 ├── **byte-copy** input → media/output/promo_<name>   (mock — current gap)
                                 └── render result.html with video URL
```

- Framework: Django 5.2, single app `promoapp`, SQLite, WhiteNoise static serving, gunicorn Procfile targeting Render/Railway.
- **The ML pipeline is not connected.** It exists as `promoapp/promo4.1.py`, a standalone script that is currently non-importable and non-runnable (current gaps): dotted filename, missing `List`/`argparse` imports, MoviePy 1.x `moviepy.editor` import against pinned MoviePy 2.2, `main`/`parse_args` indented inside the class, and an `args.no-effects` expression bug.
- `promoapp/urls.py` routes to `views.index` and `views.generate_promo`, which do not exist in `views.py` (only `home`, `generate_video`) — URLConf raises on load, so **every request 500s** (current gap).
- No DB models, no migrations, no tests, no CI, no Docker. `SECRET_KEY` hardcoded and `DEBUG=True` in `settings.py` (current gaps). `db.sqlite3` is committed.
- What *is* solid: the dependency stack (PySceneDetect, OpenCV, librosa, MoviePy 2.x, TF/Keras, scikit-learn all pinned), the upload/result templates, and a real offline eval harness (`promo4.1.evaluate.py` + `labels.csv`).

## 2. Target architecture (planned)

```
┌──────────────────────────── Django web tier ────────────────────────────┐
│  Upload view ── validates file ──► creates VideoUpload + PromoJob (DB)  │
│  Progress view ◄── polls job status/stage                               │
│  Result view ◄── PromoOutput rows (per aspect preset)                   │
└───────────────┬──────────────────────────────────────────────────────---┘
                │ enqueue
┌───────────────▼──────────── Job layer ──────────────────────────────────┐
│  MVP: in-process worker thread, status persisted to PromoJob            │
│  Production-Future: Celery workers + Redis broker                       │
└───────────────┬──────────────────────────────────────────────────────---┘
                │ run(job)
┌───────────────▼──────── Pipeline package: promoapp/pipeline/ ───────────┐
│  ingest → scenes → features (visual/motion/audio/quality/transcript)    │
│        → scoring (fusion) → narrative (segment selection) → render      │
│  (full spec: VIDEO_PIPELINE.md)                                         │
└───────────────┬──────────────────────────────────────────────────────---┘
                │ writes
        media/ (MVP)  →  object storage (Production-Future)
                │
   eval harness (promo4.1.evaluate.py methodology) reads saved scores
```

### Components and responsibilities

| Component | Responsibility | Tier | Status |
|---|---|---|---|
| `promoapp/views.py` | Upload validation, job creation, polling endpoint, result | MVP | Exists as mock (current gap) |
| `promoapp/models.py` | `VideoUpload`, `PromoJob`, `Scene`, `SegmentScore`, `PromoOutput` — see [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) | MVP | Empty (planned) |
| `promoapp/pipeline/` | Refactor of `promo4.1.py` into importable modules: `ingest.py`, `scenes.py`, `features/`, `scoring.py`, `narrative.py`, `render.py` | MVP→Advanced AI | (planned) |
| `promoapp/jobs.py` | Thread-based job runner; single interface later swapped for Celery task | MVP | (planned) |
| Eval harness | precision@K / AUPRC / bootstrap CI vs `labels.csv` | Capstone Demo | **Exists** (`promo4.1.evaluate.py`) — to be repointed at pipeline score dumps |
| Templates | `index.html` (upload), progress page (planned), `result.html` | MVP→Capstone Demo | Shells exist |

## 3. Key decisions

### D-1: Sync vs async processing, per tier
- **Phase 0/1 (MVP):** a Python `threading.Thread` started by the view, writing progress to `PromoJob`. Rationale: zero new infrastructure, survives on free-tier hosts, and the DB-status + polling contract is identical to the Celery version — the swap later is confined to `jobs.py`. Known ceiling: jobs die on dyno restart; acceptable for a demo, recorded on the job as `failed`.
- **Phase 4 (Production-Future):** Celery + Redis, separate worker dynos, task retries. Not before — it would be dead weight for a single-user demo.

### D-2: Pipeline as a plain Python package, not a service
The pipeline is CPU-bound library code; keeping it in-process (`promoapp/pipeline/`) avoids serialization, deployment, and versioning overhead of a microservice. It stays framework-free (no Django imports inside `pipeline/`) so the same package powers the web app, a CLI entry point, and the eval harness.

### D-3: Replace the untrained CNN head with hybrid zero-shot scoring
The current `PromoCNN` (EfficientNetB0 + random-initialized Dense→sigmoid head, never trained) outputs noise (current gap). Target: CLIP zero-shot semantic scoring + motion + audio + quality fusion — real signal with zero training data. EfficientNet/CLIP embeddings are reused for scene-diversity, not importance. Full rationale in [VIDEO_PIPELINE.md §4](VIDEO_PIPELINE.md). A *learned* scorer returns only in Advanced AI (F-16), gated on beating the hybrid in eval.

### D-4: Models lazy-loaded, one resident at a time
CLIP (~600 MB) and faster-whisper must fit free-tier memory alongside Django. Each pipeline stage loads its model, runs, and releases before the next stage. Cold-start (first model download) is documented, not hidden.

### D-5: Configuration via environment
`python-dotenv` is already a dependency but unused (current gap). Phase 0 moves `SECRET_KEY`, `DEBUG`, `ALLOWED_HOSTS`, and pipeline knobs (sample fps, max upload size, target-duration presets) to env vars with safe defaults.

## 4. Data flow (Capstone Demo target)

1. Upload → `VideoUpload` row + file at `media/input/` → `PromoJob(status=queued)`.
2. Job runner picks up job → per stage, updates `PromoJob.stage` and `progress`.
3. Pipeline persists `Scene` rows and per-signal `SegmentScore` rows (these power the explainability panel and the eval harness — same data, two consumers).
4. Render writes one file per requested aspect preset → `PromoOutput` rows.
5. Result page reads `PromoOutput` + `SegmentScore` for playback + explanation.

## 5. Deployment topology

- **Now:** single Render/Railway web process (gunicorn), WhiteNoise for static, media on local disk (ephemeral — outputs lost on restart; acceptable through Capstone Demo, stated in the UI).
- **Production-Future (planned):** web + Celery workers + Redis + Postgres + S3-compatible object storage; Dockerfile and GitHub Actions CI (lint, tests, eval smoke run).

## 6. Security posture

Current gaps to close in Phase 0: hardcoded `SECRET_KEY`, `DEBUG=True` on a public host, committed `db.sqlite3`, redundant unconditional media serving in `promo_project/urls.py`. MVP adds upload validation (extension allow-list, size cap, ffprobe sanity check) since file upload is the trust boundary. Auth, rate limiting, and virus scanning are Production-Future.
