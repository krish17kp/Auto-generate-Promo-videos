# DATABASE_SCHEMA — Auto-Generate Promo Videos

> Companion docs: [ARCHITECTURE.md](ARCHITECTURE.md) · [VIDEO_PIPELINE.md](VIDEO_PIPELINE.md) · [PRD.md](PRD.md)
>
> Scope tiers: **MVP → Capstone Demo → Advanced AI → Production-Future**. Unbuilt = **(planned)**.

## 1. Current state

`promoapp/models.py` is **empty** — no models, no migrations (only `__init__.py`), no job tracking (current gap). Uploads/outputs live on the filesystem with no DB record. The committed `db.sqlite3` contains only Django's built-in auth/session tables and should be removed from version control in Phase 0.

Everything below is **(planned)**. Engine: SQLite through Capstone Demo (fits the single-user demo; zero ops), Postgres in Production-Future (concurrent workers need real row locking).

## 2. Entity overview

```
VideoUpload 1──1 PromoJob 1──* Scene 1──* SegmentScore
                    │
                    └───────1──* PromoOutput
EvalRun (standalone, references a PromoJob)
```

## 3. Tables

### `VideoUpload` — MVP
| Field | Type | Notes |
|---|---|---|
| id | UUID PK | UUIDs so media paths aren't guessable |
| file | FileField → `media/input/` | |
| original_name | CharField | display only; never used as a path |
| size_bytes, duration_s, fps, width, height | ints/floats | from ffprobe at ingest |
| has_audio | Boolean | pipeline skips audio signals when false |
| created_at | DateTime, indexed | retention sweeps |

### `PromoJob` — MVP (the workflow spine)
| Field | Type | Notes |
|---|---|---|
| id | UUID PK | |
| upload | OneToOne(VideoUpload) | |
| status | CharField choices: `queued / processing / done / failed` | indexed |
| stage | CharField choices: `ingest / scenes / features / scoring / narrative / render` | drives progress UI (F-9); names match pipeline modules |
| progress | SmallInt 0–100 | |
| params | JSONField | aspect presets, target duration, weights used — full reproducibility of a run |
| error_message | TextField, nullable | user-readable (F-5) |
| started_at, finished_at | DateTime | wall-clock metric ([PRD.md §8](PRD.md)) |
| created_at | DateTime, indexed | |

### `Scene` — MVP
| Field | Type | Notes |
|---|---|---|
| id | BigAuto PK | |
| job | FK(PromoJob), indexed | |
| index | SmallInt | ordinal within video; unique together with job |
| start_s, end_s | Float | PySceneDetect boundaries |
| embedding | JSONField, nullable | mean CLIP embedding (Capstone Demo) — JSON list is fine at ≤512 floats × ≤200 scenes; a vector column is Production-Future territory |

### `SegmentScore` — Capstone Demo (explainability + eval, one row per scene)
| Field | Type | Notes |
|---|---|---|
| id | BigAuto PK | |
| scene | OneToOne(Scene) | |
| visual, audio, motion, quality, transcript | Float, nullable | normalized per-signal scores; NULL = signal not computed at this tier |
| fused | Float, indexed | selection key |
| selected | Boolean | chosen for the promo |
| narrative_role | CharField choices: `hook / build / climax / outro`, nullable | set when selected |

Per-signal columns (not a JSON blob) because they are queried independently by the eval ablations and the explainability panel (F-10).

### `PromoOutput` — MVP (one row per rendered format)
| Field | Type | Notes |
|---|---|---|
| id | UUID PK | |
| job | FK(PromoJob) | |
| file | FileField → `media/output/` | |
| aspect | CharField choices: `16:9 / 9:16 / 1:1` | MVP renders 16:9 only |
| duration_s, size_bytes | Float / BigInt | |
| created_at | DateTime | |

### `EvalRun` — Capstone Demo
| Field | Type | Notes |
|---|---|---|
| id | BigAuto PK | |
| job | FK(PromoJob), nullable | nullable: harness also runs on offline score dumps |
| config | JSONField | which signals/weights (ablation id) |
| precision_at_5, auprc | Float | from `promo4.1.evaluate.py` methodology |
| ci_low, ci_high | Float | bootstrap CI |
| created_at | DateTime | |

### Production-Future additions (planned, not designed in detail here)
`User` FK on `VideoUpload` (auth), `TranscriptSegment` table if caption editing ships, object-storage keys replacing local `FileField` paths.

## 4. Indexes & integrity

- `PromoJob(status, created_at)` — worker pickup + admin lists.
- `Scene(job, index)` unique — idempotent re-runs overwrite cleanly.
- `SegmentScore.fused` — top-K selection queries.
- All FKs `on_delete=CASCADE` from `VideoUpload` down: deleting an upload erases the whole job trail (retention, and simple GDPR-style delete later).

## 5. Retention (Capstone Demo)

Media disk is ephemeral on Render/Railway (stated in UI). Planned sweep (management command, cron in Production-Future): delete `media/input/` sources after 24 h, keep `PromoOutput` files 7 days, keep DB rows (scores, params, eval) indefinitely — they're small and they *are* the portfolio evidence.
