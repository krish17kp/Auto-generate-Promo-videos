# Auto-Generate Promo Videos

Upload a long video → the system detects scenes, scores every moment across visual, motion, and audio signals → assembles the strongest moments into a trailer-style promo (hook → build → climax → outro) → renders it for 16:9, 9:16, and 1:1.

A Django + computer-vision capstone built around one discipline: **every claim in this repo is either running code or explicitly marked (planned), and selection quality is measured, not asserted** — precision@K / AUPRC against labeled ground truth, with single-signal ablations.

## Project status — read this first

**All four phases — Repair, MVP, Capstone Demo, Advanced AI, and Production-Future — are complete.** All 20 milestones in [MILESTONES.md](MILESTONES.md) are checked, each against a real acceptance test, not a self-assessment. Honest inventory:

| Component | Status |
|---|---|
| Django app: real upload → job → progress → result flow, accounts, per-user history | ✅ Working |
| Pipeline package (`promoapp/pipeline/`): ingest, scenes, features, scoring, narrative, render, CLI | ✅ Working — framework-free, shared by web app, CLI, and eval harness |
| Hybrid scoring: CLIP zero-shot visual + optical-flow motion + librosa audio + quality gates, weighted fusion | ✅ Working (`--profile capstone`) |
| Narrative assembly: hook → build → climax → outro, diversity-constrained | ✅ Working |
| Aspect presets (16:9 / 9:16 / 1:1) + 15/30/60s duration targets, one job renders all three | ✅ Working |
| Evaluation harness — precision@K, AUPRC, bootstrap CI vs `labels.csv` (`promoapp/promo4.1.evaluate.py`) | ✅ Fed real pipeline scores — see [EVAL_REPORT.md](EVAL_REPORT.md) |
| Job tracking, live progress UI, explainability panel, title/CTA overlay, burned-in captions | ✅ Working |
| Transcript signal (faster-whisper) + keyword hooks, LLM hook picker w/ fallback | ✅ Working — opt-in (`--profile advanced`), not the shipped default (measured zero lift on the synthetic eval fixture — [EVAL_REPORT.md §5](EVAL_REPORT.md)) |
| Learned scorer (fine-tuned head over the zero-shot hybrid) | ❌ Dropped, documented — `labels.csv` (10 rows) is too small to train or validate one ([MILESTONES.md](MILESTONES.md) M3.4) |
| Celery + Redis worker swap | ✅ Code + eager-mode test; no live broker stood up in dev (see M4.1 note) |
| Postgres, optional S3 object storage, retention sweep | ✅ Working |
| Auth, per-user history, rate limiting | ✅ Working |
| Dockerfile, GitHub Actions CI (lint + test + eval-gate smoke + gated deploy) | ✅ Written; full local `docker build` deferred to CI's clean runner after it hit real memory pressure on this dev machine (see M4.4 note) |

Full acceptance criteria, what's verified where, and every honest limitation found along the way: [MILESTONES.md](MILESTONES.md). Progress tracker + decisions log: [TODO.md](TODO.md).

## Documentation map

| Doc | What it covers |
|---|---|
| [PRD.md](PRD.md) | Users, requirements (F-1…F-17), UX flow, success metrics |
| [BRD.md](BRD.md) | Positioning, KPIs, cost model, risks |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Current vs target system, key decisions (D-1…D-5) |
| [VIDEO_PIPELINE.md](VIDEO_PIPELINE.md) | **The technical core** — stage-by-stage spec, scoring math, eval design |
| [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) | Data model, field-level |
| [ROADMAP.md](ROADMAP.md) | Phase 0 Repair → 1 MVP → 2 Capstone Demo → 3 Advanced AI → 4 Production-Future |
| [MILESTONES.md](MILESTONES.md) | Checkable acceptance criteria + demo artifact per milestone |
| [EVAL_REPORT.md](EVAL_REPORT.md) | Hybrid vs single-signal ablation — the M2.2 gate |

All docs share one scope vocabulary: **MVP → Capstone Demo → Advanced AI → Production-Future**.

## How it works

```
Upload (MP4/MOV)
  → Ingest & validate (moviepy probe: duration/size/codec)
  → Scene detection (PySceneDetect ContentDetector, fixed-window fallback for static footage)
  → Per-scene signals:
      CLIP zero-shot semantics · optical-flow motion · audio energy (librosa) · quality gates
  → Weighted score fusion (+ scene-diversity via CLIP embeddings)
  → Narrative assembly: hook → build → climax → outro
  → Render (MoviePy 2.x / FFmpeg): crossfades, aspect presets, title/CTA overlays, captions
  → Evaluation: precision@K / AUPRC ablations vs labels.csv
```

Design choice worth noting: the original approach — an untrained CNN head scoring frames — produces no learned signal, so it's replaced by **zero-shot CLIP prompts + model-free motion/audio/quality signals**, which give real, explainable scores with no training data. A learned scorer was evaluated for the Advanced AI tier and explicitly **dropped**: the only ground truth (`labels.csv`, 10 rows) is too small to fit or validate one — see [MILESTONES.md](MILESTONES.md) M3.4. Full rationale: [VIDEO_PIPELINE.md §0](VIDEO_PIPELINE.md).

## Tech stack

| Layer | Technology |
|---|---|
| Web | Django 5.2, WhiteNoise, gunicorn (Render/Railway), auth + rate limiting (`django-ratelimit`) |
| Scene detection | PySceneDetect |
| Visual semantics | CLIP zero-shot (`open_clip`, ViT-B/32-quickgelu) |
| Audio | librosa (RMS, onset strength) |
| Motion / quality | OpenCV (Farnebäck optical flow, Laplacian sharpness + luma gate) |
| Transcription | faster-whisper + keyword/LLM hook scoring (opt-in, `--profile advanced`) |
| Assembly | MoviePy 2.x + FFmpeg |
| Evaluation | scikit-learn (AUPRC), bootstrap CIs, `labels.csv` ground truth |
| Async jobs | in-process thread runner (default) or Celery + Redis (`USE_CELERY=True`) — swap confined to `promoapp/jobs.py` |
| Storage | SQLite/Postgres (`DATABASE_URL`), local disk or S3-compatible object storage (`AWS_STORAGE_BUCKET_NAME`) |
| Ship | Dockerfile, GitHub Actions CI (lint, test, eval-gate smoke, gated deploy) |

## Running locally

Prerequisites: Python 3.11+ (developed on 3.13), FFmpeg on PATH.

```bash
git clone https://github.com/krish17kp/Auto-generate-Promo-videos.git
cd Auto-generate-Promo-videos
pip install -r requirements.txt
cp .env.example .env   # fill in SECRET_KEY (django.core.management.utils.get_random_secret_key())
python manage.py migrate
python manage.py test promoapp   # full suite: routes, upload safety, jobs, pipeline, eval gate
python manage.py runserver
```

CLI (no web server needed):

```bash
python -m promoapp.pipeline.cli --input your_video.mp4 --output promo.mp4 \
  --profile capstone --aspect 9:16 --duration 30 --title "My Promo" --cta "Watch Now"
```

`--profile` is `mvp` (fast: audio + frame-diff), `capstone` (CLIP + motion + quality, full narrative), or `advanced` (+ transcript + captions).

## Evaluation methodology (the part worth scrutinizing)

Selection quality is scored against ground truth (`promoapp/labels.csv`) using precision@K and AUPRC with bootstrap confidence intervals — implemented in `promoapp/promo4.1.evaluate.py` (kept unmodified, per [VIDEO_PIPELINE.md §8](VIDEO_PIPELINE.md)). `promoapp/eval_ablation.py` feeds it real per-scene scores from the pipeline's own feature modules for a purpose-built labeled fixture. Full methodology, honest caveats about the small (n=10) ground truth, and the concrete failure modes each single signal has: **[EVAL_REPORT.md](EVAL_REPORT.md)**.

**Result: the hybrid fused score beats every single-signal baseline on precision@5** (1.000 vs. audio-only 0.800, CLIP-only 0.800, motion-only 0.200) — [MILESTONES.md](MILESTONES.md) M2.2's gate.

## Results & demo

**Sample input/output pairs** (committed as fixtures, small enough to play directly):

| Pair | What it shows |
|---|---|
| `promoapp/fixtures/sample_input.mp4` → `sample_output_mvp.mp4` | MVP path: real scene-cut promo, not a mock copy (M1.1) |
| `promoapp/fixtures/demo_input.mp4` → `demo_output_16x9.mp4` | Capstone path: hybrid scoring, narrative roles, title + CTA overlay |

**Explainability + overlays** — screenshots from a real rendered output (`demo_output_16x9.mp4`):

| Title overlay | CTA end-card |
|---|---|
| ![title overlay](docs/screenshots/promo_title_overlay.png) | ![CTA overlay](docs/screenshots/promo_cta_overlay.png) |

**Ablation table** (full detail in [EVAL_REPORT.md](EVAL_REPORT.md)):

| Signal | precision@5 | AUPRC |
|---|---|---|
| **Hybrid (fused)** | **1.000** | **1.000** |
| Audio-only | 0.800 | 0.810 |
| CLIP-only | 0.800 | 0.927 |
| Motion-only | 0.200 | 0.403 |

**Measured wall-clock** (CPU-only, capstone profile — CLIP + motion + quality + full narrative — on a 20.1s / 480×270 input, model weights already cached from a prior run so this is pure compute, not download time):

| Input duration | Wall-clock | Ratio (wall-clock ÷ input-minute) |
|---|---|---|
| 0.33 min (20.1s) | 104s (features stage: 88.2s, render: 5.7s, scenes: 1.8s, rest negligible) | ~310× |

**This misses the ≤2× budget target from [VIDEO_PIPELINE.md §9](VIDEO_PIPELINE.md) by a wide margin, and that's reported plainly rather than hidden.** The `features` stage — CLIP forward passes on CPU — is 85% of total wall-clock and is dominated by fixed per-call model overhead rather than input length, so this ratio is worst-case for a *short* clip; it should amortize substantially better on longer, more realistic inputs (more scenes and frames per model-load/warm-up), but that hasn't been measured yet and isn't claimed here. GPU inference or a smaller CLIP checkpoint would be the first optimization to try before promising the CPU budget on real footage.

**2-minute demo path**: upload → live progress page (stage-by-stage) → result page with video player + per-segment explainability table (visual/audio/motion/fused per scene, narrative role) + download. No literal screen recording exists in this automated build session; the path is instead verified by passing route, job-lifecycle, and template-rendering tests (`promoapp/tests/`), plus the visually-inspected overlay screenshots above.

## Limitations

- CPU-only targets: processing budgeted at ≤2× input duration, measured not promised — and currently missed by a wide margin on short clips (see wall-clock table above); CLIP inference dominates.
- Single source video per job; no timeline editing, music generation, or live processing ([PRD.md §7](PRD.md)).
- `labels.csv` ground truth is 10 samples — enough to exercise the eval methodology, not to draw statistically tight conclusions (bootstrap CIs are wide; see [EVAL_REPORT.md §1](EVAL_REPORT.md)). A learned scorer and a rigorous transcript-signal lift measurement both need a materially larger labeled set than exists today.
- Transcript signal is opt-in (`--profile advanced`), not the shipped default — it's independently verified correct, but hasn't been shown to improve ranking on real footage (only a zero-lift result on a speech-free synthetic fixture).
- Media storage defaults to local disk (ephemeral on Render/Railway); S3-compatible object storage is implemented but requires `AWS_STORAGE_BUCKET_NAME` to be configured.
- Celery + Redis worker mode is implemented and tested in eager mode, but no live Redis broker has been run against this codebase yet — the in-process thread runner is what's actually been exercised end-to-end.
- Local `docker build` was not completed in the development environment used to build this (triggered a real memory-pressure crash on a resource-constrained machine); the Dockerfile is structurally reviewed and the GitHub Actions CI builds/tests it on a clean runner.
