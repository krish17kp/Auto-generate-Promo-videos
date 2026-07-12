# VIDEO_PIPELINE — Stage-by-Stage Specification

> The technical core of the project. Companion docs: [ARCHITECTURE.md](ARCHITECTURE.md) · [PRD.md](PRD.md) · [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md)
>
> Scope tiers: **MVP → Capstone Demo → Advanced AI → Production-Future**. Unbuilt = **(planned)**; defects = **(current gap)**.

## 0. Today vs target — the honest delta

The only pipeline code in the repo is `promoapp/promo4.1.py`, an unwired standalone script that currently cannot run (import/indentation/API bugs — see [ARCHITECTURE.md §1](ARCHITECTURE.md)). This table maps what it *attempts* to the target design:

| Stage | `promo4.1.py` today | Target (`promoapp/pipeline/`) | Verdict |
|---|---|---|---|
| Scene detection | PySceneDetect `ContentDetector(threshold=27)` | Same | **Keep** |
| Visual scoring | EfficientNetB0 (frozen) + Dense→sigmoid head, **never trained** → random scores (current gap) | CLIP zero-shot prompt scoring; embeddings reused for diversity | **Replace** |
| Audio | librosa RMS + onset strength | Same, plus per-scene aggregation | **Keep & extend** |
| Motion | — | Optical-flow magnitude (OpenCV) | **Add** |
| Quality gates | — | Sharpness (Laplacian variance) + exposure | **Add** |
| Fusion | StandardScaler + moving-average smoothing of CNN channel only | Weighted per-scene fusion of all normalized signals | **Redesign** |
| Selection | `scipy.signal.find_peaks` top-N, fixed clip length around peaks, merge overlaps | Narrative assembly: hook → build → climax → outro + diversity constraint | **Redesign** |
| Render | MoviePy `subclip`/`speedx`/crossfade (MoviePy 1.x API — breaks on pinned 2.2) (current gap) | MoviePy 2.x API, aspect presets, overlays, captions | **Rewrite** |
| Eval | `promo4.1.evaluate.py`: precision@K, AUPRC, bootstrap CI vs `labels.csv` | Same methodology, fed by pipeline score dumps, plus ablations | **Keep & extend** — this is the strongest existing asset |

## 1. Package layout (planned)

```
promoapp/pipeline/
├── ingest.py        # probe, validate, normalize input
├── scenes.py        # PySceneDetect boundaries
├── features/
│   ├── visual.py    # CLIP zero-shot scores + embeddings
│   ├── motion.py    # optical-flow magnitude
│   ├── audio.py     # librosa RMS + onset (port of existing code)
│   ├── quality.py   # sharpness / exposure gates
│   └── transcript.py# faster-whisper (Advanced AI)
├── scoring.py       # normalization + weighted fusion
├── narrative.py     # segment selection, trailer grammar
├── render.py        # MoviePy 2.x assembly, aspect presets, captions
└── cli.py           # python -m promoapp.pipeline.cli --input … --output …
```

Framework-free (no Django imports) so the web app, CLI, and eval harness share one implementation.

## 2. Stage: Ingest — MVP (planned)

- **In:** uploaded file path. **Out:** `VideoInfo(duration, fps, resolution, has_audio)`.
- ffprobe (via `imageio-ffmpeg`, already a dependency) validates the container and reads metadata. Reject: unreadable files, >30 min, >500 MB, no video stream.
- **Failure mode:** corrupt upload → job `failed` with "couldn't read this video," never a traceback page.

## 3. Stage: Scene detection — MVP (planned; logic exists in script)

- PySceneDetect `ContentDetector(threshold=27)` → list of `(start, end)` scene spans, persisted as `Scene` rows.
- **Why scenes as the unit:** cuts at scene boundaries look edited; cuts mid-scene look broken. Peak-picking on a raw frame-score curve (the current script's approach) produces mid-scene cuts — one reason its output would feel random even if the scorer worked.
- Guard: if <3 scenes detected (static lecture footage), fall back to fixed 5 s windows so the pipeline still produces output.

## 4. Stage: Feature extraction — signals per scene

Frames sampled at 1–2 fps within each scene (budget-capped so long videos don't explode compute).

| Signal | Method | Tier | Rationale |
|---|---|---|---|
| **Visual semantics** | CLIP (e.g. ViT-B/32, open weights) zero-shot similarity against a promo-prompt set: "an exciting action moment", "a person speaking directly to camera", "a product close-up", "a dramatic wide shot", vs. anti-prompts: "a blank screen", "a blurry transition frame" | Capstone Demo (planned) | Real semantic signal with **zero training data** — directly fixes the untrained-CNN gap. Prompt set is a tunable config, not code. |
| **Scene diversity** | Cosine distance between mean CLIP embeddings of scenes | Capstone Demo (planned) | Prevents five near-identical clips; embeddings are a free by-product of the CLIP pass |
| **Motion** | Mean dense optical-flow magnitude (OpenCV Farnebäck) between sampled frame pairs | Capstone Demo (planned) | High-energy moments move; cheap and model-free |
| **Audio energy** | librosa RMS + onset strength, aggregated per scene | MVP (code exists in script, to be ported) | Laughter, music hits, applause, raised voices — already implemented and proven in the script |
| **Quality gates** | Laplacian-variance sharpness + luma bounds; gate, don't score | Capstone Demo (planned) | Rejects blurry/black frames so they can't be selected no matter the other signals |
| **Transcript** | faster-whisper segments; hook-phrase scoring (questions, numbers, imperatives, keywords) | Advanced AI (planned) | Spoken hooks are the strongest promo signal for talking content; deferred because it's the heaviest dependency |

**MVP scoring** = audio energy + a simple visual proxy (frame difference), so the end-to-end loop works before CLIP lands.

## 5. Stage: Score fusion — Capstone Demo (planned)

```
scene_score = Σ wᵢ · normalize(signalᵢ)        subject to quality gate
```

- Each signal min-max normalized per video (scores are relative within a video, matching how `labels.csv` labels are defined).
- Default weights (initial, to be tuned against eval — not sacred): visual 0.4, audio 0.3, motion 0.2, transcript 0.1 when present (renormalized when absent).
- Temporal smoothing (moving average, ported from the script) applied to frame-level curves before per-scene aggregation.
- All per-signal and fused scores persisted to `SegmentScore` — one write, three consumers: selection, the UI explainability panel (F-10), and the eval harness.
- **Weight tuning:** grid-search on `labels.csv` precision@5 with a held-out split; results published in the eval report, per-video learned weights are explicitly out of scope.

## 6. Stage: Narrative assembly — Capstone Demo (planned)

Trailer grammar instead of top-K peaks:

1. **Hook** — highest-scoring 2–3 s moment in the whole video, placed first. (Advanced AI: best transcript hook line wins ties.)
2. **Build** — 2–4 mid-to-high segments in *chronological* order, each ≥ diversity threshold from already-picked scenes, preferring rising audio energy.
3. **Climax** — the peak fused-score scene (if not already used as hook).
4. **Outro** — a clean, low-motion scene + text overlay/CTA end card.

Constraints: total ≤ target duration (15/30/60 s); per-clip 1.5–6 s; no two clips from the same scene; chronological order within build (viewers notice time jumping backward). Selection is a greedy pass with the diversity penalty — an ILP would be over-engineering for ≤8 clips.

**MVP fallback:** top-3 scenes by score, chronological, hard-trimmed to 30 s — enough to prove the loop.

## 7. Stage: Render — MVP basic / Capstone Demo full (planned)

- MoviePy 2.x API (`clip.subclipped`, `CompositeVideoClip`, `vfx.CrossFadeIn`) — the script's MoviePy 1.x calls are rewritten, not patched.
- **Aspect presets:** 16:9 pass-through; 9:16 and 1:1 via center-crop MVP → content-aware crop (track the dominant CLIP-attention/face region) as Capstone Demo polish.
- Crossfades (0.3 s), optional 1.1× speed on build segments (port of script's intent), title/CTA text overlays, `libx264 + aac`, `faststart` for browser playback.
- **Captions (Advanced AI):** burn transcript segments as styled subtitles per aspect preset.
- **Failure mode:** render errors mark the job `failed` with the stage name; partial temp files cleaned up.

## 8. Stage: Evaluation — Capstone Demo (planned; harness exists)

- **Asset that already works:** `promo4.1.evaluate.py` implements precision@K, AUPRC (`sklearn.average_precision_score`), and bootstrap confidence intervals against `labels.csv`.
- Pipeline change: `scoring.py` dumps per-segment fused + per-signal scores (`.npy`/CSV) in the harness's expected format.
- **Ablation report (the resume differentiator):** hybrid vs audio-only vs CLIP-only vs motion-only, precision@5 and AUPRC with CIs, published as `EVAL_REPORT.md` (planned) and linked from README. Rule from [BRD.md §5](BRD.md): a signal that doesn't lift eval doesn't ship.
- Also logged per run: wall-clock per pipeline stage per input-minute (feeds the PRD speed metric).

## 9. Performance budget (CPU-only target)

| Stage | Budget driver | Control |
|---|---|---|
| Scene detection | full decode pass | PySceneDetect downscale factor |
| CLIP | frames × model forward | ≤2 fps sampling, ≤300 frames/video cap, batch inference |
| Optical flow | frame pairs | compute on 480p downscale |
| Whisper | audio duration | `faster-whisper` small/int8; Advanced AI only |
| Render | output duration | ≤60 s output, single pass |

Total target: ≤2× input duration wall-clock on free-tier CPU ([PRD.md §8](PRD.md)); measured, and published rather than promised.
