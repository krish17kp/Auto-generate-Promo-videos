# BRD — Auto-Generate Promo Videos

> Business Requirements Document. Companion docs: [PRD.md](PRD.md) · [ROADMAP.md](ROADMAP.md) · [MILESTONES.md](MILESTONES.md)
>
> Scope tiers: **MVP → Capstone Demo → Advanced AI → Production-Future**. This repo's primary business objective today is **portfolio/capstone value**; commercialization is sketched honestly under Production-Future. No market-size or revenue figures are invented anywhere in this document.

## 1. Business context

Short-form promos (Shorts, Reels, TikTok) are the highest-leverage distribution asset for long-form video, but producing them requires editing time most creators and small teams don't have. Commercial tools in this space (Opus Clip, Kapwing, Descript-class products) prove demand; this project demonstrates the core technology — automated high-impact moment extraction and trailer-style assembly — as an open, measurable system.

## 2. Objectives by tier

| Tier | Business objective | Evidence of success |
|---|---|---|
| MVP (planned) | A working product loop: upload → real generated promo → download | Live demo produces a genuinely edited clip (not the current mock copy — current gap) |
| Capstone Demo (planned) | A portfolio piece that survives technical scrutiny | Hybrid pipeline + published eval report (precision@K/AUPRC ablations) + polished UI; a recruiter can run it and read [VIDEO_PIPELINE.md](VIDEO_PIPELINE.md) |
| Advanced AI (planned) | Differentiation depth: transcript-aware hooks, captions, learned scorer | Eval shows measurable lift over the zero-shot hybrid before any feature ships |
| Production-Future (planned) | Optional path to a usable SaaS | Auth, durable storage, per-video compute cost known and bounded |

## 3. Target segments

1. **Solo creators** — highest volume, lowest willingness to pay, most tolerant of rough edges. Served from MVP.
2. **SMB marketing teams** — batch needs, want consistency across a library. Production-Future.
3. **Hiring panels / technical reviewers** — the real audience for the Capstone Demo tier: they value honest metrics, clean architecture, and explainability over feature count.

## 4. Value proposition & differentiation

- **vs. manual editing:** minutes instead of hours; deterministic, repeatable output.
- **vs. Opus Clip-class SaaS:** not a feature-parity competitor. Differentiators for this project: (a) **explainable scoring** — every chosen segment shows its per-signal scores; (b) **published evaluation** — precision@K/AUPRC against labeled ground truth (`labels.csv`), with single-signal ablations, using the existing `promo4.1.evaluate.py` methodology; (c) **open architecture** documented end to end.
- **Honest limitation:** commercial tools are transcript-first and template-rich; this system is signal-fusion-first and reaches transcripts only in the Advanced AI tier.

## 5. KPIs

| Tier | KPI | Notes |
|---|---|---|
| MVP | Job success rate; wall-clock per input-minute | Measured, published in README once real |
| Capstone Demo | precision@5 / AUPRC of hybrid vs baselines; demo conversion ("watched the demo → looked at the code") | Eval numbers only from actual harness runs |
| Advanced AI | Metric lift per added signal (transcript, learned head) | A feature that doesn't lift eval doesn't ship |
| Production-Future | Compute cost per processed video-minute; storage cost per user; retention | Prerequisite for any pricing decision |

## 6. Cost model (Production-Future sketch)

- **Compute:** CPU-only pipeline keeps unit cost to worker-minutes per video-minute (target ≤2×, per [PRD.md §8](PRD.md)); GPU cuts latency but raises cost — offered as a paid fast lane only if demanded.
- **Storage:** uploads are the dominant cost; retention policy (auto-delete sources after N days, keep promos) bounds it — see [DATABASE_SCHEMA.md §5](DATABASE_SCHEMA.md).
- **Models:** all models are open-weights (CLIP, faster-whisper) — zero per-call API cost; an optional LLM hook-picker (F-15) is the only metered external dependency and is degradable to a keyword heuristic.

## 7. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Selection quality plateaus (promos feel random) | Kills the value prop | Eval-gated development: hybrid must beat baselines on `labels.csv` before UI polish (see [MILESTONES.md](MILESTONES.md) M4) |
| CPU inference too slow on free-tier hosts | Demo feels broken | Frame-sampling budget, model size caps, published cold-start times; pre-generated demo video as fallback |
| Heavy ML deps (TF + CLIP + whisper) exceed host memory | Deploy fails | Lazy model loading, one model resident at a time; Advanced AI models optional at deploy time |
| `labels.csv` too small for training | F-16 (learned scorer) infeasible | Zero-shot-first strategy; F-16 explicitly gated on held-out eval |
| Copyright of user uploads | Legal exposure in Production-Future | ToS + user-owns-content model; not a concern for the capstone tiers |
| Current repo state (mock output, broken routes — current gaps) misleads reviewers | Credibility damage | Phase 0 "Repair" is the first roadmap item; README states current status plainly until fixed |

## 8. Out of scope (business)

No paid marketing, no pricing experiments, no team features, and no compliance work (GDPR/DMCA process) before Production-Future. No revenue targets are set for any tier in this document.
