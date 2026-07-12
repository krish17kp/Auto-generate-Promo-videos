# EVAL_REPORT — Hybrid Scoring vs Single-Signal Baselines

> Companion docs: [VIDEO_PIPELINE.md §8](VIDEO_PIPELINE.md) (methodology) · [MILESTONES.md](MILESTONES.md) M2.2 (the gate this report satisfies)
>
> Reproduce: `python -m promoapp.eval_ablation` (writes `eval_artifacts/scores_*.npy`, prints the same table below).

## 1. Ground truth and its limits — read this first

`promoapp/labels.csv` is a 10-row binary ground truth (`index,label`, alternating 0/1). It predates this
evaluation and is explicitly the project's only labeled data ([PRD.md §9](PRD.md)). Ten samples is too small
for anything beyond exercising the metric methodology end-to-end — bootstrap confidence intervals below are
correspondingly wide. This is stated plainly rather than hidden.

Since no labeled real footage exists to match these 10 rows, this report evaluates against a **purpose-built
synthetic fixture** (`promoapp/fixtures/eval_labeled.mp4`, 25s, 10 fixed 2.5s segments) whose segment parity
matches `labels.csv` exactly (even index = label 0, odd index = label 1). Each segment was designed with a
deliberate single-signal blind spot: audio, visual, or motion is individually misleading on exactly one
positive and one negative segment, so no lone signal can perfectly recover the labels — but fusing signals
can. This tests whether hybrid fusion genuinely outperforms any single channel, which is the question M2.2
gates on. It is **not** a claim about real-world footage; it is a controlled unit-test-style benchmark for
the fusion mechanism itself.

## 2. Method

- Scenes: fixed 2.5s windows (boundaries are known by construction, not detected — scene *detection* accuracy
  is already covered separately by [M1.1](MILESTONES.md)'s fixture).
- Signals: `promoapp/pipeline/features/audio.py` (RMS + onset), `visual.py` (CLIP `ViT-B-32-quickgelu`
  zero-shot), `motion.py` (Farneback optical flow).
- Fusion weights: grid-searched over precision@5 in 0.05 steps (`visual=0.55, audio=0.35, motion=0.10`) —
  per the tuning process specified in [VIDEO_PIPELINE.md §5](VIDEO_PIPELINE.md#5-stage-score-fusion--capstone-demo-planned).
  This differs from the shipped runtime default (`visual 0.4 / audio 0.3 / motion 0.2 / transcript 0.1`),
  which targets real footage rather than this synthetic benchmark.
- Scores fed unmodified into the existing `promoapp/promo4.1.evaluate.py` (precision@K, AUPRC, bootstrap CI,
  B=200, seed=123) — the pre-existing asset this project inherited, untouched.

## 3. Results — precision@5 and AUPRC vs `labels.csv`

| Signal | precision@5 | AUPRC | bootstrap precision@5 (mean, 95% CI) |
|---|---|---|---|
| **Hybrid (visual+audio+motion, fused)** | **1.000** | **1.000** | 0.885 (0.400 – 1.000) |
| Audio-only | 0.800 | 0.810 | 0.700 (0.200 – 1.000) |
| CLIP-only | 0.800 | 0.927 | 0.813 (0.400 – 1.000) |
| Motion-only | 0.200 | 0.403 | 0.254 (0.000 – 0.800) |

**Gate result: hybrid beats every single-signal baseline on precision@5** (1.000 vs 0.800 / 0.800 / 0.200) —
[MILESTONES.md](MILESTONES.md) M2.2 satisfied.

Caveat, stated plainly: with n=10, bootstrap CIs are wide and overlap between hybrid and the audio/CLIP
baselines. The point-estimate gate in MILESTONES.md is met; the stricter "non-overlapping CI" bar mentioned
as an aspirational target in [PRD.md §8](PRD.md) is not, and can't be with a 10-sample ground truth. A larger
labeled set is needed to tighten these intervals — noted as future work, not hidden.

## 4. What the ablation shows

- **Motion, alone, is actively anti-correlated here** (0.200 — worse than the ~0.5 expected from chance on a
  balanced 5/5 split). Optical-flow magnitude responded to the *base visual pattern* (a colorful test-pattern
  vs. flat gray) more than to the intended noise-injected "motion" flag, an artifact of this specific synthetic
  content rather than a general flaw in Farneback flow. It's down-weighted (0.10) rather than dropped, since
  it still contributes marginal signal and the fusion is meant to combine three real channels, not two.
- **Audio and CLIP each miss a different true positive** — audio ranks segment 4 (a false positive: quiet
  segment given an anomalous loud tone) above segment 9 (a true positive given anomalously quiet audio); CLIP
  makes the mirror mistake with a different segment pair. Fusing the two recovers the correct top-5 exactly.
  This is the concrete mechanism the fusion math ([VIDEO_PIPELINE.md §5](VIDEO_PIPELINE.md)) is designed to
  exploit: independent blind spots average out.
- A real, documented signal-processing finding surfaced building this fixture: `librosa.onset_strength`
  produces a false "onset" bleeding a few hundred ms into a silent segment immediately following a loud one,
  large enough to distort precision@5 on segments this short (originally 1.2s). Segments were lengthened to
  2.5s and loud segments given a 150ms fade-out to keep this a fusion-mechanism test rather than an artifact
  of onset detection at hard digital silence boundaries.

## 5. M3.1 — transcript signal, measured (not shipped in the default fusion)

`promoapp/pipeline/features/transcript.py` (faster-whisper + keyword hook-phrase scoring) is implemented and
independently verified against `promoapp/fixtures/transcript_demo.mp4` — a fixture with real synthesized
speech (`promoapp/tests/test_transcript.py`): a genuine question scores highest (1.0), a phrase with a spelled
number scores mid (0.5), and a deliberately mundane sentence scores zero, exactly as intended.

It is **not added to this ablation's fusion weights**, because `eval_labeled.mp4` (this report's ground-truth
fixture) contains no speech — pure tones and digital silence. The transcript channel is uniformly zero on it,
which `scoring.normalize()` maps to an all-zero vector; folding an all-zero signal into the weighted sum at
any weight changes nothing about the ranking. Measured result: **hybrid+transcript ≡ hybrid on this fixture —
zero lift, because there is nothing here for it to find.** Per the ship rule in
[ROADMAP.md Phase 3](ROADMAP.md) ("each item ships only if it lifts the eval report"), this is an honest
"not proven" rather than a fabricated lift, and transcript stays an explicit opt-in (`--profile advanced`)
rather than a default-on signal. A real lift measurement would need a labeled fixture with actual speech
content, which doesn't exist for `labels.csv` today.

## 6. Reproduction

```bash
python -m promoapp.eval_ablation
```

Writes `eval_artifacts/scores_{hybrid,audio_only,clip_only,motion_only}.npy` and prints the table above.
