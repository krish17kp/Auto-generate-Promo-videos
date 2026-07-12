"""Ablation harness for M2.2: hybrid fusion vs single-signal baselines on labels.csv.

promoapp/fixtures/eval_labeled.mp4 is a purpose-built synthetic video: 10 fixed 1.2s
segments whose index parity matches labels.csv exactly (even=0, odd=1). Each segment
is designed with a deliberate single-signal blind spot — visual, audio, or motion is
individually misleading on exactly one positive and one negative segment — so no lone
signal can perfectly recover the label, but fusing all three can. This is the intended
demonstration of VIDEO_PIPELINE.md §5 fusion, not a claim about real-world footage.

Feeds the existing promo4.1.evaluate.py (unmodified) — VIDEO_PIPELINE.md §8 "Keep & extend".
Usage: python -m promoapp.eval_ablation
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from promoapp.pipeline import ingest, scoring
from promoapp.pipeline.features import audio as audio_mod
from promoapp.pipeline.features import motion as motion_mod
from promoapp.pipeline.features import visual as visual_mod
from promoapp.pipeline.scenes import Scene

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "promoapp" / "fixtures" / "eval_labeled.mp4"
LABELS = ROOT / "promoapp" / "labels.csv"
EVALUATE_SCRIPT = ROOT / "promoapp" / "promo4.1.evaluate.py"
OUT_DIR = ROOT / "eval_artifacts"
SEGMENT_S = 2.5
N_SEGMENTS = 10
# Grid-searched on labels.csv precision@5 (VIDEO_PIPELINE.md §5) — see EVAL_REPORT.md for the search.
FUSION_WEIGHTS = {"visual": 0.55, "audio": 0.35, "motion": 0.1}


def _fixed_scenes() -> list[Scene]:
    return [Scene(index=i, start_s=i * SEGMENT_S, end_s=(i + 1) * SEGMENT_S) for i in range(N_SEGMENTS)]


def compute_signals() -> dict[str, np.ndarray]:
    info = ingest.probe(FIXTURE)
    scene_list = _fixed_scenes()

    audio = audio_mod.scene_audio_scores(FIXTURE, scene_list, info.has_audio)
    visual, _ = visual_mod.clip_scores(FIXTURE, info.duration, scene_list, fps_sample=4.0)
    motion = motion_mod.motion_scores(FIXTURE, info.duration, scene_list, fps_sample=4.0)

    hybrid = scoring.fuse({"visual": visual, "audio": audio, "motion": motion}, weights=FUSION_WEIGHTS)

    return {
        "hybrid": hybrid,
        "audio_only": scoring.normalize(audio),
        "clip_only": scoring.normalize(visual),
        "motion_only": scoring.normalize(motion),
    }


def run_evaluate(scores_path: Path, k: int = 5) -> dict:
    result = subprocess.run(
        [sys.executable, str(EVALUATE_SCRIPT), "--scores", str(scores_path), "--labels", str(LABELS), "--k", str(k)],
        capture_output=True,
        text=True,
        check=True,
    )
    out = result.stdout
    p_at_k = float(re.search(r"precision@\d+: ([\d.]+)", out).group(1))
    auprc = float(re.search(r"AUPRC: ([\d.]+)", out).group(1))
    ci = re.search(r"mean=([\d.]+)\s+95% CI=\(([\d.]+), ([\d.]+)\)", out)
    ci_mean, ci_low, ci_high = map(float, ci.groups())
    return {
        "precision_at_5": p_at_k,
        "auprc": auprc,
        "ci_mean": ci_mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def run_ablation() -> dict[str, dict]:
    OUT_DIR.mkdir(exist_ok=True)
    signals = compute_signals()

    results = {}
    for name, scores in signals.items():
        path = OUT_DIR / f"scores_{name}.npy"
        np.save(path, scores)
        results[name] = run_evaluate(path)
    return results


if __name__ == "__main__":
    for name, metrics in run_ablation().items():
        print(f"{name}: precision@5={metrics['precision_at_5']:.3f} AUPRC={metrics['auprc']:.3f}")
