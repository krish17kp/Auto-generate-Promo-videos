from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

from .run import PROFILE_OVERRIDES, PipelineConfig, run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a trailer-style promo video from a long-form input.")
    p.add_argument("--input", "-i", required=True, help="Input video file path")
    p.add_argument("--output", "-o", default="promo_output.mp4", help="Output promo file path")
    p.add_argument("--duration", "-d", type=int, default=30, choices=[15, 30, 60], help="Target duration (s)")
    p.add_argument("--aspect", default="16:9", choices=["16:9", "9:16", "1:1"], help="Output aspect ratio")
    p.add_argument("--fps", type=float, default=2.0, help="Feature-extraction sampling fps")
    p.add_argument("--profile", default="mvp", choices=sorted(PROFILE_OVERRIDES), help="Feature/narrative tier")
    p.add_argument("--scene-threshold", type=float, default=27.0, help="PySceneDetect content threshold")
    p.add_argument("--no-effects", action="store_true", help="Disable speed/crossfade effects")
    p.add_argument("--title", default=None, help="Title overlay text")
    p.add_argument("--cta", default=None, help="End-card CTA overlay text")
    p.add_argument("--save-scores", action="store_true", help="Persist fused scores for evaluation")
    p.add_argument("--run-tag", default="run", help="Tag for saved artifacts")
    p.add_argument("--out-dir", default="eval_artifacts", help="Directory for saved artifacts")
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    return p.parse_args(argv)


def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _set_seed(args.seed)

    if not Path(args.input).exists():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 1

    config = PipelineConfig.for_profile(
        args.profile,
        target_duration=args.duration,
        aspect=args.aspect,
        fps_sample=args.fps,
        scene_threshold=args.scene_threshold,
        add_effects=not args.no_effects,
        title=args.title,
        cta=args.cta,
        save_scores=args.save_scores,
        out_dir=Path(args.out_dir),
        run_tag=args.run_tag,
    )

    try:
        result = run_pipeline(args.input, args.output, config)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"promo saved to: {result.output_path}")
    print(f"duration: {result.output_duration_s:.1f}s across {len(result.scene_scores)} scenes")
    for stage, secs in result.stage_timings.items():
        print(f"  {stage}: {secs:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
