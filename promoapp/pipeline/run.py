from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ingest
from . import scenes as scenes_mod
from . import scoring
from . import narrative
from . import render as render_mod
from .features import audio as audio_mod
from .features import motion as motion_mod
from .features import quality as quality_mod
from .features import transcript as transcript_mod
from .features import visual as visual_mod


# Shared by the CLI (--profile) and the Django view (upload-form profile field) — a plain
# dict so it stays JSON-serializable in PromoJob.params for full run reproducibility.
PROFILE_OVERRIDES: dict[str, dict] = {
    "mvp": {},
    "capstone": {"use_clip": True, "use_motion": True, "use_quality": True, "full_narrative": True},
    "advanced": {
        "use_clip": True, "use_motion": True, "use_quality": True, "use_transcript": True,
        "full_narrative": True, "add_captions": True,
    },
}


@dataclass
class PipelineConfig:
    target_duration: int = 30
    aspect: str = "16:9"
    fps_sample: float = 2.0
    scene_threshold: float = 27.0
    use_clip: bool = False
    use_motion: bool = False
    use_quality: bool = False
    use_transcript: bool = False
    use_llm_hook: bool = False
    add_captions: bool = False
    add_effects: bool = True
    full_narrative: bool = False
    title: str | None = None
    cta: str | None = None
    save_scores: bool = False
    out_dir: Path = field(default_factory=lambda: Path("eval_artifacts"))
    run_tag: str = "run"

    @classmethod
    def for_profile(cls, profile: str, **overrides) -> "PipelineConfig":
        merged = dict(PROFILE_OVERRIDES.get(profile, {}))
        merged.update(overrides)
        return cls(**merged)


@dataclass(frozen=True)
class SceneScoreResult:
    scene_index: int
    start_s: float
    end_s: float
    visual: float | None
    audio: float | None
    motion: float | None
    quality: float | None
    transcript: float | None
    fused: float
    selected: bool
    narrative_role: str | None


@dataclass(frozen=True)
class RunResult:
    info: ingest.VideoInfo
    scene_scores: list[SceneScoreResult]
    output_path: Path
    output_duration_s: float
    stage_timings: dict[str, float]


@dataclass(frozen=True)
class Analysis:
    """Everything upstream of render: scoring + narrative selection are aspect-independent,
    so one Analysis can drive renders in multiple aspect ratios without recomputing CLIP/motion."""

    info: ingest.VideoInfo
    scene_list: list
    clips_spec: list
    scene_scores: list[SceneScoreResult]
    transcript_segments: list[dict] | None
    stage_timings: dict[str, float]


def analyze(
    input_path: str | Path,
    config: PipelineConfig,
    on_stage: Callable[[str], None] | None = None,
) -> Analysis:
    timings: dict[str, float] = {}

    def stage(name: str) -> None:
        if on_stage:
            on_stage(name)

    stage("ingest")
    with _timed(timings, "ingest"):
        info = ingest.probe(input_path)

    stage("scenes")
    with _timed(timings, "scenes"):
        scene_list = scenes_mod.detect_scenes(input_path, info.duration, threshold=config.scene_threshold)

    stage("features")
    with _timed(timings, "features"):
        audio_scores = audio_mod.scene_audio_scores(input_path, scene_list, info.has_audio)

        embeddings = None
        if config.use_clip:
            visual_scores, embeddings = visual_mod.clip_scores(
                input_path, info.duration, scene_list, config.fps_sample
            )
        else:
            visual_scores = visual_mod.frame_diff_scores(input_path, info.duration, scene_list, config.fps_sample)

        motion_scores = (
            motion_mod.motion_scores(input_path, info.duration, scene_list, config.fps_sample)
            if config.use_motion
            else None
        )
        quality_gate = (
            quality_mod.quality_gate(input_path, info.duration, scene_list, config.fps_sample)
            if config.use_quality
            else None
        )

        transcript_segments = None
        transcript_scores = None
        if config.use_transcript:
            transcript_segments = transcript_mod.transcribe(input_path)
            transcript_scores = transcript_mod.scene_transcript_scores(transcript_segments, scene_list)

    stage("scoring")
    with _timed(timings, "scoring"):
        signals = {
            "visual": visual_scores,
            "audio": audio_scores,
            "motion": motion_scores,
            "transcript": transcript_scores,
        }
        fused = scoring.fuse(signals, quality_gate=quality_gate)

    stage("narrative")
    with _timed(timings, "narrative"):
        if config.full_narrative:
            clips_spec = narrative.select_narrative(
                scene_list, fused, embeddings, audio_scores, motion_scores, config.target_duration
            )
        else:
            clips_spec = narrative.select_mvp(scene_list, fused, config.target_duration)

    if config.save_scores:
        _save_scores(config.out_dir, config.run_tag, Path(input_path).stem, fused)

    selected_map = {c.scene_index: c.role for c in clips_spec}
    scene_scores = [
        SceneScoreResult(
            scene_index=i,
            start_s=scene.start_s,
            end_s=scene.end_s,
            visual=float(visual_scores[i]) if visual_scores is not None else None,
            audio=float(audio_scores[i]) if audio_scores is not None else None,
            motion=float(motion_scores[i]) if motion_scores is not None else None,
            quality=float(quality_gate[i]) if quality_gate is not None else None,
            transcript=float(transcript_scores[i]) if transcript_scores is not None else None,
            fused=float(fused[i]),
            selected=i in selected_map,
            narrative_role=selected_map.get(i),
        )
        for i, scene in enumerate(scene_list)
    ]

    return Analysis(
        info=info,
        scene_list=scene_list,
        clips_spec=clips_spec,
        scene_scores=scene_scores,
        transcript_segments=transcript_segments,
        stage_timings=timings,
    )


def render_one(
    input_path: str | Path,
    analysis: Analysis,
    output_path: str | Path,
    config: PipelineConfig,
    aspect: str | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> RunResult:
    if on_stage:
        on_stage("render")
    timings = dict(analysis.stage_timings)
    with _timed(timings, "render"):
        caption_segs = analysis.transcript_segments if config.add_captions else None
        output_duration = render_mod.render(
            input_path,
            analysis.clips_spec,
            output_path,
            aspect=aspect or config.aspect,
            add_effects=config.add_effects,
            title=config.title,
            cta=config.cta,
            caption_segments=caption_segs,
        )

    return RunResult(
        info=analysis.info,
        scene_scores=analysis.scene_scores,
        output_path=Path(output_path),
        output_duration_s=output_duration,
        stage_timings=timings,
    )


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    config: PipelineConfig,
    on_stage: Callable[[str], None] | None = None,
) -> RunResult:
    analysis = analyze(input_path, config, on_stage=on_stage)
    return render_one(input_path, analysis, output_path, config, on_stage=on_stage)


def run_pipeline_multi_aspect(
    input_path: str | Path,
    output_paths: dict[str, str | Path],
    config: PipelineConfig,
    on_stage: Callable[[str], None] | None = None,
) -> dict[str, RunResult]:
    """Renders the same analysis in multiple aspect ratios — one job, N outputs (M2.4)."""
    analysis = analyze(input_path, config, on_stage=on_stage)
    return {
        aspect: render_one(input_path, analysis, path, config, aspect=aspect, on_stage=on_stage)
        for aspect, path in output_paths.items()
    }


class _timed:
    def __init__(self, timings: dict[str, float], key: str):
        self.timings, self.key = timings, key

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *exc):
        self.timings[self.key] = time.perf_counter() - self.start


def _save_scores(out_dir: Path, run_tag: str, stem: str, fused: np.ndarray) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"scores_{run_tag}_{stem}.npy"
    np.save(path, fused)
    return path
