from __future__ import annotations

import threading
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from .pipeline.ingest import IngestError
from .pipeline.run import PipelineConfig, run_pipeline_multi_aspect

_STAGE_PROGRESS = {
    "ingest": 5,
    "scenes": 20,
    "features": 50,
    "scoring": 65,
    "narrative": 75,
    "render": 90,
}


def start_job(job_id) -> None:
    """Entry point used by the view. Spawns a background thread to run the pipeline."""
    threading.Thread(target=run_job, args=(job_id,), daemon=True).start()


def run_job(job_id) -> None:
    from .models import PromoJob  # local import: keeps pipeline/ Django-free, jobs.py is the only bridge

    job = PromoJob.objects.select_related("upload").get(id=job_id)
    job.status = "processing"
    job.started_at = timezone.now()
    job.save(update_fields=["status", "started_at"])

    def on_stage(stage_name: str) -> None:
        job.stage = stage_name
        job.progress = _STAGE_PROGRESS.get(stage_name, job.progress)
        job.save(update_fields=["stage", "progress"])

    try:
        config = PipelineConfig(**job.params.get("config", {}))
        aspects = job.params.get("aspects") or [config.aspect]
        input_path = job.upload.file.path
        output_dir = Path(settings.MEDIA_ROOT) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = {a: output_dir / f"{job.id}_{a.replace(':', 'x')}.mp4" for a in aspects}

        results = run_pipeline_multi_aspect(input_path, output_paths, config, on_stage=on_stage)

        _persist_results(job, results)

        job.status = "done"
        job.progress = 100
        job.finished_at = timezone.now()
        job.save(update_fields=["status", "progress", "finished_at"])
    except IngestError as exc:
        _fail(job, str(exc))
    except Exception as exc:
        _fail(job, f"{job.stage or 'pipeline'} stage failed: {exc}")


def _fail(job, message: str) -> None:
    job.status = "failed"
    job.error_message = message
    job.finished_at = timezone.now()
    job.save(update_fields=["status", "error_message", "finished_at"])


def _persist_results(job, results: dict) -> None:
    from .models import PromoOutput, Scene, SegmentScore

    first = next(iter(results.values()))

    upload = job.upload
    upload.duration_s = first.info.duration
    upload.fps = first.info.fps
    upload.width = first.info.width
    upload.height = first.info.height
    upload.has_audio = first.info.has_audio
    upload.save(update_fields=["duration_s", "fps", "width", "height", "has_audio"])

    # Scene/SegmentScore are aspect-independent (same analysis feeds every render) — persist once.
    for s in first.scene_scores:
        scene = Scene.objects.create(job=job, index=s.scene_index, start_s=s.start_s, end_s=s.end_s)
        SegmentScore.objects.create(
            scene=scene,
            visual=s.visual,
            audio=s.audio,
            motion=s.motion,
            quality=s.quality,
            transcript=s.transcript,
            fused=s.fused,
            selected=s.selected,
            narrative_role=s.narrative_role,
        )

    for aspect, result in results.items():
        output_path = Path(result.output_path)
        output = PromoOutput(
            job=job,
            aspect=aspect,
            duration_s=result.output_duration_s,
            size_bytes=output_path.stat().st_size,
        )
        # render() already wrote the file under MEDIA_ROOT/output/ — point the FieldFile at it
        # instead of re-reading and re-saving the bytes through Django's storage layer.
        output.file.name = f"output/{output_path.name}"
        output.save()
