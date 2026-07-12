from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from moviepy import VideoFileClip

ALLOWED_EXTENSIONS = {".mp4", ".mov"}


class IngestError(Exception):
    """Raised for any input the pipeline can't safely process. Message is user-facing."""


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    fps: float
    width: int
    height: int
    has_audio: bool


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise IngestError(f"unsupported file type: {path.suffix or '(none)'}")
    try:
        with VideoFileClip(str(path)) as clip:
            if clip.w <= 0 or clip.h <= 0 or clip.duration <= 0:
                raise IngestError("no readable video stream")
            return VideoInfo(
                duration=float(clip.duration),
                fps=float(clip.fps or 0.0),
                width=int(clip.w),
                height=int(clip.h),
                has_audio=clip.audio is not None,
            )
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError("couldn't read this video") from exc


def validate(info: VideoInfo, file_size_bytes: int, max_mb: float, max_minutes: float) -> None:
    max_bytes = max_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise IngestError(f"file too large: {file_size_bytes / 1e6:.0f}MB (max {max_mb:.0f}MB)")
    if info.duration > max_minutes * 60:
        raise IngestError(f"video too long: {info.duration / 60:.1f} min (max {max_minutes:.0f} min)")
