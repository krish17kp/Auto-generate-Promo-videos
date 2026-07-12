from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scene:
    index: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def detect_scenes(
    path: str | Path,
    duration: float,
    threshold: float = 27.0,
    min_scenes: int = 3,
    fallback_window_s: float = 5.0,
) -> list[Scene]:
    """PySceneDetect content-aware boundaries; falls back to fixed windows for static footage."""
    spans = _detect_content_scenes(path, threshold)
    if len(spans) < min_scenes:
        spans = _fixed_windows(duration, fallback_window_s)
    return [Scene(index=i, start_s=s, end_s=e) for i, (s, e) in enumerate(spans)]


def _detect_content_scenes(path: str | Path, threshold: float) -> list[tuple[float, float]]:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector

        video = open_video(str(path))
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=threshold))
        manager.detect_scenes(video)
        return [(s.get_seconds(), e.get_seconds()) for s, e in manager.get_scene_list()]
    except Exception:
        return []


def _fixed_windows(duration: float, window_s: float) -> list[tuple[float, float]]:
    n = max(1, int(duration // window_s))
    return [(i * window_s, min(duration, (i + 1) * window_s)) for i in range(n)]
