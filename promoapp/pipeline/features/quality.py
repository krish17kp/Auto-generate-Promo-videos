from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ._frames import aggregate_per_scene, sample_frames

SHARPNESS_MIN = 15.0
LUMA_MIN, LUMA_MAX = 10.0, 245.0


def quality_gate(video_path: str | Path, duration: float, scenes, fps_sample: float) -> np.ndarray:
    """1.0 if a scene passes sharpness + exposure gates, else 0.0. Gates, doesn't score."""
    times, frames = sample_frames(video_path, duration, fps_sample)
    if not frames:
        return np.ones(len(scenes), dtype=np.float32)

    passes = np.array([_frame_passes(f) for f in frames], dtype=np.float32)
    gate = aggregate_per_scene(times, passes, scenes)
    return (gate >= 0.5).astype(np.float32)


def _frame_passes(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    luma = float(gray.mean())
    return float(sharpness >= SHARPNESS_MIN and LUMA_MIN <= luma <= LUMA_MAX)
