from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ._frames import aggregate_per_scene, sample_frames

_DOWNSCALE_WIDTH = 480


def motion_scores(video_path: str | Path, duration: float, scenes, fps_sample: float) -> np.ndarray:
    """Mean dense optical-flow magnitude (Farneback) between consecutive sampled frames, per scene."""
    times, frames = sample_frames(video_path, duration, fps_sample)
    if len(frames) < 2:
        return np.zeros(len(scenes), dtype=np.float32)

    grays = [_downscale_gray(f) for f in frames]
    mags = [0.0]
    for i in range(1, len(grays)):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i - 1], grays[i], None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mags.append(float(magnitude.mean()))

    return aggregate_per_scene(times, np.array(mags, dtype=np.float32), scenes)


def _downscale_gray(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > _DOWNSCALE_WIDTH:
        scale = _DOWNSCALE_WIDTH / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
