from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def sample_frames(
    video_path: str | Path, duration: float, fps_sample: float, max_frames: int = 300
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Uniformly sample BGR frames across the video, capped at max_frames. Returns (times_s, frames)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.array([], dtype=np.float32), []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(fps * duration)
    step = max(1, int(round(fps / max(fps_sample, 1e-6))))
    indices = list(range(0, max(total, 1), step))
    if len(indices) > max_frames:
        keep = np.linspace(0, len(indices) - 1, max_frames).round().astype(int)
        indices = [indices[i] for i in keep]

    times: list[float] = []
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(frame)
        times.append(idx / fps)

    cap.release()
    return np.array(times, dtype=np.float32), frames


def aggregate_per_scene(frame_times: np.ndarray, values: np.ndarray, scenes) -> np.ndarray:
    """Mean of per-frame values falling inside each scene's [start, end)."""
    out = np.zeros(len(scenes), dtype=np.float32)
    for i, scene in enumerate(scenes):
        mask = (frame_times >= scene.start_s) & (frame_times < scene.end_s)
        out[i] = float(values[mask].mean()) if mask.any() else 0.0
    return out
