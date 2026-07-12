from __future__ import annotations

import tempfile
from pathlib import Path

import librosa
import numpy as np

from ..scenes import Scene


def scene_audio_scores(video_path: str | Path, scenes: list[Scene], has_audio: bool) -> np.ndarray:
    """Mean RMS + onset strength per scene. Ported from promo4.1.py; ≥0, unnormalized."""
    if not has_audio or not scenes:
        return np.zeros(len(scenes), dtype=np.float32)

    from moviepy import VideoFileClip

    with VideoFileClip(str(video_path)) as clip:
        if clip.audio is None:
            return np.zeros(len(scenes), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "audio.wav"
            clip.audio.write_audiofile(str(wav_path), fps=22050, nbytes=2, codec="pcm_s16le", logger=None)
            y, sr = librosa.load(str(wav_path), sr=22050, mono=True)

    if y.size == 0:
        return np.zeros(len(scenes), dtype=np.float32)

    rms = librosa.feature.rms(y=y)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    n = min(len(rms), len(onset))
    if n == 0:
        return np.zeros(len(scenes), dtype=np.float32)

    frame_times = librosa.frames_to_time(np.arange(n), sr=sr)
    combined = rms[:n] + onset[:n]

    scores = np.zeros(len(scenes), dtype=np.float32)
    for i, scene in enumerate(scenes):
        mask = (frame_times >= scene.start_s) & (frame_times < scene.end_s)
        scores[i] = float(combined[mask].mean()) if mask.any() else 0.0
    return scores
