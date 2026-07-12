from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ._frames import aggregate_per_scene, sample_frames

# Zero-shot promo-relevance prompts (tunable config, not code — VIDEO_PIPELINE.md §4).
POSITIVE_PROMPTS = [
    "an exciting action moment",
    "a person speaking directly to camera",
    "a product close-up",
    "a dramatic wide shot",
]
NEGATIVE_PROMPTS = [
    "a blank screen",
    "a blurry transition frame",
]

_clip_cache: dict[str, tuple] = {}


def frame_diff_scores(video_path: str | Path, duration: float, scenes, fps_sample: float) -> np.ndarray:
    """MVP visual proxy: mean absolute frame-to-frame difference per scene. No training data needed."""
    times, frames = sample_frames(video_path, duration, fps_sample)
    if len(frames) < 2:
        return np.zeros(len(scenes), dtype=np.float32)

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    diffs = [0.0] + [
        float(cv2.absdiff(grays[i - 1], grays[i]).mean()) for i in range(1, len(grays))
    ]
    return aggregate_per_scene(times, np.array(diffs, dtype=np.float32), scenes)


def _load_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "openai"):
    key = f"{model_name}:{pretrained}"
    if key not in _clip_cache:
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        _clip_cache[key] = (model, preprocess, tokenizer, torch)
    return _clip_cache[key]


def clip_scores(
    video_path: str | Path, duration: float, scenes, fps_sample: float, max_frames: int = 300
) -> tuple[np.ndarray, np.ndarray]:
    """CLIP zero-shot semantic score per scene + mean CLIP embedding per scene (for diversity)."""
    times, frames = sample_frames(video_path, duration, fps_sample, max_frames=max_frames)
    n_scenes = len(scenes)
    embed_dim = 512
    if not frames:
        return np.zeros(n_scenes, dtype=np.float32), np.zeros((n_scenes, embed_dim), dtype=np.float32)

    model, preprocess, tokenizer, torch = _load_clip()
    embed_dim = model.visual.output_dim

    from PIL import Image

    with torch.no_grad():
        text_tokens = tokenizer(POSITIVE_PROMPTS + NEGATIVE_PROMPTS)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        images = torch.stack(
            [preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frames]
        )
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = image_features @ text_features.T
        n_pos = len(POSITIVE_PROMPTS)
        frame_scores = similarity[:, :n_pos].mean(dim=1) - similarity[:, n_pos:].mean(dim=1)

    frame_scores_np = frame_scores.numpy().astype(np.float32)
    embeddings_np = image_features.numpy().astype(np.float32)

    scene_scores = aggregate_per_scene(times, frame_scores_np, scenes)
    scene_embeddings = np.zeros((n_scenes, embed_dim), dtype=np.float32)
    for i, scene in enumerate(scenes):
        mask = (times >= scene.start_s) & (times < scene.end_s)
        if mask.any():
            scene_embeddings[i] = embeddings_np[mask].mean(axis=0)

    return scene_scores, scene_embeddings
