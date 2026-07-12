from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_CLIP_S = 1.5
MAX_CLIP_S = 6.0
HOOK_CLIP_S = 2.5
DIVERSITY_THRESHOLD = 0.15


@dataclass(frozen=True)
class SelectedClip:
    scene_index: int
    role: str  # hook | build | climax | outro
    start_s: float
    end_s: float


def _clip_window(scene, max_len: float) -> tuple[float, float]:
    length = min(scene.duration_s, max_len)
    return scene.start_s, scene.start_s + max(length, min(MIN_CLIP_S, scene.duration_s))


def select_mvp(scenes, fused: np.ndarray, target_duration: float) -> list[SelectedClip]:
    """MVP fallback: top-3 scenes by score, chronological order, hard-trimmed to target duration."""
    if not scenes:
        return []
    top_n = min(3, len(scenes))
    top_idx = sorted(np.argsort(fused)[::-1][:top_n].tolist())

    clips = [
        SelectedClip(i, "build", *_clip_window(scenes[i], MAX_CLIP_S))
        for i in top_idx
    ]
    return _trim_to_duration(clips, target_duration)


def select_narrative(
    scenes,
    fused: np.ndarray,
    embeddings: np.ndarray | None,
    audio: np.ndarray | None,
    motion: np.ndarray | None,
    target_duration: float,
) -> list[SelectedClip]:
    """Trailer grammar: hook -> build (chronological, diverse) -> climax -> outro."""
    if not scenes:
        return []
    if len(scenes) == 1:
        return [SelectedClip(0, "hook", *_clip_window(scenes[0], target_duration))]

    order = np.argsort(fused)[::-1].tolist()
    used: set[int] = set()

    hook_idx = order[0]
    used.add(hook_idx)

    climax_idx = next((i for i in order if i not in used), hook_idx)
    used.add(climax_idx)

    build_idx = _pick_build_scenes(scenes, fused, embeddings, used, max_clips=4)
    used.update(build_idx)

    outro_idx = _pick_outro_scene(scenes, motion, used)
    used.add(outro_idx)

    clips = [SelectedClip(hook_idx, "hook", *_clip_window(scenes[hook_idx], HOOK_CLIP_S))]
    clips += [SelectedClip(i, "build", *_clip_window(scenes[i], MAX_CLIP_S)) for i in build_idx]
    clips.append(SelectedClip(climax_idx, "climax", *_clip_window(scenes[climax_idx], MAX_CLIP_S)))
    clips.append(SelectedClip(outro_idx, "outro", *_clip_window(scenes[outro_idx], MAX_CLIP_S)))

    return _trim_to_duration(clips, target_duration, protected_roles={"hook", "climax", "outro"})


def _pick_build_scenes(scenes, fused, embeddings, used: set[int], max_clips: int) -> list[int]:
    candidates = [i for i in range(len(scenes)) if i not in used]
    candidates.sort(key=lambda i: fused[i], reverse=True)

    picked: list[int] = []
    for i in candidates:
        if len(picked) >= max_clips:
            break
        if embeddings is not None and not _is_diverse(embeddings[i], picked, embeddings):
            continue
        picked.append(i)

    return sorted(picked, key=lambda i: scenes[i].start_s)


def _is_diverse(candidate_embedding, picked: list[int], embeddings) -> bool:
    for idx in picked:
        if _cosine_distance(candidate_embedding, embeddings[idx]) < DIVERSITY_THRESHOLD:
            return False
    return True


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def _pick_outro_scene(scenes, motion: np.ndarray | None, used: set[int]) -> int:
    n = len(scenes)
    last_quarter_start = int(n * 0.75)
    candidates = [i for i in range(last_quarter_start, n) if i not in used] or [
        i for i in range(n) if i not in used
    ]
    if not candidates:
        return n - 1
    if motion is not None:
        return min(candidates, key=lambda i: motion[i])
    return max(candidates)


def _trim_to_duration(
    clips: list[SelectedClip], target_duration: float, protected_roles: set[str] | None = None
) -> list[SelectedClip]:
    protected_roles = protected_roles or set()
    total = sum(c.end_s - c.start_s for c in clips)
    if total <= target_duration:
        return clips

    trimmable = [c for c in clips if c.role not in protected_roles] or list(clips)
    while total > target_duration and len(clips) > 1 and trimmable:
        drop = trimmable.pop()
        clips = [c for c in clips if c is not drop]
        total = sum(c.end_s - c.start_s for c in clips)

    if total > target_duration and clips:
        overflow = total - target_duration
        last = clips[-1]
        new_end = max(last.start_s + MIN_CLIP_S, last.end_s - overflow)
        clips[-1] = SelectedClip(last.scene_index, last.role, last.start_s, new_end)

    return clips
