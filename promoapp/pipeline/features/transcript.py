from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

_HOOK_START = re.compile(r"^(imagine|what if|never|always|stop|listen|here's|why)\b")
_NUMBER_WORD = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|first|second|third)\b"
)

_whisper_cache: dict[str, object] = {}


def transcribe(video_path: str | Path, model_size: str = "small") -> list[dict]:
    """faster-whisper segments: [{start, end, text}, ...]. Advanced AI tier — heaviest dependency."""
    from faster_whisper import WhisperModel

    if model_size not in _whisper_cache:
        _whisper_cache[model_size] = WhisperModel(model_size, compute_type="int8")
    model = _whisper_cache[model_size]

    segments, _ = model.transcribe(str(video_path))
    return [{"start": s.start, "end": s.end, "text": s.text} for s in segments]


def hook_phrase_score(text: str) -> float:
    """Keyword heuristic: questions, numbers, and imperative openers make strong spoken hooks."""
    text = text.strip().lower()
    if not text:
        return 0.0
    score = 0.0
    if text.endswith("?"):
        score += 1.0
    if _NUMBER_WORD.search(text):
        score += 0.5
    if _HOOK_START.match(text):
        score += 1.0
    return score


def scene_transcript_scores(segments: list[dict], scenes) -> np.ndarray:
    scores = np.zeros(len(scenes), dtype=np.float32)
    for i, scene in enumerate(scenes):
        overlapping = [s for s in segments if s["start"] < scene.end_s and s["end"] > scene.start_s]
        if overlapping:
            scores[i] = max(hook_phrase_score(s["text"]) for s in overlapping)
    return scores


def best_hook_segment(segments: list[dict]) -> dict | None:
    if not segments:
        return None
    return max(segments, key=lambda s: hook_phrase_score(s["text"]))


def llm_hook_pick(segments: list[dict], api_key: str | None = None) -> dict | None:
    """Optional LLM pass to pick the single best hook line (F-15) — the only metered
    external call in the system. Falls back to the keyword heuristic whenever no API
    key is configured, or the call fails for any reason (network, quota, bad response)."""
    if not segments:
        return None

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return best_hook_segment(segments)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        lines = "\n".join(f"{i}: {s['text'].strip()}" for i, s in enumerate(segments))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    "Pick the single best hook line for a video trailer from this transcript. "
                    f"Reply with only the line number.\n\n{lines}"
                ),
            }],
            max_tokens=5,
        )
        match = re.search(r"\d+", response.choices[0].message.content or "")
        if not match:
            return best_hook_segment(segments)
        return segments[int(match.group())]
    except Exception:
        return best_hook_segment(segments)
