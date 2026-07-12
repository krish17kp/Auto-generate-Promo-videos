from __future__ import annotations

import numpy as np

# Initial weights (VIDEO_PIPELINE.md §5) — tunable, renormalized over whichever signals are present.
DEFAULT_WEIGHTS = {"visual": 0.4, "audio": 0.3, "motion": 0.2, "transcript": 0.1}


def normalize(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def fuse(
    signals: dict[str, np.ndarray | None],
    weights: dict[str, float] = DEFAULT_WEIGHTS,
    quality_gate: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted fusion of normalized per-scene signals, renormalized over present signals.
    Any scene failing quality_gate is zeroed out (gates, doesn't rank)."""
    present = {k: v for k, v in signals.items() if v is not None}
    if not present:
        raise ValueError("fuse() needs at least one signal")

    n = len(next(iter(present.values())))
    total_w = sum(weights.get(k, 0.0) for k in present) or 1.0

    fused = np.zeros(n, dtype=np.float32)
    for key, values in present.items():
        w = weights.get(key, 0.0) / total_w
        fused += w * normalize(values)

    if quality_gate is not None:
        fused = fused * quality_gate

    return fused
