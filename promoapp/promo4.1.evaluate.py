# evaluate.py
# Usage:
#   python evaluate.py --scores eval_artifacts/scores_run_movie.npy --labels labels_movie.csv --k 10
import argparse
import  numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    idx = np.argsort(scores)[::-1][:k]
    return float(labels[idx].mean())

def bootstrap_ci(metric_fn, scores, labels, k, B=200, seed=123):
    rng = np.random.default_rng(seed)
    n = len(scores)
    vals = []
    for _ in range(B):
        s = rng.integers(0, n, size=n, endpoint=False)
        vals.append(metric_fn(scores[s], labels[s], k))
    vals = np.array(vals, dtype=float)
    return float(vals.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="Path to .npy per-sample scores")
    ap.add_argument("--labels", required=True, help="CSV with columns: idx,label (0/1)")
    ap.add_argument("--k", type=int, required=True, help="K = number of clips you output")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    scores = np.load(args.scores).astype(float)
    df = pd.read_csv(args.labels)
    labels = df["label"].to_numpy().astype(int)

    n = min(len(scores), len(labels))
    scores, labels = scores[:n], labels[:n]

    p_at_k = precision_at_k(scores, labels, args.k)
    auprc = average_precision_score(labels, scores)
    mean, lo, hi = bootstrap_ci(precision_at_k, scores, labels, args.k, seed=args.seed)

    print(f"precision@{args.k}: {p_at_k:.3f}")
    print(f"AUPRC: {auprc:.3f}")
    print(f"Bootstrap precision@{args.k}: mean={mean:.3f}  95% CI=({lo:.3f}, {hi:.3f})")

if __name__ == "__main__":
    main()

# # What I changed and why (brief, no BS)

# Streaming/batched CNN inference (_extract_visual_cnn_streaming, --batch-size): avoids loading thousands of frames into RAM; faster and safer on long videos.

# Seeds everywhere (--seed, NumPy/TF): reproducible runs for your viva.

# Scene controls (--scene-threshold): tune PySceneDetect sensitivity from CLI.

# Evaluation hooks (--save-scores, --save-segments): dump exactly what you need for precision@K later — no code surgery.

# Kept your choices: CNN + RMS/onset only, PySceneDetect optional, same selection/assembly logic.