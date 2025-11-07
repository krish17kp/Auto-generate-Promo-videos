"""
AUTOMATED PROMO VIDEO GENERATOR (CNN version)
=============================================
Generates short promotional videos from long-form content using a lightweight CNN
(EfficientNetB0) + optional audio features. Replaces PCA/z-score scoring with
a semantic visual score from a pretrained model.

INSTALL (CPU is fine):
---------------------
pip install moviepy opencv-python librosa numpy scipy scikit-learn pillow tqdm imageio-ffmpeg tensorflow
# for scene-aware cuts:
pip install scenedetect[opencv]

Notes:
- Keep fps_sample small (2–4) to avoid slow inference.
- This uses transfer learning inference only (no training).
"""

from __future__ import annotations
import os
import cv2
import tempfile
from pathlib import Path


import numpy as np
import librosa
from tqdm import tqdm
from moviepy.editor import VideoFileClip, concatenate_videoclips
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks

# TF quiet + imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# PySceneDetect (optional)
_SCENEDETECT_OK = True
try:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector
except Exception:  # pragma: no cover
    _SCENEDETECT_OK = False




# ----------------------- small utilities (less noise) -----------------------

def set_global_seeds(seed: int | None) -> None:
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass


def log(*msg) -> None:
    print(*msg, flush=True)


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) == 0:
        return x
    window = max(1, min(window, len(x)))
    if window == 1:
        return x.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def _remove_overlaps(segs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(segs) <= 1:
        return segs
    segs.sort(key=lambda s: s[0])
    merged = [segs[0]]
    for a, b in segs[1:]:
        la, lb = merged[-1]
        if a <= lb:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def save_artifacts(
    out_dir: Path, run_tag: str, base_stem: str, scores: np.ndarray | None, segs: List[Tuple[float, float]] | None
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if scores is not None:
        np.save(out_dir / f"scores_{run_tag}_{base_stem}.npy", scores)
        log(f"💾 Saved per-sample scores to: {out_dir / f'scores_{run_tag}_{base_stem}.npy'}")
    if segs is not None:
        import csv
        path = out_dir / f"segments_{run_tag}_{base_stem}.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["start_sec", "end_sec"])
            for s, e in segs:
                w.writerow([round(float(s), 3), round(float(e), 3)])
        log(f"💾 Saved selected segments to: {path}")


# ------------------------------- CNN scorer -------------------------------

class PromoCNN:
    def __init__(self, img_size: int = 224, freeze_upto: int = -20):
        base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
        for l in base.layers[:freeze_upto]:
            l.trainable = False
        x = GlobalAveragePooling2D()(base.output)
        embed = Dense(128, activation="relu", name="embed")(x)
        out = Dense(1, activation="sigmoid", name="score")(embed)
        self.model = Model(inputs=base.input, outputs=[out, embed])
        self.img_size = img_size

    def _prep(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size))
        return (rgb.astype("float32") / 255.0)

    def scores(self, frames: List[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.array([], dtype=np.float32)
        batch = np.stack([self._prep(f) for f in frames], axis=0)
        s, _ = self.model.predict(batch, verbose=0)
        return s.reshape(-1).astype(np.float32)


# ----------------------------- core generator -----------------------------

class PromoVideoGenerator:
    """
    - Visual scoring: EfficientNetB0(sigmoid)
    - Audio: RMS + onset (optional fusion)
    - Selection: peaks on smoothed CNN scores
    - Optional scene snapping via PySceneDetect
    """

    def __init__(
        self,
        target_duration: int = 30,
        fps_sample: int = 2,
        scene_snap: bool = True,
        save_scores: bool = False,
        run_tag: str = "run",
        out_dir: str | Path = "eval_artifacts",
    ):
        self.target_duration = int(target_duration)
        self.fps_sample = max(1, int(fps_sample))
        self.scene_snap = bool(scene_snap)
        self.scaler = StandardScaler()
        self.cnn = PromoCNN(img_size=224)
        self.save_scores = bool(save_scores)
        self.run_tag = str(run_tag)
        self.out_dir = Path(out_dir)

    # ------------------------------ features ------------------------------

    def _extract_visual_scores(self, video_path: Path, duration: float) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log("⚠️  Could not open video for visual CNN analysis; using flat scores.")
            return np.ones(int(max(1, duration * self.fps_sample)), dtype=np.float32)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        sample_every = max(1, int(fps // self.fps_sample))
        total_frames = int(max(1, fps * duration))
        expected_steps = max(1, total_frames // sample_every)

        frames: List[np.ndarray] = []
        pbar = tqdm(total=expected_steps, desc="Visual (CNN)")

        idx = 0
        ok, frame = cap.read()
        while ok:
            if idx % sample_every == 0:
                frames.append(frame)
                pbar.update(1)
            idx += 1
            ok, frame = cap.read()

        cap.release()
        pbar.close()

        s = self.cnn.scores(frames)
        return s if s.size else np.ones(len(frames), dtype=np.float32)

    def _extract_audio_features(self, clip: VideoFileClip, duration: float) -> np.ndarray:
        if clip.audio is None:
            return np.zeros((int(max(1, duration * self.fps_sample)), 2), dtype=np.float32)

        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_path = Path(tmp.name)
            tmp.close()
            clip.audio.write_audiofile(
                str(wav_path), fps=22050, nbytes=2, codec="pcm_s16le", verbose=False, logger=None
            )
            y, _ = librosa.load(str(wav_path), sr=22050, mono=True)
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass
        except Exception:
            return np.zeros((int(max(1, duration * self.fps_sample)), 2), dtype=np.float32)

        frame_length = max(1, int(22050 / self.fps_sample))
        hop = frame_length
        try:
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
            onset = librosa.onset.onset_strength(y=y, sr=22050, hop_length=hop)
            m = min(len(rms), len(onset))
            if m == 0:
                return np.zeros((int(max(1, duration * self.fps_sample)), 2), dtype=np.float32)
            return np.column_stack([rms[:m], onset[:m]]).astype(np.float32)
        except Exception:
            return np.zeros((int(max(1, duration * self.fps_sample)), 2), dtype=np.float32)

    def extract_features(self, video_path: Path) -> Tuple[np.ndarray, float]:
        log("📊 Extracting features with CNN + Audio (no PCA)...")
        # Open once and reuse duration/audio
        with VideoFileClip(str(video_path)) as probe:
            duration = float(probe.duration)
            audio_feats = self._extract_audio_features(probe, duration)

        vis_scores = self._extract_visual_scores(video_path, duration)

        m = min(len(vis_scores), len(audio_feats))
        if m == 0:
            log("⚠️  Not enough features; using flat scores.")
            return np.ones((max(len(vis_scores), len(audio_feats), 1), 1), dtype=np.float32), duration

        fused = np.column_stack([vis_scores[:m], audio_feats[:m]])  # [cnn, rms, onset]
        fused_norm = self.scaler.fit_transform(fused)
        log("✅ Time steps:", fused_norm.shape[0])
        log("🧠 Features per step: cnn_score + audio(2) =", fused_norm.shape[1])
        return fused_norm.astype(np.float32), duration

    # ----------------------------- scoring + scenes -----------------------------

    def score_signal(self, feats: np.ndarray, window_s: float = 2.0) -> np.ndarray:
        if feats.shape[0] < 3:
            return np.ones(feats.shape[0], dtype=np.float32)
        base = feats[:, 0].astype(np.float32)
        window = max(3, int(self.fps_sample * window_s))
        return _smooth(base, window)

    def detect_scene_bounds(self, video_path: Path, threshold: float = 27.0) -> List[Tuple[float, float]]:
        if not self.scene_snap or not _SCENEDETECT_OK:
            return []
        try:
            vid = open_video(str(video_path))
            sm = SceneManager()
            sm.add_detector(ContentDetector(threshold=threshold))
            sm.detect_scenes(vid)
            scenes = sm.get_scene_list()
            return [(s.get_seconds(), e.get_seconds()) for (s, e) in scenes]
        except Exception:
            return []

    def _snap_to_scenes(
        self, segs: List[Tuple[float, float]], bounds: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        if not self.scene_snap or not bounds:
            return segs
        snapped: List[Tuple[float, float]] = []
        for a, b in segs:
            s_start, s_end = a, b
            for x, y in bounds:
                if x <= a <= y:  # start scene
                    s_start, s_end = x, y
                    break
            if b > s_end:  # extend into next scene end if needed
                for x, y in bounds:
                    if x < b <= y:
                        s_end = y
                        break
            if s_end - s_start >= 0.8:
                snapped.append((max(0.0, s_start), s_end))
        return _remove_overlaps(snapped) if snapped else segs

    # ------------------------------ selection ------------------------------

    def select_segments(self, scores: np.ndarray, video_duration: float) -> List[Tuple[float, float]]:
        log("✂️  Selecting optimal segments...")
        if video_duration <= self.target_duration:
            log("⚠️  Video shorter than target; using full video")
            return [(0.0, float(video_duration))]

        prom = max(0.15, float(scores.std()) * 0.5)
        dist = max(3, int(self.fps_sample * 2))
        peaks, _ = find_peaks(scores, prominence=prom, distance=dist)

        if peaks.size == 0:
            log("⚠️  No peaks found; using top values directly")
            k = max(3, int(self.target_duration / 5))
            peaks = np.argsort(scores)[-k:]
            peaks.sort()

        t_per = float(video_duration) / float(len(scores))
        peak_times = peaks.astype(np.float32) * t_per
        peak_scores = scores[peaks]

        num_clips = min(len(peaks), max(3, int(self.target_duration / 3)))
        sel_times = np.sort(peak_times[np.argsort(peak_scores)[-num_clips:]])

        segments: List[Tuple[float, float]] = []
        clip_len = min(video_duration / 3.0, (self.target_duration / max(1, len(sel_times))) * 1.2)

        for t in sel_times:
            a = max(0.0, float(t) - clip_len / 2.0)
            b = min(float(video_duration), float(t) + clip_len / 2.0)
            if b - a < 1.0:
                a = max(0.0, float(t) - 0.5)
                b = min(float(video_duration), float(t) + 0.5)
            segments.append((a, b))

        return _remove_overlaps(segments)

    # ------------------------------- assembly -------------------------------

        def create_promo(
            self,
            video_path: str | Path,
            output_path: str | Path,
            add_effects: bool = True,
            scene_threshold: float = 27.0,
        ) -> None:
            video_path = Path(video_path)
            output_path = Path(output_path)
            base_stem = video_path.stem

            log("\n🎬 Starting promo generation (CNN)…")
            log("=" * 60)

            feats, duration = self.extract_features(video_path)
            scores = self.score_signal(feats)
            if self.save_scores:
                save_artifacts(self.out_dir, self.run_tag, base_stem, scores, None)

            segs = self.select_segments(scores, duration)
            bounds = self.detect_scene_bounds(video_path, threshold=scene_threshold)
            if bounds:
                segs = self._snap_to_scenes(segs, bounds)
                log(f"🎞️  Snapped to {len(bounds)} detected scenes.")
            if self.save_scores:
                save_artifacts(self.out_dir, self.run_tag, base_stem, None, segs)

            log("\n🎞️  Assembling clips…")
            clips = []
            with VideoFileClip(str(video_path)) as vid:
                for i, (a, b) in enumerate(segs):
                    try:
                        clip = vid.subclip(a, b)
                        if add_effects and clip.duration > 1.0 and (i % 2 == 0):
                            clip = clip.speedx(1.1)
                        if clip.duration > 0.6:
                            clip = clip.crossfadein(0.3).crossfadeout(0.3)
                        clips.append(clip)
                    except Exception as e:
                        log(f"⚠️  Skipping segment {i+1}: {e}")

                if not clips:
                    raise ValueError("No valid clips could be extracted")

                log(f"🔗 Concatenating {len(clips)} clips…")
                final = concatenate_videoclips(clips, method="compose")
                if final.duration > self.target_duration:
                    final = final.subclip(0, self.target_duration)

                log("\n💾 Rendering final promo (secs):", round(final.duration, 1))
                final.write_videofile(
                    str(output_path),
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile="temp-audio.m4a",
                    remove_temp=True,
                    fps=24,
                    preset="medium",
                    threads=4,
                    verbose=False,
                    logger=None,
                )

            log("\n" + "=" * 60)
            log("✨ SUCCESS! Promo saved to:", str(output_path))
            log("=" * 60)
            try:
                size_mb = output_path.stat().st_size / (1024 * 1024.0)
            except Exception:
                size_mb = -1.0
            log("📊 Stats")
            log(" - Original duration (s):", round(duration, 1))
            log(" - Promo duration (s):   ", round(min(self.target_duration, duration), 1))
            log(" - Compression ratio:    ", round(duration / max(1e-6, min(self.target_duration, duration)), 1))
            log(" - Segments used:        ", len(segs))
            log(" - File size (MB):       ", round(size_mb, 2))
            log("=" * 60)


    # ----------------------------------- CLI -----------------------------------

    def parse_args() -> argparse.Namespace:
        p = argparse.ArgumentParser(description="Generate promotional videos using CNN-powered analysis")
        p.add_argument("--input", "-i", required=True, help="Input video file path")
        p.add_argument("--output", "-o", default="promo_output.mp4", help="Output promo file path")
        p.add_argument("--duration", "-d", type=int, default=30, help="Target duration in seconds")
        p.add_argument("--fps", type=int, default=2, help="Sampling fps for analysis")
        p.add_argument("--no-effects", action="store_true", help="Disable subtle speed/fade effects")
        p.add_argument("--no-scene-snap", action="store_true", help="Disable snapping to scene boundaries")
        p.add_argument("--scene-threshold", type=float, default=27.0, help="PySceneDetect content threshold")
        p.add_argument("--seed", type=int, default=123, help="Random seed")
        p.add_argument("--save-scores", action="store_true", help="Persist scores/segments for evaluation")
        p.add_argument("--run-tag", type=str, default="run", help="Tag for saved artifacts")
        p.add_argument("--out-dir", type=str, default="eval_artifacts", help="Directory for artifacts")
        return p.parse_args()


    def main() -> None:
        args = parse_args()
        in_path = Path(args.input)
        if not in_path.exists():
            log("❌ Error: Input file not found:", str(in_path))
            return

        set_global_seeds(args.seed)

        log("\n" + "=" * 60)
        log("🎥 AUTOMATED PROMO VIDEO GENERATOR (CNN)")
        log("=" * 60)
        log("Input:", str(in_path))
        log("Output:", args.output)
        log("Target Duration (s):", args.duration)
        log("Analysis FPS:", args.fps)
        log("Scene Snapping:", "ON" if not args.no_scene_snap else "OFF", "(PySceneDetect)" if _SCENEDETECT_OK else "(Unavailable)")
        log("Scene Threshold:", args.scene_threshold)
        log("Seed:", args.seed)
        log("Save Artifacts:", "ON" if args.save_scores else "OFF", "→", args.out_dir)
        log("=" * 60 + "\n")

        gen = PromoVideoGenerator(
            target_duration=args.duration,
            fps_sample=args.fps,
            scene_snap=(not args.no_scene_snap),
            save_scores=args.save_scores,
            run_tag=args.run_tag,
            out_dir=args.out_dir,
        )
        gen.create_promo(
            video_path=in_path,
            output_path=Path(args.output),
            add_effects=(not args.no-effects if hasattr(args, "no-effects") else not args.no_effects),
            scene_threshold=args.scene_threshold,
        )


    if __name__ == "__main__":
        main()
