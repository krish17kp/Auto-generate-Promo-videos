from __future__ import annotations

from pathlib import Path

from moviepy import CompositeVideoClip, TextClip, VideoFileClip, concatenate_videoclips, vfx

from .narrative import SelectedClip

ASPECT_RATIOS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0}
CROSSFADE_S = 0.3
TITLE_DURATION_S = 3.0
CTA_DURATION_S = 3.0


def render(
    video_path: str | Path,
    clips_spec: list[SelectedClip],
    output_path: str | Path,
    aspect: str = "16:9",
    add_effects: bool = True,
    title: str | None = None,
    cta: str | None = None,
    caption_segments: list[dict] | None = None,
) -> float:
    """Assembles selected clips into the final promo. Returns rendered duration in seconds."""
    if not clips_spec:
        raise ValueError("no clips to render")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with VideoFileClip(str(video_path)) as source:
        segments = []
        for i, spec in enumerate(clips_spec):
            seg = source.subclipped(spec.start_s, spec.end_s)
            if add_effects and spec.role == "build" and i % 2 == 0 and seg.duration > 1.0:
                seg = seg.with_effects([vfx.MultiplySpeed(1.1)])
            if seg.duration > 0.6:
                seg = seg.with_effects([vfx.CrossFadeIn(CROSSFADE_S)])
            segments.append(seg)

        final = concatenate_videoclips(segments, method="compose")
        final = _crop_to_aspect(final, aspect)

        overlays = [final]
        if title:
            overlays.append(_text_overlay(title, final, TITLE_DURATION_S, 0.0, "top"))
        if cta:
            start = max(0.0, final.duration - CTA_DURATION_S)
            overlays.append(_text_overlay(cta, final, final.duration - start, start, "bottom"))
        if caption_segments:
            overlays += _caption_overlays(caption_segments, clips_spec, final)

        composed = CompositeVideoClip(overlays) if len(overlays) > 1 else final

        composed.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            fps=24,
            preset="medium",
            threads=4,
            logger=None,
            ffmpeg_params=["-movflags", "+faststart"],
            # explicit path: moviepy's default temp-audio name lands in the CWD, not
            # next to the output — pinning it here keeps a killed/crashed render's
            # leftover temp file next to output_path instead of littering the CWD.
            temp_audiofile=str(output_path.with_suffix(".temp_audio.m4a")),
        )
        return float(composed.duration)


def _crop_to_aspect(clip, aspect: str):
    target_ratio = ASPECT_RATIOS[aspect]
    w, h = clip.w, clip.h
    current_ratio = w / h
    if abs(current_ratio - target_ratio) < 1e-3:
        return clip
    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x1 = (w - new_w) // 2
        return clip.with_effects([vfx.Crop(x1=x1, width=new_w, y1=0, height=h)])
    new_h = int(w / target_ratio)
    y1 = (h - new_h) // 2
    return clip.with_effects([vfx.Crop(x1=0, width=w, y1=y1, height=new_h)])


def _text_overlay(text: str, base_clip, duration: float, start: float, vpos: str):
    txt = TextClip(
        text=text,
        font_size=max(20, int(base_clip.h * 0.055)),
        color="white",
        stroke_color="black",
        stroke_width=2,
        size=(int(base_clip.w * 0.9), None),
        method="caption",
        text_align="center",
    )
    return txt.with_duration(duration).with_start(start).with_position(("center", vpos))


def _caption_overlays(caption_segments: list[dict], clips_spec: list[SelectedClip], final):
    """Burn transcript segments as styled subtitles, remapped onto the assembled promo timeline."""
    overlays = []
    cursor = 0.0
    for spec in clips_spec:
        clip_len = spec.end_s - spec.start_s
        for seg in caption_segments:
            if seg["end"] <= spec.start_s or seg["start"] >= spec.end_s:
                continue
            rel_start = max(0.0, seg["start"] - spec.start_s)
            rel_end = min(clip_len, seg["end"] - spec.start_s)
            if rel_end <= rel_start:
                continue
            txt = TextClip(
                text=seg["text"].strip(),
                font_size=max(18, int(final.h * 0.045)),
                color="white",
                stroke_color="black",
                stroke_width=2,
                size=(int(final.w * 0.85), None),
                method="caption",
                text_align="center",
            )
            overlays.append(
                txt.with_duration(rel_end - rel_start)
                .with_start(cursor + rel_start)
                .with_position(("center", "bottom"))
            )
        cursor += clip_len
    return overlays
