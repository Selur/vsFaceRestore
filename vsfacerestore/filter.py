"""The VapourSynth side: one `std.ModifyFrame` around the window engine."""

from __future__ import annotations

import torch
import vapoursynth as vs

from .engine import FaceEngine
from .restorer import Restorer
from .tracking import DetectParams, TrackParams
from .vsutil import FORMATS, bgr_to_frame, frame_to_bgr, frame_to_native


def validate(clip: vs.VideoNode, bg_clip: vs.VideoNode | None, upscale: int, radius: int, name: str = "restore") -> None:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error(f"{name}: clip is not a clip")
    if clip.format.id not in FORMATS:
        raise vs.Error(f"{name}: only RGB24, RGBS and RGBH are supported")
    if upscale < 1:
        raise vs.Error(f"{name}: upscale must be at least 1")
    if radius < 14:
        raise vs.Error(f"{name}: radius must be at least 14 (fades, alpha smoothing and the minimum track length need it)")
    if bg_clip is not None:
        if not isinstance(bg_clip, vs.VideoNode) or bg_clip.format.id != clip.format.id:
            raise vs.Error(f"{name}: bg_clip must have the format of clip ({clip.format.name})")
        if bg_clip.width != clip.width * upscale or bg_clip.height != clip.height * upscale:
            raise vs.Error(f"{name}: bg_clip must be upscale times the size of clip")
        if bg_clip.num_frames != clip.num_frames:
            raise vs.Error(f"{name}: bg_clip must have the same number of frames as clip")


def build(
    clip: vs.VideoNode,
    restorer: Restorer,
    bg_clip: vs.VideoNode | None = None,
    upscale: int = 1,
    radius: int = 15,
    tone: float = 0.5,
    device_index: int = 0,
    half: bool = True,
    detect: DetectParams | None = None,
    track: TrackParams | None = None,
    model_dir: str | None = None,
) -> tuple[vs.VideoNode, FaceEngine]:
    """The filtered clip and its engine (for statistics and tests)."""
    validate(clip, bg_clip, upscale, radius)
    if not torch.cuda.is_available():
        raise vs.Error("restore: CUDA is not available")
    device = torch.device("cuda", device_index)
    engine = FaceEngine(restorer, clip.num_frames, device, radius, upscale, tone, half, detect, track, model_dir)

    if bg_clip is None:
        bg_clip = clip if upscale == 1 else clip.resize.Bilinear(clip.width * upscale, clip.height * upscale)
    out = clip.std.BlankClip(width=clip.width * upscale, height=clip.height * upscale, keep=True)
    out = out.std.CopyFrameProps(clip)

    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        img = frame_to_bgr(f[0])

        def fetch(i: int):
            return img if i == n else frame_to_bgr(clip.get_frame(i))

        result = engine.process(n, fetch, frame_to_native(f[2]))
        return bgr_to_frame(result, f[1].copy())

    return out.std.ModifyFrame([clip, out, bg_clip], inference), engine


def restore_with(clip: vs.VideoNode, restorer: Restorer, **kwargs) -> vs.VideoNode:
    """`build()` without the engine."""
    return build(clip, restorer, **kwargs)[0]
