"""Frames <-> numpy and small VapourSynth helpers.

Two views of a frame: `frame_to_bgr()` gives uint8 levels for detection, alignment and the networks whatever the
clip format; `frame_to_native()` gives the frame as it is (uint8 for RGB24, float32 in 0..1 for RGBS and RGBH), the
array the paste blends into, so untouched pixels of a float frame come back unchanged.
"""

from __future__ import annotations

import numpy as np
import vapoursynth as vs

FORMATS = (vs.RGB24, vs.RGBS, vs.RGBH)


def is_float(frame_or_clip: vs.VideoFrame | vs.VideoNode) -> bool:
    return frame_or_clip.format.sample_type == vs.FLOAT


def frame_to_bgr(frame: vs.VideoFrame) -> np.ndarray:
    """Frame -> (h, w, 3) uint8 BGR array (OpenCV order); float frames are rounded to 8-bit levels."""
    planes = [np.asarray(frame[plane]) for plane in (2, 1, 0)]
    if is_float(frame):
        return np.stack([np.clip(p.astype(np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8) for p in planes], axis=2)
    return np.stack(planes, axis=2)


def frame_to_native(frame: vs.VideoFrame) -> np.ndarray:
    """Frame -> (h, w, 3) BGR array in the frame's own scale: uint8 for RGB24, float32 (0..1, unclipped) for float formats."""
    planes = [np.asarray(frame[plane]) for plane in (2, 1, 0)]
    if is_float(frame):
        return np.stack([p.astype(np.float32) for p in planes], axis=2)
    return np.stack(planes, axis=2)


def bgr_to_frame(img: np.ndarray, frame: vs.VideoFrame) -> vs.VideoFrame:
    """Write a (h, w, 3) BGR array in the frame's own scale into a writable frame; returns the frame."""
    for plane in range(3):
        target = np.asarray(frame[plane])
        np.copyto(target, img[:, :, 2 - plane].astype(target.dtype, copy=False))
    return frame


def pad_to_minimum(clip: vs.VideoNode, minimum: int) -> tuple[vs.VideoNode, int, int]:
    """Extend the clip at the right and bottom by repeating the edge until both sides are at least `minimum`.

    Returns the clip and the padding added (width, height); crop it off again after processing.
    """
    pad_w = max(0, minimum - clip.width)
    pad_h = max(0, minimum - clip.height)
    if pad_w or pad_h:
        w, h = clip.width + pad_w, clip.height + pad_h
        clip = clip.resize.Point(w, h, src_width=w, src_height=h)
    return clip, pad_w, pad_h
