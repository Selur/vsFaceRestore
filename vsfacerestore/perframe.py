"""Per-frame face restoration inside the VapourSynth graph (reference harness for the helper).

Every frame on its own: detect, align, restore each face with a caller-supplied function, paste back. No
tracking, no smoothing, no temporal model: this exists to check the helper and the paste against the CodeFormer
route and stays the fallback structure for the temporal filter.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock

import numpy as np
import torch
import vapoursynth as vs

from .facehelper import FACE_SIZE, Face, FaceHelper
from .vsutil import bgr_to_frame, frame_to_bgr, pad_to_minimum

RestoreFace = Callable[[np.ndarray], np.ndarray]  # aligned 512x512 BGR uint8 -> restored 512x512 BGR uint8
FaceFilter = Callable[[Face], bool]  # True keeps the detection


def per_frame(
    clip: vs.VideoNode,
    restore_face: RestoreFace,
    upscale: int = 1,
    bg_clip: vs.VideoNode | None = None,
    device_index: int = 0,
    model_dir: str | None = None,
    detect_resize: int | None = 640,
    conf_threshold: float = 0.8,
    eye_dist_threshold: float | None = 5.0,
    face_filter: FaceFilter | None = None,
) -> vs.VideoNode:
    """Restore every detected face of every frame with `restore_face`.

    :param clip:           RGB24 clip.
    :param restore_face:   function from an aligned 512x512 BGR uint8 crop to the restored crop.
    :param upscale:        output size factor; faces are pasted at that scale.
    :param bg_clip:        background already at the output size (RGB24, same frame count); bilinear otherwise.
    :param device_index:   CUDA device.
    :param model_dir:      folder with the facexlib weights; the package's `models` folder by default.
    :param detect_resize:  short side the frame is scaled up to (never down) before detection; None: as it is.
    :param conf_threshold: detector score below which a face is ignored.
    :param eye_dist_threshold: eye distance in frame pixels below which a face is ignored.
    :param face_filter:    extra test per detection; faces it returns False for are left untouched.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("vsfacerestore: clip is not a clip")
    if clip.format.id != vs.RGB24:
        raise vs.Error("vsfacerestore: only RGB24 is supported")
    if not torch.cuda.is_available():
        raise vs.Error("vsfacerestore: CUDA is not available")
    if upscale < 1:
        raise vs.Error("vsfacerestore: upscale must be at least 1")
    if bg_clip is not None:
        if not isinstance(bg_clip, vs.VideoNode) or bg_clip.format.id != vs.RGB24:
            raise vs.Error("vsfacerestore: bg_clip must be an RGB24 clip")
        if bg_clip.width != clip.width * upscale or bg_clip.height != clip.height * upscale:
            raise vs.Error("vsfacerestore: bg_clip must be upscale times the size of clip")
        if bg_clip.num_frames != clip.num_frames:
            raise vs.Error("vsfacerestore: bg_clip must have the same number of frames as clip")

    device = torch.device("cuda", device_index)
    helper = FaceHelper(device, model_dir)
    lock = Lock()  # one frame at a time on the GPU

    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        with lock, torch.inference_mode():
            img = frame_to_bgr(f[0])
            background = frame_to_bgr(f[2]) if bg_clip is not None else None
            faces = helper.detect(img, detect_resize, conf_threshold, eye_dist_threshold)
            if face_filter is not None:
                faces = [face for face in faces if face_filter(face)]
            restored = []
            for face in faces:
                affine = helper.affine(face.landmarks)
                restored.append((restore_face(helper.align(img, affine)), affine))
            out = helper.paste(img, restored, upscale, background)
            return bgr_to_frame(out, f[1].copy())

    # the detector and the parser want at least 512 pixels on each side
    clip, pad_w, pad_h = pad_to_minimum(clip, FACE_SIZE)
    if bg_clip is not None:
        bg_clip, _, _ = pad_to_minimum(bg_clip, FACE_SIZE * upscale)

    out = clip.std.BlankClip(width=clip.width * upscale, height=clip.height * upscale, keep=True)
    out = out.std.CopyFrameProps(clip)
    sources = [clip, out] if bg_clip is None else [clip, out, bg_clip]
    out = out.std.ModifyFrame(sources, inference)
    if pad_w or pad_h:
        out = out.std.Crop(right=pad_w * upscale, bottom=pad_h * upscale)
    return out
