"""vsfacerestore: temporal video face restoration for VapourSynth with DicFace, KEEP and PGTFormer."""

from __future__ import annotations

import os

import vapoursynth as vs

from .tracking import DetectParams, TrackParams

__version__ = "0.0.1"

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

MODELS = ("dicface", "keep", "pgtformer")


def restore(
    clip: vs.VideoNode,
    model: str = "dicface",
    bg_clip: vs.VideoNode | None = None,
    upscale: int = 1,
    radius: int | None = None,
    tone: float = 0.5,
    weight: float = 1.0,
    device_index: int = 0,
    half: bool = True,
    window: str = "centre",
    sample: str = "mean",
    batch: int = 1,
    crop: str = "vfhq",
    sharpness: float | None = None,
    min_eye_dist: float | None = None,
    min_visibility: float | None = None,
    fade: int | None = None,
    detect: DetectParams | None = None,
    track: TrackParams | None = None,
    model_dir: str | None = None,
) -> vs.VideoNode:
    """Restore the faces of a progressive RGB24, RGBS or RGBH clip with a temporal face model.

    :param clip:            progressive RGB24, RGBS or RGBH clip (deinterlace first). The faces are restored at
                            8 bit whatever the format; a float clip keeps its precision everywhere else.
    :param model:           'dicface', 'keep' or 'pgtformer'.
    :param bg_clip:         background already at `upscale` times the size of `clip`; bilinear otherwise.
    :param upscale:         output size factor.
    :param radius:          frames before and after the current one that tracking and smoothing may look at;
                            default 15 for DicFace, 24 for KEEP (its 20-frame blocks must fit the window).
    :param tone:            structure coarser than this many source pixels comes from the source face.
    :param weight:          fidelity weight of the model, 0 = codebook only, 1 = closest to the input (DicFace
                            and PGTFormer; 1.0 as in the original scripts).
    :param device_index:    CUDA device.
    :param half:            float16 for face detection and parsing (the restoration model stays float32).
    :param window:          DicFace: 'centre' (frames n-2 ... n+2) or 'script' (n ... n+4, as the original script).
    :param sample:          DicFace: 'mean' (deterministic) or 'random' (the original's Dirichlet sampling, seeded).
    :param batch:           DicFace and PGTFormer: faces per network call.
    :param crop:            PGTFormer: 'vfhq' (face at the scale of its training crops, steadiest) or 'ffhq' (the
                            aligned crops of DicFace and KEEP, more detail on small faces).
    :param sharpness:       faces whose sharpness score stays below this are left untouched, and the blend weight
                            ramps up from this value to twice it (default 0.04; tapewake's "do not invent detail
                            on blurry faces" setting is 0.14; a soft DV face scores about 0.08-0.10, VHS faces
                            with real detail 0.2-0.6).
    :param min_eye_dist:    faces with eyes closer than this many source pixels are left untouched (default 5);
                            the blend weight ramps up between 6 and 14 px.
    :param min_visibility:  share of the face oval the parser must see as face (default 0.45).
    :param fade:            frames over which a face fades in and out at the ends of its track (default 8).
    :param detect:          detection parameters (DetectParams).
    :param track:           tracking, filter and alpha parameters (TrackParams); the four keywords above
                            override its fields.
    :param model_dir:       folder with the weights (`detection_Resnet50_Final.pth`, `parsing_parsenet.pth`,
                            `dicface/bfr_100k.pth`, `keep/KEEP-b76feb75.pth`, `pgtformer/model.safetensors`);
                            default `vsfacerestore/models`
                            inside the package.
    """
    import dataclasses

    import torch

    from .facehelper import MODEL_DIR
    from .filter import build, validate

    if model not in MODELS:
        raise vs.Error(f"restore: model must be one of {MODELS}")
    if model == "keep":
        from .keep import radius_needed

        if radius is None:
            radius = radius_needed()
        elif radius < radius_needed():
            raise vs.Error(f"restore: KEEP needs radius >= {radius_needed()}")
    elif radius is None:
        radius = 15
    validate(clip, bg_clip, upscale, radius)
    if not 0.0 <= weight <= 1.0:
        raise vs.Error("restore: weight must be between 0.0 and 1.0")
    if crop not in ("vfhq", "ffhq"):
        raise vs.Error("restore: crop must be 'vfhq' or 'ffhq'")

    overrides = {}
    if sharpness is not None:
        if sharpness < 0:
            raise vs.Error("restore: sharpness must be 0 or more")
        overrides.update(min_detail=sharpness, detail_lo=sharpness, detail_hi=max(2 * sharpness, 1e-3))
    if min_eye_dist is not None:
        if min_eye_dist < 0:
            raise vs.Error("restore: min_eye_dist must be 0 or more")
        overrides.update(min_eye_dist=min_eye_dist)
    if min_visibility is not None:
        if not 0.0 <= min_visibility <= 1.0:
            raise vs.Error("restore: min_visibility must be between 0.0 and 1.0")
        overrides.update(min_visibility=min_visibility)
    if fade is not None:
        if not 0 <= fade <= radius - 7:
            raise vs.Error(f"restore: fade must be between 0 and radius - 7 ({radius - 7})")
        overrides.update(fade=fade)
    if overrides:
        track = dataclasses.replace(track or TrackParams(), **overrides)
    if not torch.cuda.is_available():
        raise vs.Error("restore: CUDA is not available")
    device = torch.device("cuda", device_index)
    model_dir = model_dir or MODEL_DIR
    weights = {
        "dicface": os.path.join("dicface", "bfr_100k.pth"),
        "keep": os.path.join("keep", "KEEP-b76feb75.pth"),
        "pgtformer": os.path.join("pgtformer", "model.safetensors"),
    }
    for name in ("detection_Resnet50_Final.pth", "parsing_parsenet.pth", weights[model]):
        if not os.path.isfile(os.path.join(model_dir, name)):
            raise vs.Error(f"restore: weight file {name!r} not found in model_dir {model_dir!r}")
    if model == "dicface":
        from .dicface import DicFaceRestorer

        restorer = DicFaceRestorer(device, weight, window, sample, batch, model_path=os.path.join(model_dir, weights[model]))
    elif model == "pgtformer":
        from .pgtformer import PGTFormerRestorer

        restorer = PGTFormerRestorer(device, weight, batch, crop, model_path=os.path.join(model_dir, weights[model]))
    else:
        from .keep import KEEPRestorer

        restorer = KEEPRestorer(device, model_path=os.path.join(model_dir, weights[model]))
    return build(clip, restorer, bg_clip, upscale, radius, tone, device_index, half, detect, track, model_dir)[0]
