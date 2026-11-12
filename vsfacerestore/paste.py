"""Paste a restored face back into the output frame (tapewake's `assemble.py`).

Colour and brightness stay those of the source face; the model only contributes detail finer than `tone` source
pixels. The mask is the parsing of the restored face, feathered and zero towards the crop border. Only the region
the crop covers is touched.

A face that ends up smaller than its 512 px crop is first shrunk to output resolution (area resampling, which is
the anti-alias), and the tone transfer, the mask and the warp work at that size: a 40 px face costs a 40 px blur,
not a 512 px one. tapewake blurs the full crop with a Gaussian instead and shrinks it in the warp.
"""

from __future__ import annotations

import functools

import cv2
import numpy as np

from .facehelper import FACE_SIZE, PARSE_FACE_IDS, TEMPLATE_EYE_DIST

MASK_SIGMA = 16.0  # mask feathering, in crop px
MASK_BORDER = 24  # the mask is 0 this close to the crop edge, in crop px


@functools.lru_cache(maxsize=None)
def _border_mask(size: int) -> np.ndarray:
    f = size / FACE_SIZE
    edge = np.minimum(np.arange(size), np.arange(size)[::-1]).astype(np.float32)
    edge = np.clip(edge / (MASK_BORDER * f), 0, 1)
    return np.minimum(edge[:, None], edge[None, :])


def parse_mask(classes: np.ndarray, size: int = FACE_SIZE) -> np.ndarray:
    """Soft mask (size, size) float32 from ParseNet classes: face parts, blurred, fading out at the crop edge."""
    m = np.isin(classes, PARSE_FACE_IDS).astype(np.float32)
    if size != FACE_SIZE:
        m = cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA)
    m = cv2.GaussianBlur(m, (0, 0), MASK_SIGMA * size / FACE_SIZE)
    return m * _border_mask(size)


def working_size(eye_dist: float, upscale: int) -> int:
    """Side of the working image: the crop at output resolution, at most 512."""
    shrink = TEMPLATE_EYE_DIST / max(eye_dist, 1e-3) / upscale  # crop px per output px
    return FACE_SIZE if shrink <= 1 else max(8, round(FACE_SIZE / shrink))


def paste_face(
    out: np.ndarray,
    restored: np.ndarray,
    source: np.ndarray,
    affine: np.ndarray,
    upscale: int,
    eye_dist: float,
    alpha: float,
    classes: np.ndarray,
    tone: float,
) -> None:
    """Blend `restored` into `out` (uint8 levels, or float32 in 0..1; `upscale` times the source size) in place.

    `source`: the aligned crop of the original frame (tones), `affine`: source frame -> crop, `eye_dist`: eye
    distance in source px, `alpha`: weight 0..1, `classes`: ParseNet classes of `restored`, `tone`: source px.
    """
    h_up, w_up = out.shape[:2]
    scale = TEMPLATE_EYE_DIST / max(eye_dist, 1e-3)  # crop px per source px
    size = working_size(eye_dist, upscale)
    f = size / FACE_SIZE  # working px per crop px

    face = restored.astype(np.float32)
    src = source.astype(np.float32)
    if size != FACE_SIZE:
        face = cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)
        src = cv2.resize(src, (size, size), interpolation=cv2.INTER_AREA)
    sigma = float(np.clip(tone * scale, 3, 12)) * f
    face = face - cv2.GaussianBlur(face, (0, 0), sigma) + cv2.GaussianBlur(src, (0, 0), sigma)  # keep the source tones
    face = np.clip(face, 0, 255)
    mask = parse_mask(classes, size)

    # working px -> crop px -> source px -> output px (all scalings around pixel centres)
    work_to_crop = np.array([[1 / f, 0, 0.5 / f - 0.5], [0, 1 / f, 0.5 / f - 0.5], [0, 0, 1]])
    src_to_out = np.array([[upscale, 0, (upscale - 1) / 2], [0, upscale, (upscale - 1) / 2], [0, 0, 1]], np.float64)
    inv = (src_to_out @ np.vstack([cv2.invertAffineTransform(affine), [0, 0, 1]]) @ work_to_crop)[:2]
    corners = inv @ np.array([[0, 0, 1], [size - 1, 0, 1], [0, size - 1, 1], [size - 1, size - 1, 1]]).T
    x0, y0 = max(0, int(np.floor(corners[0].min())) - 2), max(0, int(np.floor(corners[1].min())) - 2)
    x1, y1 = min(w_up, int(np.ceil(corners[0].max())) + 3), min(h_up, int(np.ceil(corners[1].max())) + 3)
    if x1 <= x0 or y1 <= y0:
        return
    inv[:, 2] -= (x0, y0)
    roi_size = (x1 - x0, y1 - y0)
    warped = cv2.warpAffine(face, inv, roi_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    weight = cv2.warpAffine(mask, inv, roi_size, flags=cv2.INTER_LINEAR, borderValue=0) * alpha
    if out.dtype != np.uint8:  # float frame in 0..1: blend in place, no rounding, the background keeps its range
        roi = out[y0:y1, x0:x1]
        roi += weight[:, :, None] * (warped * (1.0 / 255.0) - roi)
        return
    roi = out[y0:y1, x0:x1].astype(np.float32)
    roi += weight[:, :, None] * (warped - roi)
    out[y0:y1, x0:x1] = np.clip(roi + 0.5, 0, 255).astype(np.uint8)
