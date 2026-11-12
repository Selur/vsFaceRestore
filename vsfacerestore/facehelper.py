"""Face detection, alignment, parsing and paste-back on single frames.

Built on facexlib's `FaceRestoreHelper` (MIT), reduced to functions over numpy images. Images are BGR uint8
arrays (h, w, 3) as OpenCV uses them; aligned faces are 512x512 crops on the FFHQ five-point template.

Detection and `paste` reproduce the per-frame behaviour of the CodeFormer route (detection on the frame scaled up
to at least `resize` on the short side, square soft mask fused with the ParseNet mask); the filter itself pastes
with the track-based `paste.paste_face`.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .facexlib import load_parsenet, load_retinaface

FACE_SIZE = 512

# Standard five landmarks (eyes, nose, mouth corners) of FFHQ faces at 512x512, from facexlib.
FACE_TEMPLATE = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.1936],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ]
)

# ParseNet classes that count as face when pasting (index = class, 255 = face, 0 = not), from facexlib:
# 0 background, 14 neck, 16 clothes, 17 hair, 18 hat are left out.
PARSE_FACE_CLASSES = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
# the same classes as a list of ids (skin, brows, eyes, glasses, ears, nose, mouth, lips, ... up to 13)
PARSE_FACE_IDS = list(range(1, 14))

TEMPLATE_EYE_DIST = float(np.linalg.norm(FACE_TEMPLATE[0] - FACE_TEMPLATE[1]))  # 125.92 crop px

# where the face is in an aligned crop (tapewake): the oval for the visibility test, the inner part for sharpness
_yy, _xx = np.mgrid[0:FACE_SIZE, 0:FACE_SIZE]
FACE_OVAL = ((_xx - 256) / 150.0) ** 2 + ((_yy - 285) / 185.0) ** 2 <= 1
INNER_FACE = ((_xx - 256) / 120.0) ** 2 + ((_yy - 290) / 140.0) ** 2 <= 1
del _yy, _xx

MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")


def eye_distance(landmarks: np.ndarray) -> float:
    return float(np.linalg.norm(landmarks[0] - landmarks[1]))


def yaw_ratio(landmarks: np.ndarray) -> float:
    """How turned the face is: nose offset from the eye midpoint over the eye distance. 0 frontal, 0.5+ profile."""
    return float(abs(landmarks[2, 0] - (landmarks[0, 0] + landmarks[1, 0]) / 2) / max(eye_distance(landmarks), 1e-6))


@functools.lru_cache(maxsize=None)
def _inner_face(size: int) -> np.ndarray:
    f = size / FACE_SIZE
    yy, xx = np.mgrid[0:size, 0:size]
    return ((xx - 256 * f) / (120.0 * f)) ** 2 + ((yy - 290 * f) / (140.0 * f)) ** 2 <= 1


def face_detail(crop: np.ndarray, eye_dist: float) -> float:
    """Sharpness of a source face: fine detail (1 to 3 source px) against the overall contrast (tapewake).

    `crop`: aligned 512 px BGR crop, `eye_dist`: eye distance in source pixels. Blurry faces score about 0.15 or
    less; those are the ones a face model starts inventing details on. Small faces are measured on the crop
    shrunk to 4 crop px per source px, which keeps the blurs small (tapewake blurs the full crop).
    """
    s = TEMPLATE_EYE_DIST / max(eye_dist, 1e-3)  # crop px per source px
    g = cv2.cvtColor(np.asarray(crop), cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = min(1.0, 4.0 / s)
    if f < 1.0:
        size = max(16, round(FACE_SIZE * f))
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
        s *= size / FACE_SIZE
    inner = _inner_face(g.shape[0])
    b1 = cv2.GaussianBlur(g, (0, 0), 0.7 * s)
    b2 = cv2.GaussianBlur(g, (0, 0), 2.0 * s)
    return float((b1 - b2)[inner].std() / (b2[inner].std() + 8))


@dataclass(frozen=True, eq=False)
class CropGeometry:
    """How a model's 512x512 crops sit on the face: a five-point template, with or without rotation."""

    template: np.ndarray  # (5, 2) landmark positions in the crop
    rotate: bool = True  # False: scale and shift only, the crop stays axis-parallel to the frame

    @property
    def eye_dist(self) -> float:
        return eye_distance(self.template)

    def affine(self, landmarks: np.ndarray) -> np.ndarray:
        """Transform (2, 3) from frame coordinates onto the crop."""
        if self.rotate:  # cv2.LMEDS for the equivalence to skimage's transform (facexlib)
            return cv2.estimateAffinePartial2D(landmarks, self.template, method=cv2.LMEDS)[0]
        lc, tc = landmarks.mean(0), self.template.mean(0)
        s = float(np.sqrt(((self.template - tc) ** 2).sum() / max(((landmarks - lc) ** 2).sum(), 1e-9)))
        return np.array([[s, 0.0, tc[0] - s * lc[0]], [0.0, s, tc[1] - s * lc[1]]])


FFHQ_CROP = CropGeometry(FACE_TEMPLATE)


def visibility(classes: np.ndarray) -> float:
    """Share of the face oval the parser sees as face parts."""
    return float(np.isin(classes[FACE_OVAL], PARSE_FACE_IDS).mean())


@dataclass
class Face:
    """One detection: five landmarks (5, 2) and the box (x0, y0, x1, y1) in frame pixels, detector score."""

    landmarks: np.ndarray
    box: np.ndarray
    score: float

    @property
    def eye_distance(self) -> float:
        return float(np.linalg.norm(self.landmarks[0] - self.landmarks[1]))


class FaceHelper:
    """Holds the RetinaFace and ParseNet networks on one device."""

    def __init__(self, device: torch.device, model_dir: str | None = None, half: bool = False) -> None:
        """`half`: run the detector and the parser in float16 (about 1.6x faster; landmarks move well under a pixel)."""
        model_dir = model_dir or MODEL_DIR
        self.device = device
        self.half = half
        self.detector = load_retinaface(os.path.join(model_dir, "detection_Resnet50_Final.pth"), device)
        self.parser = load_parsenet(os.path.join(model_dir, "parsing_parsenet.pth"), device)

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.float16, enabled=self.half)

    # ---------------------------------------------------------------- detection

    @torch.inference_mode()
    def detect(
        self,
        img: np.ndarray,
        resize: int | None = 640,
        conf_threshold: float = 0.8,
        eye_dist_threshold: float | None = 5.0,
        scale: float | None = None,
    ) -> list[Face]:
        """Faces in `img`, sorted by score (highest first).

        `resize`: the frame is scaled up (never down) so that its short side is at least this many pixels before
        detection; None detects on the frame as it is. `scale` instead gives the factor directly (tapewake: 2, for
        small faces). Faces with eyes closer than `eye_dist_threshold` pixels (side faces, tiny faces) are dropped.
        """
        if scale is None:
            if resize is None:
                scale = 1.0
            else:
                h, w = img.shape[:2]
                scale = max(1.0, resize / min(h, w))
        if scale != 1.0:
            h, w = int(img.shape[0] * scale), int(img.shape[1] * scale)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        with self._autocast():
            rows = self.detector.detect_faces(img, conf_threshold=conf_threshold)
        if rows is None or rows.shape[0] == 0:
            return []
        rows = rows.copy()
        rows[:, 0:4] /= scale  # box and landmarks back to frame pixels; the score (column 4) stays
        rows[:, 5:15] /= scale
        faces = []
        for row in rows:
            landmarks = np.array([[row[i], row[i + 1]] for i in range(5, 15, 2)])
            face = Face(landmarks, row[0:4], float(row[4]))
            if eye_dist_threshold is not None and face.eye_distance < eye_dist_threshold:
                continue
            faces.append(face)
        return faces

    # ---------------------------------------------------------------- alignment

    @staticmethod
    def affine(landmarks: np.ndarray) -> np.ndarray:
        """Similarity transform (2, 3) from frame coordinates onto the 512x512 FFHQ template."""
        return FFHQ_CROP.affine(landmarks)

    @staticmethod
    def align(img: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """The aligned 512x512 crop of `img`; outside the frame a neutral gray."""
        return cv2.warpAffine(
            img, affine, (FACE_SIZE, FACE_SIZE), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132)
        )

    # ---------------------------------------------------------------- parsing

    PARSE_BATCH = 8  # ParseNet costs the same per face up to here; larger batches run out of fast memory

    @torch.inference_mode()
    def parse_batch(self, faces: list[np.ndarray]) -> np.ndarray:
        """ParseNet class maps (n, 512, 512) int64 of aligned faces, in batches of at most PARSE_BATCH."""
        if not faces:
            return np.zeros((0, FACE_SIZE, FACE_SIZE), np.int64)
        results = []
        for start in range(0, len(faces), self.PARSE_BATCH):
            chunk = faces[start : start + self.PARSE_BATCH]
            x = np.stack([face[:, :, ::-1].transpose(2, 0, 1) for face in chunk]).astype(np.float32) / 255.0  # RGB, CHW
            x = (torch.from_numpy(np.ascontiguousarray(x)) - 0.5) / 0.5
            with self._autocast():
                out = self.parser(x.to(self.device))[0]
            results.append(out.argmax(dim=1).cpu().numpy())
        return np.concatenate(results)

    def parse(self, face: np.ndarray) -> np.ndarray:
        """ParseNet class map (512, 512) int64 of an aligned face."""
        return self.parse_batch([face])[0]

    def parse_mask(self, face: np.ndarray) -> np.ndarray:
        """Soft face mask (512, 512) float64 in [0, 1] from the parsing, as facexlib's paste uses it."""
        classes = self.parse(face)
        mask = np.zeros(classes.shape)
        for idx, value in enumerate(PARSE_FACE_CLASSES):
            mask[classes == idx] = value
        mask = cv2.GaussianBlur(mask, (101, 101), 11)
        mask = cv2.GaussianBlur(mask, (101, 101), 11)
        border = 10
        mask[:border, :] = 0
        mask[-border:, :] = 0
        mask[:, :border] = 0
        mask[:, -border:] = 0
        return mask / 255.0

    # ---------------------------------------------------------------- paste

    def paste(
        self,
        img: np.ndarray,
        faces: list[tuple[np.ndarray, np.ndarray]],
        upscale: int = 1,
        background: np.ndarray | None = None,
    ) -> np.ndarray:
        """Paste restored faces back into the frame.

        `faces`: (restored 512x512 face, affine from `affine()`) per face. `background`: the frame already at
        `upscale` times the size; without it the frame is scaled bilinearly. Returns BGR uint8.
        """
        h, w = img.shape[:2]
        h_up, w_up = h * upscale, w * upscale
        if background is None:
            out = cv2.resize(img, (w_up, h_up), interpolation=cv2.INTER_LINEAR) if upscale != 1 else img
        else:
            out = background
        for restored, affine in faces:
            inverse = cv2.invertAffineTransform(affine) * upscale
            if upscale > 1:
                inverse[:, 2] += 0.5 * upscale  # half a source pixel, for a more precise back alignment
            inv_restored = cv2.warpAffine(restored, inverse, (w_up, h_up))

            # square mask of the crop, eroded and feathered in proportion to the face area
            square = cv2.warpAffine(np.ones((FACE_SIZE, FACE_SIZE), np.float32), inverse, (w_up, h_up))
            square = cv2.erode(square, np.ones((2 * upscale, 2 * upscale), np.uint8))
            pasted = square[:, :, None] * inv_restored
            edge = int(np.sum(square) ** 0.5) // 20
            square = cv2.erode(square, np.ones((edge * 2, edge * 2), np.uint8))
            square = cv2.GaussianBlur(square, (edge * 2 + 1, edge * 2 + 1), 0)

            # parsing mask of the restored face, warped back; the smaller of the two masks counts
            parse = cv2.warpAffine(self.parse_mask(restored), inverse, (w_up, h_up), flags=cv2.INTER_AREA)
            mask = np.minimum(parse, square)[:, :, None]
            out = mask * pasted + (1 - mask) * out
        return out.astype(np.uint8)
