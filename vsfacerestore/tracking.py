"""Detections per frame, tracks inside a window, landmark smoothing, track filter and alpha.

The rules are tapewake's (`faces_find.py`, `assemble.py`), applied to the window n-K..n+K instead of the whole
clip. Everything here is pure numpy over cached detections, so the result for a frame depends only on the window
content and not on the order frames are requested in.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter

from .facehelper import FaceHelper, eye_distance, face_detail, visibility, yaw_ratio


@dataclass
class Detection:
    """One face in one frame."""

    landmarks: np.ndarray  # (5, 2) float32, frame pixels
    box: np.ndarray  # (4,) x0, y0, x1, y1
    score: float
    visibility: float | None = None  # share of the face oval the parser sees as face (measured frames only)
    detail: float | None = None  # sharpness score (measured frames only)

    @property
    def size(self) -> float:
        return float(max(self.box[2] - self.box[0], self.box[3] - self.box[1]))


@dataclass
class DetectParams:
    scale: float = 2.0  # frame upscale before detection (small faces)
    threshold: float = 0.6  # detector score
    min_eye_dist: float = 3.0  # px in the source frame
    measure_every: int = 3  # visibility and detail are measured on every n-th frame (by frame number)
    measure_min_eye_dist: float = 4.0  # smaller detections are not measured (their tracks are too small anyway)


@dataclass
class TrackParams:
    max_gap: int = 8  # frames a face may go undetected inside a track
    min_length: int = 15  # frames a track must span (inside the window)
    smooth: int = 11  # Savitzky-Golay window for the landmarks
    # hard filter (medians over the window): below these the track is left untouched
    min_score: float = 0.85
    min_eye_dist: float = 5.0
    min_visibility: float = 0.45
    # sharpness: tapewake uses 0.14 here and a ramp 0.15..0.23, which leaves large soft faces on tape untouched
    # (a 720x576 DV face scores 0.08..0.10); Selur chose to restore those, so the cut sits well below them
    min_detail: float = 0.04
    # alpha ramps per frame
    eye_dist_lo: float = 6.0
    eye_dist_hi: float = 14.0
    yaw_lo: float = 0.25
    yaw_hi: float = 0.5
    visibility_lo: float = 0.35
    visibility_hi: float = 0.6
    detail_lo: float = 0.04
    detail_hi: float = 0.08
    detail_smooth: int = 31  # moving mean over the detail score (about half a second)
    fade: int = 8  # frames to fade in / out at the ends of a track
    alpha_smooth: int = 9  # moving mean over alpha


def detect_frame(
    helper: FaceHelper,
    img: np.ndarray,
    n: int,
    params: DetectParams,
    gpu: AbstractContextManager | None = None,
) -> list[Detection]:
    """Detect the faces of frame `n`; on measured frames also parse the crops for visibility and sharpness.

    `gpu`: a lock held around the GPU calls only (the crops and the sharpness score are CPU work).
    """
    gpu = gpu if gpu is not None else nullcontext()
    with gpu:
        faces = helper.detect(img, conf_threshold=params.threshold, eye_dist_threshold=params.min_eye_dist, scale=params.scale)
    dets = [Detection(face.landmarks, face.box, face.score) for face in faces]
    if n % params.measure_every == 0:
        measured = [d for d in dets if eye_distance(d.landmarks) >= params.measure_min_eye_dist]
        if measured:
            crops = [helper.align(img, helper.affine(d.landmarks)) for d in measured]
            with gpu:
                classes = helper.parse_batch(crops)
            for det, crop, cls in zip(measured, crops, classes):
                det.visibility = visibility(cls)
                det.detail = face_detail(crop, eye_distance(det.landmarks))
    return dets


@dataclass
class RawTrack:
    frames: list[int] = field(default_factory=list)
    dets: list[Detection] = field(default_factory=list)


def link(dets_by_frame: dict[int, list[Detection]], first: int, last: int, max_gap: int) -> list[RawTrack]:
    """Link detections of frames first..last into tracks by landmark distance and size ratio (tapewake)."""
    tracks: list[RawTrack] = []
    for f in range(first, last + 1):
        dets = dets_by_frame.get(f, [])
        active = [t for t in tracks if f - t.frames[-1] <= max_gap]
        used_d: set[int] = set()
        if active and dets:
            # all track/detection pairs at once: distance of the landmark centres against the larger size
            t_centre = np.array([t.dets[-1].landmarks.mean(0) for t in active])  # (T, 2)
            t_size = np.array([t.dets[-1].size for t in active])  # (T,)
            gap = np.array([f - t.frames[-1] for t in active])
            d_centre = np.array([d.landmarks.mean(0) for d in dets])  # (D, 2)
            d_size = np.array([d.size for d in dets])
            dist = np.linalg.norm(t_centre[:, None, :] - d_centre[None, :, :], axis=2)  # (T, D)
            larger = np.maximum(t_size[:, None], d_size[None, :])
            ratio = d_size[None, :] / t_size[:, None]
            okay = (dist < (0.5 + 0.15 * (gap[:, None] - 1)) * larger + 3) & (ratio > 0.6) & (ratio < 1.6)
            cost = np.where(okay, dist / larger, np.inf)
            used_t: set[int] = set()
            for flat in np.argsort(cost, axis=None, kind="stable"):
                ti, di = divmod(int(flat), len(dets))
                if not okay[ti, di]:
                    break
                if ti in used_t or di in used_d:
                    continue
                used_t.add(ti)
                used_d.add(di)
                active[ti].frames.append(f)
                active[ti].dets.append(dets[di])
        for di, det in enumerate(dets):
            if di not in used_d:
                tracks.append(RawTrack([f], [det]))
    return tracks


def ramp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Smoothstep from 0 at `lo` to 1 at `hi`."""
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def _fill(values: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN gaps; all-NaN stays NaN."""
    ok = ~np.isnan(values)
    if not ok.any() or ok.all():
        return values
    idx = np.arange(len(values))
    return np.interp(idx, idx[ok], values[ok])


def _moving_mean(values: np.ndarray, width: int) -> np.ndarray:
    n = len(values)
    k = min(width, n - (1 - n % 2))  # odd, at most n
    if k < 3:
        return values
    return np.convolve(np.pad(values, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


@dataclass
class Track:
    """A face over the frames first..last of a window, with per-frame values over that span."""

    first: int
    last: int
    landmarks: np.ndarray  # (len, 5, 2) smoothed
    eye_dist: np.ndarray
    yaw: np.ndarray
    visibility: np.ndarray  # NaN when no frame was measured
    detail: np.ndarray
    score: np.ndarray  # NaN in gaps
    alpha: np.ndarray  # restoration weight per frame, 0 everywhere when `reason` is set
    reason: str | None  # why the track is left untouched
    median: dict[str, float]
    raw: dict[int, np.ndarray] = field(default_factory=dict)  # frame -> unsmoothed landmarks of the detections

    def __len__(self) -> int:
        return self.last - self.first + 1

    def contains(self, frame: int) -> bool:
        return self.first <= frame <= self.last

    def index(self, frame: int) -> int:
        return min(max(frame, self.first), self.last) - self.first


def build_track(raw: RawTrack, first: int, last: int, num_frames: int, p: TrackParams) -> Track:
    """Interpolate gaps, smooth, judge and weight a raw track seen in the window first..last."""
    f0, f1 = raw.frames[0], raw.frames[-1]
    n = f1 - f0 + 1
    lms = np.full((n, 10), np.nan)
    score = np.full(n, np.nan)
    vis = np.full(n, np.nan)
    detail = np.full(n, np.nan)
    for f, det in zip(raw.frames, raw.dets):
        i = f - f0
        lms[i] = det.landmarks.reshape(10)
        score[i] = det.score
        if det.visibility is not None:
            vis[i] = det.visibility
            detail[i] = det.detail
    for c in range(10):
        lms[:, c] = _fill(lms[:, c])
    win = min(p.smooth, n - (1 - n % 2))
    if win >= 5:
        lms = savgol_filter(lms, win, 2, axis=0)
    lms = lms.reshape(n, 5, 2)
    vis = _fill(vis)
    detail = _moving_mean(_fill(detail), p.detail_smooth)
    eye_dist = np.array([eye_distance(m) for m in lms])
    yaw = np.array([yaw_ratio(m) for m in lms])

    median = {
        "score": float(np.nanmedian(score)),
        "eye_dist": float(np.median(eye_dist)),
        "yaw": float(np.median(yaw)),
        "visibility": float(np.nanmedian(vis)) if not np.all(np.isnan(vis)) else float("nan"),
        "detail": float(np.nanmedian(detail)) if not np.all(np.isnan(detail)) else float("nan"),
    }
    if n < p.min_length:
        reason = "too short"
    elif median["score"] < p.min_score:
        reason = "low confidence"
    elif median["eye_dist"] < p.min_eye_dist:
        reason = "too small"
    elif not median["visibility"] >= p.min_visibility:  # also NaN (nothing measured)
        reason = "not a visible face"
    elif not median["detail"] >= p.min_detail:
        reason = "out of focus"
    else:
        reason = None

    if reason is None:
        alpha = (
            ramp(eye_dist, p.eye_dist_lo, p.eye_dist_hi)
            * (1 - ramp(yaw, p.yaw_lo, p.yaw_hi))
            * ramp(vis, p.visibility_lo, p.visibility_hi)
            * ramp(detail, p.detail_lo, p.detail_hi)
        )
        # a track end is only known when the window shows more than max_gap frames beyond it (or the clip ends)
        k = np.arange(n)
        fade = np.ones(n)
        if f0 > 0 and (first == 0 or f0 - first > p.max_gap):
            fade = np.minimum(fade, (k + 1) / (p.fade + 1))
        if f1 < num_frames - 1 and (last == num_frames - 1 or last - f1 > p.max_gap):
            fade = np.minimum(fade, (n - k) / (p.fade + 1))
        alpha = _moving_mean(alpha * fade, p.alpha_smooth)
    else:
        alpha = np.zeros(n)
    raw_landmarks = {f: det.landmarks for f, det in zip(raw.frames, raw.dets)}
    return Track(f0, f1, lms, eye_dist, yaw, vis, detail, score, alpha, reason, median, raw_landmarks)
