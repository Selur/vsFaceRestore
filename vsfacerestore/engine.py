"""The window engine: detections cached per frame, tracks inside n-K..n+K, restoration and paste for frame n."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock

import numpy as np
import torch

from .cache import OnceCache
from .facehelper import FaceHelper
from .paste import paste_face
from .restorer import Restorer, RestoreRequest
from .tracking import DetectParams, Detection, Track, TrackParams, build_track, detect_frame, link

Fetch = Callable[[int], np.ndarray]  # frame number -> BGR uint8 frame of the source clip (detection, crops)

ALPHA_MIN = 0.02  # below this a face is not restored at all


class FaceEngine:
    def __init__(
        self,
        restorer: Restorer,
        num_frames: int,
        device: torch.device,
        radius: int = 15,
        upscale: int = 1,
        tone: float = 0.5,
        half: bool = True,
        detect: DetectParams | None = None,
        track: TrackParams | None = None,
        model_dir: str | None = None,
        cache_frames: int = 256,
    ) -> None:
        self.restorer = restorer
        self.num_frames = num_frames
        self.radius = radius
        self.upscale = upscale
        self.tone = tone
        self.detect_params = detect or DetectParams()
        self.track_params = track or TrackParams()
        self.helper = FaceHelper(device, model_dir, half=half)
        self.cache: OnceCache[int, list[Detection]] = OnceCache(cache_frames)
        self.gpu = Lock()  # all GPU work of this engine runs under it; the CPU paste does not

    # ---------------------------------------------------------------- detections and tracks

    def window(self, n: int) -> tuple[int, int]:
        return max(0, n - self.radius), min(self.num_frames - 1, n + self.radius)

    def detections(self, first: int, last: int, fetch: Fetch) -> dict[int, list[Detection]]:
        def compute(i: int) -> list[Detection]:
            with torch.inference_mode():
                return detect_frame(self.helper, fetch(i), i, self.detect_params, self.gpu)

        return self.cache.get_many(range(first, last + 1), compute)

    def tracks(self, n: int, fetch: Fetch) -> list[Track]:
        """All tracks seen in the window of frame `n` (judged and weighted), whether or not they contain `n`."""
        first, last = self.window(n)
        dets = self.detections(first, last, fetch)
        raws = link(dets, first, last, self.track_params.max_gap)
        return [build_track(raw, first, last, self.num_frames, self.track_params) for raw in raws]

    # ---------------------------------------------------------------- one output frame

    def process(self, n: int, fetch: Fetch, background: np.ndarray) -> np.ndarray:
        """The output frame `n`: `background` (uint8, upscale times the source size) with the faces restored."""
        out = background.copy()
        tracks = [t for t in self.tracks(n, fetch) if t.contains(n) and t.alpha[t.index(n)] >= ALPHA_MIN]
        if not tracks:
            return out

        frames: dict[int, np.ndarray] = {}

        def frame(i: int) -> np.ndarray:
            if i not in frames:
                frames[i] = fetch(i)
            return frames[i]

        requests = []
        for track in tracks:
            crops: dict[int, np.ndarray] = {}

            def crop(i: int, track: Track = track, crops: dict[int, np.ndarray] = crops) -> np.ndarray:
                i = min(max(i, track.first), track.last)
                if i not in crops:
                    affine = self.helper.affine(track.landmarks[track.index(i)])
                    crops[i] = self.helper.align(frame(i), affine)
                return crops[i]

            requests.append(RestoreRequest(track, n, crop))

        with self.gpu, torch.inference_mode():
            restored = self.restorer.restore(requests)
            classes = self.helper.parse_batch(restored)

        for track, request, face, cls in zip(tracks, requests, restored, classes):
            i = track.index(n)
            paste_face(
                out,
                face,
                request.crop(n),
                self.helper.affine(track.landmarks[i]),
                self.upscale,
                float(track.eye_dist[i]),
                float(track.alpha[i]),
                cls,
                self.tone,
            )
        return out
