"""The interface between the window engine and a face model.

The engine hands a restorer one request per face of the output frame. A request knows its track and can give the
aligned crop of that face at any frame of the track (edges repeated), so a per-frame model takes one crop, DicFace
a window of five, KEEP a block. Requests of one frame come together so a model can batch them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .tracking import Track


@dataclass
class RestoreRequest:
    track: Track
    frame: int  # the output frame
    crop: Callable[[int], np.ndarray]  # aligned 512x512 BGR uint8 crop of this face at a frame (clamped to the track)


class Restorer(Protocol):
    def restore(self, requests: list[RestoreRequest]) -> list[np.ndarray]:
        """Restored 512x512 BGR uint8 crops, one per request, for `request.frame`."""
        ...


class BatchRestorer:
    """A per-frame model: `fn` maps a list of crops to a list of restored crops (one network call)."""

    def __init__(self, fn: Callable[[list[np.ndarray]], list[np.ndarray]]) -> None:
        self.fn = fn

    def restore(self, requests: list[RestoreRequest]) -> list[np.ndarray]:
        return self.fn([r.crop(r.frame) for r in requests])


class IdentityRestorer:
    """Returns the crops unchanged (tests of tracking and paste without a model)."""

    def restore(self, requests: list[RestoreRequest]) -> list[np.ndarray]:
        return [r.crop(r.frame) for r in requests]


def crops_to_tensor(crops: list[np.ndarray]):
    """BGR uint8 crops -> (n, 3, 512, 512) float32 RGB in [-1, 1] (basicsr's img2tensor + normalize)."""
    import torch

    x = np.stack([c[:, :, ::-1].transpose(2, 0, 1) for c in crops]).astype(np.float32) / 255.0
    return (torch.from_numpy(np.ascontiguousarray(x)) - 0.5) / 0.5


def tensor_to_crops(y) -> list[np.ndarray]:
    """(n, 3, 512, 512) in [-1, 1] RGB -> BGR uint8 crops (basicsr's tensor2img rounding)."""
    import torch

    y = ((y.detach().float().clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8).cpu().numpy()
    return [np.ascontiguousarray(c.transpose(1, 2, 0)[:, :, ::-1]) for c in y]
