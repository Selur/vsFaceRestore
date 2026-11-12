"""KEEP: recurrent face restoration with a Kalman filter over the code features and optical flow (GMFlow).

Architecture files from jnjaby/KEEP (S-Lab License 1.0, see LICENSE and README.md here), the attention blocks
from diffusers 0.11.0 (Apache 2.0) and GMFlow (Apache 2.0).

The state runs from frame to frame, so a face is restored in blocks of `BLOCK` frames on a fixed grid by frame
number (block k covers frames k*STEP ... k*STEP+BLOCK-1); neighbouring blocks overlap by BLOCK-STEP frames and are
crossfaded there, as tapewake does. A block is computed by the first frame that needs it and cached per (block,
face), so reading in order computes every block once and a jump recomputes one or two.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

from ..cache import OnceCache
from ..facehelper import FACE_SIZE, MODEL_DIR
from ..restorer import RestoreRequest, crops_to_tensor, tensor_to_crops
from ..tracking import Track
from .arch import KEEP

BLOCK = 20  # frames per network call (the original script's --max_length)
STEP = 14  # grid step; BLOCK - STEP = 6 frames overlap, crossfaded
WEIGHTS = os.path.join(MODEL_DIR, "keep", "KEEP-b76feb75.pth")

CONFIG = dict(
    img_size=512,
    emb_dim=256,
    dim_embd=512,
    n_head=8,
    n_layers=9,
    codebook_size=1024,
    cft_list=["16", "32", "64"],
    kalman_attn_head_dim=48,
    num_uncertainty_layers=3,
    cfa_list=["16", "32"],
    cfa_nhead=4,
    cfa_dim=256,
    cond=1,
)


def radius_needed(block: int = BLOCK) -> int:
    """Window radius so that every frame of a block, plus the 5 frames the landmark smoothing needs on each side,
    lies inside the window of any frame of that block: block + 4."""
    return block + 4


def load_keep(path: str | None = None, device: torch.device | None = None) -> KEEP:
    net = KEEP(**CONFIG)
    state = torch.load(path or WEIGHTS, map_location="cpu", weights_only=True)["params_ema"]
    net.load_state_dict(state)  # includes the (fine-tuned) GMFlow weights under flownet.*
    net.eval()
    return net.to(device) if device is not None else net


def block_key(track: Track, k: int, block: int, step: int) -> tuple:
    """Identity of a face for block `k`: the unsmoothed landmarks of its first detection inside the block.

    Raw detections are cached per frame, so the same face gives the same key from whichever frame's window the
    block is computed.
    """
    first, last = k * step, k * step + block - 1
    frames = sorted(f for f in track.raw if first <= f <= last) or sorted(track.raw, key=lambda f: abs(f - first))
    f = frames[0]
    return (k, f, tuple(np.round(track.raw[f].ravel() * 4).astype(int).tolist()))


class KEEPRestorer:
    def __init__(
        self,
        device: torch.device,
        block: int = BLOCK,
        step: int = STEP,
        cache_mb: int = 1024,
        model_path: str | None = None,
    ) -> None:
        if not 1 <= step <= block:
            raise ValueError("step must be between 1 and block")
        self.device = device
        self.block = block
        self.step = step
        self.overlap = block - step
        self.net = load_keep(model_path, device)
        per_block = block * FACE_SIZE * FACE_SIZE * 3
        self.cache: OnceCache[tuple, list[np.ndarray]] = OnceCache(max(2, cache_mb * 2**20 // per_block))

    def blocks_of(self, n: int) -> list[int]:
        """Grid blocks that contain frame n (one, or two in an overlap)."""
        k_lo = max(0, math.ceil((n - self.block + 1) / self.step))
        return list(range(k_lo, n // self.step + 1))

    @torch.inference_mode()
    def compute_block(self, request: RestoreRequest, k: int) -> list[np.ndarray]:
        crops = [request.crop(k * self.step + j) for j in range(self.block)]  # clamped to the track by crop()
        x = crops_to_tensor(crops).unsqueeze(0).to(self.device)  # 1 x T x 3 x 512 x 512
        y = self.net(x, need_upscale=False)[0]
        return tensor_to_crops(y)

    def weight(self, k: int, j: int) -> float:
        """Crossfade weight of block k at its frame j: the first `overlap` frames of every block but the first ramp up."""
        if k > 0 and j < self.overlap:
            return (j + 1) / (self.overlap + 1)
        return 1.0

    def restore(self, requests: list[RestoreRequest]) -> list[np.ndarray]:
        results = []
        for request in requests:
            keys = {k: block_key(request.track, k, self.block, self.step) for k in self.blocks_of(request.frame)}
            blocks = self.cache.get_many(keys.values(), lambda key, r=request: self.compute_block(r, key[0]))
            acc = np.zeros((FACE_SIZE, FACE_SIZE, 3), np.float32)
            total = 0.0
            for k, key in keys.items():
                j = request.frame - k * self.step
                w = self.weight(k, j)
                acc += w * blocks[key][j].astype(np.float32)
                total += w
            results.append(np.clip(acc / total + 0.5, 0, 255).astype(np.uint8))
        return results
