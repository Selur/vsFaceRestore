"""DicFace: CodeFormer with temporal attention over groups of five aligned frames.

Architecture files from fudan-generative-vision/DicFace (S-Lab License 1.0, see LICENSE and README.md here).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from ..facehelper import MODEL_DIR
from ..restorer import RestoreRequest, crops_to_tensor, tensor_to_crops  # noqa: F401  (re-exported)
from .arch import TemporalCodeFormerDirDistMultiScale

FRAME_LENGTH = 5  # frames per temporal group; the network splits its batch into groups of this length
WEIGHTS = os.path.join(MODEL_DIR, "dicface", "bfr_100k.pth")

# offsets of the five frames fed per output frame, and which of them is taken
WINDOWS = {
    "centre": (range(-2, 3), 2),  # tapewake: n-2 ... n+2, the middle one
    "script": (range(0, 5), 0),  # the original script: 20 frames n-10 ... n+9, index 10 = first frame of group 3
}


def load_dicface(path: str | None = None, device: torch.device | None = None) -> TemporalCodeFormerDirDistMultiScale:
    net = TemporalCodeFormerDirDistMultiScale(
        dim_embed=512,
        n_head=8,
        n_layers=9,
        codebook_size=1024,
        connect_list=["32", "64", "128", "256"],
        frame_length=FRAME_LENGTH,
    )
    state = torch.load(path or WEIGHTS, map_location="cpu")["params_ema"]
    net.load_state_dict(state)
    net.eval()
    return net.to(device) if device is not None else net


class DicFaceRestorer:
    """Restores each requested face from the five crops around its frame, several faces per network call.

    `window`: 'centre' or 'script' (see WINDOWS). `sample`: 'mean' takes the expectation of the Dirichlet code
    distribution (deterministic); 'random' draws from it like the original, seeded per frame so that a frame gives
    the same result whenever it is requested. `batch`: faces per call (5 frames each); 1 is as fast as 2 on the
    RTX 4080 and needs half the memory, 4 spills and is three times slower.
    """

    def __init__(
        self,
        device: torch.device,
        weight: float = 1.0,
        window: str = "centre",
        sample: str = "mean",
        batch: int = 1,
        seed: int = 0,
        model_path: str | None = None,
    ) -> None:
        if window not in WINDOWS:
            raise ValueError(f"window must be one of {tuple(WINDOWS)}")
        if sample not in ("mean", "random"):
            raise ValueError("sample must be 'mean' or 'random'")
        self.device = device
        self.weight = weight
        self.window = window
        self.sample = sample
        self.batch = max(1, batch)
        self.seed = seed
        self.net = load_dicface(model_path, device)

    @torch.inference_mode()
    def restore(self, requests: list[RestoreRequest]) -> list[np.ndarray]:
        offsets, take = WINDOWS[self.window]
        results: list[np.ndarray] = []
        for start in range(0, len(requests), self.batch):
            chunk = requests[start : start + self.batch]
            crops = [r.crop(r.frame + o) for r in chunk for o in offsets]
            x = crops_to_tensor(crops).to(self.device)
            if self.sample == "random":
                with torch.random.fork_rng(devices=[self.device]):
                    torch.manual_seed(self.seed + chunk[0].frame)
                    y = self.net(x, w=self.weight, sample="random")[0]
            else:
                y = self.net(x, w=self.weight, sample="mean")[0]
            results.extend(tensor_to_crops(y[take::FRAME_LENGTH]))
        return results
