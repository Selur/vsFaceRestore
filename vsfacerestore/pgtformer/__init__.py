"""PGTFormer: parsing-guided temporal transformer over windows of three face crops.

Architecture files from kepengxu/PGTFormer (non-commercial licence, see LICENSE and README.md here).
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from ..facehelper import FACE_TEMPLATE, FFHQ_CROP, MODEL_DIR, CropGeometry
from ..restorer import RestoreRequest
from .arch import PGTFormer

FRAME_LENGTH = 3  # frames per temporal group; the network splits its batch into groups of this length
WEIGHTS = os.path.join(MODEL_DIR, "pgtformer", "model.safetensors")
OFFSETS = (-1, 0, 1)  # frames fed per output frame; the middle one is taken, as in the original script

# VFHQ-style crop fitted on the original demo video: eyes 0.79 times the FFHQ distance apart, eye centre at (256, 224)
_EYES = FACE_TEMPLATE[:2].mean(0)
VFHQ_TEMPLATE = (FACE_TEMPLATE - _EYES) * 0.79 + np.array([256.0, 224.0])
CROPS = {
    "vfhq": CropGeometry(VFHQ_TEMPLATE),  # rotated, at the scale of the training crops: steadiest, no stripe artefacts
    "ffhq": FFHQ_CROP,  # the crops of DicFace and KEEP: more detail on small faces, fine stripes on clean large ones
}

# network_g of the released model (config.json of huggingface.co/kepeng/pgtformer-base)
CONFIG = dict(
    adain=True,
    bottleneck_type="rq",
    checkpointing=False,
    code_shape=[32, 32, 1],
    connect_list=["32", "64", "128", "256"],
    ddconfig=dict(
        attn_resolutions=[32, 64, 128],
        ch=64,
        ch_mult=[1, 2, 4, 4, 8],
        depths=[2, 2, 2, 2, 2],
        double_z=False,
        dropout=0.0,
        in_channels=3,
        num_frames=3,
        num_head=8,
        num_heads=[8, 8, 8, 8, 8],
        num_res_blocks=1,
        out_ch=3,
        resolution=512,
        stages_atten=4,
        window_size=[5, 5, 5],
        window_sizes=[[4, 4], [4, 4], [4, 4], [4, 4], [4, 4]],
        z_channels=256,
    ),
    decay=0.99,
    detach_16=True,
    dim_embd=512,
    droprate=0.0,
    embed_dim=512,
    fix_modules=["quantizer", "decoder", "conditionnet"],
    latent_loss_weight=0.25,
    latent_shape=[32, 32, 512],
    loss_type="mse",
    n_embed=1024,
    n_head=8,
    n_layers=9,
    restart_unused_codes=True,
    shared_codebook=True,
    tf=3,
    w=1,
)

_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
           "U8": torch.uint8, "BOOL": torch.bool}


def read_safetensors(path: str) -> dict[str, torch.Tensor]:
    """Tensors of a .safetensors file (8-byte header length, JSON header, raw little-endian data), on the CPU."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        data = bytearray(f.read())
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        t = torch.frombuffer(data, dtype=_DTYPES[info["dtype"]], count=(end - start) // _DTYPES[info["dtype"]].itemsize,
                             offset=start) if end > start else torch.empty(0, dtype=_DTYPES[info["dtype"]])
        out[name] = t.reshape(info["shape"]).clone()
    return out


def load_pgtformer(path: str | None = None, device: torch.device | None = None) -> PGTFormer:
    net = PGTFormer(**CONFIG)
    net.load_state_dict(read_safetensors(path or WEIGHTS))
    net.eval()
    return net.to(device) if device is not None else net


def crops_to_unit(crops: list[np.ndarray]) -> torch.Tensor:
    """BGR uint8 crops -> (n, 3, 512, 512) float32 RGB in [0, 1] (the network normalises internally)."""
    x = np.stack([c[:, :, ::-1].transpose(2, 0, 1) for c in crops]).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(x))


def unit_to_crops(y: torch.Tensor) -> list[np.ndarray]:
    """(n, 3, 512, 512) RGB in [0, 1] -> BGR uint8 crops."""
    y = (y.detach().float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    return [np.ascontiguousarray(c.transpose(1, 2, 0)[:, :, ::-1]) for c in y]


class PGTFormerRestorer:
    """Restores each requested face from the crops at n-1, n, n+1 of its track, several faces per network call.

    `weight`: fidelity weight of the feature fusion (1 = closest to the input, as released; 0 = codebook only).
    `crop`: one of CROPS, how the crops sit on the face.
    """

    def __init__(
        self, device: torch.device, weight: float = 1.0, batch: int = 1, crop: str = "vfhq", model_path: str | None = None
    ) -> None:
        if crop not in CROPS:
            raise ValueError(f"crop must be one of {tuple(CROPS)}")
        self.geometry = CROPS[crop]
        self.device = device
        self.weight = weight
        self.batch = max(1, batch)
        self.net = load_pgtformer(model_path, device)

    @torch.inference_mode()
    def restore(self, requests: list[RestoreRequest]) -> list[np.ndarray]:
        results: list[np.ndarray] = []
        for start in range(0, len(requests), self.batch):
            chunk = requests[start : start + self.batch]
            crops = [r.crop(r.frame + o) for r in chunk for o in OFFSETS]
            y = self.net(crops_to_unit(crops).to(self.device), w=self.weight)[0]
            results.extend(unit_to_crops(y[1::FRAME_LENGTH]))
        return results
