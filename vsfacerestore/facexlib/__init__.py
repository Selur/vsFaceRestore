"""Vendored parts of facexlib (Xintao Wang, MIT): RetinaFace detection and ParseNet face parsing.

See README.md in this folder for what was copied and what was changed.
"""

from __future__ import annotations

import torch

from .detection.retinaface import RetinaFace
from .parsing.parsenet import ParseNet

__all__ = ["RetinaFace", "ParseNet", "load_retinaface", "load_parsenet"]


def load_retinaface(path: str, device: torch.device) -> RetinaFace:
    """RetinaFace-ResNet50 with the weights at `path` (facexlib's `detection_Resnet50_Final.pth`)."""
    model = RetinaFace(network_name="resnet50", half=False, device=device)
    state = torch.load(path, map_location="cpu")
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def load_parsenet(path: str, device: torch.device) -> ParseNet:
    """ParseNet (19 classes, 512x512) with the weights at `path` (facexlib's `parsing_parsenet.pth`)."""
    model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    return model.eval().to(device)
