"""PGTFormer acceptance: extracted network, crop mode, temporal stability and speed against DicFace.

Checks:
  network    our extracted network against the original repository's on the frames of its demo video
  crops      crop geometries (vfhq, ffhq, axis-parallel box) and DicFace on clips with head tilts: frame-to-frame
             difference inside the face oval, drift of the coarse tones from the source, detail
  speed      fps and VRAM through restore() on a one-face clip, PGTFormer against DicFace

Run with the VapourSynth addon's Python, nothing else on the GPU:

    F:\\Vapoursynth\\python.exe test\\stage5_pgtformer.py [network] [crops] [speed] [--save]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "test"))  # the embedded Python does not add the script folder
PGT_REPO = r"F:\workspace\PGTFormer"
WORK = r"J:\tmp\claude\vsfacerestore\stage5"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import vapoursynth as vs  # noqa: E402

import vsfacerestore  # noqa: E402
from vsfacerestore.dicface import DicFaceRestorer  # noqa: E402
from vsfacerestore.engine import FaceEngine  # noqa: E402
from vsfacerestore.facehelper import FACE_OVAL, FACE_SIZE, FFHQ_CROP, CropGeometry, face_detail  # noqa: E402
from vsfacerestore.pgtformer import CONFIG, CROPS, VFHQ_TEMPLATE, WEIGHTS, PGTFormerRestorer, crops_to_unit, load_pgtformer  # noqa: E402
from vsfacerestore.restorer import IdentityRestorer, RestoreRequest  # noqa: E402
from vsfacerestore.vsutil import frame_to_bgr  # noqa: E402
from testclips import source  # noqa: E402

# (clip, matrix, first, last): one usable face each; the last three tilt by about 20 degrees
SEGMENTS = [
    ("20010901_Wedding. 04.avi", "470bg", 100, 160),
    ("20010901_Wedding. 04.avi", "470bg", 262, 322),
    ("Class clip.avi", "170m", 18, 78),
    ("In_America_(2002).mkv", "709", 94, 154),
]


def original_network(device: torch.device):
    """The PGTFormer class from the original repository, imported with basicsr stubbed out."""
    registry = types.SimpleNamespace(register=lambda *a, **k: (lambda c: c))
    basicsr = types.ModuleType("basicsr")
    utils = types.ModuleType("basicsr.utils")
    utils.get_root_logger = lambda *a, **k: logging.getLogger("basicsr")
    reg = types.ModuleType("basicsr.utils.registry")
    reg.ARCH_REGISTRY = registry
    archs = types.ModuleType("basicsr.archs")
    archs.ARCH_REGISTRY = registry
    basicsr.utils, basicsr.archs, utils.registry = utils, archs, reg
    sys.modules.update({"basicsr": basicsr, "basicsr.utils": utils, "basicsr.utils.registry": reg, "basicsr.archs": archs})
    sys.path.insert(0, PGT_REPO)
    try:
        from archs.pgtformer_arch import PGTFormer as Original
    finally:
        sys.path.remove(PGT_REPO)
    from safetensors.torch import load_file

    net = Original(**CONFIG)
    net.load_state_dict(load_file(WEIGHTS))
    net.eval()
    return net.to(device)


def demo_frames(n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(os.path.join(PGT_REPO, "assets", "inputdemovideo.mp4"))
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)  # BGR uint8, 512x512
    cap.release()
    return frames


def check_network(device: torch.device, frames: int = 12) -> bool:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ours = load_pgtformer(device=device)
    orig = original_network(device)
    src = demo_frames(frames)
    print(f"network: {len(src)} demo frames {src[0].shape}")
    worst = 0.0
    with torch.inference_mode():
        for n in range(len(src)):
            win = [src[max(0, min(len(src) - 1, n + o))] for o in (-1, 0, 1)]
            x = crops_to_unit(win).to(device)
            a = ours(x, w=1.0)[0]
            b = orig(x, w=1.0)[0]
            d = (a - b).abs().max().item()
            worst = max(worst, d)
    print(f"network: max |ours - original| over {len(src)} windows = {worst:.3g}")
    return worst == 0.0


def face_track(src: vs.VideoNode, first: int, last: int, device: torch.device) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Frames and smoothed landmarks of the largest usable face over first..last, as the engine sees them."""
    engine = FaceEngine(IdentityRestorer(), src.num_frames, device)
    fetch = lambda i: frame_to_bgr(src.get_frame(i))  # noqa: E731
    frames, marks = [], []
    for n in range(first, last + 1):
        tracks = [t for t in engine.tracks(n, fetch) if t.contains(n) and t.reason is None]
        assert tracks, f"frame {n}: no usable track"
        big = max(tracks, key=lambda t: t.eye_dist[t.index(n)])  # the main face; small faces in the background are left out
        frames.append(fetch(n))
        marks.append(big.landmarks[big.index(n)])
    return frames, marks


def to_ffhq(crop: np.ndarray, geo_affine: np.ndarray, ffhq_affine: np.ndarray) -> np.ndarray:
    """A crop of any geometry re-cut onto the FFHQ crop of the same frame (so all variants compare in one space)."""
    m = np.vstack([ffhq_affine, [0, 0, 1]]) @ np.vstack([cv2.invertAffineTransform(geo_affine), [0, 0, 1]])
    return cv2.warpAffine(crop, m[:2], (FACE_SIZE, FACE_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def oval_diff(seq: list[np.ndarray]) -> float:
    """Mean absolute difference between consecutive crops inside the face oval."""
    return float(np.mean([np.abs(a.astype(np.int16) - b.astype(np.int16))[FACE_OVAL].mean() for a, b in zip(seq, seq[1:])]))


def low_diff(seq: list[np.ndarray], ref: list[np.ndarray], sigma: float = 6.0) -> float:
    """Mean absolute difference to the source inside the oval after a blur: drift of shape, colour and lighting."""
    blur = lambda c: cv2.GaussianBlur(c.astype(np.float32), (0, 0), sigma)  # noqa: E731
    return float(np.mean([np.abs(blur(a) - blur(b))[FACE_OVAL].mean() for a, b in zip(seq, ref)]))


def run_restorer(restorer, crops: list[np.ndarray]) -> list[np.ndarray]:
    def crop_at(i: int) -> np.ndarray:
        return crops[min(max(i, 0), len(crops) - 1)]

    reqs = [RestoreRequest(None, i, crop_at) for i in range(len(crops))]  # type: ignore[arg-type]
    return restorer.restore(reqs)


def check_crops(device: torch.device, save: bool) -> bool:
    """Each crop geometry on the same smoothed tracks; everything compared on the FFHQ crop of the frame."""
    torch.backends.cudnn.benchmark = False
    pgt = PGTFormerRestorer(device)
    for name, matrix, first, last in SEGMENTS:
        frames, marks = face_track(source(name, matrix), first, last, device)
        ffhq_aff = [FFHQ_CROP.affine(m) for m in marks]
        ref = [cv2.warpAffine(f, a, (FACE_SIZE, FACE_SIZE), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
               for f, a in zip(frames, ffhq_aff)]
        roll = [np.degrees(np.arctan2(m[1, 1] - m[0, 1], m[1, 0] - m[0, 0])) for m in marks]
        eye = [float(np.linalg.norm(m[0] - m[1])) for m in marks]
        print(f"crops: {name} {first}-{last}, roll {min(roll):.1f}..{max(roll):.1f} deg, eye distance {min(eye):.0f}..{max(eye):.0f} px")
        variants = {"source": ref}
        geometries = dict(CROPS, box=CropGeometry(VFHQ_TEMPLATE, rotate=False))  # box: axis-parallel like the training crops
        for crop, geo in geometries.items():
            aff = [geo.affine(m) for m in marks]
            crops = [cv2.warpAffine(f, a, (FACE_SIZE, FACE_SIZE), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
                     for f, a in zip(frames, aff)]
            pgt.geometry = geo
            out = run_restorer(pgt, crops)
            variants[f"pgtformer {crop}"] = [to_ffhq(c, a, f) for c, a, f in zip(out, aff, ffhq_aff)]
        del_dic = DicFaceRestorer(device)
        variants["dicface"] = run_restorer(del_dic, ref)
        del del_dic
        torch.cuda.empty_cache()
        print("    variant              n/n+1 diff  drift to source  detail")
        for label, seq in variants.items():
            det = np.mean([face_detail(c, e) for c, e in zip(seq, eye)])
            print(f"    {label:20s} {oval_diff(seq[1:-1]):9.3f}  {low_diff(seq, ref):14.3f}  {det:6.3f}")
        if save:
            pick = [len(ref) // 4, len(ref) // 2, 3 * len(ref) // 4]
            rows = [np.hstack([cv2.resize(seq[i], (256, 256), interpolation=cv2.INTER_AREA) for i in pick]) for seq in variants.values()]
            path = os.path.join(WORK, f"crops_{name[:6]}_{first}.png")
            cv2.imwrite(path, np.vstack(rows))
            print(f"    saved {path} (rows: {', '.join(variants)})")
            full = os.path.join(WORK, f"full_{name[:6]}_{first}.png")
            cv2.imwrite(full, np.hstack([seq[3 * len(ref) // 4] for seq in variants.values()]))
    return True


def check_speed(device: torch.device) -> bool:
    wedding = source("20010901_Wedding. 04.avi", "470bg")
    for model in ("pgtformer", "dicface"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        node = vsfacerestore.restore(wedding, model)
        node.get_frame(100)
        t0 = time.time()
        frames = 40
        for n in range(101, 101 + frames):
            node.get_frame(n)
        dt = time.time() - t0
        print(f"speed: {model} wedding clip (one face) {frames / dt:.2f} fps in order, peak VRAM "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        del node
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checks", nargs="*", default=["network"])
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    device = torch.device("cuda", 0)
    ok = True
    if "network" in args.checks:
        ok &= check_network(device)
    if "crops" in args.checks:
        ok &= check_crops(device, args.save)
    if "speed" in args.checks:
        ok &= check_speed(device)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
