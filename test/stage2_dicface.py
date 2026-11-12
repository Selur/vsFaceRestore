"""Stage 2 acceptance: DicFace extracted and wired into `restore(model='dicface')`.

Checks (plan §7, stage 2):
  network    our extracted network against the original repository's on the same crops (same weights, same seed)
  window     our 'script' window (5 frames, first taken) against the original script's 20-frame window, index 10
  stability  frame-to-frame difference inside the face oval: DicFace centre/script, mean/random sampling, and
             per-frame CodeFormer on the same crops; the source crops as the motion baseline
  speed      fps and VRAM through restore() on a one-face clip, and per-frame cost on the crowd clip per batch size

Run with the VapourSynth addon's Python, nothing else on the GPU:

    F:\\Vapoursynth\\python.exe test\\stage2_dicface.py [--save]

--save writes crops of one frame (source, CodeFormer, DicFace) to J:\\tmp\\claude\\vsfacerestore\\stage2.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "test"))  # the embedded Python does not add the script folder
DICFACE_REPO = r"F:\workspace\DicFace"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import vapoursynth as vs  # noqa: E402
from torchvision.transforms.functional import normalize  # noqa: E402

core = vs.core

import vsfacerestore  # noqa: E402
from vsfacerestore.dicface import FRAME_LENGTH, WEIGHTS, DicFaceRestorer, crops_to_tensor, load_dicface, tensor_to_crops  # noqa: E402
from vsfacerestore.engine import FaceEngine  # noqa: E402
from vsfacerestore.facehelper import FACE_OVAL  # noqa: E402
from vsfacerestore.restorer import IdentityRestorer, RestoreRequest  # noqa: E402
from testclips import source  # noqa: E402
from vsfacerestore.vsutil import frame_to_bgr  # noqa: E402

WORK = r"J:\tmp\claude\vsfacerestore\stage2"



def original_network(device: torch.device):
    """The architecture class from the DicFace repository, imported without the rest of basicsr."""
    pkg = types.ModuleType("basicsr")
    pkg.__path__ = [os.path.join(DICFACE_REPO, "basicsr")]
    utils = types.ModuleType("basicsr.utils")
    utils.__path__ = [os.path.join(DICFACE_REPO, "basicsr", "utils")]
    utils.get_root_logger = lambda *a, **k: logging.getLogger("basicsr")
    archs = types.ModuleType("basicsr.archs")
    archs.__path__ = [os.path.join(DICFACE_REPO, "basicsr", "archs")]
    sys.modules.update({"basicsr": pkg, "basicsr.utils": utils, "basicsr.archs": archs})
    importlib.import_module("basicsr.utils.registry")
    importlib.import_module("basicsr.archs.vqgan_arch")
    mod = importlib.import_module("basicsr.archs.dir_dist_codeformer_multiscale_arch")
    net = mod.TemporalCodeFormerDirDistMultiScale(
        dim_embed=512, n_head=8, n_layers=9, codebook_size=1024, connect_list=["32", "64", "128", "256"], frame_length=5
    )
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu")["params_ema"])
    return net.eval().to(device)


def face_crops(src: vs.VideoNode, first: int, last: int, device: torch.device) -> list[np.ndarray]:
    """Aligned crops of the one face of the clip over first..last, as the engine would cut them."""
    engine = FaceEngine(IdentityRestorer(), src.num_frames, device)
    fetch = lambda i: frame_to_bgr(src.get_frame(i))  # noqa: E731
    crops = []
    for n in range(first, last + 1):
        tracks = [t for t in engine.tracks(n, fetch) if t.contains(n) and t.reason is None]
        assert len(tracks) == 1, f"frame {n}: {len(tracks)} usable tracks"
        t = tracks[0]
        crops.append(engine.helper.align(fetch(n), engine.helper.affine(t.landmarks[t.index(n)])))
    return crops


def oval_diff(seq: list[np.ndarray]) -> float:
    """Mean absolute difference between consecutive crops inside the face oval."""
    d = [np.abs(a.astype(np.int16) - b.astype(np.int16))[FACE_OVAL].mean() for a, b in zip(seq, seq[1:])]
    return float(np.mean(d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    if args.save:
        os.makedirs(WORK, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", 0)
    results = {}

    wedding = source("20010901_Wedding. 04.avi", "470bg")
    first, last = 100, 140
    crops = face_crops(wedding, first, last, device)
    print(f"{len(crops)} crops of the wedding face, frames {first}-{last}")

    # ---------------------------------------------------------------- network parity
    ours = load_dicface(device=device)
    orig = original_network(device)
    x = crops_to_tensor(crops[10:15]).to(device)
    with torch.inference_mode():
        torch.manual_seed(1234)
        a = orig(x, w=1.0)[0]
        torch.manual_seed(1234)
        b = ours(x, w=1.0, sample="random")[0]
        m1 = ours(x, w=1.0, sample="mean")[0]
        m2 = ours(x, w=1.0, sample="mean")[0]
    diff_random = float((a - b).abs().max())
    diff_mean = float((m1 - m2).abs().max())
    img_diff = max(int(np.abs(p.astype(int) - q.astype(int)).max()) for p, q in zip(tensor_to_crops(a), tensor_to_crops(b)))
    results["network"] = img_diff <= 1 and diff_mean == 0
    print(f"network  original vs ours, same seed: max |diff| {diff_random:.2e} (uint8 {img_diff}); mean mode twice: {diff_mean:.1e}")
    del orig
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- windows
    def crop_at(i: int) -> np.ndarray:
        return crops[min(max(i, 0), len(crops) - 1)]

    def run(window: str, sample: str, batch: int = 2) -> list[np.ndarray]:
        r = DicFaceRestorer(device, 1.0, window, sample, batch)
        r.net = ours
        reqs = [RestoreRequest(None, i, crop_at) for i in range(len(crops))]  # type: ignore[arg-type]
        return r.restore(reqs)

    # the original script: 20 frames n-10 ... n+9 (edges repeated), index 10 of the result
    script = []
    with torch.inference_mode():
        for i in range(len(crops)):
            seq = [crop_at(i - 10 + k) for k in range(20)]
            y = ours(crops_to_tensor(seq).to(device), w=1.0, sample="mean")[0]
            script.append(tensor_to_crops(y[10:11])[0])
    ours_script = run("script", "mean")
    win_diff = [int(np.abs(p.astype(int) - q.astype(int)).max()) for p, q in zip(script, ours_script)]
    win_mean = float(np.mean([np.abs(p.astype(int) - q.astype(int)).mean() for p, q in zip(script, ours_script)]))
    results["window"] = max(win_diff) <= 2
    print(f"window   original script (20 frames, index 10) vs ours 'script': max |diff| {max(win_diff)}, mean {win_mean:.4f}")

    # ---------------------------------------------------------------- stability
    import vscodeformer
    from vscodeformer.codeformer_arch import CodeFormer
    from vscodeformer.img_util import img2tensor, tensor2img

    cf = CodeFormer()
    cf.load_state_dict(torch.load(os.path.join(vscodeformer.model_dir, "codeformer.pth"), map_location="cpu")["params_ema"])
    cf.eval().to(device)

    @torch.inference_mode()
    def codeformer(crop: np.ndarray, w: float) -> np.ndarray:
        t = img2tensor(crop / 255.0)
        normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        return tensor2img(cf(t.unsqueeze(0).to(device), w=w, adain=True)[0], min_max=(-1, 1))

    variants = {
        "source crops (motion baseline)": crops,
        "CodeFormer per frame, w=0.5": [codeformer(c, 0.5) for c in crops],
        "CodeFormer per frame, w=1.0": [codeformer(c, 1.0) for c in crops],
        "DicFace centre, mean": run("centre", "mean"),
        "DicFace centre, random": run("centre", "random"),
        "DicFace script, mean": ours_script,
        "DicFace script, random": run("script", "random"),
    }
    del cf
    print("stability: mean |frame n+1 - frame n| inside the face oval (8-bit levels), lower is steadier")
    for label, seq in variants.items():
        print(f"    {label:34s} {oval_diff(seq[2:-2]):6.3f}")
    results["stability"] = True
    if args.save:
        for label, seq in variants.items():
            cv2.imwrite(os.path.join(WORK, label.split(",")[0].replace(" ", "_") + ".png"), np.hstack(seq[18:22]))

    # ---------------------------------------------------------------- speed and VRAM through restore()
    del ours
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    node = vsfacerestore.restore(wedding, "dicface")
    node.get_frame(first)  # warm-up (window)
    t0 = time.time()
    frames = 50
    for n in range(first + 1, first + 1 + frames):
        node.get_frame(n)
    dt = time.time() - t0
    print(f"speed    wedding clip (one face): {frames / dt:.2f} fps in order, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    del node
    vhs = source("1983-i.mkv", "470bg")
    for batch in (1, 2):  # 4 measured once: 8.6 s per frame and 10.6 GiB, the batch spills
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        node = vsfacerestore.restore(vhs, "dicface", batch=batch)
        node.get_frame(480)
        t0 = time.time()
        for n in range(481, 489):
            node.get_frame(n)
        dt = (time.time() - t0) / 8
        print(f"speed    crowd clip (about 11 faces), batch {batch}: {dt:.2f} s per frame, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        del node
    results["speed"] = True

    print("stage 2:", ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
