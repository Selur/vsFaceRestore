"""Stage 3 acceptance: KEEP extracted, block grid with crossfade and cache, `restore(model='keep')`.

Checks (plan §7, stage 3):
  network   our extracted network against the original repository's on the same 20 crops
  borders   frame-to-frame steps at block borders: the original script's 20-frame pieces without overlap against
            our grid (20 frames, step 14, 6 frames crossfaded)
  jumps     restore(model='keep') read in order against a fresh engine read with jumps: identical frames
  speed     fps and VRAM on the one-face clip, cold jump time

The original network needs diffusers 0.11.0 (not installable here) and the GMFlow submodule (not in the local
checkout); the test maps those two imports to our vendored copies, which are verbatim apart from the removed
xformers path, so the comparison covers KEEP's own code.

Run with the VapourSynth addon's Python, nothing else on the GPU:

    F:\\Vapoursynth\\python.exe test\\stage3_keep.py
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "test"))
KEEP_REPO = r"F:\workspace\KEEP"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import vapoursynth as vs  # noqa: E402

core = vs.core

import vsfacerestore  # noqa: E402
import vsfacerestore.keep.attention as our_attention  # noqa: E402
import vsfacerestore.keep.gmflow.gmflow as our_gmflow  # noqa: E402
from stage2_dicface import face_crops, oval_diff  # noqa: E402
from testclips import source  # noqa: E402
from vsfacerestore.facehelper import FACE_OVAL  # noqa: E402
from vsfacerestore.keep import BLOCK, CONFIG, STEP, WEIGHTS, KEEPRestorer, load_keep  # noqa: E402
from vsfacerestore.restorer import RestoreRequest, crops_to_tensor, tensor_to_crops  # noqa: E402
from vsfacerestore.vsutil import frame_to_bgr  # noqa: E402


def original_network(device: torch.device):
    """KEEP's own architecture file, imported with the rest of basicsr, diffusers and GMFlow shimmed."""
    def pkg(name, path=None, **attrs):
        m = types.ModuleType(name)
        if path:
            m.__path__ = [path]
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    pkg("basicsr", os.path.join(KEEP_REPO, "basicsr"))
    pkg("basicsr.utils", os.path.join(KEEP_REPO, "basicsr", "utils"), get_root_logger=lambda *a, **k: logging.getLogger("basicsr"))
    pkg("basicsr.archs", os.path.join(KEEP_REPO, "basicsr", "archs"))
    pkg("basicsr.ops")
    pkg("basicsr.ops.dcn", ModulatedDeformConvPack=type("ModulatedDeformConvPack", (torch.nn.Module,), {}), modulated_deform_conv=None)
    pkg("distutils")
    pkg("distutils.version", LooseVersion=lambda v: tuple(int(p) if p.isdigit() else 0 for p in str(v).split(".")[:3]))
    pkg("diffusers")
    pkg("diffusers.models")
    sys.modules["diffusers.models.attention"] = our_attention
    pkg("basicsr.archs.gmflow")
    pkg("basicsr.archs.gmflow.gmflow")
    sys.modules["basicsr.archs.gmflow.gmflow.gmflow"] = our_gmflow
    importlib.import_module("basicsr.utils.registry")
    import re

    if not hasattr(re, "T"):  # keep_arch.py has a stray `from re import T`; re.T is gone in Python 3.13+
        re.T = 0
    mod = importlib.import_module("basicsr.archs.keep_arch")
    net = mod.KEEP(**CONFIG)
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True)["params_ema"])
    return net.eval().to(device)


def block_key_for(restorer: KEEPRestorer, track, k: int) -> tuple:
    from vsfacerestore.keep import block_key

    return block_key(track, k, restorer.block, restorer.step)


def digest(frame: vs.VideoFrame) -> str:
    return hashlib.sha256(frame_to_bgr(frame).tobytes()).hexdigest()[:12]


def steps(seq: list[np.ndarray]) -> np.ndarray:
    return np.array([np.abs(a.astype(np.int16) - b.astype(np.int16))[FACE_OVAL].mean() for a, b in zip(seq, seq[1:])])


def main() -> int:
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", 0)
    results = {}

    wedding = source("20010901_Wedding. 04.avi")
    first, last = 100, 160
    crops = face_crops(wedding, first, last, device)
    print(f"{len(crops)} crops of the wedding face, frames {first}-{last}")

    # ---------------------------------------------------------------- network parity
    ours = load_keep(device=device)
    orig = original_network(device)
    x = crops_to_tensor(crops[:BLOCK]).unsqueeze(0).to(device)
    with torch.inference_mode():
        a = orig(x, need_upscale=False)[0]
        b = ours(x, need_upscale=False)[0]
        b2 = ours(x, need_upscale=False)[0]
    diff = float((a - b).abs().max())
    img_diff = max(int(np.abs(p.astype(int) - q.astype(int)).max()) for p, q in zip(tensor_to_crops(a), tensor_to_crops(b)))
    repeat = float((b - b2).abs().max())
    results["network"] = img_diff <= 1
    print(f"network  original vs ours on {BLOCK} crops: max |diff| {diff:.2e} (uint8 {img_diff}); same input twice: {repeat:.1e}")
    del orig
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- block borders
    def crop_at(i: int) -> np.ndarray:
        return crops[min(max(i, 0), len(crops) - 1)]

    script = []  # the original script: pieces of 20 without overlap
    with torch.inference_mode():
        for s in range(0, len(crops), BLOCK):
            piece = [crop_at(i) for i in range(s, min(s + BLOCK, len(crops)))]
            if len(piece) == 1:  # the script feeds a lone frame twice and keeps the first result
                piece = piece * 2
            y = ours(crops_to_tensor(piece).unsqueeze(0).to(device), need_upscale=False)[0]
            script.extend(tensor_to_crops(y)[: min(BLOCK, len(crops) - s)])
    restorer = KEEPRestorer(device)
    restorer.net = ours
    reqs = [RestoreRequest(None, i, crop_at) for i in range(len(crops))]  # type: ignore[arg-type]
    fake_track = types.SimpleNamespace(raw={0: np.zeros((5, 2), np.float32)})
    for r in reqs:
        r.track = fake_track  # one face: a constant key per block is enough here
    grid = restorer.restore(reqs)
    s_script, s_grid, s_src = steps(script), steps(grid), steps(crops)
    border_script = [BLOCK - 1, 2 * BLOCK - 1]  # steps 19->20, 39->40
    overlap_grid = [j for j in range(len(s_grid)) if any(k * STEP - 1 <= j < k * STEP + BLOCK - STEP for k in range(1, 4))]
    inside = [j for j in range(len(s_grid)) if j not in border_script and j not in overlap_grid]
    # how much the two blocks of an overlap disagree, per overlap frame (the crossfade blends these)
    disagree = []
    for k in range(1, len(crops) // STEP + 1):
        a = restorer.cache.get_many([block_key_for(restorer, fake_track, k - 1)], lambda key: None)
        b = restorer.cache.get_many([block_key_for(restorer, fake_track, k)], lambda key: None)
        ya, yb = next(iter(a.values())), next(iter(b.values()))
        if ya is None or yb is None:
            continue
        disagree.append([np.abs(ya[STEP + j].astype(np.int16) - yb[j].astype(np.int16))[FACE_OVAL].mean() for j in range(BLOCK - STEP)])
    disagree = np.array(disagree)
    print("borders  mean |frame n+1 - frame n| inside the face oval (the source moves too, so steps are compared with it):")
    print(f"    source                        inside {s_src[inside].mean():.3f}   in the overlaps {s_src[overlap_grid].mean():.3f}")
    print(f"    script, 20 no overlap         inside {s_script[inside].mean():.3f}   at its borders {s_script[border_script].mean():.3f}")
    print(f"    grid 20/14 crossfaded         inside {s_grid[inside].mean():.3f}   in the overlaps {s_grid[overlap_grid].mean():.3f}")
    print("    disagreement of the two blocks per overlap frame (mean over overlaps): " + " ".join(f"{v:.2f}" for v in disagree.mean(axis=0)))
    excess_inside = (s_grid - s_src)[inside]
    excess_overlap = (s_grid - s_src)[overlap_grid]
    results["borders"] = excess_overlap.max() <= excess_inside.max() + 1.0 and disagree[:, 1:].max() <= 1.5
    print(f"    restored step minus source step: inside max {excess_inside.max():.2f}, overlaps max {excess_overlap.max():.2f}; KEEP overall {oval_diff(grid):.3f}")
    del ours, restorer
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- jumps vs in order through restore()
    core.num_threads = 4
    node = vsfacerestore.restore(wedding, "keep")
    frames = list(range(first, first + 40))
    t0 = time.time()
    in_order = {n: digest(node.get_frame(n)) for n in frames}
    dt = time.time() - t0
    print(f"speed    wedding clip (one face), KEEP in order: {len(frames) / dt:.2f} fps, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    del node
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    node = vsfacerestore.restore(wedding, "keep")
    order = [first + 30, first + 5, first + 39, first + 17, first + 0, first + 24]
    t0 = time.time()
    node.get_frame(order[0])
    cold = time.time() - t0
    jumped = {n: digest(node.get_frame(n)) for n in order}
    same = all(jumped[n] == in_order[n] for n in order)
    results["jumps"] = same
    print(f"jumps    fresh engine with jumps {order}: {'identical' if same else 'DIFFERENT'} to in-order reading; cold jump {cold:.1f}s")
    results["speed"] = True

    print("stage 3:", ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
