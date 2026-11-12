"""Stage 1 acceptance: window engine, tracks, filter, alpha and paste.

Checks (plan §7, stage 1):
  order     same frames when reading forwards, backwards and in random order, with 1 and with 6 threads
  detect    about one detection per frame when reading in order
  jump      time of a cold frame after a jump against a warm one
  identity  the paste leaves a frame almost unchanged when the model returns the crop itself
  faces     with CodeFormer as the model: untouched tracks stay untouched, only face regions change, and the
            per-track judgement is printed

Run with the VapourSynth addon's Python, nothing else on the GPU:

    F:\\Vapoursynth\\python.exe test\\stage1_engine.py [--save] [--skip-model]

--save writes PNGs of the model run to J:\\tmp\\claude\\vsfacerestore\\stage1 (delete afterwards).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "test"))  # the embedded Python does not add the script folder

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import vapoursynth as vs  # noqa: E402
from torchvision.transforms.functional import normalize  # noqa: E402

core = vs.core

from vsfacerestore.filter import build  # noqa: E402
from vsfacerestore.restorer import BatchRestorer, IdentityRestorer  # noqa: E402
from testclips import source  # noqa: E402
from vsfacerestore.vsutil import frame_to_bgr  # noqa: E402

WORK = r"J:\tmp\claude\vsfacerestore\stage1"



def digest(frame: vs.VideoFrame) -> str:
    return hashlib.sha256(frame_to_bgr(frame).tobytes()).hexdigest()[:12]


def read_all(node: vs.VideoNode, prefetch: int) -> list[str]:
    return [digest(f) for f in node.frames(prefetch=prefetch, backlog=2 * prefetch)]


def check_order(src: vs.VideoNode, first: int, count: int) -> bool:
    """Fresh engine per order; every order must give the same frames."""
    ok = True
    reference = None
    orders = {
        "forward": list(range(first, first + count)),
        "backward": list(range(first + count - 1, first - 1, -1)),
        "random": random.Random(7).sample(range(first, first + count), count),
    }
    for threads in (1, 6):
        core.num_threads = threads
        for label, order in orders.items():
            node, engine = build(src, IdentityRestorer())
            spliced = core.std.Splice([node[i] for i in order])
            t0 = time.time()
            digests = read_all(spliced, prefetch=threads)
            dt = time.time() - t0
            by_frame = dict(zip(order, digests))
            if reference is None:
                reference = by_frame
            same = all(by_frame[i] == reference[i] for i in order)
            ok &= same
            print(
                f"order  {label:8s} threads={threads}  {count} frames {dt:5.1f}s  detections {engine.cache.computed:4d} "
                f"{'same' if same else 'DIFFERENT'}"
            )
    core.num_threads = 6
    return ok


def check_detections_and_jump(src: vs.VideoNode, first: int, count: int, jump_to: int) -> bool:
    node, engine = build(src, IdentityRestorer())
    core.num_threads = 1
    for i in range(first, first + count):
        node.get_frame(i)
    computed = engine.cache.computed
    expected = count + 2 * engine.radius
    ok = computed <= expected + 2
    print(f"detect in-order {count} frames: {computed} detections (expected about {expected}) {'ok' if ok else 'TOO MANY'}")
    t0 = time.time()
    node.get_frame(jump_to)
    cold = time.time() - t0
    t0 = time.time()
    node.get_frame(jump_to + 1)
    warm = time.time() - t0
    print(f"jump   cold frame {jump_to}: {cold:.2f}s (window of {2 * engine.radius + 1} detections), next frame {warm:.2f}s")
    core.num_threads = 6
    return ok


def check_identity(src: vs.VideoNode, frames: list[int]) -> bool:
    """With the crop pasted back unchanged the frame must stay nearly the same (resampling only)."""
    node, engine = build(src, IdentityRestorer())
    ok = True
    for n in frames:
        a = frame_to_bgr(src.get_frame(n)).astype(np.int16)
        b = frame_to_bgr(node.get_frame(n)).astype(np.int16)
        d = np.abs(a - b)
        tracks = [t for t in engine.tracks(n, lambda i: frame_to_bgr(src.get_frame(i))) if t.contains(n) and t.alpha[t.index(n)] >= 0.02]
        good = d.mean() < 0.5  # the anti-alias and the resampling smooth the face a little; the frame stays the same
        ok &= good
        print(f"identity frame {n}: {len(tracks)} faces pasted, mean diff {d.mean():.3f}, max {d.max()} {'ok' if good else 'TOO LARGE'}")
    return ok


def crop_quad(affine: np.ndarray, upscale: int) -> np.ndarray:
    """The crop's corners in output pixels, (4, 2) int32."""
    inv = cv2.invertAffineTransform(affine)
    corners = np.array([[0, 0, 1], [511, 0, 1], [511, 511, 1], [0, 511, 1]], np.float64)
    pts = (inv @ corners.T).T * upscale + (upscale - 1) / 2
    return np.round(pts).astype(np.int32)


def check_faces(src: vs.VideoNode, name: str, frames: list[int], restorer, save: bool) -> bool:
    node, engine = build(src, restorer, upscale=1)
    ok = True
    fetch = lambda i: frame_to_bgr(src.get_frame(i))  # noqa: E731
    for n in frames:
        a = fetch(n)
        b = frame_to_bgr(node.get_frame(n))
        changed = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2) > 0
        tracks = [t for t in engine.tracks(n, fetch) if t.contains(n)]
        print(f"faces {name[:12]} frame {n}: {len(tracks)} tracks in frame, {int(changed.sum())} pixels changed")
        # pixels a restored face may change: its crop quad (the mask is zero at the crop border)
        touched = np.zeros(changed.shape, np.uint8)
        for t in tracks:
            i = t.index(n)
            if t.alpha[i] >= 0.02:
                cv2.fillConvexPoly(touched, crop_quad(engine.helper.affine(t.landmarks[i]), 1), 1)
        touched = cv2.dilate(touched, np.ones((5, 5), np.uint8)).astype(bool)
        for t in tracks:
            i = t.index(n)
            alpha = t.alpha[i]
            m = t.median
            note = ""
            if alpha < 0.02:
                # the inner face of an untouched track must be unchanged, except where a restored neighbour overlaps
                cx, cy = t.landmarks[i].mean(0)
                r = max(2.0, t.eye_dist[i])
                x0, x1 = int(max(0, cx - r)), int(min(a.shape[1], cx + r))
                y0, y1 = int(max(0, cy - r)), int(min(a.shape[0], cy + r))
                if (changed[y0:y1, x0:x1] & ~touched[y0:y1, x0:x1]).any():
                    ok = False
                    note = "  CHANGED ALTHOUGH UNTOUCHED"
            print(
                f"    track {t.first:4d}-{t.last:4d} ed {m['eye_dist']:5.1f} score {m['score']:.2f} vis {m['visibility']:.2f} "
                f"yaw {m['yaw']:.2f} detail {m['detail']:.3f} alpha {alpha:.2f} {t.reason or 'restored'}{note}"
            )
        outside = changed & ~touched
        if outside.any():
            ok = False
            print(f"    {int(outside.sum())} pixels changed outside restored faces: FAIL")
        if save:
            base = os.path.join(WORK, f"{os.path.splitext(name)[0][:12]}_{n}")
            cv2.imwrite(base + "_src.png", a)
            cv2.imwrite(base + "_out.png", b)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--skip-model", action="store_true", help="skip the CodeFormer run")
    args = ap.parse_args()
    if args.save:
        os.makedirs(WORK, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    vhs = source("1983-i.mkv", "470bg")
    wedding = source("20010901_Wedding. 04.avi", "470bg")

    results = {}
    results["order"] = check_order(vhs, 200, 40)
    results["detect+jump"] = check_detections_and_jump(vhs, 300, 40, 900)
    results["identity"] = check_identity(vhs, [240, 480]) and check_identity(wedding, [120])

    if not args.skip_model:
        import vscodeformer
        from vscodeformer.codeformer_arch import CodeFormer
        from vscodeformer.img_util import img2tensor, tensor2img

        device = torch.device("cuda", 0)
        net = CodeFormer()
        net.load_state_dict(torch.load(os.path.join(vscodeformer.model_dir, "codeformer.pth"), map_location="cpu")["params_ema"])
        net.eval().to(device)

        @torch.inference_mode()
        def codeformer(crops):
            x = torch.stack([img2tensor(c / 255.0) for c in crops])
            normalize(x, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            out = net(x.to(device), w=0.5, adain=True)[0]
            return [tensor2img(o, min_max=(-1, 1)) for o in out]

        restorer = BatchRestorer(codeformer)
        results["faces"] = check_faces(vhs, "1983-i.mkv", [240, 480, 720], restorer, args.save) and check_faces(
            wedding, "20010901_Wedding. 04.avi", [120, 280], restorer, args.save
        )

    print("stage 1:", ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
