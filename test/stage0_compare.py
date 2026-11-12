"""Stage 0 acceptance: the facexlib-based helper and the per-frame paste reproduce vscodeformer 1.0.1.

Runs the CodeFormer network (taken from the installed vscodeformer at test time, as the yardstick only) through
`vsfacerestore.perframe.per_frame` and compares every pixel with `vscodeformer.codeformer()` on fixed frames.
Also compares the helper level: landmarks, affine matrices and crops.

Run with the VapourSynth addon's Python, nothing else on the GPU:

    F:\\Vapoursynth\\python.exe test\\stage0_compare.py [--save]

--save writes reference, result and difference images to J:\\tmp\\claude\\vsfacerestore\\stage0.
"""

from __future__ import annotations

import argparse
import os
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

import vscodeformer  # noqa: E402
from vscodeformer.codeformer_arch import CodeFormer  # noqa: E402
from vscodeformer.face_restoration_helper import FaceRestoreHelper  # noqa: E402
from vscodeformer.img_util import img2tensor, tensor2img  # noqa: E402

from vsfacerestore.facehelper import Face, FaceHelper  # noqa: E402
from vsfacerestore.perframe import per_frame  # noqa: E402
from testclips import source  # noqa: E402
from vsfacerestore.vsutil import frame_to_bgr, pad_to_minimum  # noqa: E402

WORK = r"J:\tmp\claude\vsfacerestore\stage0"
WEIGHT = 0.5

# file, matrix, frames, upscale, background clip given (Spline36 at the output size)
CASES = [
    ("1983-i.mkv", "470bg", [0, 240, 480, 720], 1, False),
    ("1983-i.mkv", "470bg", [240], 2, False),
    ("1983-i.mkv", "470bg", [480], 2, True),
    ("20010901_Wedding. 04.avi", "470bg", [120, 280], 1, False),
    ("20010901_Wedding. 04.avi", "470bg", [120], 2, False),
    ("20010901_Wedding. 04.avi", "470bg", [280], 2, True),
    ("Class clip.avi", "170m", [100], 1, False),
    ("Class clip.avi", "170m", [100], 2, True),
    ("In_America_(2002).mkv", "709", [300, 900, 1350], 1, False),
]



def codeformer_eye_filter(face: Face) -> bool:
    """CodeFormer's eye-distance test, reproduced for the comparison only.

    Its helper takes the wrong landmark indices: it measures the vector (left eye y - right eye y,
    right eye x - nose x), about half the real eye distance on a frontal face, and drops faces where that is
    below 5 px. The package uses the real eye distance; this filter makes both sides keep the same faces.
    """
    lm = face.landmarks
    return float(np.linalg.norm([lm[0, 1] - lm[1, 1], lm[1, 0] - lm[2, 0]])) >= 5.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="write reference/result/difference PNGs to WORK")
    args = ap.parse_args()
    if args.save:
        os.makedirs(WORK, exist_ok=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")  # vscodeformer sets this too
    device = torch.device("cuda", 0)

    net = CodeFormer()
    state = torch.load(os.path.join(vscodeformer.model_dir, "codeformer.pth"), map_location="cpu")["params_ema"]
    net.load_state_dict(state)
    net.eval().to(device)

    @torch.inference_mode()
    def restore_face(crop: np.ndarray) -> np.ndarray:
        t = img2tensor(crop / 255.0)
        normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        out = net(t.unsqueeze(0).to(device), w=WEIGHT, adain=True)[0]
        return tensor2img(out, min_max=(-1, 1))

    ref_helper = FaceRestoreHelper(1, det_model="retinaface_resnet50", use_parse=True, device=device)
    our_helper = FaceHelper(device)

    failures = 0
    print(
        f"{'clip':26s} {'frame':>5s} {'up':>2s} {'faces':>9s} {'lm':>8s} {'affine':>8s} {'crop':>4s} | "
        f"{'max':>3s} {'mean':>7s} {'>0':>8s} {'>1':>6s}"
    )
    for name, matrix, frames, upscale, with_bg in CASES:
        src = source(name, matrix)
        bg = src.resize.Spline36(src.width * upscale, src.height * upscale) if with_bg else None
        ref_clip = vscodeformer.codeformer(src, upscale=upscale, weight=WEIGHT, bg_clip=bg)
        our_clip = per_frame(src, restore_face, upscale=upscale, bg_clip=bg, face_filter=codeformer_eye_filter)
        padded = pad_to_minimum(src, 512)[0]  # both filters pad frames under 512 px before detecting
        for n in frames:
            img = frame_to_bgr(padded.get_frame(n))

            # helper level: detections, affines, crops
            ref_helper.clean_all()
            ref_helper.read_image(img)
            ref_helper.get_face_landmarks_5(resize=640, eye_dist_threshold=5)
            ref_helper.align_warp_face()
            faces = [f for f in our_helper.detect(img, 640, 0.8, 5.0) if codeformer_eye_filter(f)]
            lm_diff = aff_diff = crop_diff = float("nan")
            if len(faces) == len(ref_helper.all_landmarks_5) and faces:
                lm_diff = max(float(np.abs(f.landmarks - r).max()) for f, r in zip(faces, ref_helper.all_landmarks_5))
                affines = [our_helper.affine(f.landmarks) for f in faces]
                aff_diff = max(float(np.abs(a - r).max()) for a, r in zip(affines, ref_helper.affine_matrices))
                crops = [our_helper.align(img, a) for a in affines]
                crop_diff = max(
                    int(np.abs(c.astype(int) - r.astype(int)).max()) for c, r in zip(crops, ref_helper.cropped_faces)
                )
            face_info = f"{len(faces)}/{len(ref_helper.all_landmarks_5)}"

            # frame level: the whole filter
            t0 = time.time()
            ref = frame_to_bgr(ref_clip.get_frame(n))
            t1 = time.time()
            ours = frame_to_bgr(our_clip.get_frame(n))
            t2 = time.time()
            diff = np.abs(ref.astype(np.int16) - ours.astype(np.int16))
            dmax, dmean = int(diff.max()), float(diff.mean())
            n0, n1 = int((diff > 0).sum()), int((diff > 1).sum())
            ok = dmax <= 1 and len(faces) == len(ref_helper.all_landmarks_5)
            failures += not ok
            print(
                f"{name[:26]:26s} {n:5d} {upscale:2d}{'b' if with_bg else ' '} {face_info:>8s} {lm_diff:8.4f} "
                f"{aff_diff:8.5f} {crop_diff:4.0f} | "
                f"{dmax:3d} {dmean:7.4f} {n0:8d} {n1:6d}  ref {t1 - t0:4.1f}s ours {t2 - t1:4.1f}s "
                f"{'ok' if ok else 'DIFF'}"
            )
            if args.save:
                base = os.path.join(WORK, f"{os.path.splitext(name)[0][:12]}_{n}_x{upscale}")
                cv2.imwrite(base + "_ref.png", ref)
                cv2.imwrite(base + "_ours.png", ours)
                cv2.imwrite(base + "_diff.png", np.clip(diff.max(axis=2) * 64, 0, 255).astype(np.uint8))
    print("stage 0:", "PASS" if failures == 0 else f"FAIL ({failures} frames)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
