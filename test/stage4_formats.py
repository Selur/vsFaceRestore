"""Stage 4 acceptance: RGBS and RGBH input next to RGB24.

Checks (plan §7, stage 4):
  rgb24      the repository code gives the same RGB24 frames as the installed package (byte-identical)
  rgbs       RGBS output: a frame without a face comes back exactly; with faces, away from the box around the pixels
             the RGB24 path changed nothing moves by half a level or more (the blend fades out over the whole aligned
             crop, so a float still moves where an 8-bit value rounds back), and the whole frame stays within one
             8-bit level of the RGB24 output, since the face itself is 8-bit either way
  rgbh       RGBH output within fp16 precision of the RGBS output

Run with the VapourSynth addon's Python, nothing else on the GPU:

    F:\\Vapoursynth\\python.exe test\\stage4_formats.py [--frames 41-46] [--keep]

Each render runs in its own process (one model load each); frames land in J:\\tmp\\claude\\vsfacerestore\\stage4
and are deleted at the end unless --keep is given.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = r"J:\tmp\claude\vsfacerestore\stage4"
CLIP = "20010901_Wedding. 04.avi"


def render(source: str, fmt: str, first: int, last: int, out: str) -> None:
    """Child process: `source` is 'repo' (this tree) or 'installed' (site-packages); writes one .npy per frame."""
    if source == "repo":
        sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "test"))
    import numpy as np
    import vapoursynth as vs

    import vsfacerestore
    from testclips import source as load

    clip = load(CLIP)
    if fmt != "RGB24":
        clip = clip.resize.Point(format=getattr(vs, fmt))
    clip = vsfacerestore.restore(clip, model="dicface")
    os.makedirs(out, exist_ok=True)
    for n in range(first, last + 1):
        f = clip.get_frame(n)
        arr = np.stack([np.asarray(f[p]) for p in range(3)], axis=0)
        np.save(os.path.join(out, "%05d.npy" % n), arr)
    print("rendered", source, fmt, vsfacerestore.__file__)


def run_render(source: str, fmt: str, first: int, last: int) -> str:
    out = os.path.join(WORK, "%s-%s" % (source, fmt))
    subprocess.run([sys.executable, os.path.abspath(__file__), "--render", source, fmt, str(first), str(last), out], check=True)
    return out


def load(folder: str, first: int, last: int):
    import numpy as np

    return [np.load(os.path.join(folder, "%05d.npy" % n)) for n in range(first, last + 1)]


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--render":
        render(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="41-46")  # two frames before the face appears, four with it
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    first, last = (int(x) for x in args.frames.split("-"))

    import numpy as np

    findings = []
    folders = {key: run_render(*key, first, last) for key in (("installed", "RGB24"), ("repo", "RGB24"), ("repo", "RGBS"), ("repo", "RGBH"))}
    sys.path.insert(0, os.path.join(ROOT, "test"))
    import vapoursynth as vs  # noqa: E402
    from testclips import source as load_clip  # noqa: E402

    src_clip = load_clip(CLIP)
    src_float = src_clip.resize.Point(format=vs.RGBS)  # the exact floats the RGBS render started from
    installed = load(folders[("installed", "RGB24")], first, last)
    rgb24 = load(folders[("repo", "RGB24")], first, last)
    rgbs = load(folders[("repo", "RGBS")], first, last)
    rgbh = load(folders[("repo", "RGBH")], first, last)

    MARGIN = 8  # px around the 8-bit changes within which float changes may lie
    faces = 0
    print("%-8s %-10s %-10s %-10s %-10s %-10s" % ("frame", "rgb24=inst", "inside", "max|S-24|", "max|H-S|", "changed 8bit/float"))
    for i, n in enumerate(range(first, last + 1)):
        f = src_clip.get_frame(n)
        src = np.stack([np.asarray(f[p]) for p in range(3)], axis=0)
        same = np.array_equal(rgb24[i], installed[i])
        changed24 = (rgb24[i] != src).any(axis=0)  # pixels the 8-bit path changed
        ff = src_float.get_frame(n)
        src_f = np.stack([np.asarray(ff[p]) for p in range(3)], axis=0)
        changed_s = (rgbs[i] != src_f).any(axis=0)
        if changed24.any():
            ys, xs = np.nonzero(changed24)
            box = np.zeros_like(changed24)
            box[max(0, ys.min() - MARGIN):ys.max() + MARGIN + 1, max(0, xs.min() - MARGIN):xs.max() + MARGIN + 1] = True
            # the blend fades out over the whole aligned crop, so sub-level float changes reach beyond the visible
            # 8-bit changes; away from the box nothing may move by half a level or more
            stray = (np.abs(rgbs[i] - src_f).max(axis=0) * 255.0)[~box]
            inside = stray.size == 0 or float(stray.max()) < 0.5
            faces += 1
        else:
            inside = not changed_s.any()  # no face: the float frame must come back untouched
        diff_s = float(np.abs(rgbs[i] * 255.0 - rgb24[i].astype(np.float32)).max())
        diff_h = float(np.abs(rgbh[i].astype(np.float32) - rgbs[i]).max()) * 255.0
        print("%-8d %-10s %-10s %-10.3f %-10.3f %d/%d" % (n, same, inside, diff_s, diff_h, int(changed24.sum()), int(changed_s.sum())))
        if not same:
            findings.append("frame %d: RGB24 differs from the installed package" % n)
        if not inside:
            findings.append("frame %d: RGBS changed pixels away from the restored faces" % n)
        if diff_s > 1.0:
            findings.append("frame %d: RGBS more than one level from RGB24 (%.3f)" % (n, diff_s))
        if diff_h > 0.5:
            findings.append("frame %d: RGBH more than half a level from RGBS (%.3f)" % (n, diff_h))
    if faces == 0:
        findings.append("no frame with a restored face, the check proves nothing")

    if not args.keep:
        shutil.rmtree(WORK, ignore_errors=True)
    print("findings:", findings or "none")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
