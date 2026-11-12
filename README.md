# vsfacerestore

Temporal video face restoration for VapourSynth with [DicFace](https://github.com/fudan-generative-vision/DicFace)
and [KEEP](https://github.com/jnjaby/KEEP). Faces are found, linked across frames, restored and pasted back inside
the VapourSynth graph; nothing is written to disk.

```python
import vsfacerestore
clip = vsfacerestore.restore(clip, model='dicface')  # progressive RGB24, RGBS or RGBH, CUDA
```

The input must be progressive: deinterlace before this filter, as before any other. The faces are restored at
8 bit whatever the format (detector, parser and both models take uint8 crops); with RGBS or RGBH the rest of the
frame and the background clip keep their precision, the paste blends in float without rounding.

**Status:** `restore(model='dicface')` and `restore(model='keep')` work.

Parameters of `restore()`: `bg_clip` (background at the output size), `upscale`, `radius` (15), `tone` (0.5),
`weight` (DicFace fidelity, 1.0), `device_index`, `half` (fp16 detection and parsing), `window` ('centre'),
`sample` ('mean'), `batch` (1), `sharpness` (0.04; faces scoring below it stay untouched), `min_eye_dist` (5 px),
`min_visibility` (0.45), `fade` (8 frames), `detect` (`DetectParams`), `track` (`TrackParams`: every filter
threshold and alpha ramp), `model_dir` (folder with the weights; default `vsfacerestore/models` in the package).

## Layout

| Path | Content |
|---|---|
| `vsfacerestore/__init__.py` | `restore()` |
| `vsfacerestore/filter.py` | the VapourSynth filter around the engine (`build()`, `restore_with()`) |
| `vsfacerestore/engine.py` | window engine: detections for n−K…n+K, tracks, restoration, paste of frame n |
| `vsfacerestore/tracking.py` | detections, linking, landmark smoothing, track filter, alpha |
| `vsfacerestore/paste.py` | paste-back with tone transfer and parsing mask (tapewake) |
| `vsfacerestore/restorer.py` | the model interface (`Restorer`, `RestoreRequest`) |
| `vsfacerestore/cache.py` | compute-once LRU for the detections |
| `vsfacerestore/facehelper.py` | detection (RetinaFace-ResNet50), alignment, parsing (ParseNet), stage 0 paste |
| `vsfacerestore/perframe.py` | per-frame filter around a face function (stage 0 harness) |
| `vsfacerestore/vsutil.py` | frames <-> numpy, padding |
| `vsfacerestore/facexlib/` | vendored networks from facexlib 0.3.0 (MIT), see its README |
| `vsfacerestore/dicface/` | DicFace architecture (S-Lab) and its restorer, see its README |
| `vsfacerestore/keep/` | KEEP architecture (S-Lab) with the diffusers 0.11.0 attention blocks and GMFlow (Apache 2.0), block grid, crossfade and cache, see its README |
| `vsfacerestore/models/` | weights (not in git): facexlib's `detection_Resnet50_Final.pth`, `parsing_parsenet.pth`; `dicface/bfr_100k.pth`; `keep/KEEP-b76feb75.pth` |
| `test/stage0_compare.py` | stage 0 acceptance: same output as vscodeformer 1.0.1 on fixed frames |
| `test/stage1_engine.py` | stage 1 acceptance: order independence, detections per frame, jump time, untouched faces |
| `test/stage2_dicface.py` | stage 2 acceptance: network parity, window, stability, speed and VRAM |
| `test/stage3_keep.py` | stage 3 acceptance: network parity, block borders, jumps against in-order reading, speed |
| `test/stage4_formats.py` | stage 4 acceptance: RGB24 unchanged against the installed package, RGBS exact outside the faces and within one level inside, RGBH within fp16 of RGBS |
| `test/testclips.py` | the test clips, deinterlaced where needed |

## Requirements

Python 3.12+, VapourSynth R66+, torch with CUDA, torchvision, OpenCV, numpy, scipy, einops.

## Licences

Package code: MIT. `vsfacerestore/facexlib`: MIT (Xintao Wang). The DicFace and KEEP model code and weights
(stages 2 and 3) are S-Lab License 1.0 (non-commercial) and get their own licence files.
