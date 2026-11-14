vsfacerestore - temporal video face restoration for VapourSynth
================================================================

Restores faces in video with [DicFace](https://github.com/fudan-generative-vision/DicFace),
[KEEP](https://github.com/jnjaby/KEEP) or [PGTFormer](https://github.com/kepengxu/PGTFormer). Faces are found,
linked across frames, restored and pasted back inside the VapourSynth graph.

**Goals**

- Temporal: every face is tracked over a window of frames, so the restoration stays stable from frame to frame.
- Self-contained: no pre-pass, no image files on disk, no downloads at run time.
- Gentle: faces that are too small, too blurry, turned away or hidden are left untouched and faded in and out
  instead of popping.

**Note on input:** the clip must be progressive RGB24, RGBS or RGBH. Deinterlace first, as before any other
filter. The faces are always restored at 8 bit (detector, parser and models take uint8 crops); with RGBS or RGBH
the rest of the frame and the background clip keep their precision, the paste blends in float.

**Note on hardware:** an NVIDIA GPU with CUDA is required.

Table of Contents
-----------------

- [Requirements](#requirements)
- [Installation](#installation)
- [Model files](#model-files)
- [Usage](#usage)
  - [restore](#restore)
  - [DetectParams](#detectparams)
  - [TrackParams](#trackparams)
- [Choosing a model](#choosing-a-model)
- [Development](#development)
- [Licences](#licences)

Requirements
------------

- Python 3.12+
- VapourSynth R66+
- PyTorch 2.5+ with CUDA, torchvision 0.20+
- numpy, scipy, OpenCV (`opencv-python`), einops

Installation
------------

From a wheel (the wheel already contains all model files, about 2.2 GB):

    pip install vsfacerestore-0.0.1-py3-none-any.whl

From a source checkout (the model files are **not** in git, see [Model files](#model-files)):

    pip install .

Model files
-----------

vsfacerestore never downloads anything. All weights are read from one folder, by default `vsfacerestore/models`
inside the installed package; `model_dir=` points `restore()` to another folder with the same layout.

The face detector and the face parser are needed for every model; of the three restoration models only the one
you select has to be present.

| File (relative to the model folder) | Needed for | Source | Size |
|---|---|---|---|
| `detection_Resnet50_Final.pth` | all models | [facexlib release v0.1.0](https://github.com/xinntao/facexlib/releases/tag/v0.1.0) | 110 MB |
| `parsing_parsenet.pth` | all models | [facexlib release v0.2.2](https://github.com/xinntao/facexlib/releases/tag/v0.2.2) | 85 MB |
| `dicface/bfr_100k.pth` | `model='dicface'` | [fudan-generative-ai/DicFace_model](https://huggingface.co/fudan-generative-ai/DicFace_model), file `CodeFormer/bfr_100k.pth` | 930 MB |
| `keep/KEEP-b76feb75.pth` | `model='keep'` | [KEEP release v1.0.0](https://github.com/jnjaby/KEEP/releases/tag/v1.0.0) | 634 MB |
| `pgtformer/model.safetensors` | `model='pgtformer'` | [kepeng/pgtformer-base](https://huggingface.co/kepeng/pgtformer-base), file `model.safetensors` | 522 MB |

The resulting layout:

```
vsfacerestore/models/
├── detection_Resnet50_Final.pth
├── parsing_parsenet.pth
├── dicface/
│   ├── LICENSE
│   └── bfr_100k.pth
├── keep/
│   ├── LICENSE
│   └── KEEP-b76feb75.pth
└── pgtformer/
    ├── LICENSE
    └── model.safetensors
```

Keep the file names exactly as listed (the KEEP file name contains the start of its hash). Only the files are
needed: the release page of KEEP also offers `gmflow_sintel-0c07dcb3.pth`, which is not used (the KEEP checkpoint
contains its own fine-tuned flow network). To find the package folder:

    python -c "import vsfacerestore, os; print(os.path.join(os.path.dirname(vsfacerestore.__file__), 'models'))"

A missing file is reported when `restore()` is called, with its name and the folder searched.

<details>
<summary>SHA-256 checksums</summary>

| File | SHA-256 |
|---|---|
| `detection_Resnet50_Final.pth` | `6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d` |
| `parsing_parsenet.pth` | `3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2` |
| `dicface/bfr_100k.pth` | `fb14b9d74f6b629d0b3f1f824cc7213eaa68a4759fbf2aa6a559a58cd7e297af` |
| `keep/KEEP-b76feb75.pth` | `b76feb75f3ed782f4e91eda85c03018241827bba8693532dc10351bab5052c1a` |
| `pgtformer/model.safetensors` | `37f9a42b9e35e0ccb89016ca6cea6728a7bf783e70b99e886ac52bb37c6855f3` |

</details>

Usage
-----

```python
import vapoursynth as vs
import vsfacerestore

core = vs.core
clip = core.bs.VideoSource("input.mkv")
clip = core.resize.Bicubic(clip, format=vs.RGB24, matrix_in_s="709")
clip = vsfacerestore.restore(clip, model="dicface")
```

### restore

Finds the faces of every frame, links them into tracks over `radius` frames before and after the current one,
filters out faces that should stay untouched, restores the rest with the selected model and pastes them back
with a tone transfer and a parsing mask.

```python
vsfacerestore.restore(clip clip[, str model = 'dicface', clip bg_clip = None, int upscale = 1, int radius = None,
                      float tone = 0.5, float weight = 1.0, int device_index = 0, bool half = True,
                      str window = 'centre', str sample = 'mean', int batch = 1, str crop = 'vfhq',
                      float sharpness = 0.04, float min_eye_dist = 5, float min_visibility = 0.45, int fade = 8,
                      DetectParams detect = None, TrackParams track = None, str model_dir = None])
```

| Parameter | Type | Options (Default) | Description |
|---|---|---|---|
| `clip` | clip | progressive RGB24, RGBS, RGBH | Clip to process. |
| `model` | str | `'dicface'`, `'keep'`, `'pgtformer'` (`'dicface'`) | Restoration model, see [Choosing a model](#choosing-a-model). |
| `bg_clip` | clip | same format, `upscale` × the size of `clip` (None) | Background the faces are pasted onto. Without it, `clip` is scaled bilinearly. |
| `upscale` | int | 1 - … (1) | Output size factor. |
| `radius` | int | DicFace / PGTFormer: (15); KEEP: 24 - … (24) | Frames before and after the current one that tracking and smoothing may look at. KEEP needs at least 24 (its 20-frame blocks must fit the window). Larger values cost memory and start-up time. |
| `tone` | float | 0.0 - … (0.5) | Structure coarser than this many source pixels is taken from the source face, so colour and lighting stay those of the source. |
| `weight` | float | 0.0 - 1.0 (1.0) | DicFace and PGTFormer: fidelity. 0 = codebook only (more invented detail), 1 = closest to the input. |
| `device_index` | int | 0 - … (0) | CUDA device. |
| `half` | bool | (True) | float16 for face detection and parsing. The restoration model always runs in float32. |
| `window` | str | `'centre'`, `'script'` (`'centre'`) | DicFace: frames fed to the network, `'centre'` = n-2 … n+2, `'script'` = n … n+4 as in the original script. |
| `sample` | str | `'mean'`, `'random'` (`'mean'`) | DicFace: `'mean'` is deterministic, `'random'` draws the codebook weights like the original (seeded). |
| `batch` | int | 1 - … (1) | DicFace and PGTFormer: faces per network call. Higher is faster and needs more VRAM. |
| `crop` | str | `'vfhq'`, `'ffhq'` (`'vfhq'`) | PGTFormer: `'vfhq'` = face at the scale of its training crops (steadiest); `'ffhq'` = the aligned crops of DicFace and KEEP (more detail on small faces, can draw fine stripes on clean large faces). |
| `sharpness` | float | 0.0 - … (0.04) | Faces with a sharpness score below this stay untouched; the blend weight ramps up to full at twice the value. A soft DV face scores about 0.08 - 0.10, VHS faces with real detail 0.2 - 0.6. Raise it (e.g. 0.14) to leave blurry faces alone. |
| `min_eye_dist` | float | 0.0 - … (5) | Faces whose eyes are closer than this many source pixels stay untouched; the blend weight ramps up between 6 and 14 px. |
| `min_visibility` | float | 0.0 - 1.0 (0.45) | Share of the face oval the parser must see as face; hidden or turned-away faces stay untouched. |
| `fade` | int | 0 - `radius` - 7 (8) | Frames over which a face fades in and out at the ends of its track. |
| `detect` | DetectParams | (None) | Detection settings, see [DetectParams](#detectparams). |
| `track` | TrackParams | (None) | Tracking, filter and blend settings, see [TrackParams](#trackparams). `sharpness`, `min_eye_dist`, `min_visibility` and `fade` override its fields. |
| `model_dir` | str | path (None) | Folder with the weights, see [Model files](#model-files). Default: `vsfacerestore/models` in the package. |

#### Examples

```python
# KEEP, 2x output on top of a separately upscaled background
bg = core.resize.Spline36(clip, clip.width * 2, clip.height * 2)
out = vsfacerestore.restore(clip, model="keep", upscale=2, bg_clip=bg)

# PGTFormer, a bit more invented detail, leave blurry faces alone
out = vsfacerestore.restore(clip, model="pgtformer", weight=0.7, sharpness=0.14)

# weights in a shared folder instead of the package
out = vsfacerestore.restore(clip, model_dir=r"D:\models\vsfacerestore")
```

### DetectParams

```python
from vsfacerestore import DetectParams
vsfacerestore.restore(clip, detect=DetectParams(scale=1.5, threshold=0.7))
```

| Parameter | Type | Options (Default) | Description |
|---|---|---|---|
| `scale` | float | (2.0) | The frame is upscaled by this before detection, which finds small faces. |
| `threshold` | float | 0.0 - 1.0 (0.6) | Detector score a face must reach. |
| `min_eye_dist` | float | (3.0) | Detections with eyes closer than this many source pixels are dropped. |
| `measure_every` | int | 1 - … (3) | Visibility and sharpness are measured on every n-th frame. |
| `measure_min_eye_dist` | float | (4.0) | Smaller detections are not measured. |

### TrackParams

```python
from vsfacerestore import TrackParams
vsfacerestore.restore(clip, track=TrackParams(max_gap=4, yaw_hi=0.6))
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_gap` | int | 8 | Frames a face may go undetected inside a track. |
| `min_length` | int | 15 | Frames a track must span inside the window. |
| `smooth` | int | 11 | Savitzky-Golay window for the landmarks. |
| `min_score` | float | 0.85 | Hard filter: median detector score below this leaves the track untouched. |
| `min_eye_dist` | float | 5.0 | Hard filter: median eye distance (source px). Same as `restore(min_eye_dist=)`. |
| `min_visibility` | float | 0.45 | Hard filter: median visibility. Same as `restore(min_visibility=)`. |
| `min_detail` | float | 0.04 | Hard filter: median sharpness. Set by `restore(sharpness=)`. |
| `eye_dist_lo`, `eye_dist_hi` | float | 6.0, 14.0 | Blend ramp over the eye distance (source px). |
| `yaw_lo`, `yaw_hi` | float | 0.25, 0.5 | Blend ramp over the head turn; profiles fade out. |
| `visibility_lo`, `visibility_hi` | float | 0.35, 0.6 | Blend ramp over the visibility. |
| `detail_lo`, `detail_hi` | float | 0.04, 0.08 | Blend ramp over the sharpness. Set by `restore(sharpness=)`. |
| `detail_smooth` | int | 31 | Moving mean over the sharpness score (frames). |
| `fade` | int | 8 | Fade in / out at the ends of a track. Same as `restore(fade=)`. |
| `alpha_smooth` | int | 9 | Moving mean over the blend weight (frames). |

Choosing a model
----------------

| Model | Frames per face | Character |
|---|---|---|
| `dicface` | 5 (window) | Default. Codebook model with Dirichlet sampling; `weight` trades fidelity against detail. |
| `keep` | blocks of 20, optical flow | Uses optical flow between frames; needs `radius` ≥ 24 and more VRAM; no `weight`. |
| `pgtformer` | 3 (n-1, n, n+1) | Trained on video face crops; `crop` selects the crop geometry. |

Development
-----------

| Path | Content |
|---|---|
| `vsfacerestore/__init__.py` | `restore()` |
| `vsfacerestore/filter.py` | the VapourSynth filter around the engine |
| `vsfacerestore/engine.py` | window engine: detections for n−K…n+K, tracks, restoration, paste of frame n |
| `vsfacerestore/tracking.py` | detections, linking, landmark smoothing, track filter, alpha |
| `vsfacerestore/paste.py` | paste-back with tone transfer and parsing mask |
| `vsfacerestore/restorer.py` | the model interface (`Restorer`, `RestoreRequest`) |
| `vsfacerestore/facehelper.py` | detection, alignment and crop geometries, parsing, per-frame paste |
| `vsfacerestore/facexlib/` | vendored networks from facexlib 0.3.0 (MIT), see its README |
| `vsfacerestore/dicface/`, `keep/`, `pgtformer/` | model architectures and their restorers, each with a README on origin and changes |
| `test/` | parity, engine, per-model and format tests (`python test/<name>.py`) |

Licences
--------

- Package code: MIT. `vsfacerestore/facexlib`: MIT (Xintao Wang).
- DicFace and KEEP model code and weights: S-Lab License 1.0, **non-commercial**.
- PGTFormer model code and weights: BSD-style licence of Xidian University, **non-commercial**.

Each model folder carries its licence file. Using the restoration models commercially is not permitted by their
licences.
