# PGTFormer

Architecture taken from [kepengxu/PGTFormer](https://github.com/kepengxu/PGTFormer), commit `e78a26c`
(2025-03-21), under its non-commercial licence (`LICENSE`, Xidian University).

| File | Origin | Changes |
|---|---|---|
| `arch.py` | `archs/pgtformer_arch.py` | basicsr registry and logger and the `sys.path` tweak removed; the ResNet18 ImageNet download (`model_zoo`, unused by the released model) removed; `torch.jit.script` on `swish` removed; the YAML helper and the `__main__` block dropped; imports point to the files here |
| `tdcrqvae3_arch.py` | `archs/tdcrqvae3_arch.py`, itself based on Kakao Brain's RQ-VAE (Apache License 2.0, header kept, `LICENSE-rqvae`) | basicsr registry and `huggingface_hub`'s `PyTorchModelHubMixin` removed (the weights load from the local file); a print in the decoder removed; timm's `trunc_normal_` replaced by torch's; the YAML helper and the `__main__` block dropped |
| `rstt_layers.py` | `modules/rstt_layers.py` | timm's `DropPath` replaced by the identity (inference only) and `trunc_normal_` by torch's; `meshgrid` with explicit `indexing="ij"` (its default) |
| `codeformer_parts.py` | `archs/codeformer_arch.py` | only `calc_mean_std`, `adaptive_instance_normalization`, `_get_activation_fn` and `TransformerSALayer`; the rest of the file (and `vqgan_arch.py` it imports) is not needed |
| `__init__.py` | own | configuration of the released model, `.safetensors` reader, crop geometries, `PGTFormerRestorer` |

Not taken: `archs/rqvae_arch.py` (only `nonlinearity`, identical in `rstt_layers.py`), `archs/tdrqvae_arch.py` and
`modules/swin.py` (not imported; `swin.py` needs mmcv). Nothing is downloaded at run time.

Weights: `vsfacerestore/models/pgtformer/model.safetensors` (521,861,436 bytes, SHA-256
`37f9a42b9e35e0ccb89016ca6cea6728a7bf783e70b99e886ac52bb37c6855f3`) from
[huggingface.co/kepeng/pgtformer-base](https://huggingface.co/kepeng/pgtformer-base), read without the safetensors
package. The configuration in `__init__.py` is that repository's `config.json`. The model card names no licence; the
repository's `LICENSE` applies (`models/pgtformer/LICENSE`).

The network takes three frames (n-1, n, n+1) of 512×512 RGB in [0, 1] and returns all three; the middle one is
used, as in the original `inference.py`. It runs in float32.

## Crops

The released weights were trained on VFHQ's own face crops, not on crops warped to a landmark template. In the
original demo video the eyes are about 95 px apart (FFHQ template: 126) and the head is tilted by up to 29°. Two
crop geometries are offered (`crop=` of `restore()`):

- `vfhq` (default): the FFHQ five-point template scaled by 0.79 with the eye centre at (256, 224), fitted on the
  demo video, rotated with the face like the other models. Steadiest from frame to frame on every test clip, no
  stripe artefacts.
- `ffhq`: the aligned crops of DicFace and KEEP. Stays closest to the source's coarse tones and gives more detail on
  small faces, but draws fine vertical stripes on clean large faces (film source).

An axis-parallel crop at the VFHQ scale (like the training crops) was measured as well and brought nothing over
`vfhq`.
