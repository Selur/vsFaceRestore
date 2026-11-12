# DicFace

Architecture taken from [fudan-generative-vision/DicFace](https://github.com/fudan-generative-vision/DicFace),
commit `df08bb7` (2025-07-25).

| File | Origin | Changes |
|---|---|---|
| `vqgan_arch.py` | `basicsr/archs/vqgan_arch.py` | basicsr registry and logger replaced |
| `arch.py` | `basicsr/archs/dir_dist_codeformer_multiscale_arch.py` | registry, logger and two prints removed; `forward(..., sample=)`: `'random'` draws the codebook weights from the Dirichlet distribution as the original does, `'mean'` takes its expectation |
| `__init__.py` | own | loader and the `Restorer` for the window engine |

Weights: `vsfacerestore/models/dicface/bfr_100k.pth` (929,937,246 bytes, SHA-256
`fb14b9d74f6b629d0b3f1f824cc7213eaa68a4759fbf2aa6a559a58cd7e297af`) from
[huggingface.co/fudan-generative-ai/DicFace_model](https://huggingface.co/fudan-generative-ai/DicFace_model),
file `CodeFormer/bfr_100k.pth`.

## Licence

The repository's README says "open sourced under NTU S-Lab License 1.0" but ships no licence file; the Hugging
Face model card says `license: mit`. Treated as S-Lab License 1.0 (non-commercial), the stricter reading; the
text in `LICENSE` is the S-Lab License 1.0 as published with KEEP (same licence, copyright S-Lab).
