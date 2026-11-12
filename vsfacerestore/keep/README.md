# KEEP

Architecture taken from [jnjaby/KEEP](https://github.com/jnjaby/KEEP), commit `f382730` (2025-05-14), S-Lab
License 1.0 (`LICENSE`).

| File | Origin | Changes |
|---|---|---|
| `arch.py` | `basicsr/archs/keep_arch.py` | basicsr registry and logger, `pdb`, `torch.profiler` and a stray `from re import T` removed; the xformers switch removed; the parameter-count helper and the `__main__` block dropped; imports point to the files here |
| `vqgan_arch.py` | `basicsr/archs/vqgan_arch.py` | registry, logger and spectral norm replaced; the discriminators and the HQ encoder at the end of the file dropped |
| `flow.py` | `basicsr/archs/gmflow_arch.py` (`FlowGenerator`) and `basicsr/archs/arch_util.py` (`flow_warp`, `resize_flow`) | `pdb` and the `__main__` block dropped; `arch_util.py` itself is not taken (it needs `distutils` and a compiled deformable-convolution op that KEEP never uses) |
| `attention.py` | diffusers 0.11.0 `diffusers/models/attention.py` (Apache License 2.0, `LICENSE-diffusers`) | `CrossAttention`, `FeedForward`, `GELU`, `GEGLU`, `ApproximateGELU`, `AdaLayerNorm`; the xformers path removed. Newer diffusers no longer have these classes. |
| `gmflow/` | [haofeixu/gmflow](https://github.com/haofeixu/gmflow) commit `b512343`, the `gmflow` package (Apache License 2.0, `gmflow/LICENSE`) | none; KEEP has it as a git submodule, which the local checkout had not fetched |
| `__init__.py` | own | loader, block grid, crossfade, cache (`KEEPRestorer`) |

Weights: `vsfacerestore/models/keep/KEEP-b76feb75.pth` (634,093,768 bytes, SHA-256
`b76feb75f3ed782f4e91eda85c03018241827bba8693532dc10351bab5052c1a`) from the
[v1.0.0 release](https://github.com/jnjaby/KEEP/releases/tag/v1.0.0). The checkpoint contains the GMFlow weights
under `flownet.*`; they differ from the release's `gmflow_sintel-0c07dcb3.pth` (fine-tuned with KEEP), so that
file is neither needed nor shipped. The `Asian` variant is not public (its URL returns 404).

The network runs in float32 only: in float16 it outputs black (tapewake).
