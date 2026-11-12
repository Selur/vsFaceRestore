# Vendored parts of facexlib

Copied from [facexlib 0.3.0](https://github.com/xinntao/facexlib) (Xintao Wang, MIT, see `LICENSE`), the wheel
`facexlib-0.3.0-py3-none-any.whl` from PyPI.

| File | Origin | Changes |
|---|---|---|
| `detection/retinaface.py` | `facexlib/detection/retinaface.py` | imports made relative |
| `detection/retinaface_net.py` | `facexlib/detection/retinaface_net.py` | none |
| `detection/retinaface_utils.py` | `facexlib/detection/retinaface_utils.py` | `PriorBox.forward` caches the anchors per image size |
| `detection/align_trans.py`, `detection/matlab_cp2tform.py` | same names | none (only needed by `RetinaFace.align_multi`) |
| `parsing/parsenet.py` | `facexlib/parsing/parsenet.py` | none |
| `__init__.py` | own | loads the two networks from a file path, no download |

Not taken: the download helpers, BiSeNet, alignment, head pose, matting, recognition, tracking, visualisation,
`FaceRestoreHelper` (its logic lives in `vsfacerestore/facehelper.py`).

Weights (`vsfacerestore/models/`), from the facexlib releases:

| File | Release | SHA-256 |
|---|---|---|
| `detection_Resnet50_Final.pth` | v0.1.0 | `6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d` |
| `parsing_parsenet.pth` | v0.2.2 | `3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2` |
