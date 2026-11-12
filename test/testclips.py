"""Test clips: BestSource, deinterlaced where the source is interlaced, converted to RGB24.

The filter expects progressive frames (the host deinterlaces first, as for any other filter), so the tests do the
same: Bwdif at single rate keeping the first field, which leaves the frame numbers of the fixed test frames alone.
"""

from __future__ import annotations

import vapoursynth as vs

core = vs.core
core.std.LoadPlugin(r"F:\Vapoursynth\Lib\site-packages\vapoursynth\plugins2\libbestsource.dll")
core.std.LoadPlugin(r"F:\Vapoursynth\Lib\site-packages\vapoursynth\plugins2\bwdif.dll")

CLIPS = "G:/TestClips&Co/"

# file -> colour matrix; the field order comes from the source's frame properties
MATRIX = {
    "20010901_Wedding. 04.avi": "470bg",
    "1983-i.mkv": "470bg",
    "Class clip.avi": "170m",
    "In_America_(2002).mkv": "709",
}


def source(name: str, matrix: str | None = None) -> vs.VideoNode:
    clip = core.bs.VideoSource(CLIPS + name, cachemode=0)
    field_based = clip.get_frame(0).props.get("_FieldBased", 0)
    if field_based == 1:  # bottom field first: keep the bottom field
        clip = core.bwdif.Bwdif(clip, field=0)
    elif field_based == 2:  # top field first: keep the top field
        clip = core.bwdif.Bwdif(clip, field=1)
    return clip.resize.Bicubic(format=vs.RGB24, matrix_in_s=matrix or MATRIX[name])
