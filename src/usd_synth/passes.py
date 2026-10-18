"""Read the auxiliary render passes the Blender worker writes to ``aux.exr``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Passes:
    depth: np.ndarray  # (H, W) float32, planar depth in metres along the view axis
    index: np.ndarray  # (H, W) int32, Cycles object pass index (instance ID; 0 = unlabelled)


def _find(channels: dict, *suffixes: str) -> np.ndarray:
    for name, channel in channels.items():
        if name.endswith(suffixes):
            return channel.pixels
    raise KeyError(f"no channel ending in {suffixes} (have {sorted(channels)})")


def read_passes(path: Path | str) -> Passes:
    import OpenEXR

    channels = {}
    with OpenEXR.File(str(path), separate_channels=True) as exr:
        for part in exr.parts:
            channels.update(part.channels)
        depth = np.array(_find(channels, ".Depth.Z", ".Z"), dtype=np.float32)
        raw_index = np.array(_find(channels, "Object Index.X", "IndexOB.X"), dtype=np.float64)
    index = np.rint(raw_index).astype(np.int32)
    if not np.allclose(raw_index, index, rtol=0.0, atol=1e-3):
        raise ValueError(f"{path}: object index pass is not integral (edge blending?)")
    return Passes(depth=depth, index=index)
