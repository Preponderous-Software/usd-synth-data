"""Ground truth from render passes: instance masks (COCO RLE), tight boxes, depth PNGs.

Labels come from the Object Index pass, not from re-projecting geometry, so occlusion is exact.
Per-scene labels are written without COCO IDs; IDs are assigned when a dataset is assembled
(:mod:`usd_synth.coco`), ordered by scene index, so they do not depend on how work was split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config

NS = "userProperties:usd_synth:"
DISTRACTOR_CATEGORY = "distractor"
NO_HIT_M = 1e9  # Cycles writes ~1e10 for rays that hit nothing


@dataclass(frozen=True)
class InstanceInfo:
    instance_id: int
    kind: str  # "object" | "distractor"
    category_name: str
    projected_area_px: float  # unoccluded, unclipped silhouette estimate (0 = unknown)


def read_instances(usd_path: Path | str) -> list[InstanceInfo]:
    """Labelled and distractor instances recorded in a generated stage."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    out = []
    for parent in ("/World/Objects", "/World/Distractors"):
        root = stage.GetPrimAtPath(parent)
        if not root:
            continue
        for prim in root.GetChildren():
            area = prim.GetAttribute(NS + "projected_area_px").Get()
            out.append(
                InstanceInfo(
                    int(prim.GetAttribute(NS + "instance_id").Get()),
                    str(prim.GetAttribute(NS + "kind").Get()),
                    str(prim.GetAttribute(NS + "category_name").Get()),
                    float(area) if area is not None else 0.0,
                )
            )
    return out


def coco_categories(cfg: Config) -> list[dict]:
    """COCO categories: the config's labelled categories in order, plus ``distractor`` when
    distractors are labelled ``ignore`` (their annotations are ``iscrowd`` regions)."""
    names = cfg.categories()
    if cfg.distractors.label == "ignore":
        names = [*names, DISTRACTOR_CATEGORY]
    return [{"id": i + 1, "name": n, "supercategory": "object"} for i, n in enumerate(names)]


def encode_rle(mask: np.ndarray) -> dict:
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(s) for s in rle["size"]], "counts": rle["counts"].decode("ascii")}


def tight_bbox(mask: np.ndarray) -> list[float]:
    """COCO ``[x, y, w, h]`` of the visible pixels (pixel edges, so a 1-pixel mask is 1x1)."""
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def label_scene(index: np.ndarray, instances: list[InstanceInfo], cfg: Config) -> list[dict]:
    """Annotations (without ``id``/``image_id``) for one rendered scene.

    An instance is dropped when it has fewer than ``labels.min_visible_pixels`` visible pixels,
    or when its visible area is below ``labels.min_visible_fraction`` of its unoccluded,
    unclipped projected area (so heavily occluded and heavily truncated instances both go).
    Distractors become ``iscrowd`` regions (``label: ignore``) or are omitted (``label: omit``).
    """
    labels = cfg.labels
    cat_ids = {c["name"]: c["id"] for c in coco_categories(cfg)}
    out = []
    for inst in sorted(instances, key=lambda i: i.instance_id):
        is_distractor = inst.kind == "distractor"
        if is_distractor and cfg.distractors.label == "omit":
            continue
        mask = index == inst.instance_id
        visible = int(mask.sum())
        if visible == 0 or visible < labels.min_visible_pixels:
            continue
        if inst.projected_area_px > 0:
            if visible / inst.projected_area_px < labels.min_visible_fraction:
                continue
        name = DISTRACTOR_CATEGORY if is_distractor else inst.category_name
        out.append(
            {
                "category_id": cat_ids[name],
                "segmentation": encode_rle(mask),
                "area": float(visible),
                "bbox": tight_bbox(mask),
                "iscrowd": 1 if is_distractor else 0,
                "instance_id": inst.instance_id,
            }
        )
    return out


def depth_to_mm(depth_m: np.ndarray, max_m: float) -> np.ndarray:
    """16-bit millimetres; 0 = no hit, non-finite, or beyond ``max_m``."""
    depth = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0) & (depth <= max_m) & (depth < NO_HIT_M)
    mm = np.where(valid, np.rint(depth * 1000.0), 0)
    return np.clip(mm, 0, 65535).astype(np.uint16)


def write_depth_png(depth_m: np.ndarray, max_m: float, path: Path | str) -> None:
    from PIL import Image

    Image.fromarray(depth_to_mm(depth_m, max_m)).save(path)  # uint16 -> 16-bit greyscale PNG


def read_depth_png(path: Path | str) -> np.ndarray:
    """Depth in metres (NaN where the PNG holds 0)."""
    from PIL import Image

    mm = np.asarray(Image.open(path)).astype(np.float64)
    return np.where(mm > 0, mm / 1000.0, np.nan)


def label_render(usd_path: Path | str, render_dir: Path | str, cfg: Config):
    """Annotations and planar depth (metres) for one scene rendered into ``render_dir``."""
    from .passes import read_passes

    passes = read_passes(Path(render_dir) / "aux.exr")
    return label_scene(passes.index, read_instances(usd_path), cfg), passes.depth
