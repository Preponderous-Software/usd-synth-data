"""Assemble per-scene label records into a COCO dataset and a provenance manifest; validate.

A *scene record* (``labels.json`` next to each rendered scene) holds the image entry and its
annotations without COCO IDs. IDs are assigned here, ordered by scene index: the image ID is the
scene index and annotation IDs count up from 1 in (scene index, instance ID) order. Any split of
the same scenes into shards therefore assembles to the same ``annotations.json``.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path

import numpy as np

from . import __version__
from .config import Config
from .labels import coco_categories
from .seeds import render_seed, scene_seed


def build_coco(
    records: list[dict],
    cfg: Config | None,
    description: str = "usd-synth-data",
    categories: list[dict] | None = None,
) -> dict:
    """COCO dataset from scene records. Categories come from ``cfg`` unless ``categories`` (a
    COCO ``categories`` list) is given, which lets scenarios that are not driven by a
    :class:`Config` (such as ``examples/minecraft``) reuse the writer; ``cfg`` may then be None.
    """
    if categories is None:
        if cfg is None:
            raise ValueError("build_coco needs a config or an explicit categories list")
        categories = coco_categories(cfg)
    indices = [r["scene_index"] for r in records]
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate scene index in records")
    images, annotations = [], []
    for rec in sorted(records, key=lambda r: r["scene_index"]):
        image_id = rec["scene_index"]
        images.append(
            {
                "id": image_id,
                "file_name": rec["file_name"],
                "width": rec["width"],
                "height": rec["height"],
                "depth_file": rec["depth_file"],
                "scene_file": rec["scene_file"],
                "scene_index": rec["scene_index"],
            }
        )
        for ann in sorted(rec["annotations"], key=lambda a: a["instance_id"]):
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, **ann})
    return {
        "info": {"description": description, "version": __version__},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def build_manifest(
    cfg: Config,
    dataset_seed: int,
    records: list[dict],
    render_info: dict,
    config_dir: str,
) -> dict:
    """Everything needed to reproduce any image: config, seeds, tool versions, provenance."""
    return {
        "tool": "usd-synth-data",
        "tool_version": __version__,
        "config": cfg.resolved(),
        "config_hash": cfg.config_hash(),
        "config_dir": config_dir,
        "dataset_seed": dataset_seed,
        "engine": render_info.get("engine", cfg.render.engine),
        "blender": render_info.get("blender"),
        **(
            {k: render_info.get(k) for k in ("strata", "strata_abi", "gl", "depth", "ignored")}
            if cfg.render.engine == "strata"
            else {}
        ),
        "device": render_info.get("device"),
        "samples": cfg.render.samples if cfg.render.engine == "cycles" else None,
        "denoise": cfg.render.denoise if cfg.render.engine == "cycles" else None,
        "resolution": [cfg.image.width, cfg.image.height],
        "images": [
            {
                "image_id": r["scene_index"],
                "scene_index": r["scene_index"],
                "scene_seed": scene_seed(dataset_seed, r["scene_index"]),
                "render_seed": render_seed(dataset_seed, r["scene_index"]),
                "file_name": r["file_name"],
                "scene_file": r["scene_file"],
            }
            for r in sorted(records, key=lambda r: r["scene_index"])
        ],
    }


def dump_json(data: dict, path: Path | str) -> None:
    Path(path).write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def validate_coco(path: Path | str) -> list[str]:
    """MVP acceptance criterion 7: problems found in ``annotations.json`` (empty = valid).

    Loads with ``pycocotools.coco.COCO``; checks every annotation references an existing image
    and category, every mask decodes to its image's size, and every box encloses its mask.
    """
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO

    errors = []
    with contextlib.redirect_stdout(io.StringIO()):  # COCO() prints progress chatter
        coco = COCO(str(path))
    images, cats = coco.imgs, coco.cats
    for ann in coco.dataset["annotations"]:
        aid = ann["id"]
        img = images.get(ann["image_id"])
        if img is None:
            errors.append(f"annotation {aid}: unknown image_id {ann['image_id']}")
            continue
        if ann["category_id"] not in cats:
            errors.append(f"annotation {aid}: unknown category_id {ann['category_id']}")
        try:
            mask = mask_utils.decode(coco.annToRLE(ann))
        except (ValueError, TypeError) as err:
            errors.append(f"annotation {aid}: mask does not decode to the image size ({err})")
            continue
        if mask.shape != (img["height"], img["width"]):
            errors.append(f"annotation {aid}: mask {mask.shape} != image size")
            continue
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            errors.append(f"annotation {aid}: empty mask")
            continue
        numbers = [*ann["bbox"], ann["area"]]
        if len(ann["bbox"]) != 4 or not all(
            isinstance(v, int | float) and math.isfinite(v) for v in numbers
        ):
            errors.append(f"annotation {aid}: non-finite or malformed bbox/area {numbers}")
            continue
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            errors.append(f"annotation {aid}: bbox {ann['bbox']} has no extent")
            continue
        if xs.min() < x or ys.min() < y or xs.max() + 1 > x + w or ys.max() + 1 > y + h:
            errors.append(f"annotation {aid}: bbox {ann['bbox']} does not enclose its mask")
        if abs(float(mask.sum()) - ann["area"]) > 0.5:
            errors.append(f"annotation {aid}: area {ann['area']} != mask pixels {mask.sum()}")
    return errors
