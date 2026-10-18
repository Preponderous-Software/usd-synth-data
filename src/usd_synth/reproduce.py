"""Regenerate one image of a dataset from its manifest and compare it with the original.

The stage and every geometry-derived label are byte-reproducible; the RGB image is reproducible
to the extent the renderer is, so it is compared by mean absolute difference against
:data:`RGB_MAD_THRESHOLD` (8-bit levels).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import parse_config
from .pipeline import assemble, image_name, run_shard
from .scenegen import scene_filename

RGB_MAD_THRESHOLD = 1.0  # mean absolute difference, 0-255 levels, averaged over all channels


def _annotations_for(coco: dict, image_id: int) -> list[dict]:
    anns = [a for a in coco["annotations"] if a["image_id"] == image_id]
    return [{k: v for k, v in a.items() if k != "id"} for a in anns]


def reproduce(dataset: Path, image_id: int, out: Path, blender: str, threads=None) -> dict:
    dataset, out = Path(dataset), Path(out)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    cfg = parse_config(manifest["config"])
    if cfg.config_hash() != manifest["config_hash"]:
        raise ValueError("manifest config does not match its recorded hash")
    entry = next((e for e in manifest["images"] if e["image_id"] == image_id), None)
    if entry is None:
        raise ValueError(f"image id {image_id} is not in {dataset / 'manifest.json'}")
    index, seed = entry["scene_index"], manifest["dataset_seed"]
    config_dir = Path(manifest["config_dir"])

    run_shard(cfg, config_dir, seed, [index], out, blender, threads=threads)
    new_coco = assemble(cfg, config_dir, seed, out, [index])
    old_coco = json.loads((dataset / "annotations.json").read_text(encoding="utf-8"))

    from PIL import Image

    old_rgb = np.asarray(Image.open(dataset / "images" / image_name(index)), dtype=np.float64)
    new_rgb = np.asarray(Image.open(out / "images" / image_name(index)), dtype=np.float64)
    diff = np.abs(old_rgb - new_rgb)
    report = {
        "image_id": image_id,
        "scene_index": index,
        "usda_identical": (dataset / "scenes" / scene_filename(index)).read_bytes()
        == (out / "scenes" / scene_filename(index)).read_bytes(),
        "annotations_identical": _annotations_for(old_coco, image_id)
        == _annotations_for(new_coco, image_id),
        "rgb_mean_abs_diff": float(diff.mean()),
        "rgb_max_abs_diff": float(diff.max()),
        "rgb_threshold": RGB_MAD_THRESHOLD,
    }
    report["ok"] = (
        report["usda_identical"]
        and report["annotations_identical"]
        and report["rgb_mean_abs_diff"] <= RGB_MAD_THRESHOLD
    )
    (out / "reproduce.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report
