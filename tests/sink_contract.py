"""Shared helpers for storage-sink tests: fake shard outputs written through the Sink API only."""

from __future__ import annotations

import json

import numpy as np

from usd_synth.config import Config
from usd_synth.labels import encode_rle
from usd_synth.storage import Sink, scene_prefix, shard_prefix


def fake_shard(sink: Sink, cfg: Config, k: int, n: int, start: int, stop: int, count: int,
               seed: int = 7) -> None:  # fmt: skip
    info = {
        "shard_index": k, "shard_count": n, "start": start, "stop": stop, "count": count,
        "dataset_seed": seed, "config_hash": cfg.config_hash(), "config": cfg.resolved(),
        "config_dir": "/cfg",
    }  # fmt: skip
    sink.put_bytes(shard_prefix(k) + "shard.json", json.dumps(info).encode())


def fake_scene(sink: Sink, k: int, i: int, done: bool = True) -> None:
    """A finished scene as the pipeline would leave it (DONE last), with one annotation."""
    p = scene_prefix(k, i)
    mask = np.zeros((12, 16), dtype=bool)
    mask[i % 6 : i % 6 + 4, 2:6] = True
    record = {
        "scene_index": i, "file_name": f"images/scene_{i:06d}.png",
        "depth_file": f"depth/scene_{i:06d}.png", "scene_file": f"scenes/scene_{i:06d}.usda",
        "width": 16, "height": 12,
        "annotations": [{"category_id": 1, "segmentation": encode_rle(mask), "area": 16.0,
                         "bbox": [2.0, float(i % 6), 4.0, 4.0], "iscrowd": 0, "instance_id": 1}],
    }  # fmt: skip
    timings = {s: 0.1 * (i + 1) for s in ("generate", "blender_start", "import", "render",
                                          "label", "write")}  # fmt: skip
    meta = {"timings": timings, "started": 1000.0 + i, "finished": 1001.0 + i,
            "render": {"blender": "5.2.2", "device": "cpu", "threads": 2, "samples": 4,
                       "seed": 1, "cpu_model": "test", "host": f"pod-{k}"}}  # fmt: skip
    for name in ("rgb.png", "depth.png", "scene.usda"):
        sink.put_bytes(p + name, f"{name}:{i}".encode())
    sink.put_bytes(p + "labels.json", json.dumps(record).encode())
    sink.put_bytes(p + "timings.json", json.dumps(meta).encode())
    if done:
        sink.mark_done(k, i)
