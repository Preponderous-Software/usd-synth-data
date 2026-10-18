"""Combine shard outputs into one dataset, through the :mod:`usd_synth.storage` interface only.

Checks that every shard describes the same run, that shard ranges neither overlap nor leave
gaps, and that every scene has its ``DONE`` object. Then copies each scene's files into the
dataset layout and writes ``annotations.json`` (IDs ordered by scene index, so any shard split
gives the same file), ``manifest.json`` and ``throughput.json``.
"""

from __future__ import annotations

import json

from .coco import build_coco, build_manifest
from .config import parse_config
from .storage import Sink, scene_prefix, shard_prefix
from .throughput import summarize


class MergeError(ValueError):
    pass


class MergeNotReady(MergeError):
    """Shards are missing or unfinished — possibly still running (``merge --wait`` retries)."""


def _dumps(data: dict) -> bytes:
    return (json.dumps(data, indent=1, sort_keys=True) + "\n").encode()


def read_shards(sink: Sink) -> list[dict]:
    infos = [json.loads(sink.get_bytes(shard_prefix(k) + "shard.json")) for k in sink.shards()]
    if not infos:
        raise MergeNotReady(f"no shards found under {sink.describe('shards/')}")
    first = infos[0]
    for info in infos[1:]:
        for key in ("config_hash", "dataset_seed", "count", "shard_count"):
            if info[key] != first[key]:
                raise MergeError(
                    f"shard {info['shard_index']} has {key}={info[key]!r} but shard "
                    f"{first['shard_index']} has {first[key]!r}; they are not one run"
                )
    return infos


def check_ranges(infos: list[dict], expected: range) -> None:
    """Shard ranges must be disjoint and cover ``expected`` exactly."""
    spans = sorted(
        (i["start"], i["stop"], i["shard_index"]) for i in infos if i["stop"] > i["start"]
    )
    for (s0, e0, k0), (s1, e1, k1) in zip(spans, spans[1:], strict=False):
        if s1 < e0:
            raise MergeError(f"shards {k0} [{s0}, {e0}) and {k1} [{s1}, {e1}) overlap")
    covered = set()
    for s, e, _ in spans:
        covered.update(range(s, e))
    missing = sorted(set(expected) - covered)
    extra = sorted(covered - set(expected))
    if missing:
        raise MergeNotReady(f"scene ranges are missing {len(missing)} scene(s): {_ranges(missing)}")
    if extra:
        raise MergeError(f"shards cover scenes outside the run: {_ranges(extra)}")


def _ranges(values: list[int]) -> str:
    out, start = [], values[0]
    for a, b in zip(values, values[1:] + [None], strict=False):
        if b != a + 1:
            out.append(f"{start}" if start == a else f"{start}-{a}")
            start = b
    return ", ".join(out)


def merge(sink: Sink, expected: range | None = None) -> dict:
    """Merge every shard in ``sink`` into the dataset at ``sink.dataset_prefix``."""
    infos = read_shards(sink)
    first = infos[0]
    cfg = parse_config(first["config"])
    if cfg.config_hash() != first["config_hash"]:
        raise MergeError("shard config does not match its recorded hash")
    expected = expected if expected is not None else range(first["count"])
    check_ranges(infos, expected)

    owner = {}
    for info in infos:
        for i in range(info["start"], info["stop"]):
            owner[i] = info["shard_index"]
    done = {info["shard_index"]: sink.done_scenes(info["shard_index"]) for info in infos}
    incomplete = [i for i in expected if i not in done[owner[i]]]
    if incomplete:
        raise MergeNotReady(f"{len(incomplete)} scene(s) not complete: {_ranges(incomplete)}")

    ds = sink.dataset_prefix
    records, scenes = [], []
    for i in expected:
        prefix = scene_prefix(owner[i], i)
        record = json.loads(sink.get_bytes(prefix + "labels.json"))
        sink.copy(prefix + "rgb.png", ds + record["file_name"])
        sink.copy(prefix + "depth.png", ds + record["depth_file"])
        sink.copy(prefix + "scene.usda", ds + record["scene_file"])
        records.append(record)
        scenes.append({"shard": owner[i], **json.loads(sink.get_bytes(prefix + "timings.json"))})

    coco = build_coco(records, cfg)
    manifest = build_manifest(
        cfg, first["dataset_seed"], records, scenes[0]["render"], first["config_dir"]
    )
    manifest["shards"] = [
        {k: info[k] for k in ("shard_index", "shard_count", "start", "stop")}
        for info in sorted(infos, key=lambda i: i["shard_index"])
    ]
    throughput = summarize(scenes, first["config"])
    sink.put_bytes(ds + "annotations.json", _dumps(coco))
    sink.put_bytes(ds + "manifest.json", _dumps(manifest))
    sink.put_bytes(ds + "throughput.json", _dumps(throughput))
    return {"coco": coco, "manifest": manifest, "throughput": throughput}
