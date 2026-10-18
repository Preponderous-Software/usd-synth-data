"""End-to-end pipeline for one shard: generate -> render -> label, per scene, with resume.

A shard is a contiguous range of scene indices. Scene *i* of shard *k* is rendered in a local
scratch directory, its files are put into the sink under ``shards/<k>/scenes/<i>/``, and its
``DONE`` object is put last. A re-run lists the ``DONE`` objects and skips those scenes.
:mod:`usd_synth.merge` later combines the shards into one dataset.
"""

from __future__ import annotations

import json
import math
import platform
import socket
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .labels import label_render, write_depth_png
from .render import render_batch, scene_spec
from .scenegen import scene_filename, write_scene
from .storage import Sink, scene_prefix, shard_prefix


def image_name(index: int) -> str:
    return f"scene_{index:06d}.png"


def shard_range(count: int, shard_count: int, shard_index: int) -> range:
    """Shard *k* owns ``[k * size, (k + 1) * size)`` with ``size = ceil(count / shard_count)``."""
    if count < 1 or shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError(
            f"invalid shard {shard_index} of {shard_count} for {count} scenes "
            "(need count >= 1, shard count >= 1, 0 <= shard index < shard count)"
        )
    size = math.ceil(count / shard_count)
    return range(min(shard_index * size, count), min((shard_index + 1) * size, count))


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def prepare_shard(
    sink: Sink,
    cfg: Config,
    config_dir: Path,
    dataset_seed: int,
    count: int,
    shard_count: int,
    shard_index: int,
    indices: range | None = None,
) -> range:
    """Write ``shard.json`` (or check that an existing one describes the same run)."""
    rng = indices if indices is not None else shard_range(count, shard_count, shard_index)
    info = {
        "shard_index": shard_index,
        "shard_count": shard_count,
        "start": rng.start,
        "stop": rng.stop,
        "count": count,
        "dataset_seed": dataset_seed,
        "config_hash": cfg.config_hash(),
        "config": cfg.resolved(),
        "config_dir": str(config_dir),
    }
    key = shard_prefix(shard_index) + "shard.json"
    if sink.exists(key):
        previous = json.loads(sink.get_bytes(key))
        mismatch = {k for k in info if k != "config_dir" and previous.get(k) != info[k]}
        if mismatch:
            raise ValueError(
                f"{sink.describe(key)} describes a different run (differs in {sorted(mismatch)});"
                " use a new --out to start another"
            )
    else:
        sink.put_bytes(key, json.dumps(info, indent=1, sort_keys=True).encode())
    return rng


@dataclass
class ShardResult:
    rendered: list[int] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)


def run_shard(
    cfg: Config,
    config_dir: Path,
    dataset_seed: int,
    indices,
    sink: Sink,
    blender: str,
    *,
    shard: int = 0,
    threads: int | None = None,
    batch_size: int | None = None,
) -> ShardResult:
    """Generate, render and label ``indices`` into ``shard``; skip scenes whose DONE exists."""
    result = ShardResult()
    done = sink.done_scenes(shard)
    todo = []
    for i in indices:
        (result.skipped if i in done else todo).append(i)
    size = batch_size or cfg.render.batch_size
    for start in range(0, len(todo), size):
        batch = todo[start : start + size]
        batch_started = time.time()
        with tempfile.TemporaryDirectory(prefix="usd-synth-shard-") as scratch:
            scratch = Path(scratch)
            specs, gen_s = [], {}
            for i in batch:
                t0 = time.perf_counter()
                usd = write_scene(cfg, dataset_seed, i, scratch / "scenes", config_dir)
                gen_s[i] = time.perf_counter() - t0
                specs.append(scene_spec(usd, scratch / "render" / str(i)))
            rendered = render_batch(cfg, specs, blender, threads=threads)
            per_start = rendered.blender_start_s / len(batch)
            for i, info in zip(batch, rendered.scenes, strict=True):
                _finish_scene(
                    cfg, sink, shard, i, scratch, info, gen_s[i], per_start, batch_started
                )
                result.rendered.append(i)
    return result


def _finish_scene(cfg, sink, shard, i, scratch, info, gen_s, blender_start_s, started) -> None:
    usd = scratch / "scenes" / scene_filename(i)
    render_dir = scratch / "render" / str(i)
    out = scratch / "out" / str(i)
    out.mkdir(parents=True)

    t0 = time.perf_counter()
    annotations, depth = label_render(usd, render_dir, cfg)
    label_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    write_depth_png(depth, cfg.labels.depth.max_m, out / "depth.png")
    record = {
        "scene_index": i,
        "file_name": f"images/{image_name(i)}",
        "depth_file": f"depth/{image_name(i)}",
        "scene_file": f"scenes/{scene_filename(i)}",
        "width": cfg.image.width,
        "height": cfg.image.height,
        "annotations": annotations,
    }
    prefix = scene_prefix(shard, i)
    files = {"scene.usda": usd, "rgb.png": render_dir / "rgb.png", "depth.png": out / "depth.png"}
    if cfg.labels.depth.keep_exr:
        files["aux.exr"] = render_dir / "aux.exr"
    for name, path in files.items():
        sink.put_file(prefix + name, path)
    sink.put_bytes(prefix + "labels.json", json.dumps(record, sort_keys=True).encode())
    write_s = time.perf_counter() - t1
    timings = {
        "generate": gen_s,
        "blender_start": blender_start_s,
        "import": info["timings"]["import"],
        "render": info["timings"]["render"],
        "label": label_s,
        "write": write_s + info["timings"]["write"],
    }
    render_meta = {k: info[k] for k in ("blender", "device", "threads", "samples", "seed")}
    render_meta.update(cpu_model=cpu_model(), host=socket.gethostname())
    meta = {"timings": timings, "render": render_meta, "started": started, "finished": time.time()}
    sink.put_bytes(prefix + "timings.json", json.dumps(meta, sort_keys=True).encode())
    sink.mark_done(shard, i)  # put last: the completion signal
