"""End-to-end local pipeline: generate -> render -> label, per scene, with resume.

Work for scene *i* of shard *k* lands in ``<out>/shards/<k>/scenes/<i>/``; the ``DONE`` file is
written last, so a scene directory without it is redone on the next run. :func:`assemble` then
copies the per-scene outputs into the dataset layout and writes ``annotations.json`` and
``manifest.json``.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .coco import build_coco, build_manifest, dump_json
from .config import Config
from .labels import label_render, write_depth_png
from .render import render_batch, scene_spec
from .scenegen import scene_filename, write_scene

RUN_FILE = "run.json"


def scene_dir(out: Path, shard: int, index: int) -> Path:
    return Path(out) / "shards" / str(shard) / "scenes" / f"{index:06d}"


def image_name(index: int) -> str:
    return f"scene_{index:06d}.png"


@dataclass
class ShardResult:
    rendered: list[int] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)
    blender_start_s: list[float] = field(default_factory=list)


def check_run(out: Path, cfg: Config, dataset_seed: int, count: int) -> None:
    """Refuse to resume into an output directory made with another config, seed or count."""
    out.mkdir(parents=True, exist_ok=True)
    run = {"config_hash": cfg.config_hash(), "dataset_seed": dataset_seed, "count": count}
    path = out / RUN_FILE
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != run:
            raise ValueError(
                f"{out} holds a different run ({previous}); use a new --out to start another"
            )
    else:
        path.write_text(json.dumps(run, sort_keys=True) + "\n", encoding="utf-8")


def run_shard(
    cfg: Config,
    config_dir: Path,
    dataset_seed: int,
    indices: list[int],
    out: Path,
    blender: str,
    *,
    shard: int = 0,
    threads: int | None = None,
    batch_size: int | None = None,
) -> ShardResult:
    """Generate, render and label ``indices`` into shard ``shard``; skip scenes already DONE."""
    result = ShardResult()
    todo = []
    for i in indices:
        (result.skipped if (scene_dir(out, shard, i) / "DONE").exists() else todo).append(i)
    size = batch_size or cfg.render.batch_size
    for start in range(0, len(todo), size):
        batch = todo[start : start + size]
        with tempfile.TemporaryDirectory(prefix="usd-synth-render-") as scratch:
            scratch = Path(scratch)
            specs, gen_s = [], {}
            for i in batch:
                t0 = time.perf_counter()
                usd = write_scene(cfg, dataset_seed, i, scratch / "scenes", config_dir)
                gen_s[i] = time.perf_counter() - t0
                specs.append(scene_spec(usd, scratch / "render" / str(i)))
            rendered = render_batch(cfg, specs, blender, threads=threads)
            result.blender_start_s.append(rendered.blender_start_s)
            per_start = rendered.blender_start_s / len(batch)
            for i, info in zip(batch, rendered.scenes, strict=True):
                _finish_scene(cfg, out, shard, i, scratch, info, gen_s[i], per_start)
                result.rendered.append(i)
    return result


def _finish_scene(cfg, out, shard, i, scratch, info, gen_s, blender_start_s) -> None:
    dest = scene_dir(out, shard, i)
    if dest.exists():
        shutil.rmtree(dest)  # leftovers of an interrupted attempt
    dest.mkdir(parents=True)
    usd = scratch / "scenes" / scene_filename(i)
    render_dir = scratch / "render" / str(i)

    t0 = time.perf_counter()
    annotations, depth = label_render(usd, render_dir, cfg)
    label_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    shutil.copyfile(usd, dest / "scene.usda")
    shutil.copyfile(render_dir / "rgb.png", dest / "rgb.png")
    write_depth_png(depth, cfg.labels.depth.max_m, dest / "depth.png")
    if cfg.labels.depth.keep_exr:
        shutil.copyfile(render_dir / "aux.exr", dest / "aux.exr")
    record = {
        "scene_index": i,
        "file_name": f"images/{image_name(i)}",
        "depth_file": f"depth/{image_name(i)}",
        "scene_file": f"scenes/{scene_filename(i)}",
        "width": cfg.image.width,
        "height": cfg.image.height,
        "annotations": annotations,
    }
    (dest / "labels.json").write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
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
    (dest / "timings.json").write_text(
        json.dumps({"timings": timings, "render": render_meta, "finished": time.time()}),
        encoding="utf-8",
    )
    (dest / "DONE").write_text("", encoding="utf-8")  # written last: the completion signal


def assemble(
    cfg: Config, config_dir: Path, dataset_seed: int, out: Path, indices: list[int], shard: int = 0
) -> dict:
    """Copy finished scenes of ``shard`` into the dataset layout and write the COCO files."""
    records, render_info = [], {}
    for sub in ("images", "depth", "scenes"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for i in indices:
        src = scene_dir(out, shard, i)
        if not (src / "DONE").exists():
            raise ValueError(f"scene {i} is not complete in {src}")
        record = json.loads((src / "labels.json").read_text(encoding="utf-8"))
        shutil.copyfile(src / "rgb.png", out / record["file_name"])
        shutil.copyfile(src / "depth.png", out / record["depth_file"])
        shutil.copyfile(src / "scene.usda", out / record["scene_file"])
        records.append(record)
        render_info = render_info or json.loads((src / "timings.json").read_text())["render"]
    coco = build_coco(records, cfg)
    dump_json(coco, out / "annotations.json")
    dump_json(
        build_manifest(cfg, dataset_seed, records, render_info, str(config_dir)),
        out / "manifest.json",
    )
    return coco
