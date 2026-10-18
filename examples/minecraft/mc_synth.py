"""Synthetic, auto-labelled training data from a Minecraft world.

Renders randomized views over a `Minecraft-USD-Exporter`_ terrain export with the stock
usd-synth-data Blender worker and writes, per view, an RGB image, a semantic mask, a depth
map and COCO instance annotations (boxes + RLE masks) for the block classes in the config.

    python examples/minecraft/mc_synth.py out/world.usda \\
        --config examples/minecraft/minecraft.yaml --views 24 --seed 8675309 \\
        --out out/mc --threads 4

Labels are read from the render's Object Index pass (``aux.exr``): each connected component
of a labelled class is one mesh with its own pass index, so occlusion, the image border and
the camera's view are all handled by the renderer. Nothing is drawn or projected by hand.

Output (``--out``)::

    scene.usdc               terrain as meshes; labelled components under /World/Objects
    views/view_NNNN.usda     sublayers scene.usdc; camera, sun and sky of one view
    images/NNNN.png          RGB
    semantic/NNNN.png        8-bit class map (0 = unlabelled, k = k-th class in the config)
    depth/NNNN.png           16-bit planar depth in centimetres (0 = sky / beyond 655 m)
    annotations.json         COCO: images, instance annotations, categories (validated)
    meta.json                views, classes, instance counts, render settings, images/hour

.. _Minecraft-USD-Exporter: https://github.com/Dans-Plugins/Minecraft-USD-Exporter
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mc_scene  # noqa: E402

from usd_synth.coco import build_coco, dump_json, validate_coco  # noqa: E402
from usd_synth.config import ImageConfig, RenderConfig  # noqa: E402
from usd_synth.labels import encode_rle, read_instances, tight_bbox  # noqa: E402
from usd_synth.passes import read_passes  # noqa: E402
from usd_synth.pipeline import cpu_model  # noqa: E402
from usd_synth.render import find_blender, render_batch, scene_spec  # noqa: E402

DEPTH_SCALE_M = 0.01  # depth PNG unit: centimetres (outdoor ranges exceed 16-bit millimetres)


def label_view(index: np.ndarray, instances, categories: list[str], min_pixels: int):
    """COCO annotations (no IDs yet) and the semantic class map for one rendered view."""
    cat_of = {i.instance_id: categories.index(i.category_name) + 1 for i in instances}
    lut = np.zeros(max(cat_of, default=0) + 1, dtype=np.uint8)
    for iid, cat in cat_of.items():
        lut[iid] = cat
    semantic = lut[np.clip(index, 0, len(lut) - 1)]
    semantic[(index < 0) | (index >= len(lut))] = 0
    ids, counts = np.unique(index[index > 0], return_counts=True)
    anns = []
    for iid, n in zip(ids.tolist(), counts.tolist(), strict=True):
        if iid not in cat_of or n < min_pixels:
            continue
        mask = index == iid
        anns.append(
            {
                "category_id": cat_of[iid],
                "segmentation": encode_rle(mask),
                "area": float(n),
                "bbox": tight_bbox(mask),
                "iscrowd": 0,
                "instance_id": int(iid),
            }
        )
    return anns, semantic


def palette_png(classes: np.ndarray, path: Path, colours: list[tuple[int, int, int]]) -> None:
    im = Image.fromarray(classes.astype(np.uint8), mode="P")
    flat = [0, 0, 0] + [c for rgb in colours for c in rgb]
    im.putpalette(flat + [0] * (768 - len(flat)))
    im.save(path, optimize=True)


CLASS_COLOURS = [(46, 160, 67), (64, 140, 255), (255, 99, 71), (255, 214, 10), (170, 170, 190)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("export", type=Path, help="Minecraft-USD-Exporter world.usda or terrain.usdc")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("minecraft.yaml"))
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--seed", type=int, default=8675309)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=0, help="Blender threads (0 = all)")
    ap.add_argument("--samples", type=int, help="override the config's Cycles samples")
    ap.add_argument("--batch-size", type=int, default=6, help="views per Blender process")
    ap.add_argument("--blender", help="Blender binary (default: $USD_SYNTH_BLENDER)")
    args = ap.parse_args(argv)

    settings = mc_scene.load_settings(args.config)
    if args.samples:
        settings = settings.model_copy(update={"samples": args.samples})
    blender = find_blender(args.blender)
    out = args.out
    for sub in ("views", "images", "semantic", "depth", "render"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    terrain = mc_scene.read_export(args.export)
    info = mc_scene.build_scene(terrain, settings, out / "scene.usdc")
    t_scene = time.perf_counter() - t0
    print(
        f"scene: {info.blocks_kept} blocks, {info.faces} faces, {len(info.instances)} labelled "
        f"instances in {t_scene:.1f} s"
    )

    views = []
    for i in range(args.views):
        view = mc_scene.sample_view(info, settings, args.seed, i)
        mc_scene.write_view(view, out / "scene.usdc", out / "views" / f"view_{i:04d}.usda",
                            args.seed, settings)  # fmt: skip
        views.append(view)

    shim = SimpleNamespace(  # the two Config sections render_batch reads
        render=RenderConfig(samples=settings.samples, denoise=settings.denoise),
        image=ImageConfig(width=settings.width, height=settings.height),
    )
    categories = info.categories
    records, per_view, render_s = [], [], []
    t_start = time.perf_counter()
    blender_info = {}
    for b in range(0, args.views, args.batch_size):
        batch = range(b, min(b + args.batch_size, args.views))
        specs = [scene_spec(out / "views" / f"view_{i:04d}.usda", out / "render" / f"{i:04d}")
                 for i in batch]  # fmt: skip
        result = render_batch(shim, specs, blender, threads=args.threads or None, timeout=7200)
        for rj in result.scenes:
            blender_info = {k: rj[k] for k in ("blender", "device", "threads")}
        for i, rj in zip(batch, result.scenes, strict=True):
            render_s.append(rj["timings"]["render"])
            rdir = out / "render" / f"{i:04d}"
            usd = out / "views" / f"view_{i:04d}.usda"
            passes = read_passes(rdir / "aux.exr")
            anns, semantic = label_view(
                passes.index, read_instances(usd), categories, settings.min_visible_pixels
            )
            shutil.copyfile(rdir / "rgb.png", out / "images" / f"{i:04d}.png")
            palette_png(semantic, out / "semantic" / f"{i:04d}.png", CLASS_COLOURS)
            depth = np.asarray(passes.depth, dtype=np.float64)
            ok = np.isfinite(depth) & (depth > 0) & (depth < 655.0)
            cm = np.where(ok, np.rint(depth / DEPTH_SCALE_M), 0).astype(np.uint16)
            Image.fromarray(cm).save(out / "depth" / f"{i:04d}.png")
            records.append(
                {
                    "scene_index": i,
                    "file_name": f"images/{i:04d}.png",
                    "depth_file": f"depth/{i:04d}.png",
                    "scene_file": f"views/view_{i:04d}.usda",
                    "width": settings.width,
                    "height": settings.height,
                    "annotations": anns,
                }
            )
            counts = {c: 0 for c in categories}
            for a in anns:
                counts[categories[a["category_id"] - 1]] += 1
            pixels = {c: int((semantic == k).sum()) for k, c in enumerate(categories, start=1)}
            per_view.append({"view": i, "instances": counts, "pixels": pixels})
            print(f"view {i:04d}: {rj['timings']['render']:.1f} s render, {len(anns)} instances")
    wall = time.perf_counter() - t_start

    coco_categories = [
        {"id": k, "name": c, "supercategory": "minecraft"}
        for k, c in enumerate(categories, start=1)
    ]
    coco = build_coco(records, None, "usd-synth-data minecraft example", coco_categories)
    coco["info"]["depth_scale_m"] = DEPTH_SCALE_M
    dump_json(coco, out / "annotations.json")
    errors = validate_coco(out / "annotations.json")
    if errors:
        raise SystemExit("invalid COCO:\n" + "\n".join(errors[:20]))

    totals = {c: sum(v["instances"][c] for v in per_view) for c in categories}
    meta = {
        "export": str(args.export),
        "export_bbox": terrain.bbox,
        "seed": args.seed,
        "views": args.views,
        "classes": [
            {
                "id": k,
                "name": c.name,
                "blocks": c.blocks,
                "connectivity": c.connectivity,
                "rgb": CLASS_COLOURS[k - 1],
            }
            for k, c in enumerate(settings.classes, start=1)
        ],
        "scene": {
            "blocks_exported": info.blocks_in,
            "blocks_rendered": info.blocks_kept,
            "faces": info.faces,
            "labelled_components": {
                c: sum(1 for i in info.instances if i.category == c) for c in categories
            },
            "blocks_by_class": info.blocks_by_category,
            "build_seconds": round(t_scene, 2),
        },
        "annotations": {"total": len(coco["annotations"]), "per_class": totals},
        "render": {
            "engine": "cycles",
            **blender_info,
            "samples": settings.samples,
            "denoise": settings.denoise,
            "resolution": [settings.width, settings.height],
            "render_s_mean": round(float(np.mean(render_s)), 2),
            "render_s_p95": round(float(np.percentile(render_s, 95)), 2),
            "cpu": cpu_model(),
        },
        "throughput": {
            "images": args.views,
            "wall_seconds": round(wall, 1),
            "images_per_hour": round(args.views / wall * 3600, 1),
            "note": "render + label wall time (Blender start-up, USD import, render, labels); "
            "scene build excluded",
        },
        "view_params": [
            {
                **{k: v for k, v in asdict(view).items() if k != "camera"},
                "camera_position": [round(float(x), 3) for x in view.camera.position],
                "focal_length_mm": round(view.camera.focal_length_mm, 3),
            }
            for view in views
        ],
        "per_view": per_view,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(
        f"{args.views} views in {wall:.0f} s = {meta['throughput']['images_per_hour']:.0f} "
        f"images/hour; annotations per class: {totals}"
    )


if __name__ == "__main__":
    main()
