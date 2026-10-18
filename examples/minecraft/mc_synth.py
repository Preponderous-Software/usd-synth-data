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
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mc_scene  # noqa: E402  (also puts examples/common on sys.path)
import worldviews  # noqa: E402
from worldviews import DEPTH_SCALE_M, label_view, palette_png  # noqa: E402, F401

from usd_synth.render import find_blender  # noqa: E402

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
    (out / "views").mkdir(parents=True, exist_ok=True)

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

    categories = info.categories
    run = worldviews.render_views(
        out, args.views, categories, CLASS_COLOURS, width=settings.width, height=settings.height,
        samples=settings.samples, denoise=settings.denoise,
        min_visible_pixels=settings.min_visible_pixels, blender=blender,
        threads=args.threads or None, batch_size=args.batch_size,
    )  # fmt: skip
    shared = worldviews.write_dataset(
        out, run, categories, "usd-synth-data minecraft example", "minecraft",
        width=settings.width, height=settings.height, samples=settings.samples,
        denoise=settings.denoise,
    )  # fmt: skip
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
        **shared,
        "view_params": [
            {
                **{k: v for k, v in asdict(view).items() if k != "camera"},
                "camera_position": [round(float(x), 3) for x in view.camera.position],
                "focal_length_mm": round(view.camera.focal_length_mm, 3),
            }
            for view in views
        ],
        "per_view": run.per_view,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(
        f"{args.views} views in {run.wall_s:.0f} s = "
        f"{meta['throughput']['images_per_hour']:.0f} images/hour; annotations per class: "
        f"{shared['annotations']['per_class']}"
    )


if __name__ == "__main__":
    main()
