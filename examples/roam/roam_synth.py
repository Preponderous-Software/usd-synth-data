"""Synthetic, auto-labelled training data from Roam worlds.

Renders randomized views over one or more `roam-usd`_ world stages with the stock
usd-synth-data Blender worker and writes, per view, an RGB image, a semantic mask, a depth map
and COCO instance annotations (boxes + RLE masks) for the entity classes in the config.

    roam-usd generate --roam <roam-checkout> --seeds 1..6 --radius 3 --cave-depth 0 \\
        --binary -o out/roam-worlds
    python examples/roam/roam_synth.py out/roam-worlds/seed_{1..6}.usdc \\
        --views 24 --seed 1 --out out/roam --threads 4

Labels are read from the render's Object Index pass (``aux.exr``): each labelled tree, outcrop,
vein, creature, structure or food item is one mesh with its own pass index, so occlusion, the
image border and the camera's view are all handled by the renderer.

Output (``--out``)::

    scene_K.usdc             world K as meshes; labelled instances under /World/Objects
    views/view_NNNN.usda     sublayers one scene; camera, sun and sky of one view
    images/NNNN.png          RGB
    semantic/NNNN.png        8-bit class map (0 = unlabelled, k = k-th class in the config)
    depth/NNNN.png           16-bit planar depth in centimetres (0 = sky)
    annotations.json         COCO: images, instance annotations, categories (validated)
    meta.json                worlds, views, classes, instance counts, render settings, images/hour

Only roam-usd's catalog colours are rendered (creature sprite materials are ignored), so the
images contain no Roam artwork.

.. _roam-usd: https://github.com/Preponderous-Software/roam-usd
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import roam_scene  # noqa: E402  (also puts examples/common on sys.path)
import worldviews  # noqa: E402

from usd_synth.render import find_blender  # noqa: E402

CLASS_COLOURS = [
    (46, 160, 67),  # tree
    (170, 170, 190),  # stone
    (255, 99, 71),  # ore
    (64, 140, 255),  # creature
    (255, 214, 10),  # structure
    (230, 80, 200),  # food
]

LAYER_KEYS = ("roam:seed", "roam:radius", "roam:commit", "roam:version", "roam:source")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("worlds", type=Path, nargs="+", help="roam-usd stages (.usda / .usdc)")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("roam.yaml"))
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=0, help="Blender threads (0 = all)")
    ap.add_argument("--samples", type=int, help="override the config's Cycles samples")
    ap.add_argument("--batch-size", type=int, default=6, help="views per Blender process")
    ap.add_argument("--blender", help="Blender binary (default: $USD_SYNTH_BLENDER)")
    args = ap.parse_args(argv)

    settings = roam_scene.load_settings(args.config)
    if args.samples:
        settings = settings.model_copy(update={"samples": args.samples})
    if len(settings.classes) > len(CLASS_COLOURS):
        raise SystemExit(f"at most {len(CLASS_COLOURS)} classes (add colours to CLASS_COLOURS)")
    blender = find_blender(args.blender)
    out = args.out
    (out / "views").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    scenes, worlds = [], []
    for k, path in enumerate(args.worlds):
        world = roam_scene.read_world(path)
        info = roam_scene.build_scene(world, settings, out / f"scene_{k}.usdc")
        scenes.append(info)
        worlds.append(world)
        print(
            f"world {k} ({path.name}): {info.entities} entities, {info.faces} faces, "
            f"{len(info.instances)} labelled instances"
        )
    t_scene = time.perf_counter() - t0

    views = []
    for i in range(args.views):
        view = roam_scene.sample_view(scenes, settings, args.seed, i)
        worldviews.write_view(view.camera, view.light, out / f"scene_{view.world}.usdc",
                              out / "views" / f"view_{i:04d}.usda", args.seed, i)  # fmt: skip
        views.append(view)

    categories = scenes[0].categories
    run = worldviews.render_views(
        out, args.views, categories, CLASS_COLOURS, width=settings.width, height=settings.height,
        samples=settings.samples, denoise=settings.denoise,
        min_visible_pixels=settings.min_visible_pixels, blender=blender,
        threads=args.threads or None, batch_size=args.batch_size,
    )  # fmt: skip
    shared = worldviews.write_dataset(
        out, run, categories, "usd-synth-data roam example", "roam",
        width=settings.width, height=settings.height, samples=settings.samples,
        denoise=settings.denoise,
    )  # fmt: skip
    meta = {
        "worlds": [
            {
                "file": str(w.path),
                "scene": f"scene_{k}.usdc",
                **{
                    key.split(":")[1]: w.layer_data[key]
                    for key in LAYER_KEYS
                    if key in w.layer_data
                },  # fmt: skip
                "rooms": len(w.rooms),
                "bounds": list(s.bounds),
                "entities": s.entities,
                "faces": s.faces,
                "labelled_instances": {
                    c: sum(1 for i in s.instances if i.category == c) for c in categories
                },
                "entities_by_class": s.entities_by_class,
            }
            for k, (w, s) in enumerate(zip(worlds, scenes, strict=True))
        ],
        "seed": args.seed,
        "views": args.views,
        "classes": [
            {
                "id": k,
                "name": c.name,
                "entities": c.entities,
                "canopy": c.canopy,
                "connectivity": None if c.per_entity or c.canopy else c.connectivity,
                "per_entity": c.per_entity,
                "rgb": CLASS_COLOURS[k - 1],
            }
            for k, c in enumerate(settings.classes, start=1)
        ],
        "scene": {
            "worlds": len(scenes),
            "entities": sum(s.entities for s in scenes),
            "faces": sum(s.faces for s in scenes),
            "labelled_instances": {
                c: sum(1 for s in scenes for i in s.instances if i.category == c)
                for c in categories
            },
            "build_seconds": round(t_scene, 2),
        },
        **shared,
        "view_params": [
            {
                "index": v.index,
                "world": v.world,
                "target_class": v.target_class,
                "target": list(v.target),
                "yaw_deg": v.yaw_deg,
                "pitch_deg": v.pitch_deg,
                "distance": v.distance,
                "camera_position": [round(float(x), 3) for x in v.camera.position],
                "focal_length_mm": round(v.camera.focal_length_mm, 3),
                "sun_elevation_deg": v.light.sun_elevation_deg,
                "sun_intensity": v.light.sun_intensity,
                "sky_intensity": v.light.sky_intensity,
            }
            for v in views
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
