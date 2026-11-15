"""Synthetic, auto-labelled aerial views of a heightmap world package.

Builds a terrain mesh from a heightmap world package (``world.json``, ``height.png``,
``ground.png`` and optionally ``vectors.json``), renders randomized aerial views of it with the
stock usd-synth-data Blender worker and writes, per view, an RGB image, a semantic mask, a
depth map and COCO instance annotations (boxes + RLE masks) for the labelled ground classes.

    python examples/heightmap_world/hm_synth.py examples/heightmap_world/sample \\
        --views 6 --seed 11 --out out/hm --threads 4

Labels are read from the render's Object Index pass (``aux.exr``): each region of a labelled
class is one mesh with its own pass index, so occlusion, the image border and the camera's
view are all handled by the renderer.

Output (``--out``)::

    scene.usdc               terrain as meshes; labelled regions under /World/Objects
    views/view_NNNN.usda     sublayers scene.usdc; camera, sun and sky of one view
    images/NNNN.png          RGB
    semantic/NNNN.png        8-bit class map (0 = unlabelled, k = k-th labelled class)
    depth/NNNN.png           16-bit planar depth in centimetres (0 = sky / beyond 655 m)
    annotations.json         COCO: images, instance annotations, categories (validated)
    meta.json                package, classes, instances, views, render settings, images/hour
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hm_scene  # noqa: E402  (also puts examples/common on sys.path)
import worldviews  # noqa: E402

from usd_synth.render import find_blender  # noqa: E402

# Semantic-mask colour per ground class (classes not here get grey)
CLASS_COLOURS = {
    "water": (64, 140, 255),
    "forest": (46, 160, 67),
    "snow": (230, 236, 245),
    "sand": (255, 214, 10),
    "stone": (150, 150, 165),
    "badland": (214, 110, 50),
    "ice": (150, 220, 255),
    "lava": (255, 70, 20),
    "built": (230, 80, 200),
    "farm": (190, 170, 60),
}


def colours_for(categories: list[str]) -> list[tuple[int, int, int]]:
    return [CLASS_COLOURS.get(c, (128, 128, 128)) for c in categories]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("package", type=Path, help="heightmap world package folder")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("heightmap_world.yaml"))
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=0, help="Blender threads (0 = all)")
    ap.add_argument("--samples", type=int, help="override the config's Cycles samples")
    ap.add_argument("--batch-size", type=int, default=6, help="views per Blender process")
    ap.add_argument("--blender", help="Blender binary (default: $USD_SYNTH_BLENDER)")
    args = ap.parse_args(argv)

    settings = hm_scene.load_settings(args.config)
    if args.samples:
        settings = settings.model_copy(update={"samples": args.samples})
    blender = find_blender(args.blender)
    out = args.out
    (out / "views").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    world = hm_scene.read_package(args.package, settings.stride)
    info = hm_scene.build_scene(world, settings, out / "scene.usdc", args.seed)
    t_scene = time.perf_counter() - t0
    h, w = world.shape
    print(
        f"scene: {w}×{h} px at {world.blocks_per_px} blocks/px, {info.faces} faces, "
        f"{info.trees} trees, {len(info.instances)} labelled regions in {t_scene:.1f} s"
    )

    views = []
    for i in range(args.views):
        view = hm_scene.sample_view(info, settings, args.seed, i)
        worldviews.write_view(view.camera, view.light, out / "scene.usdc",
                              out / "views" / f"view_{i:04d}.usda", args.seed, i)  # fmt: skip
        views.append(view)

    categories = info.categories
    colours = colours_for(categories)
    run = worldviews.render_views(
        out, args.views, categories, colours, width=settings.width, height=settings.height,
        samples=settings.samples, denoise=settings.denoise,
        min_visible_pixels=settings.min_visible_pixels, blender=blender,
        threads=args.threads or None, batch_size=args.batch_size,
    )  # fmt: skip
    shared = worldviews.write_dataset(
        out, run, categories, "usd-synth-data heightmap world example", "ground",
        width=settings.width, height=settings.height, samples=settings.samples,
        denoise=settings.denoise,
    )  # fmt: skip
    meta = {
        "package": str(args.package),
        "grid": {
            "origin_x": world.origin_x,
            "origin_z": world.origin_z,
            "blocks_per_px": world.blocks_per_px,
            "size_px": [w, h],
            "stride": settings.stride,
        },
        "seed": args.seed,
        "views": args.views,
        "classes": [
            {"id": k, "name": c, "rgb": colours[k - 1]} for k, c in enumerate(categories, start=1)
        ],
        "scene": {
            "faces": info.faces,
            "trees": info.trees,
            "labelled_regions": {
                c: sum(1 for i in info.instances if i.category == c) for c in categories
            },
            "pixels_by_class": info.pixels_by_category,
            "build_seconds": round(t_scene, 2),
        },
        **shared,
        "view_params": [
            {
                "index": v.index,
                "target_class": v.target_class,
                "target": list(v.target),
                "yaw_deg": v.yaw_deg,
                "pitch_deg": v.pitch_deg,
                "altitude": v.altitude,
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
