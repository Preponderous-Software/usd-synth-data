"""Synthetic, auto-labelled training data from Barony game states.

Builds a 3D board for each saved Barony game state (``capture_states.py`` output), renders
randomized views over the boards with the stock usd-synth-data Blender worker and writes, per
view, an RGB image, a semantic mask, a depth map and COCO instance annotations (boxes + RLE
masks) for castles, villages and armies.

    python examples/barony/barony_synth.py out/barony-states/*.json \\
        --views 24 --seed 3 --out out/barony --threads 4

Labels are read from the render's Object Index pass (``aux.exr``): each castle, village and
army is one mesh with its own pass index, so occlusion, the image border and the camera's view
are all handled by the renderer.

Output (``--out``)::

    scene_K.usdc             board K as meshes; labelled instances under /World/Objects
    views/view_NNNN.usda     sublayers one scene; camera, sun and sky of one view
    images/NNNN.png          RGB
    semantic/NNNN.png        8-bit class map (0 = unlabelled, 1 castle, 2 village, 3 army)
    depth/NNNN.png           16-bit planar depth in centimetres (0 = sky)
    annotations.json         COCO: images, instance annotations, categories (validated)
    meta.json                boards, views, instance counts, render settings, images/hour
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import barony_scene  # noqa: E402  (also puts examples/common on sys.path)
import worldviews  # noqa: E402

from usd_synth.render import find_blender  # noqa: E402

CLASS_COLOURS = [(255, 214, 10), (46, 160, 67), (230, 80, 200)]  # castle, village, army


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("states", type=Path, nargs="+", help="Barony game state JSON files")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("barony.yaml"))
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=0, help="Blender threads (0 = all)")
    ap.add_argument("--samples", type=int, help="override the config's Cycles samples")
    ap.add_argument("--batch-size", type=int, default=6, help="views per Blender process")
    ap.add_argument("--blender", help="Blender binary (default: $USD_SYNTH_BLENDER)")
    args = ap.parse_args(argv)

    settings = barony_scene.load_settings(args.config)
    if args.samples:
        settings = settings.model_copy(update={"samples": args.samples})
    blender = find_blender(args.blender)
    out = args.out
    (out / "views").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    scenes, boards = [], []
    for k, path in enumerate(args.states):
        board = barony_scene.read_state(path)
        info = barony_scene.build_scene(board, settings, out / f"scene_{k}.usdc")
        scenes.append(info)
        boards.append(board)
        print(
            f"board {k} ({path.name}): {board.width}×{board.height} tiles, {info.faces} faces, "
            f"{len(info.instances)} labelled instances"
        )
    t_scene = time.perf_counter() - t0

    views = []
    for i in range(args.views):
        view = barony_scene.sample_view(scenes, settings, args.seed, i)
        worldviews.write_view(view.camera, view.light, out / f"scene_{view.board}.usdc",
                              out / "views" / f"view_{i:04d}.usda", args.seed, i)  # fmt: skip
        views.append(view)

    categories = barony_scene.CATEGORIES
    run = worldviews.render_views(
        out, args.views, categories, CLASS_COLOURS, width=settings.width, height=settings.height,
        samples=settings.samples, denoise=settings.denoise,
        min_visible_pixels=settings.min_visible_pixels, blender=blender,
        threads=args.threads or None, batch_size=args.batch_size,
    )  # fmt: skip
    shared = worldviews.write_dataset(
        out, run, categories, "usd-synth-data barony example", "barony",
        width=settings.width, height=settings.height, samples=settings.samples,
        denoise=settings.denoise,
    )  # fmt: skip

    def count(s, c):
        return sum(1 for i in s.instances if i.category == c)

    meta = {
        "boards": [
            {
                "file": str(b.path),
                "scene": f"scene_{k}.usdc",
                "tiles": [b.width, b.height],
                "tick": b.tick,
                "faces": s.faces,
                "instances": {c: count(s, c) for c in categories},
            }
            for k, (b, s) in enumerate(zip(boards, scenes, strict=True))
        ],
        "seed": args.seed,
        "views": args.views,
        "classes": [
            {"id": k, "name": c, "rgb": CLASS_COLOURS[k - 1]}
            for k, c in enumerate(categories, start=1)
        ],
        "scene": {
            "boards": len(scenes),
            "faces": sum(s.faces for s in scenes),
            "labelled_instances": {c: sum(count(s, c) for s in scenes) for c in categories},
            "build_seconds": round(t_scene, 2),
        },
        **shared,
        "view_params": [
            {
                "index": v.index,
                "board": v.board,
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
