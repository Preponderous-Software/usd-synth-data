# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.4", "Pillow==13.0.0", "pycocotools==2.0.11", "matplotlib==3.10.7"]
# ///
"""Compose a README contact sheet from a generated dataset.

One row per scene: the rendered RGB, then its ground truth as written by ``usd-synth generate``
— boxes with category labels, instance masks, and the depth map. Every label is read from the
dataset's ``annotations.json`` and ``depth/`` PNGs; nothing is recomputed here. The caption
reports the seed, resolution, samples and render time from ``manifest.json`` and
``throughput.json``.

    usd-synth generate --config configs/tabletop.yaml --seed 2026 --count 12 \\
        --out out/sheet --threads 6
    uv run scripts/contact_sheet.py out/sheet --images 0 2 3 4 7 1 \\
        --out docs/images/contact-sheet.png
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils

PANEL_W, PANEL_H = 400, 300
GAP = 8
MARGIN = 20
BG = (24, 26, 31)
FG = (235, 237, 242)
MUTED = (160, 166, 178)
DISTRACTOR = (150, 150, 150)
CATEGORY_COLOURS = {
    "box": (255, 99, 71),
    "cylinder": (64, 196, 255),
    "sphere": (255, 214, 10),
    "capsule": (190, 120, 255),
    "torus": (70, 230, 140),
}
COLUMNS = ["RGB render", "boxes + categories", "instance masks", "depth"]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in ("/usr/share/fonts/truetype/dejavu", "/Library/Fonts", "C:/Windows/Fonts"):
        p = Path(d) / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size)


def instance_colour(k: int) -> tuple[int, int, int]:
    """Distinct, saturated colours via golden-ratio hue stepping."""
    r, g, b = colorsys.hsv_to_rgb((k * 0.618034) % 1.0, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def boxes_panel(rgb: Image.Image, anns: list[dict], cats: dict[int, str], sx: float, sy: float):
    im = rgb.resize((PANEL_W, PANEL_H), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    f = font(11, bold=True)
    labelled = []
    for a in sorted(anns, key=lambda a: cats[a["category_id"]] != "distractor"):
        x, y, w, h = a["bbox"]
        box = (x * sx, y * sy, (x + w) * sx, (y + h) * sy)
        name = cats[a["category_id"]]
        if name == "distractor":
            d.rectangle(box, outline=DISTRACTOR, width=1)
            continue
        colour = CATEGORY_COLOURS.get(name, FG)
        d.rectangle(box, outline=colour, width=2)
        labelled.append((box, name, colour))
    placed: list[tuple[float, float, float, float]] = []
    for box, name, colour in labelled:
        tw, th = d.textbbox((0, 0), name, font=f)[2:]
        w, h = tw + 4, th + 2
        lx = min(max(box[0], 0), PANEL_W - w)
        # above the box, then inside its top edge, then below it: first spot no other tag covers
        spots = [box[1] - h - 1, box[1] + 1, box[3] + 1]
        spots = [min(max(y, 0), PANEL_H - h) for y in spots]
        ly = next(
            (y for y in spots if not any(_overlaps((lx, y, lx + w, y + h), r) for r in placed)),
            spots[0],
        )
        placed.append((lx, ly, lx + w, ly + h))
        d.rectangle((lx, ly, lx + w, ly + h), fill=colour)
        d.text((lx + 2, ly), name, font=f, fill=(0, 0, 0))
    return im


def _overlaps(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def masks_panel(rgb: Image.Image, anns: list[dict], cats: dict[int, str]) -> Image.Image:
    base = np.asarray(rgb.convert("L").convert("RGB"), dtype=np.float32) * 0.35
    out = base.copy()
    for k, a in enumerate(anns):
        m = mask_utils.decode(a["segmentation"]).astype(bool)
        c = DISTRACTOR if cats[a["category_id"]] == "distractor" else instance_colour(k)
        out[m] = base[m] * 0.25 + np.array(c, dtype=np.float32) * 0.75
    return Image.fromarray(out.clip(0, 255).astype(np.uint8)).resize(
        (PANEL_W, PANEL_H), Image.LANCZOS
    )


def depth_panel(depth_png: Path) -> tuple[Image.Image, float, float]:
    mm = np.asarray(Image.open(depth_png), dtype=np.float32)
    lo, hi = np.percentile(mm[mm > 0], [0.5, 99.5])
    # rank (histogram-equalized) scale: the table and the objects on it get most of the colours
    t = np.searchsorted(np.sort(mm.ravel()), mm, side="right").astype(np.float32) / mm.size
    rgb = (colormaps["turbo"](1.0 - t)[..., :3] * 255).astype(np.uint8)  # near = warm
    return Image.fromarray(rgb).resize((PANEL_W, PANEL_H), Image.LANCZOS), lo / 1000, hi / 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--images", type=int, nargs="+", required=True, help="COCO image ids, in order")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--colours", type=int, default=0, help="quantize to N colours (0 = full colour)"
    )
    args = ap.parse_args()

    ds = args.dataset
    coco = json.loads((ds / "annotations.json").read_text())
    thr = json.loads((ds / "throughput.json").read_text())
    manifest = json.loads((ds / "manifest.json").read_text())
    env = thr["environment"]
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    by_image: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)

    header_h, colhead_h, footer_h = 74, 26, 64
    rows = len(args.images)
    width = MARGIN * 2 + 4 * PANEL_W + 3 * GAP
    height = header_h + colhead_h + rows * PANEL_H + (rows - 1) * GAP + footer_h + MARGIN
    sheet = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(sheet)

    d.text((MARGIN, 16), "usd-synth-data", font=font(26, bold=True), fill=FG)
    d.text(
        (MARGIN + 232, 24),
        "randomized OpenUSD tabletop scenes, rendered headless in Blender Cycles, "
        "with exact ground truth",
        font=font(15),
        fill=MUTED,
    )
    x = MARGIN
    for name, colour in CATEGORY_COLOURS.items():
        d.rectangle((x, 52, x + 12, 64), fill=colour)
        d.text((x + 17, 50), name, font=font(13), fill=FG)
        x += 17 + d.textlength(name, font=font(13)) + 18
    d.rectangle((x, 52, x + 12, 64), outline=DISTRACTOR, width=2)
    d.text((x + 17, 50), "distractor (no tag)", font=font(13), fill=MUTED)

    y0 = header_h
    for c, title in enumerate(COLUMNS):
        d.text((MARGIN + c * (PANEL_W + GAP), y0 + 4), title, font=font(14, bold=True), fill=FG)
    y = y0 + colhead_h
    for image_id in args.images:
        meta = images[image_id]
        rgb = Image.open(ds / meta["file_name"]).convert("RGB")
        anns = by_image.get(image_id, [])
        sx, sy = PANEL_W / meta["width"], PANEL_H / meta["height"]
        depth, near, far = depth_panel(ds / meta["depth_file"])
        panels = [
            rgb.resize((PANEL_W, PANEL_H), Image.LANCZOS),
            boxes_panel(rgb, anns, cats, sx, sy),
            masks_panel(rgb, anns, cats),
            depth,
        ]
        for c, p in enumerate(panels):
            sheet.paste(p, (MARGIN + c * (PANEL_W + GAP), y))
        n_obj = sum(cats[a["category_id"]] != "distractor" for a in anns)
        n_dis = len(anns) - n_obj
        tag = f"scene {meta['scene_index']:06d} · {n_obj} objects, {n_dis} distractor"
        tag += "" if n_dis == 1 else "s"
        _badge(d, (MARGIN + 6, y + 6), tag)
        _badge(
            d,
            (MARGIN + 3 * (PANEL_W + GAP) + 6, y + 6),
            f"near {near:.2f} m → far {far:.2f} m (equalized)",
        )
        y += PANEL_H + GAP

    render = thr["stages"]["render"]
    w, h = env["resolution"]
    line1 = (
        f"seed {manifest['dataset_seed']} · {w}×{h} · {env['samples']} samples, "
        f"{'denoised' if env['denoise'] else 'no denoise'} · Cycles {env['device'].upper()} "
        f"({env['threads']} threads, {env['cpu_model']}) · Blender {env['blender']}"
    )
    line2 = (
        f"measured: render {render['mean_s']:.1f} s/image mean (p95 {render['p95_s']:.1f} s) · "
        f"{thr['images']} images in {thr['wall_seconds']:.0f} s = {thr['images_per_hour']:.0f} "
        f"images/hour · all labels from annotations.json + depth/, unedited"
    )
    fy = y - GAP + 14
    d.text((MARGIN, fy), line1, font=font(14), fill=FG)
    d.text((MARGIN, fy + 22), line2, font=font(14), fill=MUTED)

    if args.colours:
        sheet = sheet.quantize(
            colors=args.colours, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, optimize=True)
    print(f"{args.out}: {width}×{height}, {args.out.stat().st_size / 1e6:.2f} MB")


def _badge(d: ImageDraw.ImageDraw, xy: tuple[float, float], text: str) -> None:
    f = font(12)
    x, y = xy
    tw = d.textlength(text, font=f)
    d.rectangle((x, y, x + tw + 10, y + 19), fill=(0, 0, 0))
    d.text((x + 5, y + 2), text, font=f, fill=FG)


if __name__ == "__main__":
    main()
