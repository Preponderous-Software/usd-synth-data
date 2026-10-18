# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.4", "Pillow==13.0.0", "pycocotools==2.0.11", "matplotlib==3.10.7"]
# ///
"""Compose a contact sheet from a world example run (``roam_synth.py``, ``barony_synth.py``).

One row per view: the rendered RGB, then the ground truth the example wrote for it — boxes
with class tags (from ``annotations.json``), the semantic mask (``semantic/``) and the depth
map (``depth/``). Nothing is recomputed here; the caption reports settings and the measured
throughput from ``meta.json``, plus the title and source line given on the command line. Same
visual language as ``scripts/contact_sheet.py`` and ``examples/minecraft/contact_sheet.py``.

    uv run examples/common/contact_sheet.py out/roam --views 0 1 2 3 4 5 \\
        --title "usd-synth-data × Roam" --subtitle "random views over roam-usd worlds" \\
        --source "6 worlds from roam-usd generate --seeds 1..6" \\
        --out docs/images/roam-contact-sheet.png --colours 192
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

PANEL_W, PANEL_H = 400, 300
GAP = 8
MARGIN = 20
BG = (24, 26, 31)
FG = (235, 237, 242)
MUTED = (160, 166, 178)
COLUMNS = ["RGB render", "boxes + classes", "semantic mask", "depth"]
MIN_TAG_AREA = 900  # px at render resolution; smaller boxes are outlined but not tagged


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in ("/usr/share/fonts/truetype/dejavu", "/Library/Fonts", "C:/Windows/Fonts"):
        p = Path(d) / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size)


def _overlaps(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def boxes_panel(rgb, anns, names, colours, sx, sy) -> Image.Image:
    im = rgb.resize((PANEL_W, PANEL_H), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    f = font(11, bold=True)
    placed: list[tuple[float, float, float, float]] = []
    for a in sorted(anns, key=lambda a: -a["area"]):  # big first, so small boxes draw on top
        x, y, w, h = a["bbox"]
        box = (x * sx, y * sy, (x + w) * sx - 1, (y + h) * sy - 1)
        colour = colours[a["category_id"]]
        d.rectangle(box, outline=colour, width=2 if w * h >= MIN_TAG_AREA else 1)
    for a in sorted(anns, key=lambda a: -a["area"]):
        x, y, w, h = a["bbox"]
        if w * h < MIN_TAG_AREA:
            continue
        box = (x * sx, y * sy, (x + w) * sx, (y + h) * sy)
        name, colour = names[a["category_id"]], colours[a["category_id"]]
        tw, th = d.textbbox((0, 0), name, font=f)[2:]
        bw, bh = tw + 4, th + 2
        lx = min(max(box[0], 0), PANEL_W - bw)
        spots = [min(max(v, 0), PANEL_H - bh) for v in (box[1] - bh - 1, box[1] + 1, box[3] + 1)]
        free = [v for v in spots if not any(_overlaps((lx, v, lx + bw, v + bh), r) for r in placed)]
        if not free:
            continue  # crowded: the outline is enough
        ly = free[0]
        placed.append((lx, ly, lx + bw, ly + bh))
        d.rectangle((lx, ly, lx + bw, ly + bh), fill=colour)
        d.text((lx + 2, ly), name, font=f, fill=(0, 0, 0))
    return im


def semantic_panel(rgb: Image.Image, semantic: np.ndarray, colours: dict) -> Image.Image:
    base = np.asarray(rgb.convert("L").convert("RGB"), dtype=np.float32) * 0.35
    out = base.copy()
    for k, c in colours.items():
        m = semantic == k
        out[m] = base[m] * 0.2 + np.array(c, dtype=np.float32) * 0.8
    return Image.fromarray(out.clip(0, 255).astype(np.uint8)).resize(
        (PANEL_W, PANEL_H), Image.NEAREST
    )


def depth_panel(depth_png: Path, scale_m: float) -> tuple[Image.Image, float, float]:
    raw = np.asarray(Image.open(depth_png), dtype=np.float32)
    valid = raw > 0
    lo, hi = np.percentile(raw[valid], [0.5, 99.5]) * scale_m
    vals = np.sort(raw[valid])
    t = np.searchsorted(vals, raw, side="right").astype(np.float32) / max(len(vals), 1)
    rgb = (colormaps["turbo"](1.0 - t)[..., :3] * 255).astype(np.uint8)  # near = warm
    rgb[~valid] = (12, 13, 16)  # sky: no depth
    return Image.fromarray(rgb).resize((PANEL_W, PANEL_H), Image.LANCZOS), lo, hi


def _badge(d: ImageDraw.ImageDraw, xy: tuple[float, float], text: str) -> None:
    f = font(12)
    x, y = xy
    tw = d.textlength(text, font=f)
    d.rectangle((x, y, x + tw + 10, y + 19), fill=(0, 0, 0))
    d.text((x + 5, y + 2), text, font=f, fill=FG)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="the example driver's --out directory")
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", default="random views, labelled from the render")
    ap.add_argument("--legend-note", default="one instance = one labelled object")
    ap.add_argument("--source", required=True, help="first footer line: where the world came from")
    ap.add_argument("--views", type=int, nargs="+", required=True, help="view indices, in order")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--colours", type=int, default=0, help="quantize to N colours (0 = full)")
    args = ap.parse_args()

    run = args.run
    coco = json.loads((run / "annotations.json").read_text())
    meta = json.loads((run / "meta.json").read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}
    colours = {c["id"]: tuple(c["rgb"]) for c in meta["classes"]}
    scale_m = coco["info"].get("depth_scale_m", 0.001)
    images = {im["id"]: im for im in coco["images"]}
    by_image: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    params = {v["index"]: v for v in meta["view_params"]}

    header_h, colhead_h, footer_h = 74, 26, 64
    rows = len(args.views)
    width = MARGIN * 2 + 4 * PANEL_W + 3 * GAP
    height = header_h + colhead_h + rows * PANEL_H + (rows - 1) * GAP + footer_h + MARGIN
    sheet = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(sheet)

    title = args.title
    d.text((MARGIN, 16), title, font=font(26, bold=True), fill=FG)
    d.text(
        (MARGIN + d.textlength(title, font=font(26, bold=True)) + 16, 24),
        args.subtitle,
        font=font(15),
        fill=MUTED,
    )
    x = MARGIN
    for k, name in names.items():
        d.rectangle((x, 52, x + 12, 64), fill=colours[k])
        d.text((x + 17, 50), name, font=font(13), fill=FG)
        x += 17 + d.textlength(name, font=font(13)) + 18
    d.text((x, 50), args.legend_note, font=font(13), fill=MUTED)

    for c, col in enumerate(COLUMNS):
        d.text((MARGIN + c * (PANEL_W + GAP), header_h + 4), col, font=font(14, bold=True), fill=FG)
    y = header_h + colhead_h
    for vid in args.views:
        im_meta = images[vid]
        rgb = Image.open(run / im_meta["file_name"]).convert("RGB")
        anns = by_image.get(vid, [])
        sx, sy = PANEL_W / im_meta["width"], PANEL_H / im_meta["height"]
        semantic = np.asarray(Image.open(run / "semantic" / Path(im_meta["file_name"]).name))
        depth, near, far = depth_panel(run / im_meta["depth_file"], scale_m)
        panels = [
            rgb.resize((PANEL_W, PANEL_H), Image.LANCZOS),
            boxes_panel(rgb, anns, names, colours, sx, sy),
            semantic_panel(rgb, semantic, colours),
            depth,
        ]
        for c, p in enumerate(panels):
            sheet.paste(p, (MARGIN + c * (PANEL_W + GAP), y))
        counts = {}
        for a in anns:
            counts[names[a["category_id"]]] = counts.get(names[a["category_id"]], 0) + 1
        v = params[vid]
        _badge(
            d,
            (MARGIN + 6, y + 6),
            f"view {vid:02d} · sun {v['sun_elevation_deg']:.0f}° · pitch {v['pitch_deg']:.0f}° · "
            f"{v['focal_length_mm']:.0f} mm",
        )
        _badge(
            d,
            (MARGIN + PANEL_W + GAP + 6, y + 6),
            " · ".join(f"{n} {counts[n]}" for n in names.values() if n in counts) or "none",
        )
        _badge(d, (MARGIN + 3 * (PANEL_W + GAP) + 6, y + 6), f"{near:.0f} m → {far:.0f} m")
        y += PANEL_H + GAP

    r, t = meta["render"], meta["throughput"]
    line1 = (
        f"{args.source} · {r['resolution'][0]}×{r['resolution'][1]} · "
        f"{r['samples']} samples{', denoised' if r['denoise'] else ''} · "
        f"Cycles CPU ({r['threads']} threads, {r['cpu']}) · Blender {r['blender']}"
    )
    line2 = (
        f"measured: render {r['render_s_mean']:.1f} s/view mean · {t['images']} views in "
        f"{t['wall_seconds']:.0f} s = {t['images_per_hour']:.0f} images/hour · "
        f"{meta['annotations']['total']} instances · labels from the Object Index pass, unedited"
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


if __name__ == "__main__":
    main()
