# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.4", "Pillow==13.0.0", "pycocotools==2.0.11"]
# ///
"""Compose a Cycles-vs-Strata comparison grid from two datasets of the same scenes.

Generate the same config and seed twice, once per ``render.engine``; the scenes are identical
because the generator does not depend on the renderer. One row per scene: the Cycles image with
its COCO instance masks, the Strata image with its masks, and where the two label maps disagree.
Every mask is decoded from each dataset's ``annotations.json``; nothing is recomputed here. The
caption reports render times from each ``throughput.json``.

    usd-synth generate --config configs/tabletop.yaml --seed 2026 --count 12 --out out/cycles
    usd-synth generate --config strata.yaml --seed 2026 --count 12 --out out/strata
    uv run scripts/engine_compare.py out/cycles out/strata --images 0 4 6 9 \\
        --out docs/images/cycles-vs-strata.png
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils

PANEL_W, PANEL_H = 400, 300
GAP = 8
MARGIN = 20
BG = (24, 26, 31)
FG = (235, 237, 242)
MUTED = (160, 166, 178)
DISTRACTOR = (150, 150, 150)
DISAGREE = (255, 64, 64)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in ("/usr/share/fonts/truetype/dejavu", "/Library/Fonts", "C:/Windows/Fonts"):
        p = Path(d) / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size)


def instance_colour(instance_id: int) -> tuple[int, int, int]:
    """Keyed by instance ID, so one object has one colour in both columns."""
    r, g, b = colorsys.hsv_to_rgb((instance_id * 0.618034) % 1.0, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


class Dataset:
    def __init__(self, root: Path):
        self.root = root
        self.coco = json.loads((root / "annotations.json").read_text())
        self.thr = json.loads((root / "throughput.json").read_text())
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.cats = {c["id"]: c["name"] for c in self.coco["categories"]}
        self.images = {im["id"]: im for im in self.coco["images"]}
        self.anns: dict[int, list[dict]] = {}
        for a in self.coco["annotations"]:
            self.anns.setdefault(a["image_id"], []).append(a)

    def rgb(self, image_id: int) -> Image.Image:
        return Image.open(self.root / self.images[image_id]["file_name"]).convert("RGB")

    def label_map(self, image_id: int) -> np.ndarray:
        meta = self.images[image_id]
        out = np.zeros((meta["height"], meta["width"]), dtype=np.int64)
        for a in self.anns.get(image_id, []):
            out[mask_utils.decode(a["segmentation"]).astype(bool)] = a["instance_id"]
        return out

    def overlay(self, image_id: int) -> Image.Image:
        base = np.asarray(self.rgb(image_id), dtype=np.float32)
        out = base * 0.55
        for a in self.anns.get(image_id, []):
            m = mask_utils.decode(a["segmentation"]).astype(bool)
            distractor = self.cats[a["category_id"]] == "distractor"
            c = np.array(DISTRACTOR if distractor else instance_colour(a["instance_id"]))
            out[m] = base[m] * 0.5 + c * 0.5
            edge = m & ~_erode(m)
            out[edge] = c * 0.5 + 127
        return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def _erode(m: np.ndarray) -> np.ndarray:
    e = m.copy()
    e[1:] &= m[:-1]
    e[:-1] &= m[1:]
    e[:, 1:] &= m[:, :-1]
    e[:, :-1] &= m[:, 1:]
    return e


def diff_panel(a: np.ndarray, b: np.ndarray, rgb: Image.Image) -> tuple[Image.Image, float]:
    grey = np.asarray(rgb.convert("L").convert("RGB"), dtype=np.float32) * 0.3
    labelled = (a > 0) | (b > 0)
    agree = labelled & (a == b)
    grey[agree] = grey[agree] * 0.4 + np.array((70, 200, 120)) * 0.6
    grey[a != b] = DISAGREE
    share = float((a != b).sum() / max(labelled.sum(), 1))
    return Image.fromarray(grey.clip(0, 255).astype(np.uint8)), share


def _badge(d: ImageDraw.ImageDraw, xy: tuple[float, float], text: str) -> None:
    f = font(12)
    x, y = xy
    tw = d.textlength(text, font=f)
    d.rectangle((x, y, x + tw + 10, y + 19), fill=(0, 0, 0))
    d.text((x + 5, y + 2), text, font=f, fill=FG)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cycles", type=Path, help="dataset rendered with render.engine: cycles")
    ap.add_argument("strata", type=Path, help="the same config and seed, render.engine: strata")
    ap.add_argument("--images", type=int, nargs="+", required=True, help="COCO image ids")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cyc, sta = Dataset(args.cycles), Dataset(args.strata)
    if cyc.manifest["dataset_seed"] != sta.manifest["dataset_seed"]:
        raise SystemExit("the two datasets have different seeds, so different scenes")
    scene_files = {e["image_id"]: e["scene_file"] for e in cyc.manifest["images"]}
    for image_id in args.images:
        name = scene_files[image_id]
        if (cyc.root / name).read_bytes() != (sta.root / name).read_bytes():
            raise SystemExit(f"image {image_id}: the two {name} stages differ")

    ce, se = cyc.thr["environment"], sta.thr["environment"]
    columns = [
        f"Cycles · {ce['samples']} samples · {cyc.thr['stages']['render']['mean_s']:.1f} s/image",
        f"Strata {se['strata']} · llvmpipe · {sta.thr['stages']['render']['mean_s']:.2f} s/image",
        "instance labels: agree / disagree",
    ]
    header_h, colhead_h, footer_h = 62, 26, 64
    rows = len(args.images)
    width = MARGIN * 2 + 3 * PANEL_W + 2 * GAP
    height = header_h + colhead_h + rows * PANEL_H + (rows - 1) * GAP + footer_h + MARGIN
    sheet = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(sheet)
    title_font = font(26, bold=True)
    d.text((MARGIN, 14), "Cycles vs Strata", font=title_font, fill=FG)
    d.text(
        (MARGIN + d.textlength("Cycles vs Strata", font=title_font) + 16, 23),
        "the same randomized OpenUSD scenes, each with the COCO instance masks its dataset wrote",
        font=font(15),
        fill=MUTED,
    )
    for c, title in enumerate(columns):
        d.text((MARGIN + c * (PANEL_W + GAP), header_h + 4), title, font=font(14, True), fill=FG)
    y = header_h + colhead_h
    shares = []
    for image_id in args.images:
        la, lb = cyc.label_map(image_id), sta.label_map(image_id)
        diff, share = diff_panel(la, lb, cyc.rgb(image_id))
        shares.append(share)
        panels = [cyc.overlay(image_id), sta.overlay(image_id), diff]
        for c, p in enumerate(panels):
            sheet.paste(
                p.resize((PANEL_W, PANEL_H), Image.LANCZOS), (MARGIN + c * (PANEL_W + GAP), y)
            )
        n = len(set(np.unique(la)) | set(np.unique(lb))) - 1
        _badge(d, (MARGIN + 6, y + 6), f"scene {cyc.images[image_id]['file_name'][-10:-4]}")
        _badge(
            d,
            (MARGIN + 2 * (PANEL_W + GAP) + 6, y + 6),
            f"{n} instances · {100 * share:.3f} % of labelled pixels differ",
        )
        y += PANEL_H + GAP
    w, h = ce["resolution"]
    gl = se["gl"].split(" |")[0].replace(", 256 bits", "")
    line1 = (
        f"seed {cyc.manifest['dataset_seed']} · {w}×{h} · Blender {ce['blender']} Cycles CPU, "
        f"{ce['threads']} threads · Strata {se['strata']} on {gl}, {se['threads']} threads · "
        f"{ce['cpu_model'].replace('(R)', '').replace('(TM)', '').replace(' CPU', '')}"
    )
    line2 = (
        "Strata ignores roughness, metallic, HDRIs and soft shadows (recorded in manifest.json); "
        f"labels match: worst row {100 * max(shares):.3f} % of labelled pixels differ"
    )
    fy = y - GAP + 14
    d.text((MARGIN, fy), line1, font=font(14), fill=FG)
    d.text((MARGIN, fy + 22), line2, font=font(14), fill=MUTED)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, optimize=True)
    print(f"{args.out}: {width}×{height}, {args.out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
