# Example: auto-labelled training data from a Minecraft world

This example turns a Minecraft world into a labelled computer-vision dataset. It reads a
terrain export written by
[Minecraft-USD-Exporter](https://github.com/Dans-Plugins/Minecraft-USD-Exporter), renders
randomized camera views of it with the stock usd-synth-data Blender worker, and writes for
every view:

- an RGB image;
- a **semantic mask** (one class index per pixel);
- a **depth map**;
- **COCO instance annotations** (tight boxes and RLE masks).

![Six random views of a Minecraft world with their boxes, semantic masks and depth](../../docs/images/minecraft-contact-sheet.png)

## Classes

The labelled classes are block categories, defined in [`minecraft.yaml`](minecraft.yaml) as
glob patterns on namespaced block IDs:

| Class | Blocks | Instance = |
|---|---|---|
| `tree` | `*_log`, `*_wood`, `*_stem`, `*_leaves`, … | a 26-connected group of trunk and leaf blocks (one tree, or a clump whose canopies touch) |
| `water` | `water`, plus waterlogged plants and bubble columns | a 6-connected body of water |
| `ore` | `*_ore` (including deepslate variants), `ancient_debris` | a 26-connected vein |
| `sand` | `sand`, `red_sand`, `sandstone` | a 6-connected beach or bank |
| `stone` | `stone`, `andesite`, `diorite`, `granite`, `deepslate`, `tuff`, `cobblestone`, … | a 6-connected exposed outcrop or cliff |

Every other block (grass, dirt, gravel, clay and so on) is rendered but left unlabelled.
Non-cube decoration such as grass tufts, flowers and leaf litter is left out of the scene,
because it would otherwise render as full blocks.

## How the labels are made

All labels come from the render; none are drawn by hand or re-projected:

1. `mc_scene.py` reads the export's `PointInstancer`s with `usd-core`. It uses the
   per-instance `primvars:mcusd:blockId` to classify every block, then splits each class into
   connected components.
2. It writes `scene.usdc` in the usd-synth stage layout (Z-up). Each component becomes one
   `/World/Objects/inst_NNNNN` mesh with the usual `userProperties:usd_synth:*` attributes
   (`instance_id`, `category_name`, …). The other blocks become unlabelled `/World/Env`
   meshes. Only block faces next to an empty cell are emitted. Each block type gets a
   `UsdPreviewSurface` in the exporter's display colour, bound through a `GeomSubset`.
3. For each view, a small `views/view_NNNN.usda` sublayers `scene.usdc` and adds the camera,
   a `DistantLight` sun and a dome sky colour (written by the shared
   [`../common/worldviews.py`](../common/worldviews.py), which the Roam and Barony examples
   use too). The unchanged `usd_synth.render.scene_spec` and
   `render_batch` render it. The worker gives every instance its own Cycles Object Index.
4. `mc_synth.py` reads the Object Index pass from `aux.exr` (`usd_synth.passes`). Each visible
   instance becomes a COCO annotation: its mask is the instance's pixels, and its box is the
   tight box of that mask. Instances with fewer than `min_visible_pixels` visible pixels are
   skipped. The semantic mask maps every instance to its class. The dataset is written with
   `usd_synth.coco.build_coco` and checked with `validate_coco`.

Occlusion, truncation at the image border and the camera's field of view are therefore exact.
A mask is exactly the pixels where that component is the first surface hit.

## Randomization

Each view is sampled from `stage_rng(seed, view, "camera" | "lighting")`, so view *i* depends
only on the seed and *i*:

- **Target:** classes take turns by view index. The camera is aimed near a random block of
  that class that can be seen from above (the top of its column, or part of a cliff or bank
  face), offset by up to `aim_jitter_blocks`. Rare classes such as ore therefore get as many
  views as forests do.
- **Camera:** bearing, horizontal distance, pitch and focal length are drawn from the config's
  ranges. A pose is redrawn if its line of sight to the target is blocked, if it is inside the
  terrain, or if any downward-looking ray in a 9 × 7 grid leaves the export before it reaches
  the ground. The last check keeps the cut edges of the exported box out of the picture.
- **Time of day:** sun elevation and azimuth, and sun strength. A low sun gives a warm sun
  colour and a dusk-coloured sky; a high sun gives white light under a blue sky.

Fog and textured blocks are not modelled; blocks use flat display colours.

## Run it

You need an export (any Minecraft-USD-Exporter `world.usda` or `terrain.usdc`), the package
installed (`pip install -e .`) and Blender 5.2 (`USD_SYNTH_BLENDER`):

```bash
mc-usd-export /path/to/world --bbox -128 44 -128 31 120 31 --out out/export   # Minecraft-USD-Exporter
python examples/minecraft/mc_synth.py out/export/world.usda --views 24 --seed 8675309 \
    --out out/mc --threads 4
uv run examples/minecraft/contact_sheet.py out/mc --views 0 2 8 9 14 23 \
    --out docs/images/minecraft-contact-sheet.png --colours 192
```

Output in `out/mc/`:

| Path | Contents |
|---|---|
| `scene.usdc` | the terrain as meshes; labelled components under `/World/Objects` |
| `views/view_NNNN.usda` | one view: sublayers `scene.usdc` and adds the camera, sun and sky |
| `images/NNNN.png` | RGB |
| `semantic/NNNN.png` | 8-bit palette PNG; pixel value *k* = the *k*-th class, 0 = unlabelled |
| `depth/NNNN.png` | 16-bit planar depth in **centimetres** (0 = sky). Outdoor distances exceed the core pipeline's 16-bit millimetres. |
| `annotations.json` | COCO: images, instance annotations (`instance_id` kept), categories; `info.depth_scale_m` = 0.01 |
| `meta.json` | classes, connected components, annotation counts, per-view parameters and pixel counts, render settings, measured images/hour |
| `render/NNNN/` | the worker's raw output (`rgb.png`, `aux.exr`, `render.json`) |

## The image above

The image above shows a fresh vanilla 26.3 world (seed `8675309`, no players). It is the same
160 × 160-block export (`--bbox -128 44 -128 31 120 31`) that Minecraft-USD-Exporter uses for
its README preview. The figures below were measured on one run, not benchmarked:

- **Scene:** 165,727 exported blocks; 157,926 after dropping decoration. These give 222,874
  faces and 1,869 labelled components (1,185 stone, 319 sand, 237 ore, 106 tree, 22 water).
  Building the scene took about 4 s.
- **Rendering:** 24 views at 640 × 480, 64 samples, denoised, with Cycles on the CPU limited to
  4 threads (Intel Core i9-10850K, shared with other jobs). Rendering took 17.6 s per view on
  average (p95 20.7 s). The 24 views took 507 s including Blender start-up, USD import and
  labelling, which is **170 images/hour**.
- **Labels:** 676 instance annotations (347 tree, 150 sand, 147 stone, 22 water, 10 ore).
  Water is usually one large lake per view. Ore is rare at the surface, so its instances are
  a few pixels to a few blocks across.
- **Not ground truth:** the terrain below Y 44 and the decoration are not in the export, so
  cave walls, hidden ore and plants are not labelled.

## Tests

`tests/test_minecraft_example.py` builds small exports in the exporter's layout. It checks
reading (including region transforms), classification, connectivity, face winding,
deterministic views and labelling. The Blender-marked test renders a known top-down view and
checks that the class at the projected centre of each placed block is the right one. It also
checks that the water mask covers blue pixels.
