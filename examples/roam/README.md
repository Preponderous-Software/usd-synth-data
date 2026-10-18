# Example: auto-labelled training data from Roam worlds

This example turns worlds from the game [Roam](https://github.com/Preponderous-Software/roam)
into a labelled computer-vision dataset. It reads world stages written by
[roam-usd](https://github.com/Preponderous-Software/roam-usd), renders randomized camera
views of them with the stock usd-synth-data Blender worker, and writes for every view:

- an RGB image;
- a **semantic mask** (one class index per pixel);
- a **depth map**;
- **COCO instance annotations** (tight boxes and RLE masks).

![Six random views over Roam worlds with their boxes, semantic masks and depth](../../docs/images/roam-contact-sheet.png)

## Classes

The labelled classes are groups of Roam entity classes, defined in [`roam.yaml`](roam.yaml) as
glob patterns on the `roam:entityClass` that roam-usd records on every prototype and prim:

| Class | Roam entities | Instance = |
|---|---|---|
| `tree` | `OakWood`, `JungleWood`, with the `Leaves` on the four tiles around them | one tree: a trunk and its canopy |
| `stone` | `Stone` | a 4-connected outcrop of stone tiles |
| `ore` | `CoalOre`, `IronOre`, `GoldOre` | an 8-connected vein |
| `creature` | `Bear`, `Chicken`, `Deer`, `Rabbit`, `Snake`, `Wolf`, `Npc` | one animal or NPC |
| `structure` | floors, beds, chests, torches, campfires, fences, cave entrances, crops, … | an 8-connected group (the starting home is one instance) |
| `food` | `Apple`, `Banana`, `ChickenMeat`, `BearMeat`, `Wheat`, `WheatSeed` | one item on the ground |

Grass, cave floor, the room grounds and leaf cover are rendered but not labelled.

## How the scene is built

roam-usd draws a Roam world as boxes in its own catalog colours: one tile is 1 × 1 unit,
ground cover is a thin slab, stone and ore are unit cubes, a trunk is a tall narrow box and
`Leaves` is a canopy slab lifted above the ground. `roam_scene.py` reads that stage with
`usd-core` (rooms, the `Tiles` `PointInstancer` of each room, `Creatures` and `Structures`),
keeps only the surface level, and makes three changes before building meshes:

- **Creatures get a body.** roam-usd writes creatures as flat sprite cards. Each becomes a
  box (`creature_box`, 0.7 × 0.7 × 0.55) in the card's catalog colour.
- **Canopies are resolved.** Roam spawns a tree as a trunk with leaves on the four tiles
  around it. Forest rooms also scatter `Leaves` over almost every tile, and roam-usd stacks
  every entity of a tile on the one below, so a forest tile can become a column several
  canopies tall. Here the first `Leaves` on a tile next to a trunk is that tree's canopy, and
  further `Leaves` on the same tile are dropped. `Leaves` with no trunk next to them are leaf
  cover (`ground_cover`), drawn as a flat unlabelled slab.
- **Tiles are re-stacked.** Everything else on a tile is stacked from the ground up in its
  original order. A canopy floats at its catalog lift and does not raise what comes after it,
  so the apples under an oak lie on the ground.

Each labelled instance becomes one `/World/Objects/inst_NNNNN` mesh with the usual
`userProperties:usd_synth:*` attributes. Everything else becomes one unlabelled `/World/Env`
mesh per room, on a ground quad in this example's own soil colour (`ground_color`). Downward
faces are left out. Each Roam entity class gets a `UsdPreviewSurface` in roam-usd's catalog
colour, bound through a `GeomSubset`.

**License boundary.** Roam is under the Preponderous Non-Commercial License; this repository
is MIT. Nothing from Roam is stored here: no code, schemas, art or saves. The tests build
their worlds by hand. `roam-usd generate` binds Roam's own sprites to creature cards; this
example ignores those materials and renders only roam-usd's catalog colours, which roam-usd
picked by hand rather than sampling from Roam's art. The committed contact sheet therefore
contains no Roam artwork. Generating worlds runs Roam's code, which is governed by Roam's
license.

## Randomization

Views are sampled as in the Minecraft example. View *i* depends only on the seed and *i*:

- **Target:** classes take turns by view index, and so do the worlds that contain the class.
  The camera is aimed near a random entity of that class, offset by up to `aim_jitter_tiles`.
- **Camera:** bearing, horizontal distance, pitch and focal length come from the config's
  ranges. A pose is redrawn if the camera would be outside the world (less `margin_tiles`),
  lower than `min_height`, or if any ray in a 9 × 7 pixel grid would meet the ground outside
  the world. The last check keeps the world's edge and the sky out of the picture.
- **Time of day:** sun elevation, azimuth and strength, as in the Minecraft example
  (`examples/common/worldviews.py`).

## Run it

You need roam-usd, a Roam checkout with Roam's Python requirements, the package installed
(`pip install -e .`) and Blender 5.2 (`USD_SYNTH_BLENDER`):

```bash
roam-usd generate --roam /path/to/roam --seeds 1..6 --radius 3 --cave-depth 0 --binary \
    -o out/roam-worlds
python examples/roam/roam_synth.py out/roam-worlds/seed_{1..6}.usdc --views 24 --seed 1 \
    --out out/roam --threads 4
uv run examples/common/contact_sheet.py out/roam --views 9 12 22 7 17 15 \
    --title "usd-synth-data × Roam" \
    --subtitle "random views over Roam worlds exported by roam-usd, labelled from the render" \
    --legend-note "one instance = one tree, outcrop, vein, home or creature" \
    --source "6 generated Roam worlds (seeds 1–6, 7×7 rooms each): 91,817 entities → 459,379 faces" \
    --out docs/images/roam-contact-sheet.png
```

A save works too: `roam-usd export <save-dir> --levels surface -o world.usda`, then pass
`world.usda` to `roam_synth.py`.

Output in `out/roam/`:

| Path | Contents |
|---|---|
| `scene_K.usdc` | world *K* as meshes; labelled instances under `/World/Objects` |
| `views/view_NNNN.usda` | one view: sublayers its world's scene and adds the camera, sun and sky |
| `images/NNNN.png` | RGB |
| `semantic/NNNN.png` | 8-bit palette PNG; pixel value *k* = the *k*-th class, 0 = unlabelled |
| `depth/NNNN.png` | 16-bit planar depth in **centimetres** (0 = sky) |
| `annotations.json` | COCO: images, instance annotations (`instance_id` kept), categories; `info.depth_scale_m` = 0.01 |
| `meta.json` | worlds (seed, Roam commit), classes, instance counts, per-view parameters and pixel counts, render settings, measured images/hour |
| `render/NNNN/` | the worker's raw output (`rgb.png`, `aux.exr`, `render.json`) |

## The image above

The image above shows six of 24 views over six worlds that `roam-usd generate --seeds 1..6
--radius 3 --cave-depth 0` produced with Roam's own generator (Roam commit `bd4da73`, Python
3.12). Each world is 7 × 7 rooms of 17 × 17 tiles, with the starting home in the middle. The
sheet is not quantized, because quantizing it would merge the small food and creature colours.
The figures below were measured on one run, not benchmarked:

- **Scenes:** 91,817 entities after re-stacking give 459,379 faces and 10,861 labelled
  instances (3,691 stone, 3,404 food, 2,046 tree, 1,102 ore, 579 creature, 39 structure).
  Building the six scenes took 85 s in all.
- **Rendering:** 24 views at 640 × 480, 64 samples, denoised, with Cycles on the CPU limited to
  4 threads (Intel Core i9-10850K, shared with other jobs at a load average of about 30 on 20
  threads). Rendering took 18.8 s per view on average (p95 24.2 s). The 24 views took 557 s
  including Blender start-up, USD import and labelling, which is **155 images/hour**.
- **Labels:** 3,190 instance annotations (1,423 stone, 771 tree, 415 ore, 360 food, 194
  creature, 27 structure). Mountain rooms are fields of single stone and ore cubes, so views
  over them hold many small stone instances. Food items are 0.3 units across and are often
  only a few dozen pixels.
- **Not modelled:** cave levels, Roam's own sprites and textures, and Roam's room background
  colours (the ground uses `ground_color`).

## Tests

`tests/test_roam_example.py` builds a small world in roam-usd's layout by hand. It checks
reading (tile coordinates, room transforms, creature cards and structures), canopy
resolution, leaf cover and re-stacking, connectivity, instance counts, and deterministic views.
The Blender-marked test runs the driver on one view. It then renders a known top-down view
and checks that the class at the projected position of each placed object is the right one:
trunk, canopy, stone, ore, chicken, chest, floor, apple, leaf cover (unlabelled) and plain
grass (unlabelled).
