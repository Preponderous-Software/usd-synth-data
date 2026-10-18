# Example: auto-labelled training data from Barony boards

This example turns game states from
[Barony](https://github.com/Preponderous-Software/barony), a turn-based strategy game on a
grid, into a labelled computer-vision dataset. It saves states from a running Barony backend,
builds a 3D board for each, renders randomized camera views with the stock usd-synth-data
Blender worker, and writes for every view:

- an RGB image;
- a **semantic mask** (one class index per pixel);
- a **depth map**;
- **COCO instance annotations** (tight boxes and RLE masks).

![Six random views over Barony boards with their boxes, semantic masks and depth](../../docs/images/barony-contact-sheet.png)

## Where the world comes from

Barony's backend serves the whole game as JSON (`GET /api/session/state`). That JSON holds
the tile grid, with each tile's type (`CASTLE`, `VILLAGE` or `EMPTY`) and owner (0 = nobody,
1 = the player, 2 = the AI), and every army's tile, soldier count, owner and destination.
[`capture_states.py`](capture_states.py) plays guest games against the backend over that API
and saves the JSON at chosen ticks. Guest mode needs neither UserAuth nor a database server,
and nothing in Barony is changed. The script plays player 1: every few ticks it sends each
idle army towards a random village or tile and sometimes splits large armies. Barony's own AI
plays player 2. The backend does not take a map seed through the API, so the saved JSON files
are the inputs that make a run repeatable.

## Classes and the board

Barony draws its board in 2D and has no 3D models, so the 3D board is this example's own
(`barony_scene.py`). Every shape and colour comes from the state and a per-board random
stream; nothing is taken from Barony.

| Class | Built as | Instance = |
|---|---|---|
| `castle` | four corner towers, walls with a gate, a keep, and a banner in the owner's colour | one castle |
| `village` | three or four houses with gable roofs at random spots on the tile; a banner if owned | one village |
| `army` | soldier figures (body, head, spear) in the owner's colour, one per `soldiers_per_figure` soldiers (at most `max_figures`), behind a standard; a moving army faces its destination | one army |

Each tile is a slab, tinted towards its owner's colour when owned, on a meadow that extends
`surround_m` beyond the board. Empty tiles without an army get unlabelled scenery
(`scenery`: clumps of trees, rocks or a field) so the background is not uniform. An army on a
castle or village tile stands on the next tile towards the middle of the board, and armies
sharing a tile stand side by side.

## Randomization

View *i* depends only on the seed and *i*. Classes take turns, and so do the boards that
contain the class. The camera is aimed near a random object of that class, with a random
bearing, distance, pitch and focal length. A pose is redrawn if the camera would leave the
board or if any ray in a 9 × 7 pixel grid would miss the board and meadow (no sky). Sun and
sky are drawn as in the other world examples (`examples/common/worldviews.py`).

## Run it

You need a Barony checkout with JDK 17 and Maven, the package installed (`pip install -e .`)
and Blender 5.2 (`USD_SYNTH_BLENDER`):

```bash
(cd /path/to/barony/backend && mvn -q package -DskipTests && \
    USAGE_REPORTING_ENABLED=false java -jar target/backend-*.jar --server.port=18089) &
python examples/barony/capture_states.py --url http://localhost:18089 --games 3 \
    --ticks 0 15 40 80 --seed 7 --out out/barony-states
python examples/barony/barony_synth.py out/barony-states/*.json --views 24 --seed 3 \
    --out out/barony --threads 4
uv run examples/common/contact_sheet.py out/barony --views 17 19 0 22 7 3 \
    --title "usd-synth-data × Barony" \
    --subtitle "random views over boards built from Barony game states, labelled from the render" \
    --legend-note "one instance = one castle, village or army" \
    --source "12 game states (3 guest games at ticks 0, 15, 40, 80; 11×14 to 20×19 tiles): 72,581 faces" \
    --out docs/images/barony-contact-sheet.png
```

Output in `out/barony/` follows the other world examples: `scene_K.usdc` per board,
`views/view_NNNN.usda`, `images/`, `semantic/` (1 castle, 2 village, 3 army), `depth/`
(centimetres), `annotations.json` (COCO, validated) and `meta.json` (boards, instance counts,
per-view parameters, render settings, measured images/hour).

## The image above

The image above shows six of 24 views. The states came from three guest games against a
local Barony backend (commit `3ae215a`), saved at ticks 0, 15, 40 and 80 with the command
above. The boards are 11 × 14, 19 × 20 and 20 × 19 tiles. The figures below were measured on
one run, not benchmarked:

- **Scenes:** 12 boards with 72,581 faces and 196 labelled instances (24 castles, 116 villages,
  56 armies). Building them took 13 s.
- **Rendering:** 24 views at 640 × 480, 64 samples, denoised, with Cycles on the CPU limited to
  4 threads (Intel Core i9-10850K, shared with other jobs). Rendering took 13.7 s per view on
  average (p95 22.1 s). The 24 views took 347 s including Blender start-up, USD import and
  labelling, which is **249 images/hour**.
- **Labels:** 63 instance annotations (28 villages, 24 armies, 11 castles). A board has only
  two castles and a handful of villages and armies, so a view holds two or three objects. 24
  views cover 8 of the 12 boards, because views cycle through the classes first.

## Tests

`tests/test_barony_example.py` writes small game states by hand, using only the fields the
example reads. It checks reading, instance counts and owners, that a board is reproducible,
army size and placement, and deterministic views. The Blender-marked test runs the driver on
one view. It then renders a known top-down view and checks the class at each castle's keep,
at each village's highest roof ridge (read back from the scene) and at each army's soldier.
It also checks that an empty tile is unlabelled.
