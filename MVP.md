# MVP

## The smallest end-to-end slice

> One config file and one seed produce N randomized **tabletop** scenes as USD, each rendered
> headless with Blender Cycles into an RGB image plus depth, and the whole set is written as a
> COCO-style dataset with 2D boxes and instance masks. Throughput is reported. The same run can
> be split into shards and executed as Kubernetes Jobs on any cluster, and the merged result is
> identical in content to the local run.

Concretely, this command should work at the end of the MVP:

```bash
usd-synth generate --config configs/tabletop.yaml --seed 42 --count 200 --out out/ds
# → out/ds/scenes/*.usda, out/ds/images/*.png, out/ds/depth/*.png,
#   out/ds/annotations.json, out/ds/manifest.json, out/ds/throughput.json
```

and so should regenerating one image from the manifest:

```bash
usd-synth reproduce --dataset out/ds --image-id 137 --out /tmp/repro
```

## Why Kubernetes fan-out is in the MVP (as the last milestone)

Throughput is one of the project's first-class outputs, and the only meaningful throughput
claim for path-traced rendering is a parallel one. Leaving fan-out until after the MVP would
mean the sharding design (how scene indices are split, where outputs go, how shards merge) is
never exercised, and is likely to need breaking changes later.

It is placed **last** so that every earlier milestone works and is tested on a single machine
without a cluster. The Kubernetes milestone adds only a thin layer: one image, one Indexed Job
manifest, and a merge step that already exists for local multi-process runs (milestone 6).
If it slips, milestones 1–6 are still a usable, releasable tool. This was confirmed as a
decision on 2026-10-17 (ARCHITECTURE.md, Decisions): M7 stays in the MVP and is tested on a
local `kind` cluster.

## In scope

- Python package + CLI (`usd-synth`) with a declarative YAML config validated against a schema.
- Deterministic seed derivation: `(dataset seed, scene index) → scene seed → per-stage seeds`.
- One scenario: **tabletop / bin-picking** — a support surface (table, optionally a bin), a set
  of objects scattered on it with clutter, randomized lighting, materials and camera.
- An object library made of procedural primitives (box, cylinder, sphere, capsule, torus-like
  mesh) plus optional user-supplied USD/USDZ assets referenced by path.
- USD stage writing with `usd-core`, flattened to a single `.usda` file per scene.
- Headless rendering with Blender (current LTS) Cycles, CPU by default, GPU optional.
- Ground truth: RGB (PNG), per-pixel depth, instance segmentation (COCO RLE), 2D bounding boxes
  (COCO `[x, y, w, h]`), category per instance.
- COCO-style `annotations.json` validated with `pycocotools`.
- `manifest.json` recording config hash, seeds, tool versions and per-image provenance.
- Throughput measurement: images/hour and per-stage timing, written to `throughput.json` and
  printed at the end of every run.
- Sharding: local multi-process sharding and Kubernetes Indexed Job fan-out with a merge step.
- Output sinks: a local filesystem directory (M1–M6) and, for the Kubernetes path, an
  S3-compatible object store (M7; MinIO on `kind` for tests). No ReadWriteMany volume is needed.
- Container image for the render worker.
- Tests at every stage (see ARCHITECTURE.md, Testing strategy).

## Out of scope (post-MVP)

- Additional scenarios: the **outdoor terrain** scenario is the planned second scenario (see
  [Post-MVP roadmap](#post-mvp-roadmap)); **indoor rooms** (walls, floor, furniture placement)
  remain a possible later one. The config schema and scene-builder interface are designed so
  scenarios can be added without changing the other stages.
- Physics-settled object piles (planned directly after the MVP; see
  [Post-MVP roadmap](#post-mvp-roadmap)).
- Alternative render backends (Omniverse Replicator, Blender EEVEE, others). Since the MVP, an
  optional Strata backend exists (`render.engine: strata`; ARCHITECTURE.md, Decision 5); the
  others remain out of scope.
- Other label types: 6-DoF object poses, keypoints, surface normals, optical flow, amodal masks.
- Other output formats (BOP, KITTI, YOLO txt); converters can come later.
- Camera intrinsics/lens distortion models beyond a pinhole camera.
- Autoscaling, job queues, or a scheduler beyond a single Indexed Job.
- Storage backends beyond the local filesystem and one S3-compatible sink (e.g. provider-specific
  APIs, other protocols).
- A web UI or dataset browser.

## Acceptance criteria

The MVP is done when all of the following are true and checked by automated tests or a
documented, repeatable manual check:

1. **Config validation.** An invalid config (unknown key, out-of-range value, empty object
   library) is rejected with an error message naming the offending field; the bundled
   `configs/tabletop.yaml` validates.
2. **Scene determinism.** Generating scene index *i* twice with the same config and seed
   produces byte-identical `.usda` files; changing the seed changes them.
3. **Shard independence.** Scene *i* is byte-identical whether generated alone, as part of a
   full run, or inside any shard.
4. **USD validity.** Every generated stage opens with `pxr.Usd.Stage.Open`, has a single
   camera, at least one light, and one prim per object instance carrying its instance ID and
   category.
5. **Rendering.** For the bundled config, a 640×480 render of one scene completes headless on a
   CPU-only machine in the CI container, producing an RGB PNG and a depth output.
6. **Label correctness.** On a synthetic fixture scene with known geometry (e.g. a single cube
   seen head-on), the computed bounding box matches the analytically projected box within
   2 px, and the mask area matches within 2 %.
7. **COCO validity.** `annotations.json` loads with `pycocotools.coco.COCO`, every annotation
   references an existing image and category, every mask decodes to the image size, and
   boxes enclose their masks.
8. **Reproduce.** `usd-synth reproduce` for any image ID produces identical annotations and
   an RGB image whose mean absolute pixel difference from the original is below a documented
   threshold (exact equality is only expected on the same Blender build and device; see
   ARCHITECTURE.md).
9. **Throughput report.** Every run writes `throughput.json` with total images, wall time,
   images/hour, and per-stage mean/p95 times, and prints a one-line summary.
10. **Kubernetes fan-out.** On a local `kind` cluster, a 4-shard Indexed Job renders a
    small dataset into an S3-compatible bucket (MinIO running in the cluster); after the merge step the merged `annotations.json` is equivalent (same
    images, same annotations up to ID renumbering rules) to a single-process local run with
    the same config and seed.
11. **Docs.** README quick-start commands run as written.

## Milestones

Each milestone maps 1:1 to a GitHub issue labelled `mvp`.

### M1. Project skeleton, config schema and seed derivation

Tracked in [#1](https://github.com/Preponderous-Software/usd-synth-data/issues/1). Implemented in [PR #9](https://github.com/Preponderous-Software/usd-synth-data/pull/9).

Python package layout, CLI entry point, `pyproject.toml`, lint/test CI. Pydantic config model
for the tabletop scenario with validation. Deterministic seed tree
(`dataset seed → scene seed → per-stage streams`) with tests proving stability and
independence. Covers acceptance criterion 1 and the determinism foundations of 2–3.

### M2. Tabletop scene generator writing USD

Tracked in [#2](https://github.com/Preponderous-Software/usd-synth-data/issues/2). Implemented in [PR #10](https://github.com/Preponderous-Software/usd-synth-data/pull/10).

Sampling of support surface, objects, poses (with non-overlap rejection), clutter, lights,
materials (UsdPreviewSurface) and camera (look-at with randomized distance/elevation/azimuth).
Writes a flattened `.usda` per scene with instance IDs and categories as custom attributes.
`usd-synth scenes` command generates scenes only. Covers criteria 2, 3 and 4.

### M3. Headless Blender render worker

Tracked in [#3](https://github.com/Preponderous-Software/usd-synth-data/issues/3). Implemented in [PR #11](https://github.com/Preponderous-Software/usd-synth-data/pull/11).

A script run by Blender in background mode: import the USD stage, map instance IDs to Cycles
object pass indices, configure Cycles (samples, denoiser, device, seed), render RGB + depth +
object-index passes to files. Container image with a pinned Blender LTS. Covers criterion 5.

### M4. Ground-truth labels and COCO writer

Tracked in [#4](https://github.com/Preponderous-Software/usd-synth-data/issues/4). Implemented in [PR #12](https://github.com/Preponderous-Software/usd-synth-data/pull/12).

Convert the index pass into per-instance masks (RLE), derive tight boxes, handle occlusion
and truncation (visible-pixel threshold), encode depth, and write `annotations.json` plus
`manifest.json`. Validated with `pycocotools`. Covers criteria 6 and 7.

### M5. End-to-end local pipeline and `reproduce`

Tracked in [#5](https://github.com/Preponderous-Software/usd-synth-data/issues/5). Implemented in [PR #13](https://github.com/Preponderous-Software/usd-synth-data/pull/13).

`usd-synth generate` runs generate → render → label for N scenes in one process and writes a
complete dataset; `usd-synth reproduce` regenerates one image from the manifest. Covers
criterion 8 and the first half of 11.

### M6. Throughput measurement, sharding and merge

Tracked in [#6](https://github.com/Preponderous-Software/usd-synth-data/issues/6). Implemented in [PR #14](https://github.com/Preponderous-Software/usd-synth-data/pull/14).

Per-stage timers, `throughput.json`, end-of-run summary; `--shard-index/--shard-count` to run a
subset; local multi-process sharding; `usd-synth merge` to combine shard outputs into one COCO
dataset. Resume and merge go through a small output-sink interface (local filesystem only
here) so M7 can add an S3-compatible sink without changing them. Covers criteria 3 (at scale)
and 9.

### M7. Kubernetes Indexed Job fan-out

Tracked in [#7](https://github.com/Preponderous-Software/usd-synth-data/issues/7). Implemented in [PR #15](https://github.com/Preponderous-Software/usd-synth-data/pull/15).

Indexed Job manifest template (one pod per shard, `JOB_COMPLETION_INDEX` → shard index),
an S3-compatible object-storage sink for shard outputs and the merged dataset (MinIO on
`kind` for tests; no ReadWriteMany volume), a merge step after completion, and a `kind`-based
integration test and throughput report from the cluster run. Covers criteria 10 and the rest
of 11.

## Post-MVP roadmap

Decided 2026-10-17 (ARCHITECTURE.md, Decisions). Not tracked as `mvp` issues.

- **Physics settling (after M7).** Drop objects and simulate them to rest before rendering,
  so piles and bins look physically plausible. Engine not chosen yet (Blender rigid-body or
  PyBullet). The settled poses are written into the `.usda`, keeping scenes seed-reproducible.
- **Outdoor terrain scenario (second scenario).** Procedural terrain with scattered
  vegetation and props, as a second scene-builder module next to tabletop. Expected to cost
  noticeably more render time per image than tabletop (larger scenes, far more geometry);
  the throughput report will measure by how much.
- **Indoor rooms (later idea).** Walls, floor and furniture placement; not scheduled.
- **Strata fast tier (added 2026-10-23).** `render.engine: strata` renders the same stages
  rasterized on CPU, about 68× faster per render step (measured under load) with exact masks
  and boxes; Cycles stays the default. Depth comes from Strata's ABI 1.9 depth readback, in
  Strata 0.5.0, pinned since 2026-10-24 (ARCHITECTURE.md, Strata backend).
