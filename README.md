# usd-synth-data

**usd-synth-data** is a **procedural synthetic-dataset generator** for computer-vision and
robotics perception training.
It writes randomized **OpenUSD** scenes, renders them headless with **Blender Cycles**, and
emits RGB images with exact ground truth — **2D bounding boxes, instance masks and depth** — as
a **COCO-style** dataset. Rendering fans out as **Kubernetes Jobs** on any cluster, and
**throughput (images/hour)** is measured and reported on every run.

> **Status: in progress — MVP milestones are landing in order.** Implemented so far: the
> package, config schema and seed derivation (M1). Commands below that are not implemented
> yet print a "not implemented yet" message.

---

## 🚀 Why usd-synth-data Exists

Labelling real images by hand is slow and inconsistent, and some labels (per-pixel depth,
pixel-exact masks) are impractical to label at all. When a program builds the scene, every label
is known exactly — but only if the pipeline also gives you **variety**, **reproducibility** and
**throughput**.

With usd-synth-data:
- Every scene is randomized from a declarative config: objects, poses, clutter, lighting,
  materials and camera placement (domain randomization).
- Every image is reproducible from the config, the dataset seed and its scene index.
- Every scene is a plain `.usda` file you can open in any USD viewer or diff as text.
- Every run tells you how many images per hour it produced, and where the time went.

---

## 🌍 A Day in the Life with usd-synth-data

1. **Describe the scene space** in `configs/tabletop.yaml`: which objects, how many, how
   cluttered, what lighting and camera ranges.
2. **Generate a pilot set locally**: `usd-synth generate --config configs/tabletop.yaml --seed 42 --count 200 --out out/ds`.
3. **Inspect one odd image** by opening its `scene_000137.usda`, or regenerate it exactly with
   `usd-synth reproduce --image-id 137`.
4. **Scale out** by rendering the same config as a sharded Kubernetes Indexed Job, then merge
   the shards into one COCO dataset — identical in content to a local run.
5. **Compare throughput** across sample counts or resolutions from `throughput.json`.

---

## 🎯 Purpose

Core responsibilities:
- Sampling randomized tabletop / bin-picking scenes and writing them as flattened OpenUSD stages.
- Rendering each stage headless (Blender Cycles; CPU by default, GPU optional).
- Deriving ground truth from render passes: 2D boxes, instance segmentation (COCO RLE), depth.
- Writing a validated COCO-style dataset plus a provenance manifest.
- Sharding work across processes or Kubernetes Jobs and merging the results.
- Measuring and reporting throughput as a first-class output.

---

## 📦 Features (MVP Scope)

See [MVP.md](MVP.md) for the full scope, acceptance criteria and milestones.

- **Scenario:** tabletop / bin-picking (an outdoor terrain scenario and physics-settled
  placement are planned after the MVP).
- **Randomization:** object selection and count, poses, distractor clutter, support surface,
  materials, key/dome lighting, camera distance/elevation/azimuth/focal length.
- **Labels:** RGB PNG, 16-bit depth PNG (millimetres), instance masks (RLE), tight boxes,
  categories.
- **Format:** COCO-style `annotations.json` validated with `pycocotools`.
- **Determinism:** seed-addressable scenes, independent of sharding.
- **Scale-out:** Kubernetes Indexed Job (one pod per shard) writing to S3-compatible object
  storage + merge step; local multi-process sharding (local filesystem) for single machines.
- **Throughput:** `throughput.json` with images/hour and per-stage timings.

---

## 🛠 Tech Stack (planned)

- Python 3.12+
- OpenUSD (`usd-core`)
- Blender 5.2 LTS (Cycles), run headless
- NumPy, Pydantic, PyYAML, OpenEXR, Pillow
- pycocotools
- Docker (render worker image)
- Kubernetes Indexed Jobs (any cluster ≥ v1.33; `kind` for local testing)
- S3-compatible object storage for cluster runs (MinIO for local testing)
- pytest, ruff

---

## 📂 Documentation

| Document | What it covers |
|---|---|
| [CONCEPT.md](CONCEPT.md) | The problem, who it is for, what makes it interesting, non-goals, and prior art (BlenderProc, Kubric, Omniverse Replicator, Infinigen, Unity Perception). |
| [MVP.md](MVP.md) | The smallest end-to-end slice, in/out of scope, acceptance criteria, and the milestone list that maps 1:1 to GitHub issues. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Components, data flow, USD stage layout, config schema, label generation, renderer choice, Kubernetes topology, dependencies, testing, throughput, decisions, open questions. |

MVP work is tracked in the issues labelled
[`mvp`](https://github.com/Preponderous-Software/usd-synth-data/issues?q=label%3Amvp).

---

## 🚀 Getting Started

Works today (M1):

```bash
pip install -e '.[dev]'
usd-synth config --config configs/tabletop.yaml   # validate a config, print its SHA-256
pytest                                            # run the test suite
```

Intended interface (later milestones):

```bash
usd-synth generate --config configs/tabletop.yaml --seed 42 --count 200 --out out/ds
usd-synth reproduce --dataset out/ds --image-id 137 --out /tmp/repro
usd-synth k8s-manifest --config configs/tabletop.yaml --seed 42 --count 10000 --shards 50 \
  --out s3://datasets/tabletop-42 | kubectl apply -f -
```

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

**Copyright © 2026 Daniel McCoy Stephenson.**

---

## 📬 Contact

For inquiries, feature requests, or contributions, please open an issue:
https://github.com/Preponderous-Software/usd-synth-data/issues
