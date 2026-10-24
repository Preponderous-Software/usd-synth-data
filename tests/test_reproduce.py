"""``reproduce``'s manifest checks and its comparison verdict, without Blender: the pipeline
stages it calls are replaced by fakes that write a chosen regenerated image, stage and labels,
so each comparison (stage bytes, labels, RGB difference against the threshold) is exercised."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import usd_synth.reproduce as rep
from usd_synth.config import load_config
from usd_synth.reproduce import RGB_MAD_THRESHOLD, reproduce

SMALL = Path(__file__).resolve().parent / "data" / "small.yaml"
INDEX = 2
USDA = b"#usda 1.0\n"
RGB = np.full((6, 8, 3), 100, dtype=np.uint8)


def ann(ann_id, image_id, bbox=(1.0, 2.0, 3.0, 4.0)):
    return {"id": ann_id, "image_id": image_id, "category_id": 1, "bbox": list(bbox),
            "area": 12.0, "iscrowd": 0}  # fmt: skip


def write_scene(root: Path, rgb: np.ndarray, usda: bytes) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "scenes").mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(root / "images" / f"scene_{INDEX:06d}.png")
    (root / "scenes" / f"scene_{INDEX:06d}.usda").write_bytes(usda)


@pytest.fixture
def dataset(tmp_path):
    """A four-image dataset whose image ``INDEX`` is the one reproduced."""
    cfg = load_config(SMALL)
    ds = tmp_path / "ds"
    write_scene(ds, RGB, USDA)
    manifest = {
        "config": cfg.resolved(), "config_hash": cfg.config_hash(), "config_dir": "/cfg",
        "dataset_seed": 7, "images": [{"image_id": i, "scene_index": i} for i in range(4)],
    }  # fmt: skip
    (ds / "manifest.json").write_text(json.dumps(manifest))
    coco = {"annotations": [ann(5, INDEX), ann(6, 1, bbox=(0.0, 0.0, 1.0, 1.0))]}
    (ds / "annotations.json").write_text(json.dumps(coco))
    return ds


@pytest.fixture
def rerun(monkeypatch):
    """Replace the pipeline stages; returns the recorded calls and the output to regenerate."""
    state = {"rgb": RGB, "usda": USDA, "annotations": [ann(0, INDEX)], "calls": []}

    def prepare_shard(sink, cfg, config_dir, seed, count, *shard, indices):
        state["calls"].append(("prepare", config_dir, seed, count, *shard, indices))

    def run_shard(cfg, config_dir, seed, indices, sink, blender, threads=None):
        state["calls"].append(("run", config_dir, seed, indices, blender, threads))
        write_scene(sink.root, state["rgb"], state["usda"])

    def merge(sink, expected=None):
        state["calls"].append(("merge", expected))
        return {"coco": {"annotations": state["annotations"]}}

    monkeypatch.setattr(rep, "prepare_shard", prepare_shard)
    monkeypatch.setattr(rep, "run_shard", run_shard)
    monkeypatch.setattr(rep, "merge", merge)
    return state


def test_identical_rerun_is_ok_and_report_is_written(dataset, rerun, tmp_path):
    out = tmp_path / "repro"
    report = reproduce(dataset, INDEX, out, "blender-bin", threads=3)
    assert report == {
        "image_id": INDEX, "scene_index": INDEX, "usda_identical": True,
        "annotations_identical": True, "rgb_mean_abs_diff": 0.0, "rgb_max_abs_diff": 0.0,
        "rgb_threshold": RGB_MAD_THRESHOLD, "ok": True,
    }  # fmt: skip
    assert json.loads((out / "reproduce.json").read_text()) == report


def test_only_the_requested_scene_is_regenerated(dataset, rerun, tmp_path):
    reproduce(dataset, INDEX, tmp_path / "repro", "blender-bin", threads=3)
    only = range(INDEX, INDEX + 1)
    assert rerun["calls"] == [
        ("prepare", Path("/cfg"), 7, 4, 1, 0, only),
        ("run", Path("/cfg"), 7, only, "blender-bin", 3),
        ("merge", only),
    ]


def test_annotation_ids_and_other_images_do_not_count(dataset, rerun, tmp_path):
    # the original has id 5 for this image and an annotation on image 1; the rerun has id 0 only
    assert reproduce(dataset, INDEX, tmp_path / "repro", "b")["annotations_identical"] is True


def test_changed_annotation_is_a_mismatch(dataset, rerun, tmp_path):
    rerun["annotations"] = [ann(0, INDEX, bbox=(1.0, 2.0, 3.0, 5.0))]
    report = reproduce(dataset, INDEX, tmp_path / "repro", "b")
    assert report["annotations_identical"] is False and report["ok"] is False


def test_changed_stage_is_a_mismatch(dataset, rerun, tmp_path):
    rerun["usda"] = USDA + b"\n"
    report = reproduce(dataset, INDEX, tmp_path / "repro", "b")
    assert report["usda_identical"] is False and report["ok"] is False


@pytest.mark.parametrize(
    "delta, ok", [(1, True), (2, False)], ids=["at-threshold", "over-threshold"]
)
def test_rgb_mean_difference_is_compared_with_the_threshold(dataset, rerun, tmp_path, delta, ok):
    rerun["rgb"] = RGB + np.uint8(delta)
    report = reproduce(dataset, INDEX, tmp_path / "repro", "b")
    assert report["rgb_mean_abs_diff"] == float(delta) and report["rgb_max_abs_diff"] == delta
    assert report["ok"] is ok


def test_one_far_off_pixel_passes_on_the_mean(dataset, rerun, tmp_path):
    rgb = RGB.copy()
    rgb[0, 0] = 255
    rerun["rgb"] = rgb
    report = reproduce(dataset, INDEX, tmp_path / "repro", "b")
    assert report["rgb_max_abs_diff"] == 155.0
    assert report["rgb_mean_abs_diff"] == pytest.approx(155 * 3 / RGB.size)
    assert report["ok"] is True


def test_tampered_manifest_config_is_rejected_before_any_stage(dataset, rerun, tmp_path):
    manifest = json.loads((dataset / "manifest.json").read_text())
    manifest["config"]["image"]["width"] += 1
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match its recorded hash"):
        reproduce(dataset, INDEX, tmp_path / "repro", "b")
    assert rerun["calls"] == [] and not (tmp_path / "repro").exists()


def test_unknown_image_id_is_rejected_before_any_stage(dataset, rerun, tmp_path):
    with pytest.raises(ValueError, match=r"image id 9 is not in .*manifest\.json"):
        reproduce(dataset, 9, tmp_path / "repro", "b")
    assert rerun["calls"] == [] and not (tmp_path / "repro").exists()
