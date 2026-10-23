import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from pycocotools import mask as mask_utils

from usd_synth.coco import build_coco, build_manifest, dump_json, validate_coco
from usd_synth.config import parse_config
from usd_synth.labels import (
    InstanceInfo,
    coco_categories,
    depth_to_mm,
    encode_rle,
    label_scene,
    read_depth_png,
    tight_bbox,
    write_depth_png,
)
from usd_synth.seeds import render_seed, scene_seed

SMALL = Path(__file__).resolve().parent / "data" / "small.yaml"


def _cfg(**labels):
    raw = yaml.safe_load(SMALL.read_text())
    raw["labels"] = {"min_visible_pixels": 4, "min_visible_fraction": 0.25, **labels}
    return parse_config(raw)


def _obj(iid, area=0.0, cat="box"):
    return InstanceInfo(iid, "object", cat, area)


def test_rle_round_trip_and_tight_box():
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:9, 10:17] = True
    rle = encode_rle(mask)
    decoded = mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode()})
    assert np.array_equal(decoded.astype(bool), mask)
    assert tight_bbox(mask) == [10.0, 5.0, 7.0, 4.0]


def test_occlusion_masks_only_visible_pixels():
    index = np.zeros((20, 20), dtype=np.int32)
    index[2:12, 2:12] = 1  # 100 px of instance 1 ...
    index[6:12, 6:12] = 2  # ... of which 36 are covered by instance 2 in front
    anns = label_scene(index, [_obj(1, 100), _obj(2, 36)], _cfg())
    a1, a2 = anns
    assert a1["area"] == 64 and a1["bbox"] == [2.0, 2.0, 10.0, 10.0]
    assert a2["area"] == 36 and a2["bbox"] == [6.0, 6.0, 6.0, 6.0]


def test_min_visible_fraction_drops_heavily_occluded():
    index = np.zeros((20, 20), dtype=np.int32)
    index[0:4, 0:4] = 1  # 16 visible px of an instance whose full silhouette is 100 px
    assert label_scene(index, [_obj(1, 100)], _cfg()) == []
    assert len(label_scene(index, [_obj(1, 60)], _cfg())) == 1  # 16/60 >= 0.25


def test_truncation_at_image_border():
    # an instance mostly outside the image: its unclipped projected area is large
    index = np.zeros((10, 10), dtype=np.int32)
    index[:, 8:] = 1  # 20 px visible at the right border
    assert label_scene(index, [_obj(1, 200)], _cfg()) == []  # 10 % visible -> dropped
    (ann,) = label_scene(index, [_obj(1, 40)], _cfg())  # 50 % visible -> kept
    assert ann["bbox"] == [8.0, 0.0, 2.0, 10.0]


def test_min_visible_pixels():
    index = np.zeros((10, 10), dtype=np.int32)
    index[0, 0:3] = 1
    assert label_scene(index, [_obj(1, 3)], _cfg()) == []
    assert len(label_scene(index, [_obj(1, 3)], _cfg(min_visible_pixels=3))) == 1


def test_not_visible_and_unknown_area():
    index = np.zeros((10, 10), dtype=np.int32)
    index[0:5, 0:5] = 2
    anns = label_scene(index, [_obj(1, 50), _obj(2, 0.0)], _cfg())
    assert [a["instance_id"] for a in anns] == [2]  # 1 not visible; 2 has no area estimate


def test_distractors_ignore_vs_omit():
    index = np.zeros((10, 10), dtype=np.int32)
    index[0:5, 0:5] = 3
    dis = InstanceInfo(3, "distractor", "distractor", 25)
    cfg = _cfg()
    (ann,) = label_scene(index, [dis], cfg)
    cats = {c["name"]: c["id"] for c in coco_categories(cfg)}
    assert ann["iscrowd"] == 1 and ann["category_id"] == cats["distractor"]
    raw = yaml.safe_load(SMALL.read_text())
    raw["distractors"] = {"label": "omit"}
    omit = parse_config(raw)
    assert label_scene(index, [dis], omit) == []
    assert "distractor" not in [c["name"] for c in coco_categories(omit)]


def test_categories_in_config_order():
    assert [c["name"] for c in coco_categories(_cfg())] == [
        "box",
        "cylinder",
        "torus",
        "distractor",
    ]
    assert [c["id"] for c in coco_categories(_cfg())] == [1, 2, 3, 4]


def test_depth_encoding(tmp_path):
    depth = np.array([[0.5, 1.2345, 9.999], [10.5, np.inf, 1e10]], dtype=np.float32)
    mm = depth_to_mm(depth, max_m=10.0)
    assert mm.dtype == np.uint16
    assert mm.tolist() == [[500, 1234, 9999], [0, 0, 0]] or mm.tolist() == [
        [500, 1235, 9999],
        [0, 0, 0],
    ]
    write_depth_png(depth, 10.0, tmp_path / "d.png")
    back = read_depth_png(tmp_path / "d.png")
    assert back[0, 0] == pytest.approx(0.5) and np.isnan(back[1, 0])


def _record(i, anns, w=10, h=10):
    return {
        "scene_index": i,
        "file_name": f"images/scene_{i:06d}.png",
        "depth_file": f"depth/scene_{i:06d}.png",
        "scene_file": f"scenes/scene_{i:06d}.usda",
        "width": w,
        "height": h,
        "annotations": anns,
    }


def _anns(seed):
    rng = np.random.default_rng(seed)
    index = np.zeros((10, 10), dtype=np.int32)
    for iid in (1, 2):
        x, y = rng.integers(0, 6, size=2)
        index[y : y + 4, x : x + 4] = iid
    return label_scene(index, [_obj(1, 16), _obj(2, 16, "torus")], _cfg(min_visible_fraction=0))


def test_coco_ids_follow_scene_order_not_record_order(tmp_path):
    recs = [_record(i, _anns(i)) for i in (3, 0, 7)]
    a = build_coco(recs, _cfg())
    b = build_coco(list(reversed(recs)), _cfg())
    assert a == b
    assert [im["id"] for im in a["images"]] == [0, 3, 7]
    assert [an["id"] for an in a["annotations"]] == list(range(1, len(a["annotations"]) + 1))
    with pytest.raises(ValueError, match="duplicate"):
        build_coco(recs + recs[:1], _cfg())


def test_valid_coco_passes_and_corruptions_are_caught(tmp_path):
    data = build_coco([_record(i, _anns(i)) for i in range(4)], _cfg())
    path = tmp_path / "annotations.json"
    dump_json(data, path)
    assert validate_coco(path) == []

    bad = json.loads(path.read_text())
    bad["annotations"][0]["bbox"] = [9.0, 9.0, 1.0, 1.0]
    bad["annotations"][1]["category_id"] = 99
    bad["annotations"][2]["segmentation"]["size"] = [5, 5]
    dump_json(bad, path)
    errors = validate_coco(path)
    assert any("does not enclose" in e for e in errors)
    assert any("unknown category_id 99" in e for e in errors)
    assert any("mask" in e and "image size" in e for e in errors)


def test_non_finite_bbox_or_area_is_caught(tmp_path):
    data = build_coco([_record(i, _anns(i)) for i in range(2)], _cfg())
    data["annotations"][0]["bbox"][2] = float("nan")
    data["annotations"][1]["area"] = float("inf")
    path = tmp_path / "annotations.json"
    dump_json(data, path)  # Python's json writes NaN/Infinity and reads them back
    errors = validate_coco(path)
    assert sum("non-finite" in e for e in errors) == 2


def test_distractor_category_name_is_reserved():
    from usd_synth.config import ConfigError

    raw = yaml.safe_load(SMALL.read_text())
    raw["objects"]["library"][0]["category"] = "distractor"
    with pytest.raises(ConfigError, match="objects.library: category 'distractor' is reserved"):
        parse_config(raw)
    raw["distractors"] = {"label": "omit"}
    assert parse_config(raw).categories()[0] == "distractor"  # fine when distractors are omitted


def test_manifest_records_provenance():
    cfg = _cfg()
    m = build_manifest(cfg, 42, [_record(5, [])], {"blender": "5.2.2", "device": "cpu"}, "/cfg")
    assert m["config_hash"] == cfg.config_hash() and m["dataset_seed"] == 42
    assert m["blender"] == "5.2.2" and m["device"] == "cpu" and m["samples"] == cfg.render.samples
    assert m["images"][0]["scene_index"] == 5 and m["images"][0]["image_id"] == 5


def test_manifest_seeds_come_from_the_seed_tree_in_scene_order():
    cfg = _cfg()
    m = build_manifest(cfg, 42, [_record(i, []) for i in (7, 2)], {}, "/cfg")
    assert [im["scene_index"] for im in m["images"]] == [2, 7]
    for im in m["images"]:
        assert im["scene_seed"] == scene_seed(42, im["scene_index"])
        assert im["render_seed"] == render_seed(42, im["scene_index"])
    assert m["resolution"] == [cfg.image.width, cfg.image.height]
    assert m["blender"] is None and m["device"] is None  # absent from render_info


def test_dump_json_is_sorted_indented_and_newline_terminated(tmp_path):
    dump_json({"b": 1, "a": [1]}, tmp_path / "x.json")
    assert (tmp_path / "x.json").read_text() == '{\n "a": [\n  1\n ],\n "b": 1\n}\n'


def _separated_anns():
    # two 4x4 instances that never overlap, so each annotation's mask, box and area are known
    index = np.zeros((10, 10), dtype=np.int32)
    index[0:4, 0:4] = 1
    index[5:9, 5:9] = 2
    return label_scene(index, [_obj(1, 16), _obj(2, 16, "torus")], _cfg())


def test_each_remaining_validator_check_reports_its_annotation(tmp_path):
    data = build_coco([_record(i, _separated_anns()) for i in range(3)], _cfg())
    path = tmp_path / "annotations.json"
    dump_json(data, path)
    assert validate_coco(path) == []

    anns = data["annotations"]
    anns[0]["image_id"] = 999
    anns[1]["segmentation"] = encode_rle(np.zeros((10, 10), dtype=bool))
    anns[2]["bbox"] = [0.0, 0.0, 4.0, 4.0, 1.0]
    anns[3]["bbox"] = [5.0, 5.0, 0.0, 4.0]
    anns[4]["area"] = anns[4]["area"] + 5
    dump_json(data, path)
    assert validate_coco(path) == [
        "annotation 1: unknown image_id 999",
        "annotation 2: empty mask",
        "annotation 3: non-finite or malformed bbox/area [0.0, 0.0, 4.0, 4.0, 1.0, 16.0]",
        "annotation 4: bbox [5.0, 5.0, 0.0, 4.0] has no extent",
        "annotation 5: area 21.0 != mask pixels 16",
    ]
