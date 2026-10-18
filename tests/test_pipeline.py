"""End-to-end local pipeline, resume and reproduce (MVP criteria 7, 8 and the first half of 11)."""

import json
import os
import re
import shutil
from pathlib import Path

import pytest
import yaml

from usd_synth.cli import main
from usd_synth.coco import validate_coco
from usd_synth.config import parse_config
from usd_synth.pipeline import check_run, run_shard, scene_dir
from usd_synth.reproduce import RGB_MAD_THRESHOLD, reproduce

pytestmark = pytest.mark.blender
ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "tests" / "data" / "small.yaml"
THREADS = int(os.environ.get("USD_SYNTH_TEST_THREADS", "4"))


@pytest.fixture(scope="module")
def dataset(tmp_path_factory, blender):
    """`usd-synth generate --count 4` at 160x120 with 4 samples."""
    tmp = tmp_path_factory.mktemp("ds")
    cfg_path = tmp / "small.yaml"
    raw = yaml.safe_load(SMALL.read_text())
    raw["render"].update(threads=THREADS, batch_size=3)
    cfg_path.write_text(yaml.safe_dump(raw))
    out = tmp / "ds"
    rc = main(["generate", "--config", str(cfg_path), "--seed", "42", "--count", "4",
               "--out", str(out)])  # fmt: skip
    assert rc == 0
    return cfg_path, out


def test_generate_writes_complete_valid_dataset(dataset):
    _, out = dataset
    for sub, n in [("images", 4), ("depth", 4), ("scenes", 4)]:
        assert len(list((out / sub).iterdir())) == n
    assert validate_coco(out / "annotations.json") == []
    coco = json.loads((out / "annotations.json").read_text())
    assert [im["id"] for im in coco["images"]] == [0, 1, 2, 3]
    assert len(coco["annotations"]) > 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["dataset_seed"] == 42 and manifest["blender"].startswith("5.2")
    assert [e["scene_index"] for e in manifest["images"]] == [0, 1, 2, 3]
    timings = json.loads((scene_dir(out, 0, 0) / "timings.json").read_text())["timings"]
    assert set(timings) == {"generate", "blender_start", "import", "render", "label", "write"}


def test_reproduce_matches_original(dataset, tmp_path, blender, capsys):
    _, out = dataset
    report = reproduce(out, 2, tmp_path / "repro", blender, threads=THREADS)
    assert report["usda_identical"] and report["annotations_identical"]
    assert report["rgb_mean_abs_diff"] <= RGB_MAD_THRESHOLD
    print(f"\nreproduce RGB mean abs diff: {report['rgb_mean_abs_diff']}")  # recorded in CI logs


def test_rerun_completes_only_remaining_scenes(dataset, blender):
    cfg_path, out = dataset
    cfg = parse_config(yaml.safe_load(cfg_path.read_text()))
    # interrupted before DONE (scene 1) and before anything was written (scene 3)
    (scene_dir(out, 0, 1) / "DONE").unlink()
    shutil.rmtree(scene_dir(out, 0, 3))
    before = (scene_dir(out, 0, 0) / "timings.json").read_text()
    result = run_shard(cfg, cfg_path.parent, 42, [0, 1, 2, 3], out, blender, threads=THREADS)
    assert result.rendered == [1, 3] and result.skipped == [0, 2]
    assert (scene_dir(out, 0, 0) / "timings.json").read_text() == before  # untouched
    rc = main(["generate", "--config", str(cfg_path), "--seed", "42", "--count", "4",
               "--out", str(out)])  # fmt: skip
    assert rc == 0 and validate_coco(out / "annotations.json") == []


def test_resume_refuses_a_different_run(dataset):
    cfg_path, out = dataset
    cfg = parse_config(yaml.safe_load(cfg_path.read_text()))
    with pytest.raises(ValueError, match="different run"):
        check_run(out, cfg, 43, 4)


def test_readme_quickstart_runs_as_written(tmp_path, monkeypatch, blender):
    """Every `usd-synth` line of the README quick-start block runs as written (criterion 11)."""
    readme = (ROOT / "README.md").read_text()
    block = re.search(r"<!-- quickstart -->\s*```bash\n(.*?)```", readme, re.S).group(1)
    commands = [line.split() for line in block.splitlines() if line.startswith("usd-synth ")]
    assert any(c[1] == "generate" for c in commands) and any(c[1] == "reproduce" for c in commands)
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    monkeypatch.chdir(tmp_path)
    for cmd in commands:
        extra = ["--threads", str(THREADS)] if cmd[1] in ("generate", "reproduce") else []
        assert main(cmd[1:] + extra) == 0, " ".join(cmd)
    assert (tmp_path / "out" / "ds" / "annotations.json").exists()
