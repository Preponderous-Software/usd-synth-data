import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from scene_fixtures import top_down_scene, write

from usd_synth.cli import main
from usd_synth.config import load_config, parse_config
from usd_synth.passes import read_passes
from usd_synth.render import RenderError, render_batch, scene_spec
from usd_synth.scenegen import write_scene

pytestmark = pytest.mark.blender
ROOT = Path(__file__).resolve().parents[1]
BUNDLED = ROOT / "configs" / "tabletop.yaml"
SMALL = ROOT / "tests" / "data" / "small.yaml"
THREADS = int(os.environ.get("USD_SYNTH_TEST_THREADS", "4"))


def _fixture_cfg(width=160, height=120):
    raw = yaml.safe_load(SMALL.read_text())
    raw["image"] = {"width": width, "height": height}
    return parse_config(raw)


@pytest.fixture(scope="module")
def three_cubes(tmp_path_factory, blender):
    """Three separated cubes seen straight down from 1 m, rendered once for several tests."""
    tmp = tmp_path_factory.mktemp("cubes")
    scene = top_down_scene([(-0.15, 0.0, 0.08), (0.0, 0.08, 0.06), (0.15, -0.05, 0.1)])
    usd = write(scene, tmp / "scene_000000.usda")
    no_props = tmp / "noprops" / "scene_000000.usda"
    no_props.parent.mkdir()
    lines = usd.read_text().splitlines()
    no_props.write_text("\n".join(x for x in lines if "usd_synth:instance_id" not in x) + "\n")
    specs = [scene_spec(usd, tmp / "out"), scene_spec(no_props, tmp / "out_noprops")]
    result = render_batch(_fixture_cfg(), specs, blender, threads=THREADS)
    return scene, tmp, result


def test_bundled_config_640x480_renders_on_cpu(tmp_path, blender):
    """MVP acceptance criterion 5 with the bundled config as shipped (640x480, its samples)."""
    cfg = load_config(BUNDLED)
    usd = write_scene(cfg, 42, 0, tmp_path / "scenes", BUNDLED.parent)
    result = render_batch(cfg, [scene_spec(usd, tmp_path / "out")], blender, threads=THREADS)
    info = result.scenes[0]
    assert info["device"] == "cpu" and info["resolution"] == [640, 480]
    rgb = Image.open(tmp_path / "out" / "rgb.png")
    assert rgb.size == (640, 480) and rgb.mode == "RGB"
    passes = read_passes(tmp_path / "out" / "aux.exr")
    assert passes.depth.shape == passes.index.shape == (480, 640)
    assert passes.index.max() >= 1  # at least one labelled instance is visible
    assert 10 < np.asarray(rgb).mean() < 245  # neither black nor blown out


def test_index_channel_contains_exactly_the_instance_ids(three_cubes):
    _, tmp, _ = three_cubes
    index = read_passes(tmp / "out" / "aux.exr").index
    assert set(np.unique(index).tolist()) == {0, 1, 2, 3}


def test_index_pass_has_no_edge_blending(three_cubes):
    """read_passes rejects non-integral index values; here we also check raw edge pixels."""
    import OpenEXR

    _, tmp, _ = three_cubes
    with OpenEXR.File(str(tmp / "out" / "aux.exr"), separate_channels=True) as exr:
        channels = {n: c for part in exr.parts for n, c in part.channels.items()}
        raw = next(c.pixels for n, c in channels.items() if "Object Index" in n)
    assert np.array_equal(raw, np.rint(raw))


def test_instance_ids_come_from_custom_properties_on_the_object(three_cubes):
    _, _, result = three_cubes
    with_props, without_props = result.scenes
    expected = {"inst_0001_geom": 1, "inst_0002_geom": 2, "inst_0003_geom": 3}
    assert with_props["pass_indices"] == expected
    # prim-name fallback: same IDs when the userProperties attribute is absent
    assert without_props["pass_indices"] == expected


def test_depth_is_planar(three_cubes):
    """Camera 1 m above the table top: planar depth is 1 m everywhere on the table, even in the
    image corners, where radial distance would be about 19 % larger."""
    _, tmp, _ = three_cubes
    passes = read_passes(tmp / "out" / "aux.exr")
    table = passes.index == 0
    assert table[0, 0] and table[-1, -1]
    np.testing.assert_allclose(passes.depth[table], 1.0, atol=1e-3)
    # top face of cube 3 (0.1 m tall) is 0.9 m from the camera
    np.testing.assert_allclose(np.median(passes.depth[passes.index == 3]), 0.9, atol=1e-3)


def test_render_json_records_environment(three_cubes):
    _, tmp, _ = three_cubes
    info = json.loads((tmp / "out" / "render.json").read_text())
    assert info["blender"].startswith("5.2")
    assert info["samples"] == 4 and info["threads"] == THREADS
    assert set(info["timings"]) == {"import", "render", "write"}


def test_cycles_seed_comes_from_render_stream(three_cubes):
    from usd_synth.seeds import render_seed

    _, _, result = three_cubes
    assert result.scenes[0]["seed"] == render_seed(0, 0)


def test_missing_scene_is_a_render_error(tmp_path, blender):
    spec = {
        "usd": str(tmp_path / "missing.usda"),
        "out_dir": str(tmp_path / "o"),
        "seed": 1,
        "camera": {"focal_length_mm": 35.0, "sensor_width_mm": 36.0},
        "dome": None,
    }
    with pytest.raises(RenderError):
        render_batch(_fixture_cfg(), [spec], blender, threads=2)


def test_cli_render(tmp_path, blender, capsys):
    cfg_path = tmp_path / "c.yaml"
    raw = yaml.safe_load(SMALL.read_text())
    raw["render"]["threads"] = THREADS
    cfg_path.write_text(yaml.safe_dump(raw))
    assert main(["scenes", "--config", str(cfg_path), "--seed", "1", "--count", "2",
                 "--out", str(tmp_path / "s")]) == 0  # fmt: skip
    assert main(["render", "--config", str(cfg_path), "--scenes", str(tmp_path / "s"),
                 "--out", str(tmp_path / "r")]) == 0  # fmt: skip
    for name in ("scene_000000", "scene_000001"):
        assert (tmp_path / "r" / name / "rgb.png").exists()
        assert (tmp_path / "r" / name / "aux.exr").exists()


def test_distractors_get_their_own_pass_index(tmp_path, blender):
    scene = top_down_scene([(-0.1, 0.0, 0.08), (0.1, 0.0, 0.06)])
    scene.instances[1].kind = "distractor"
    scene.instances[1].category_id = 0
    scene.instances[1].category_name = "distractor"
    usd = write(scene, tmp_path / "scene_000000.usda")
    assert "dis_0002" in usd.read_text()
    result = render_batch(_fixture_cfg(), [scene_spec(usd, tmp_path / "o")], blender, threads=2)
    assert result.scenes[0]["pass_indices"] == {"inst_0001_geom": 1, "dis_0002_geom": 2}
    assert set(np.unique(read_passes(tmp_path / "o" / "aux.exr").index).tolist()) == {0, 1, 2}


def test_timeout_is_a_render_error(tmp_path, blender):
    usd = write(top_down_scene([(0.0, 0.0, 0.1)]), tmp_path / "scene_000000.usda")
    with pytest.raises(RenderError, match="timed out"):
        render_batch(_fixture_cfg(), [scene_spec(usd, tmp_path / "o")], blender, timeout=0.01)
