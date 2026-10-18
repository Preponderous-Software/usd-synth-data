import copy
from pathlib import Path

import pytest
import yaml

from usd_synth.cli import main
from usd_synth.config import ConfigError, load_config, parse_config

ROOT = Path(__file__).resolve().parents[1]
BUNDLED = ROOT / "configs" / "tabletop.yaml"


@pytest.fixture
def raw():
    return yaml.safe_load(BUNDLED.read_text())


def test_bundled_config_validates():
    cfg = load_config(BUNDLED)
    assert cfg.scenario == "tabletop"
    assert cfg.categories() == ["box", "cylinder", "sphere", "capsule", "torus"]


def test_minimal_config_fills_defaults():
    cfg = parse_config(
        {
            "version": 1,
            "scenario": "tabletop",
            "objects": {
                "library": [
                    {
                        "name": "b",
                        "source": "primitive:box",
                        "category": "box",
                        "size_m": [0.05, 0.1],
                    }
                ]
            },
        }
    )
    resolved = cfg.resolved()
    assert resolved["image"] == {"width": 640, "height": 480}
    assert resolved["labels"]["min_visible_pixels"] == 50
    assert resolved["camera"]["distance_m"] == [0.6, 1.5]


def _errors_for(data):
    with pytest.raises(ConfigError) as exc:
        parse_config(data)
    return str(exc.value)


@pytest.mark.parametrize(
    "mutate, field",
    [
        (lambda d: d.__setitem__("colour", 1), "colour"),
        (lambda d: d["camera"].__setitem__("fov", 3), "camera.fov"),
        (lambda d: d["objects"].__setitem__("count", [12, 3]), "objects.count"),
        (lambda d: d["objects"].__setitem__("library", []), "objects.library"),
        (lambda d: d["image"].__setitem__("width", 0), "image.width"),
        (lambda d: d["materials"].__setitem__("roughness", [0.2, 1.5]), "materials.roughness"),
        (lambda d: d["materials"]["metallic"].__setitem__("p", [0.5, 0.6]), "materials.metallic"),
        (lambda d: d["lighting"].__setitem__("key_lights", [0, 2]), "lighting.key_lights"),
        (lambda d: d["camera"].__setitem__("elevation_deg", [10, 95]), "camera.elevation_deg"),
        (lambda d: d["render"].__setitem__("device", "tpu"), "render.device"),
        (lambda d: d["support"].__setitem__("kind", ["shelf"]), "support.kind.0"),
        (
            lambda d: d["objects"]["library"][0].__setitem__("source", "primitive:cone"),
            "objects.library.0",
        ),
        (lambda d: d["objects"]["library"][0].pop("size_m"), "objects.library.0"),
        (lambda d: d["objects"]["library"][1].__setitem__("name", "box"), "objects"),
        (lambda d: d.__setitem__("version", 2), "version"),
        (
            lambda d: d["materials"].__setitem__("roughness", [float("nan"), 0.9]),
            "materials.roughness.0",
        ),
        (
            lambda d: d["camera"].__setitem__("distance_m", [0.6, float("inf")]),
            "camera.distance_m.1",
        ),
        (lambda d: d["render"].__setitem__("samples", float("inf")), "render.samples"),
    ],
)
def test_invalid_config_names_offending_field(raw, mutate, field):
    data = copy.deepcopy(raw)
    mutate(data)
    message = _errors_for(data)
    assert f"\n  {field}:" in message, message


def test_top_level_must_be_mapping():
    assert "<root>" in _errors_for([1, 2])


def test_invalid_yaml_is_config_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("objects: [unclosed")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(bad)


def test_config_hash_is_stable_and_sensitive(raw):
    a = parse_config(raw)
    assert a.config_hash() == parse_config(copy.deepcopy(raw)).config_hash()
    changed = copy.deepcopy(raw)
    changed["render"]["samples"] = 8
    assert parse_config(changed).config_hash() != a.config_hash()


def test_hash_ignores_key_order_and_explicit_defaults(raw):
    reordered = dict(reversed(list(raw.items())))
    assert parse_config(reordered).config_hash() == parse_config(raw).config_hash()
    explicit = copy.deepcopy(raw)
    explicit["camera"]["jitter_target_m"] = 0.05  # same as the default
    assert parse_config(explicit).config_hash() == parse_config(raw).config_hash()


def test_cli_config_ok(capsys):
    assert main(["config", "--config", str(BUNDLED)]) == 0
    assert "config OK" in capsys.readouterr().out


def test_cli_config_invalid_names_field(tmp_path, raw, capsys):
    raw["objects"]["count"] = [5, 1]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    assert main(["config", "--config", str(bad)]) == 1
    assert "objects.count" in capsys.readouterr().err


def test_cli_missing_config_file_is_clean_error(tmp_path, capsys):
    assert main(["config", "--config", str(tmp_path / "nope.yaml")]) == 1
    assert "file not found" in capsys.readouterr().err


def test_cli_config_flag_is_required():
    with pytest.raises(SystemExit):
        main(["config"])
