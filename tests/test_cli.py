"""CLI argument checks and error paths that run before any renderer is needed."""

import json
from pathlib import Path

import pytest

from usd_synth import __version__
from usd_synth.cli import main
from usd_synth.config import load_config

ROOT = Path(__file__).resolve().parents[1]
BUNDLED = ROOT / "configs" / "tabletop.yaml"
SMALL = ROOT / "tests" / "data" / "small.yaml"


def _generate(out, *extra):
    return main(["generate", "--config", str(SMALL), "--seed", "1", "--count", "4",
                 "--out", str(out), *extra])  # fmt: skip


@pytest.mark.parametrize(
    "extra, message",
    [(["--count", "0"], "--count must be >= 1"),
     (["--workers", "0"], "--workers must be >= 1"),
     (["--shard-index", "1"], "--shard-index and --shard-count must be given together"),
     (["--shard-count", "4"], "--shard-index and --shard-count must be given together"),
     (["--seed", "-1"], "dataset seed must be in [0, 2**63 - 1] (got -1)")],
)  # fmt: skip
def test_generate_rejects_bad_arguments_before_writing(
    tmp_path, monkeypatch, capsys, extra, message
):
    monkeypatch.delenv("JOB_COMPLETION_INDEX", raising=False)
    monkeypatch.setattr("usd_synth.render.find_blender", lambda explicit=None: pytest.fail())
    out = tmp_path / "out"
    assert _generate(out, *extra) == 1  # later flags override the defaults above
    assert capsys.readouterr().err == f"usd-synth: {message}\n"
    assert not out.exists()


def test_render_with_no_scenes_is_a_clean_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("usd_synth.render.find_blender", lambda explicit=None: pytest.fail())
    empty = tmp_path / "scenes"
    empty.mkdir()
    rc = main(["render", "--config", str(SMALL), "--scenes", str(empty),
               "--out", str(tmp_path / "out")])  # fmt: skip
    assert rc == 1
    assert capsys.readouterr().err == f"usd-synth render: no .usda files in {empty}\n"


def test_merge_without_wait_fails_at_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("time.sleep", lambda s: pytest.fail("should not wait"))
    assert main(["merge", "--out", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("usd-synth: no shards found under ")
    assert "waiting for shards" not in captured.out


def test_merge_wait_gives_up_at_the_deadline(tmp_path, monkeypatch, capsys):
    clock = [100.0]
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    monkeypatch.setattr("time.sleep", fake_sleep)
    assert main(["merge", "--out", str(tmp_path), "--wait", "5", "--poll", "2"]) == 1
    assert sleeps == [2.0, 2.0, 2.0]  # checks at t=0, 2, 4 wait; the check at t=6 gives up
    captured = capsys.readouterr()
    assert captured.out.count("waiting for shards: no shards found") == 3
    assert captured.err.startswith("usd-synth: no shards found under ")


def test_reproduce_without_manifest_names_the_missing_file(tmp_path, capsys):
    rc = main(["reproduce", "--dataset", str(tmp_path), "--image-id", "1",
               "--out", str(tmp_path / "out")])  # fmt: skip
    assert rc == 1
    assert capsys.readouterr().err == f"usd-synth: file not found: {tmp_path / 'manifest.json'}\n"


def test_config_resolved_prints_json_then_hash(capsys):
    assert main(["config", "--config", str(BUNDLED), "--resolved"]) == 0
    *body, last = capsys.readouterr().out.splitlines()
    cfg = load_config(BUNDLED)
    assert json.loads("\n".join(body)) == json.loads(json.dumps(cfg.resolved()))
    assert last == f"config OK: {BUNDLED} sha256={cfg.config_hash()}"


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out == f"usd-synth {__version__}\n"
