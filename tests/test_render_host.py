"""Host side of rendering, without Blender: finding the binary, the job, the worker's events.

``tests/test_render.py`` runs the real worker, and only under the Blender job. Here a fake
``subprocess.run`` returns canned worker output, so the ways a batch can fail are checked in
``lint and test`` with exact messages, including ones a healthy Blender never produces.
"""

import json
import subprocess
import time
from pathlib import Path

import pytest
import yaml
from scene_fixtures import top_down_scene, write

from usd_synth.config import parse_config
from usd_synth.render import (
    WORKER,
    RenderError,
    find_blender,
    find_renderer,
    render_batch,
    scene_spec,
)
from usd_synth.scenegen import write_scene
from usd_synth.seeds import render_seed

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "tests" / "data" / "small.yaml"


def _raw():
    return yaml.safe_load(SMALL.read_text())


def _cfg(**render):
    raw = _raw()
    raw["render"].update(render)
    return parse_config(raw)


def _specs(tmp_path, count=1):
    return [{"out_dir": str(tmp_path / f"scene_{k}")} for k in range(count)]


def _event(event, **fields):
    return "USD_SYNTH " + json.dumps({"event": event, **fields}, sort_keys=True)


READY = _event("ready", time=0.0)


def _done(spec, **info):
    """The ``done`` event, plus the render.json the worker writes before logging it."""
    out_dir = Path(spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "render.json").write_text(json.dumps(info), encoding="utf-8")
    return _event("done", out_dir=spec["out_dir"])


def _error(spec, message):
    return _event("error", out_dir=spec["out_dir"], message=message)


class FakeWorker:
    """Replaces ``subprocess.run``: records the call and the job, returns canned output."""

    def __init__(self, stdout, stderr, returncode):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode

    def __call__(self, cmd, **kwargs):
        self.cmd, self.kwargs = cmd, kwargs
        self.job = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, self.stderr)


@pytest.fixture
def worker(monkeypatch):
    def install(lines, stderr="", returncode=0):
        fake = FakeWorker("".join(f"{line}\n" for line in lines), stderr, returncode)
        monkeypatch.setattr(subprocess, "run", fake)
        return fake

    return install


# ------------------------------------------------------------------------------- the job


def test_job_carries_the_config_and_specs(tmp_path, worker):
    specs = _specs(tmp_path, 2)
    fake = worker([READY, _done(specs[0]), _done(specs[1])])
    render_batch(_cfg(), specs, "blender-bin")
    assert fake.job == {
        "device": "cpu",
        "samples": 4,
        "denoise": False,
        "threads": 2,
        "width": 160,
        "height": 120,
        "scenes": specs,
    }


def test_worker_runs_headless_and_its_job_file_is_cleaned_up(tmp_path, worker):
    specs = _specs(tmp_path)
    fake = worker([READY, _done(specs[0])])
    render_batch(_cfg(), specs, "blender-bin", timeout=12.5)
    assert fake.cmd[:-1] == [
        "blender-bin", "-b", "--factory-startup", "-noaudio",
        "--python-exit-code", "1", "--python", str(WORKER), "--",
    ]  # fmt: skip
    assert fake.kwargs == {"capture_output": True, "text": True, "timeout": 12.5}
    assert not Path(fake.cmd[-1]).exists()


@pytest.mark.parametrize("threads, expected", [(None, 2), (0, 0), (7, 7)])
def test_threads_argument_overrides_the_config_even_when_zero(tmp_path, worker, threads, expected):
    specs = _specs(tmp_path)
    fake = worker([READY, _done(specs[0])])
    render_batch(_cfg(threads=2), specs, "blender-bin", threads=threads)
    assert fake.job["threads"] == expected


# ---------------------------------------------------------------------- the worker's events


def test_done_events_collect_render_json_and_other_output_is_ignored(tmp_path, worker):
    specs = _specs(tmp_path, 2)
    lines = [
        "Blender 5.2.2",
        READY,
        "Fra:1 Mem:12.00M | Sample 1/4",
        "a line that mentions USD_SYNTH {not json",
        _done(specs[0], seed=1),
        _done(specs[1], seed=2),
    ]
    worker(lines)
    result = render_batch(_cfg(), specs, "blender-bin")
    assert result.scenes == [{"seed": 1}, {"seed": 2}]
    assert result.errors == []


def test_blender_start_is_the_time_until_ready(tmp_path, worker):
    specs = _specs(tmp_path)
    worker([_event("ready", time=time.time() + 60.0), _done(specs[0])])
    assert 50.0 < render_batch(_cfg(), specs, "blender-bin").blender_start_s <= 60.0


def test_blender_start_is_never_negative(tmp_path, worker):
    specs = _specs(tmp_path)
    worker([READY, _done(specs[0])])  # the ready timestamp is long before the call started
    assert render_batch(_cfg(), specs, "blender-bin").blender_start_s == 0.0


def test_no_ready_event_reports_the_exit_code_and_the_output_tail(tmp_path, worker):
    worker([f"line {k}" for k in range(40)], stderr="boom\n", returncode=3)
    with pytest.raises(RenderError, match=r"did not start \(exit 3\)") as exc:
        render_batch(_cfg(), _specs(tmp_path), "blender-bin")
    message = str(exc.value)
    assert "boom" in message and "line 11" in message  # the last 30 lines of stdout + stderr
    assert "line 10" not in message


def test_failed_scenes_are_all_named_even_when_others_finished(tmp_path, worker):
    specs = _specs(tmp_path, 3)
    worker(
        [READY, _error(specs[0], "ValueError: a"), _done(specs[1]), _error(specs[2], "OSError: c")],
        returncode=1,
    )
    with pytest.raises(RenderError, match=r"worker failed \(exit 1\)") as exc:
        render_batch(_cfg(), specs, "blender-bin")
    named = f"{specs[0]['out_dir']}: ValueError: a; {specs[2]['out_dir']}: OSError: c"
    assert named in str(exc.value)


def test_nonzero_exit_fails_the_batch_even_when_every_scene_finished(tmp_path, worker):
    specs = _specs(tmp_path)
    worker([READY, _done(specs[0])], stderr="Segmentation fault\n", returncode=139)
    with pytest.raises(RenderError, match=r"exit 139") as exc:
        render_batch(_cfg(), specs, "blender-bin")
    assert "Segmentation fault" in str(exc.value)


def test_a_scene_with_no_event_fails_the_batch_despite_exit_zero(tmp_path, worker):
    specs = _specs(tmp_path, 2)
    worker([READY, _done(specs[0])])
    with pytest.raises(RenderError, match=r"worker failed \(exit 0\)"):
        render_batch(_cfg(), specs, "blender-bin")


# ---------------------------------------------------------------------- finding the renderer


def _file(path):
    path.write_bytes(b"")
    return str(path)


def test_find_blender_prefers_the_explicit_path_over_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("USD_SYNTH_BLENDER", _file(tmp_path / "from-env"))
    assert find_blender(_file(tmp_path / "explicit")) == str(tmp_path / "explicit")


def test_find_blender_prefers_the_environment_over_path(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _file(bin_dir / "blender")
    (bin_dir / "blender").chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("USD_SYNTH_BLENDER", _file(tmp_path / "from-env"))
    assert find_blender() == str(tmp_path / "from-env")


def test_find_blender_falls_back_to_path(tmp_path, monkeypatch):
    _file(tmp_path / "blender")
    (tmp_path / "blender").chmod(0o755)
    monkeypatch.delenv("USD_SYNTH_BLENDER", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert find_blender() == str(tmp_path / "blender")


def test_find_blender_missing_explicit_path_does_not_fall_through(tmp_path, monkeypatch):
    monkeypatch.setenv("USD_SYNTH_BLENDER", _file(tmp_path / "from-env"))
    with pytest.raises(RenderError, match="Blender not found"):
        find_blender(str(tmp_path / "nope"))


def test_find_blender_nothing_found_says_what_to_do(tmp_path, monkeypatch):
    monkeypatch.delenv("USD_SYNTH_BLENDER", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(RenderError, match="pass --blender, set USD_SYNTH_BLENDER"):
        find_blender()


def test_find_renderer_uses_blender_for_cycles(tmp_path):
    assert find_renderer(_cfg(), blender=_file(tmp_path / "b")) == str(tmp_path / "b")


def test_find_renderer_turns_a_missing_strata_into_a_render_error(tmp_path):
    with pytest.raises(RenderError, match="Strata library not found at"):
        find_renderer(_cfg(engine="strata"), strata=str(tmp_path / "nope.so"))


# ------------------------------------------------------------------------------ scene_spec


def _scene_with_dome(tmp_path, dome):
    raw = _raw()
    raw["lighting"] = {"dome": dome}
    return write_scene(parse_config(raw), 1, 0, tmp_path / "scenes", tmp_path)


def test_scene_spec_reads_the_stage(tmp_path):
    scene = top_down_scene([(0.0, 0.0, 0.1)], focal_length_mm=50.0, index=5, dataset_seed=9)
    usd = write(scene, tmp_path / "scene_000005.usda")
    spec = scene_spec(usd, tmp_path / "out")
    assert spec["usd"] == str(usd.resolve())
    assert spec["out_dir"] == str((tmp_path / "out").resolve())
    assert spec["scene_index"] == 5
    assert spec["seed"] == render_seed(9, 5)
    assert spec["camera"] == {"focal_length_mm": 50.0, "sensor_width_mm": 36.0}
    assert spec["dome"] == {"intensity": 0.5, "color": [1.0, 1.0, 1.0], "texture": None}


def test_scene_spec_without_a_dome_light(tmp_path):
    usd = _scene_with_dome(tmp_path, {"enabled": False})
    assert scene_spec(usd, tmp_path / "out")["dome"] is None


@pytest.mark.parametrize("exists", [True, False], ids=["resolved", "unresolved"])
def test_scene_spec_returns_the_dome_texture_path(tmp_path, exists):
    hdri = tmp_path / "sky.hdr"
    if exists:
        _file(hdri)
    usd = _scene_with_dome(tmp_path, {"hdris": ["sky.hdr"]})
    assert scene_spec(usd, tmp_path / "out")["dome"]["texture"] == str(hdri)
