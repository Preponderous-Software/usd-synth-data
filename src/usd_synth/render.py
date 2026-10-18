"""Host side of rendering: describe a batch of scenes as JSON and run the Blender worker on it."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .seeds import render_seed

WORKER = Path(__file__).with_name("blender_worker.py")
NS = "userProperties:usd_synth:"


class RenderError(RuntimeError):
    pass


def find_blender(explicit: str | None = None) -> str:
    """Blender binary: ``--blender`` > ``$USD_SYNTH_BLENDER`` > ``blender`` on ``PATH``."""
    candidate = explicit or os.environ.get("USD_SYNTH_BLENDER") or shutil.which("blender")
    if not candidate or not Path(candidate).exists():
        raise RenderError(
            "Blender not found: pass --blender, set USD_SYNTH_BLENDER, or put blender on PATH"
        )
    return str(candidate)


def scene_spec(usd_path: Path | str, out_dir: Path | str) -> dict:
    """Per-scene render inputs read from the stage (camera intrinsics, dome light, seed)."""
    from pxr import Usd, UsdLux

    stage = Usd.Stage.Open(str(usd_path))
    world = stage.GetPrimAtPath("/World")
    index = world.GetAttribute(NS + "scene_index").Get()
    seed = world.GetAttribute(NS + "dataset_seed").Get()
    cam = stage.GetPrimAtPath("/World/Camera")
    spec = {
        "usd": str(Path(usd_path).resolve()),
        "out_dir": str(Path(out_dir).resolve()),
        "scene_index": int(index),
        "seed": render_seed(int(seed), int(index)),
        "camera": {
            "focal_length_mm": cam.GetAttribute(NS + "focal_length_mm").Get(),
            "sensor_width_mm": cam.GetAttribute(NS + "sensor_width_mm").Get(),
        },
        "dome": None,
    }
    dome = stage.GetPrimAtPath("/World/Lights/Dome")
    if dome and dome.IsA(UsdLux.DomeLight):
        light = UsdLux.DomeLight(dome)
        texture = light.GetTextureFileAttr().Get()
        spec["dome"] = {
            "intensity": float(light.GetIntensityAttr().Get()),
            "color": [float(c) for c in light.GetColorAttr().Get()],
            "texture": texture.resolvedPath or texture.path if texture else None,
        }
    return spec


@dataclass
class BatchResult:
    blender_start_s: float
    scenes: list[dict] = field(default_factory=list)  # one render.json per finished scene
    errors: list[dict] = field(default_factory=list)


def render_batch(
    cfg: Config, specs: list[dict], blender: str, threads: int | None = None, timeout: float = 3600
) -> BatchResult:
    """Render ``specs`` in one Blender process; raises RenderError if any scene failed."""
    job = {
        "device": cfg.render.device,
        "samples": cfg.render.samples,
        "denoise": cfg.render.denoise,
        "threads": cfg.render.threads if threads is None else threads,
        "width": cfg.image.width,
        "height": cfg.image.height,
        "scenes": specs,
    }
    with tempfile.TemporaryDirectory(prefix="usd-synth-job-") as tmp:
        job_path = Path(tmp) / "job.json"
        job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
        cmd = [
            blender, "-b", "--factory-startup", "-noaudio",
            "--python-exit-code", "1", "--python", str(WORKER), "--", str(job_path),
        ]  # fmt: skip
        start = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    events = []
    for line in proc.stdout.splitlines():
        if line.startswith("USD_SYNTH "):
            events.append(json.loads(line[len("USD_SYNTH ") :]))
    ready = next((e for e in events if e["event"] == "ready"), None)
    if ready is None:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-30:])
        raise RenderError(f"Blender worker did not start (exit {proc.returncode}):\n{tail}")
    result = BatchResult(blender_start_s=max(ready["time"] - start, 0.0))
    for e in events:
        if e["event"] == "done":
            info = json.loads((Path(e["out_dir"]) / "render.json").read_text(encoding="utf-8"))
            result.scenes.append(info)
        elif e["event"] == "error":
            result.errors.append(e)
    if result.errors or proc.returncode != 0 or len(result.scenes) != len(specs):
        detail = "; ".join(f"{e['out_dir']}: {e['message']}" for e in result.errors)
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-20:])
        raise RenderError(f"Blender worker failed (exit {proc.returncode}): {detail}\n{tail}")
    return result
