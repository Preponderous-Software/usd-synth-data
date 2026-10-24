"""Throughput report: images/hour overall and per shard, and per-stage mean/p95 timings."""

from __future__ import annotations

import numpy as np

STAGES = ("generate", "blender_start", "import", "render", "label", "write")


def _rate(images: int, seconds: float) -> float:
    return round(images / seconds * 3600.0, 2) if seconds > 0 else 0.0


def summarize(scenes: list[dict], cfg_resolved: dict) -> dict:
    """``scenes``: one dict per finished scene with ``shard``, ``started``, ``finished``,
    ``timings`` (per stage, seconds) and ``render`` (environment facts).

    Wall time is first start to last finish, so it includes start-up and scheduling gaps; a
    resumed run spans its interruption too.
    """
    if not scenes:
        raise ValueError("no finished scenes to summarize")
    first = min(s["started"] for s in scenes)
    last = max(s["finished"] for s in scenes)
    wall = last - first
    stages = {}
    for stage in STAGES:
        values = np.array([s["timings"][stage] for s in scenes], dtype=np.float64)
        stages[stage] = {
            "mean_s": round(float(values.mean()), 4),
            "p95_s": round(float(np.percentile(values, 95)), 4),
        }
    shards = []
    for shard in sorted({s["shard"] for s in scenes}):
        mine = [s for s in scenes if s["shard"] == shard]
        span = max(s["finished"] for s in mine) - min(s["started"] for s in mine)
        shards.append(
            {
                "shard": shard,
                "images": len(mine),
                "wall_seconds": round(span, 3),
                "images_per_hour": _rate(len(mine), span),
                "host": mine[0]["render"].get("host"),
            }
        )
    env = scenes[0]["render"]
    cycles = cfg_resolved["render"].get("engine", "cycles") == "cycles"
    return {
        "images": len(scenes),
        "wall_seconds": round(wall, 3),
        "images_per_hour": _rate(len(scenes), wall),
        "stages": stages,
        "shards": shards,
        "environment": {
            "engine": env.get("engine", cfg_resolved["render"].get("engine", "cycles")),
            "blender": env.get("blender"),
            "strata": env.get("strata"),
            "gl": env.get("gl"),
            "device": env.get("device"),
            "threads": env.get("threads"),
            "cpu_model": env.get("cpu_model"),
            "gpu": None if env.get("device") == "cpu" else env.get("device"),
            "samples": cycles and cfg_resolved["render"]["samples"] or None,
            "denoise": cfg_resolved["render"]["denoise"] if cycles else None,
            "resolution": [cfg_resolved["image"]["width"], cfg_resolved["image"]["height"]],
            "shard_count": len(shards),
            "hosts": sorted({s["render"].get("host") for s in scenes} - {None}),
        },
    }


def summary_line(t: dict) -> str:
    st = t["stages"]
    return (
        f"{t['images']} images in {t['wall_seconds']:.1f} s -> {t['images_per_hour']:.1f} "
        f"images/hour ({len(t['shards'])} shard(s); render mean {st['render']['mean_s']:.2f} s, "
        f"p95 {st['render']['p95_s']:.2f} s; {_starter(t)} start mean "
        f"{st['blender_start']['mean_s']:.2f} s)"
    )


def _starter(t: dict) -> str:
    return "strata" if t.get("environment", {}).get("engine") == "strata" else "blender"
