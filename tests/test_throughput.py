import pytest

from usd_synth.pipeline import shard_range
from usd_synth.throughput import STAGES, summarize, summary_line

CFG = {"render": {"samples": 4, "denoise": False}, "image": {"width": 160, "height": 120}}


def _scene(shard, k, render_s):
    timings = dict.fromkeys(STAGES, 0.01) | {"render": render_s}
    env = {"blender": "5.2.2", "device": "cpu", "threads": 4, "cpu_model": "x", "host": "h"}
    return {"shard": shard, "started": 100.0 + k, "finished": 101.0 + k, "timings": timings,
            "render": env}  # fmt: skip


def test_summary_shape_matches_documented_report():
    t = summarize([_scene(0, k, float(k + 1)) for k in range(20)], CFG)
    assert set(t) == {"images", "wall_seconds", "images_per_hour", "stages", "shards",
                      "environment"}  # fmt: skip
    assert set(t["stages"]) == set(STAGES)
    assert t["stages"]["render"]["mean_s"] == pytest.approx(10.5)
    assert t["stages"]["render"]["p95_s"] == pytest.approx(19.05)
    assert t["images"] == 20 and t["wall_seconds"] == 20.0 and t["images_per_hour"] == 3600.0
    env = t["environment"]
    assert env["resolution"] == [160, 120] and env["samples"] == 4 and env["gpu"] is None
    assert "images/hour" in summary_line(t)


def test_summarize_requires_scenes():
    with pytest.raises(ValueError):
        summarize([], CFG)


@pytest.mark.parametrize("count, n", [(10, 3), (7, 7), (3, 5), (200, 50), (1, 1)])
def test_shard_ranges_partition_the_scenes(count, n):
    ranges = [shard_range(count, n, k) for k in range(n)]
    flat = [i for r in ranges for i in r]
    assert flat == list(range(count))  # contiguous, ordered, disjoint, complete


def test_shard_range_validation():
    with pytest.raises(ValueError):
        shard_range(10, 2, 2)
    with pytest.raises(ValueError):
        shard_range(0, 1, 0)
