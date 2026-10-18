"""Storage-sink contract, resume bookkeeping and merge — through the Sink interface only, so
the same tests run against every sink (local filesystem here; S3-compatible from M7)."""

import json
from pathlib import Path

import pytest
from sink_contract import fake_scene, fake_shard

from usd_synth.config import load_config
from usd_synth.merge import MergeError, merge
from usd_synth.storage import LocalSink, open_sink

SMALL = Path(__file__).resolve().parent / "data" / "small.yaml"


@pytest.fixture
def cfg():
    return load_config(SMALL)


@pytest.fixture(params=["local"])
def make_sink(request, tmp_path):
    """Factory of empty sinks; each call returns a fresh, isolated sink."""
    counter = iter(range(1000))
    return lambda: LocalSink(tmp_path / f"sink{next(counter)}")


def test_put_get_list_exists_copy(make_sink):
    sink = make_sink()
    sink.put_bytes("a/b/c.txt", b"hello")
    assert sink.get_bytes("a/b/c.txt") == b"hello"
    assert sink.exists("a/b/c.txt") and not sink.exists("a/b/missing")
    sink.copy("a/b/c.txt", "x/y.txt")
    assert sink.get_bytes("x/y.txt") == b"hello"
    assert sink.list("a/") == ["a/b/c.txt"]
    assert sink.list("nothing/") == []


def test_done_scenes_counts_only_done_objects(make_sink):
    sink = make_sink()
    fake_scene(sink, 0, 1)
    fake_scene(sink, 0, 2, done=False)  # files present, DONE missing: interrupted
    fake_scene(sink, 1, 5)
    assert sink.done_scenes(0) == {1}
    assert sink.done_scenes(1) == {5}


def _merged_annotations(sink):
    return json.loads(sink.get_bytes(sink.dataset_prefix + "annotations.json"))


def test_merge_is_independent_of_the_shard_split(make_sink, cfg):
    one = make_sink()
    fake_shard(one, cfg, 0, 1, 0, 6, 6)
    for i in range(6):
        fake_scene(one, 0, i)
    two = make_sink()
    fake_shard(two, cfg, 0, 2, 0, 3, 6)
    fake_shard(two, cfg, 1, 2, 3, 6, 6)
    for i in range(6):
        fake_scene(two, i // 3, i)
    assert merge(one)["coco"] == merge(two)["coco"]
    assert _merged_annotations(one) == _merged_annotations(two)
    assert two.get_bytes(two.dataset_prefix + "images/scene_000004.png") == b"rgb.png:4"


def test_merge_rejects_overlapping_ranges(make_sink, cfg):
    sink = make_sink()
    fake_shard(sink, cfg, 0, 2, 0, 4, 6)
    fake_shard(sink, cfg, 1, 2, 3, 6, 6)
    with pytest.raises(MergeError, match=r"shards 0 \[0, 4\) and 1 \[3, 6\) overlap"):
        merge(sink)


def test_merge_rejects_missing_ranges(make_sink, cfg):
    sink = make_sink()
    fake_shard(sink, cfg, 0, 3, 0, 2, 6)
    fake_shard(sink, cfg, 2, 3, 4, 6, 6)  # shard 1 never ran
    with pytest.raises(MergeError, match="missing 2 scene\\(s\\): 2-3"):
        merge(sink)


def test_merge_rejects_incomplete_scenes(make_sink, cfg):
    sink = make_sink()
    fake_shard(sink, cfg, 0, 1, 0, 3, 3)
    fake_scene(sink, 0, 0)
    fake_scene(sink, 0, 1, done=False)
    fake_scene(sink, 0, 2)
    with pytest.raises(MergeError, match="1 scene\\(s\\) not complete: 1"):
        merge(sink)


def test_merge_rejects_shards_of_different_runs(make_sink, cfg):
    sink = make_sink()
    fake_shard(sink, cfg, 0, 2, 0, 3, 6, seed=1)
    fake_shard(sink, cfg, 1, 2, 3, 6, 6, seed=2)
    with pytest.raises(MergeError, match="dataset_seed"):
        merge(sink)


def test_merge_with_no_shards(make_sink):
    with pytest.raises(MergeError, match="no shards"):
        merge(make_sink())


def test_merge_writes_throughput_with_cluster_and_per_shard_figures(make_sink, cfg):
    sink = make_sink()
    fake_shard(sink, cfg, 0, 2, 0, 2, 4)
    fake_shard(sink, cfg, 1, 2, 2, 4, 4)
    for i in range(4):
        fake_scene(sink, i // 2, i)
    merge(sink)
    t = json.loads(sink.get_bytes(sink.dataset_prefix + "throughput.json"))
    assert t["images"] == 4 and t["wall_seconds"] == 4.0  # 1000.0 -> 1004.0
    assert t["images_per_hour"] == 3600.0
    assert [s["images_per_hour"] for s in t["shards"]] == [3600.0, 3600.0]
    assert t["environment"]["hosts"] == ["pod-0", "pod-1"]
    manifest = json.loads(sink.get_bytes(sink.dataset_prefix + "manifest.json"))
    assert [s["shard_index"] for s in manifest["shards"]] == [0, 1]


def test_open_sink(tmp_path):
    assert isinstance(open_sink(tmp_path), LocalSink)
