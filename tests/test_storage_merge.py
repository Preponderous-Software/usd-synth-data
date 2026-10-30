"""Storage-sink contract, resume bookkeeping and merge — through the Sink interface only, so
the same tests run against every sink (local filesystem here; S3-compatible from M7)."""

import json
import os
import uuid
from pathlib import Path

import pytest
from sink_contract import fake_scene, fake_shard

from usd_synth.config import load_config
from usd_synth.merge import MergeError, MergeNotReady, merge
from usd_synth.s3 import S3Sink
from usd_synth.storage import LocalSink, open_sink

SMALL = Path(__file__).resolve().parent / "data" / "small.yaml"


@pytest.fixture
def cfg():
    return load_config(SMALL)


@pytest.fixture(params=["local", "s3"])
def make_sink(request, tmp_path):
    """Factory of empty sinks; each call returns a fresh, isolated sink.

    The ``s3`` variant needs an S3-compatible endpoint (``USD_SYNTH_TEST_S3_ENDPOINT`` plus
    ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY``; CI runs MinIO) and is skipped without one,
    unless ``USD_SYNTH_REQUIRE_S3=1``.
    """
    counter = iter(range(1000))
    if request.param == "local":
        return lambda: LocalSink(tmp_path / f"sink{next(counter)}")
    endpoint = os.environ.get("USD_SYNTH_TEST_S3_ENDPOINT")
    if not endpoint:
        if os.environ.get("USD_SYNTH_REQUIRE_S3") == "1":
            pytest.fail("S3 sink tests required but USD_SYNTH_TEST_S3_ENDPOINT is unset")
        pytest.skip("no S3-compatible endpoint (set USD_SYNTH_TEST_S3_ENDPOINT)")
    import boto3

    os.environ["AWS_ENDPOINT_URL"] = endpoint
    client = boto3.client("s3", endpoint_url=endpoint, region_name="us-east-1")
    bucket = f"test-{uuid.uuid4().hex[:12]}"
    client.create_bucket(Bucket=bucket)
    return lambda: S3Sink(bucket, f"run{next(counter)}/data")


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


def test_done_scenes_ignores_done_objects_outside_a_scene_directory(make_sink):
    sink = make_sink()
    sink.put_bytes("shards/0/scenes/000003/extra/DONE", b"")
    sink.put_bytes("shards/0/scenes/DONE", b"")
    sink.put_bytes("shards/0/DONE", b"")
    assert sink.done_scenes(0) == set()


@pytest.mark.parametrize("key", ["/abs/key", "../outside", "a/../../b", ".."])
def test_keys_escaping_the_sink_root_are_rejected(make_sink, key):
    sink = make_sink()
    for call in (
        lambda: sink.put_bytes(key, b"x"),
        lambda: sink.get_bytes(key),
        lambda: sink.exists(key),
    ):
        with pytest.raises(ValueError, match="invalid key"):
            call()


def test_shards_lists_only_numbered_shards_with_a_description(make_sink):
    sink = make_sink()
    for key in (
        "shards/10/shard.json",
        "shards/2/shard.json",
        "shards/0/shard.json",
        "shards/1/scenes/000000/DONE",  # outputs but no shard.json
        "shards/tmp/shard.json",  # not a shard number
        "shards/3/sub/shard.json",  # not directly under the shard
    ):
        sink.put_bytes(key, b"{}")
    assert sink.shards() == [0, 2, 10]


def test_local_sink_hides_partial_files_and_lists_a_single_key(tmp_path):
    sink = LocalSink(tmp_path)
    sink.put_bytes("a/done.txt", b"x")
    (tmp_path / "a" / "torn.txt.part").write_bytes(b"half")
    assert sink.list("a/") == ["a/done.txt"]
    assert sink.list("") == ["a/done.txt"]
    assert sink.list("a/done.txt") == ["a/done.txt"]
    assert sink.describe("a/done.txt") == str(tmp_path / "a" / "done.txt")


@pytest.mark.parametrize(
    "uri, bucket, prefix",
    [
        ("s3://bucket", "bucket", ""),
        ("s3://bucket/", "bucket", ""),
        ("s3://bucket/run", "bucket", "run/"),
        ("s3://bucket//nested/run//", "bucket", "nested/run/"),
    ],
)
def test_s3_uri_parsing_normalizes_the_prefix(monkeypatch, uri, bucket, prefix):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
    sink = S3Sink.from_uri(uri)
    assert (sink.bucket, sink.prefix) == (bucket, prefix)
    assert sink.describe("k") == f"s3://{bucket}/{prefix}k"


@pytest.mark.parametrize(
    "uri, match",
    [("s3:///prefix", "needs a bucket"), ("http://bucket/prefix", "not an s3:// URI")],
)
def test_s3_uri_without_a_bucket_or_scheme_is_rejected(uri, match):
    with pytest.raises(ValueError, match=match):
        S3Sink.from_uri(uri)


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


def _missing_shard(sink, cfg):
    fake_shard(sink, cfg, 0, 2, 0, 3, 6)  # shard 1 has not written shard.json yet


def _unfinished_scene(sink, cfg):
    fake_shard(sink, cfg, 0, 1, 0, 2, 2)
    fake_scene(sink, 0, 0)
    fake_scene(sink, 0, 1, done=False)


def _overlapping_shards(sink, cfg):
    fake_shard(sink, cfg, 0, 2, 0, 4, 6)
    fake_shard(sink, cfg, 1, 2, 3, 6, 6)


def _mixed_runs(sink, cfg):
    fake_shard(sink, cfg, 0, 2, 0, 3, 6, seed=1)
    fake_shard(sink, cfg, 1, 2, 3, 6, 6, seed=2)


def _scenes_outside_the_run(sink, cfg):
    fake_shard(sink, cfg, 0, 1, 0, 8, 6)


def _tampered_config_hash(sink, cfg):
    fake_shard(sink, cfg, 0, 1, 0, 1, 1)
    key = "shards/0/shard.json"
    info = json.loads(sink.get_bytes(key))
    info["config_hash"] = "0" * len(info["config_hash"])
    sink.put_bytes(key, json.dumps(info).encode())


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda sink, cfg: None, "no shards found"),
        (_missing_shard, r"missing 3 scene\(s\): 3-5"),
        (_unfinished_scene, r"1 scene\(s\) not complete: 1"),
    ],
    ids=["no-shards", "missing-shard", "unfinished-scene"],
)
def test_merge_reports_shards_that_may_still_finish_as_not_ready(make_sink, cfg, build, match):
    """``merge --wait`` retries on MergeNotReady: a shard that has not started or finished."""
    sink = make_sink()
    build(sink, cfg)
    with pytest.raises(MergeNotReady, match=match):
        merge(sink)


@pytest.mark.parametrize(
    "build, match",
    [
        (_overlapping_shards, r"shards 0 \[0, 4\) and 1 \[3, 6\) overlap"),
        (_mixed_runs, "dataset_seed=2 but shard 0 has 1; they are not one run"),
        (_scenes_outside_the_run, "shards cover scenes outside the run: 6-7"),
        (_tampered_config_hash, "shard config does not match its recorded hash"),
    ],
    ids=["overlap", "mixed-runs", "outside-the-run", "tampered-hash"],
)
def test_merge_fails_at_once_on_shards_that_can_never_merge(make_sink, cfg, build, match):
    """Waiting cannot fix these, so they are plain MergeErrors that ``merge --wait`` raises."""
    sink = make_sink()
    build(sink, cfg)
    with pytest.raises(MergeError, match=match) as err:
        merge(sink)
    assert not isinstance(err.value, MergeNotReady)


def test_merge_lists_missing_scenes_as_compact_ranges(make_sink, cfg):
    sink = make_sink()
    for k in (0, 2, 5):  # shards 1, 3 and 4 of 6 (one scene each) never ran
        fake_shard(sink, cfg, k, 6, k, k + 1, 6)
    with pytest.raises(MergeNotReady, match=r"missing 3 scene\(s\): 1, 3-4$"):
        merge(sink)


def test_merge_accepts_an_empty_trailing_shard(make_sink, cfg):
    sink = make_sink()
    for k in range(4):  # 3 scenes over 4 shards: shard 3 owns [3, 3)
        start, stop = min(k, 3), min(k + 1, 3)
        fake_shard(sink, cfg, k, 4, start, stop, 3)
    for i in range(3):
        fake_scene(sink, i, i)
    merged = merge(sink)
    assert len(merged["coco"]["images"]) == 3
    assert merged["manifest"]["shards"][-1] == {
        "shard_index": 3,
        "shard_count": 4,
        "start": 3,
        "stop": 3,
    }


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


def test_open_sink(tmp_path, monkeypatch):
    assert isinstance(open_sink(tmp_path), LocalSink)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
    sink = open_sink("s3://bucket/some/prefix/")
    assert isinstance(sink, S3Sink) and sink.bucket == "bucket" and sink.prefix == "some/prefix/"
    assert sink.dataset_prefix == "dataset/"
    assert (
        sink.describe("dataset/annotations.json")
        == "s3://bucket/some/prefix/dataset/annotations.json"
    )
