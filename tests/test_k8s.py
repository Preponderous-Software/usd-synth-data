"""Kubernetes manifests and the cluster entrypoints, tested without a cluster."""

import json
from pathlib import Path

import pytest
import yaml
from sink_contract import fake_scene, fake_shard

from usd_synth import cli
from usd_synth.config import load_config, parse_config
from usd_synth.k8s import render_manifests
from usd_synth.storage import LocalSink

ROOT = Path(__file__).resolve().parents[1]
BUNDLED = ROOT / "configs" / "tabletop.yaml"
SMALL = ROOT / "tests" / "data" / "small.yaml"


def _docs(**kw):
    cfg = kw.pop("cfg", None) or load_config(BUNDLED)
    args = dict(seed=42, count=100, shards=4, out="s3://datasets/run-42", image="usd-synth:test")
    args.update(kw)
    return cfg, list(yaml.safe_load_all(render_manifests(cfg, **args)))


def test_indexed_job_shape():
    cfg, (cm, job, merge) = _docs(parallelism=2, cpu=3, backoff_limit_per_index=5)
    assert cm["kind"] == "ConfigMap" and job["kind"] == merge["kind"] == "Job"
    spec = job["spec"]
    assert spec["completionMode"] == "Indexed"
    assert spec["completions"] == 4 and spec["parallelism"] == 2
    assert spec["backoffLimitPerIndex"] == 5
    pod = spec["template"]["spec"]
    assert pod["restartPolicy"] == "Never"
    (c,) = pod["containers"]
    args = c["args"]
    assert args[:3] == ["generate", "--config", "/config/config.yaml"]
    assert args[args.index("--shard-count") + 1] == "4" and "--shard-index" not in args
    assert args[args.index("--out") + 1] == "s3://datasets/run-42"
    # Blender threads == CPU limit == CPU request
    assert args[args.index("--threads") + 1] == "3"
    assert c["resources"]["limits"]["cpu"] == c["resources"]["requests"]["cpu"] == "3"
    assert "nvidia.com/gpu" not in c["resources"]["limits"]
    assert c["envFrom"] == [{"secretRef": {"name": "usd-synth-s3"}}]
    assert {"name": "TMPDIR", "value": "/scratch"} in c["env"]
    assert {"name": "scratch", "emptyDir": {}} in pod["volumes"]
    margs = merge["spec"]["template"]["spec"]["containers"][0]["args"]
    assert margs[:3] == ["merge", "--out", "s3://datasets/run-42"] and "--wait" in margs


def test_no_persistent_volume_claim_anywhere():
    _, docs = _docs()
    text = json.dumps(docs)
    assert "persistentVolumeClaim" not in text and "PersistentVolumeClaim" not in text


def test_config_travels_in_configmap_with_the_same_hash():
    cfg, (cm, _, _) = _docs()
    shipped = parse_config(yaml.safe_load(cm["data"]["config.yaml"]))
    assert shipped.config_hash() == cfg.config_hash()
    # credentials never appear in the manifests
    assert "AWS_SECRET_ACCESS_KEY" not in render_manifests(
        cfg, seed=1, count=4, shards=2, out="s3://b/p", image="i"
    )


def test_gpu_request_only_for_gpu_device():
    raw = yaml.safe_load(SMALL.read_text())
    raw["render"]["device"] = "gpu"
    _, (_, job, _) = _docs(cfg=parse_config(raw))
    limits = job["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]
    assert limits["nvidia.com/gpu"] == "1"


@pytest.mark.parametrize(
    "kw, match",
    [({"out": "/local/dir"}, "s3://"), ({"name": "Bad_Name"}, "DNS-1123"),
     ({"shards": 0}, "shards"), ({"shards": 101}, "shards")],
)  # fmt: skip
def test_invalid_manifest_arguments(kw, match):
    with pytest.raises(ValueError, match=match):
        _docs(**kw)


def test_cli_k8s_manifest_prints_yaml(capsys):
    rc = cli.main(["k8s-manifest", "--config", str(BUNDLED), "--seed", "42", "--count", "10",
                   "--shards", "2", "--out", "s3://datasets/x"])  # fmt: skip
    assert rc == 0
    kinds = [d["kind"] for d in yaml.safe_load_all(capsys.readouterr().out)]
    assert kinds == ["ConfigMap", "Job", "Job"]


def test_shard_index_comes_from_job_completion_index(monkeypatch, tmp_path):
    seen = {}

    def fake_run_shard(cfg, config_dir, seed, rng, sink, blender, *, shard, **kw):
        seen.update(shard=shard, scenes=list(rng))
        return type("R", (), {"rendered": list(rng), "skipped": []})()

    monkeypatch.setattr("usd_synth.render.find_blender", lambda explicit=None: "blender")
    monkeypatch.setattr("usd_synth.pipeline.run_shard", fake_run_shard)
    monkeypatch.setenv("JOB_COMPLETION_INDEX", "2")
    rc = cli.main(["generate", "--config", str(SMALL), "--seed", "1", "--count", "10",
                   "--shard-count", "4", "--out", str(tmp_path)])  # fmt: skip
    assert rc == 0
    assert seen == {"shard": 2, "scenes": [6, 7, 8]}  # size ceil(10/4)=3


def test_merge_wait_retries_until_shards_finish(monkeypatch, tmp_path):
    cfg = load_config(SMALL)
    sink = LocalSink(tmp_path)
    fake_shard(sink, cfg, 0, 2, 0, 2, 4)
    fake_scene(sink, 0, 0)
    fake_scene(sink, 0, 1)
    sleeps = []

    def finish_other_shard(seconds):
        sleeps.append(seconds)
        fake_shard(sink, cfg, 1, 2, 2, 4, 4)  # the slower pod finishes while merge waits
        fake_scene(sink, 1, 2)
        fake_scene(sink, 1, 3)

    monkeypatch.setattr("time.sleep", finish_other_shard)
    assert cli.main(["merge", "--out", str(tmp_path), "--wait", "60", "--poll", "1"]) == 0
    assert sleeps == [1.0]
    assert (tmp_path / "annotations.json").exists() and (tmp_path / "throughput.json").exists()


def test_merge_does_not_wait_on_permanent_errors(monkeypatch, tmp_path, capsys):
    cfg = load_config(SMALL)
    sink = LocalSink(tmp_path)
    fake_shard(sink, cfg, 0, 2, 0, 3, 4)
    fake_shard(sink, cfg, 1, 2, 2, 4, 4)  # overlap: waiting cannot fix it
    monkeypatch.setattr("time.sleep", lambda s: pytest.fail("should not wait"))
    assert cli.main(["merge", "--out", str(tmp_path), "--wait", "60"]) == 1
    assert "overlap" in capsys.readouterr().err
