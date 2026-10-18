"""Kubernetes manifests for cluster fan-out: one Indexed Job (a pod per shard) and a merge Job.

Plain YAML generated from Python data — no Helm, no operator — so the output works with
``kubectl apply -f -`` on any conformant cluster (v1.33+ for ``backoffLimitPerIndex``).

* The dataset config travels in a ConfigMap; endpoint and credentials of the S3-compatible
  store come from a Secret (``envFrom``), never from the config, so the config hash does not
  depend on where the output goes.
* Each render pod gets ``JOB_COMPLETION_INDEX`` from Kubernetes and uses it as its shard index.
  It renders into an ``emptyDir`` scratch volume and uploads each finished scene, ``DONE`` last;
  a retried pod skips the scenes whose ``DONE`` objects exist. No PersistentVolumeClaim.
* Blender's thread count equals the pod's CPU limit, so pods do not oversubscribe a node.
* The merge Job polls the bucket until every shard is complete, then merges (``merge --wait``),
  so both Jobs can be applied together.
"""

from __future__ import annotations

import re

import yaml

from .config import Config

NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9]{0,40}[a-z0-9])?$")


class _Dumper(yaml.SafeDumper):
    """Writes multi-line strings (the embedded config) as readable literal blocks."""


_Dumper.add_representer(
    str,
    lambda d, s: d.represent_scalar("tag:yaml.org,2002:str", s, style="|" if "\n" in s else None),
)


def render_manifests(
    cfg: Config,
    *,
    seed: int,
    count: int,
    shards: int,
    out: str,
    image: str,
    name: str = "usd-synth",
    parallelism: int | None = None,
    cpu: int | None = None,
    memory: str = "2Gi",
    secret: str = "usd-synth-s3",
    backoff_limit_per_index: int = 3,
    image_pull_policy: str = "IfNotPresent",
    merge_wait_seconds: int = 86400,
) -> str:
    if not out.startswith("s3://"):
        raise ValueError("--out must be an s3://bucket/prefix URI for cluster runs")
    if not NAME_RE.match(name):
        raise ValueError(f"--name {name!r} must be a lowercase DNS-1123 label of <= 42 chars")
    if count < 1 or shards < 1 or shards > count:
        raise ValueError("need count >= 1 and 1 <= shards <= count")
    parallelism = parallelism or shards
    cpu = cpu or cfg.render.threads or 2
    labels = {"app.kubernetes.io/name": "usd-synth", "app.kubernetes.io/instance": name}
    volumes = [
        {"name": "config", "configMap": {"name": f"{name}-config"}},
        {"name": "scratch", "emptyDir": {}},
    ]
    mounts = [
        {"name": "config", "mountPath": "/config", "readOnly": True},
        {"name": "scratch", "mountPath": "/scratch"},
    ]
    env = [{"name": "TMPDIR", "value": "/scratch"}]
    limits = {"cpu": str(cpu), "memory": memory}
    if cfg.render.device == "gpu":
        limits["nvidia.com/gpu"] = "1"

    def container(args, resources):
        return {
            "name": "usd-synth",
            "image": image,
            "imagePullPolicy": image_pull_policy,
            "args": [str(a) for a in args],
            "env": env,
            "envFrom": [{"secretRef": {"name": secret}}],
            "volumeMounts": mounts,
            "resources": resources,
        }

    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": f"{name}-config", "labels": labels},
        "data": {"config.yaml": yaml.safe_dump(cfg.resolved(), sort_keys=True)},
    }
    render_job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": f"{name}-render", "labels": labels},
        "spec": {
            "completionMode": "Indexed",
            "completions": shards,
            "parallelism": parallelism,
            "backoffLimitPerIndex": backoff_limit_per_index,
            "template": {
                "metadata": {"labels": {**labels, "usd-synth/role": "render"}},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        container(
                            [
                                "generate",
                                "--config",
                                "/config/config.yaml",
                                "--seed",
                                seed,
                                "--count",
                                count,
                                "--shard-count",
                                shards,
                                "--out",
                                out,
                                "--threads",
                                cpu,
                            ],
                            {"requests": {"cpu": str(cpu), "memory": memory}, "limits": limits},
                        )  # fmt: skip
                    ],
                    "volumes": volumes,
                },
            },
        },
    }
    merge_job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": f"{name}-merge", "labels": labels},
        "spec": {
            "backoffLimit": 3,
            "template": {
                "metadata": {"labels": {**labels, "usd-synth/role": "merge"}},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        container(
                            ["merge", "--out", out, "--wait", merge_wait_seconds],
                            {
                                "requests": {"cpu": "250m", "memory": "512Mi"},
                                "limits": {"cpu": "1", "memory": "1Gi"},
                            },
                        )  # fmt: skip
                    ],
                    "volumes": volumes,
                },
            },
        },
    }
    return yaml.dump_all([config_map, render_job, merge_job], Dumper=_Dumper, sort_keys=False)
