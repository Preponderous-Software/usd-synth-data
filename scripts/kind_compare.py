"""Compare a merged dataset in an S3-compatible bucket with a local run (kind e2e helper).

Usage: python scripts/kind_compare.py s3://bucket/prefix <local dataset dir> <shard count>
Endpoint and credentials come from AWS_ENDPOINT_URL / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
"""

import json
import sys
from pathlib import Path

from usd_synth.storage import open_sink


def main(uri: str, local: str, shards: int) -> int:
    sink = open_sink(uri)
    ds = sink.dataset_prefix
    remote = json.loads(sink.get_bytes(ds + "annotations.json"))
    mine = json.loads((Path(local) / "annotations.json").read_text())
    if remote != mine:
        print("FAIL: merged cluster annotations.json differs from the local run")
        return 1
    print(
        f"annotations.json identical: {len(mine['images'])} images, "
        f"{len(mine['annotations'])} annotations"
    )
    t = json.loads(sink.get_bytes(ds + "throughput.json"))
    print(
        f"cluster: {t['images']} images in {t['wall_seconds']} s -> "
        f"{t['images_per_hour']} images/hour"
    )
    for s in t["shards"]:
        print(
            f"  pod {s['host']} (shard {s['shard']}): {s['images']} images, "
            f"{s['images_per_hour']} images/hour"
        )
    if len(t["shards"]) != shards or t["images_per_hour"] <= 0:
        print("FAIL: throughput.json lacks per-pod figures")
        return 1
    print(json.dumps(t, indent=1))
    keys = sink.list("")
    assert any(k.startswith("shards/1/scenes/") and k.endswith("/DONE") for k in keys)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], int(sys.argv[3])))
