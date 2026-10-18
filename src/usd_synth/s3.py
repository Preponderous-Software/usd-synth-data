"""S3-compatible object-storage sink (``--out s3://bucket/prefix``).

Uses boto3 with a custom endpoint, so any S3-compatible store works (MinIO in the ``kind``
test). Endpoint and credentials come from the environment — in a cluster, from the Kubernetes
Secret the manifest references — never from the dataset config, so the config hash does not
depend on the storage target:

* ``AWS_ENDPOINT_URL`` (or ``AWS_ENDPOINT_URL_S3``): e.g. ``http://minio:9000``; omit for AWS
* ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``
* ``AWS_DEFAULT_REGION`` (optional, default ``us-east-1``)

Shard outputs live under ``<prefix>/shards/`` and the merged dataset under ``<prefix>/dataset/``.
Object stores have no atomic rename, so completion relies only on the ``DONE`` object being put
after a scene's other objects.
"""

from __future__ import annotations

import os
from pathlib import Path

from .storage import Sink


def _client():
    import boto3
    from botocore.config import Config as BotoConfig

    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    return boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        config=BotoConfig(
            s3={"addressing_style": "path"},  # works with MinIO and other non-AWS stores
            retries={"max_attempts": 8, "mode": "standard"},
        ),
    )


class S3Sink(Sink):
    dataset_prefix = "dataset/"

    def __init__(self, bucket: str, prefix: str = "", client=None):
        if not bucket:
            raise ValueError("an S3 URI needs a bucket: s3://<bucket>/<prefix>")
        self.bucket = bucket
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
        self.client = client or _client()

    @classmethod
    def from_uri(cls, uri: str) -> S3Sink:
        if not uri.startswith("s3://"):
            raise ValueError(f"not an s3:// URI: {uri}")
        bucket, _, prefix = uri[len("s3://") :].partition("/")
        return cls(bucket, prefix)

    def _key(self, key: str) -> str:
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError(f"invalid key {key!r}")
        return self.prefix + key

    def put_file(self, key: str, local_path: Path | str) -> None:
        with open(local_path, "rb") as fh:
            self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=fh)

    def put_bytes(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def get_bytes(self, key: str) -> bytes:
        obj = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        return obj["Body"].read()

    def copy(self, src: str, dst: str) -> None:
        self.client.copy_object(
            Bucket=self.bucket,
            Key=self._key(dst),
            CopySource={"Bucket": self.bucket, "Key": self._key(src)},
        )

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except ClientError as err:
            if err.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list(self, prefix: str) -> list[str]:
        keys = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._key(prefix)):
            keys += [o["Key"][len(self.prefix) :] for o in page.get("Contents", [])]
        return sorted(keys)

    def describe(self, key: str = "") -> str:
        return f"s3://{self.bucket}/{self.prefix}{key}"
