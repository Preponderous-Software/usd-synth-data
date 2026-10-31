"""S3 sink logic that needs no S3 endpoint: key prefixing, ``exists``' not-found
classification, listing across pages, and how the boto3 client is configured from the
environment. The sink contract itself runs against MinIO in ``test_storage_merge.py``."""

import boto3
import pytest
from botocore.exceptions import ClientError

from usd_synth import s3
from usd_synth.s3 import S3Sink


class _Body:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data


class _Paginator:
    def __init__(self, client):
        self.client = client

    def paginate(self, **kwargs):
        self.client.calls.append(("paginate", kwargs))
        return iter(self.client.pages)


class FakeClient:
    """Records every call; ``head_object`` fails with ``error_code`` when one is set."""

    def __init__(self, error_code=None, pages=()):
        self.calls = []
        self.error_code = error_code
        self.pages = list(pages)

    def put_object(self, **kwargs):
        body = kwargs["Body"]
        kwargs["Body"] = body if isinstance(body, bytes) else body.read()
        self.calls.append(("put_object", kwargs))

    def get_object(self, **kwargs):
        self.calls.append(("get_object", kwargs))
        return {"Body": _Body(b"payload")}

    def copy_object(self, **kwargs):
        self.calls.append(("copy_object", kwargs))

    def head_object(self, **kwargs):
        self.calls.append(("head_object", kwargs))
        if self.error_code is not None:
            raise ClientError({"Error": {"Code": self.error_code, "Message": "x"}}, "HeadObject")
        return {}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _Paginator(self)


def test_every_operation_addresses_keys_under_the_sink_prefix(tmp_path):
    client = FakeClient()
    sink = S3Sink("bucket", "run/data", client=client)
    local = tmp_path / "f.bin"
    local.write_bytes(b"from file")

    sink.put_file("a/f.bin", local)
    sink.put_bytes("a/b.txt", b"hello")
    assert sink.get_bytes("a/b.txt") == b"payload"
    sink.copy("a/b.txt", "x/y.txt")
    assert sink.exists("x/y.txt")

    assert client.calls == [
        ("put_object", {"Bucket": "bucket", "Key": "run/data/a/f.bin", "Body": b"from file"}),
        ("put_object", {"Bucket": "bucket", "Key": "run/data/a/b.txt", "Body": b"hello"}),
        ("get_object", {"Bucket": "bucket", "Key": "run/data/a/b.txt"}),
        (
            "copy_object",
            {
                "Bucket": "bucket",
                "Key": "run/data/x/y.txt",
                "CopySource": {"Bucket": "bucket", "Key": "run/data/a/b.txt"},
            },
        ),
        ("head_object", {"Bucket": "bucket", "Key": "run/data/x/y.txt"}),
    ]


def test_a_sink_without_a_prefix_uses_keys_as_given():
    client = FakeClient()
    S3Sink("bucket", "", client=client).put_bytes("shards/0/shard.json", b"{}")
    assert client.calls[0][1]["Key"] == "shards/0/shard.json"


def test_copy_rejects_an_escaping_source_or_destination():
    client = FakeClient()
    sink = S3Sink("bucket", "run", client=client)
    for src, dst in (("../a", "b"), ("a", "/b")):
        with pytest.raises(ValueError, match="invalid key"):
            sink.copy(src, dst)
    assert client.calls == []


@pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound"])
def test_exists_reads_not_found_errors_as_missing(code):
    assert S3Sink("bucket", client=FakeClient(error_code=code)).exists("k") is False


@pytest.mark.parametrize("code", ["403", "AccessDenied", "NoSuchBucket", "500"])
def test_exists_raises_every_other_client_error(code):
    # A permission or bucket error must not read as "scene not done": resume would redo the
    # scene and merge would report it missing instead of surfacing the real failure.
    sink = S3Sink("bucket", client=FakeClient(error_code=code))
    with pytest.raises(ClientError) as info:
        sink.exists("k")
    assert info.value.response["Error"]["Code"] == code


def test_list_joins_pages_strips_the_sink_prefix_and_sorts():
    pages = [
        {"Contents": [{"Key": "run/data/shards/1/b"}, {"Key": "run/data/shards/0/z"}]},
        {},  # a page with no Contents (an empty listing) is skipped
        {"Contents": [{"Key": "run/data/shards/0/a"}]},
    ]
    client = FakeClient(pages=pages)
    sink = S3Sink("bucket", "run/data/", client=client)
    assert sink.list("shards/") == ["shards/0/a", "shards/0/z", "shards/1/b"]
    assert client.calls == [("paginate", {"Bucket": "bucket", "Prefix": "run/data/shards/"})]


def test_list_of_an_empty_prefix_returns_nothing():
    assert S3Sink("bucket", "run", client=FakeClient(pages=[{}])).list("shards/") == []


@pytest.fixture
def aws_env(monkeypatch):
    # boto3.client() goes through a module-level default session that caches the first
    # credentials it resolves; give each test its own so the dummy keys below do not leak
    # into later tests that talk to a real endpoint.
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)
    for name in ("AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
    return monkeypatch


def test_client_uses_the_endpoint_path_addressing_and_retries(aws_env):
    aws_env.setenv("AWS_ENDPOINT_URL", "http://minio.test:9000")
    client = s3._client()
    assert client.meta.endpoint_url == "http://minio.test:9000"
    assert client.meta.region_name == "us-east-1"
    assert client.meta.config.s3["addressing_style"] == "path"
    retries = client.meta.config.retries
    assert retries["mode"] == "standard"
    # botocore may rewrite max_attempts (retries) as total_max_attempts (retries + first try)
    assert retries.get("max_attempts") == 8 or retries.get("total_max_attempts") == 9


def test_client_prefers_the_s3_specific_endpoint_and_honours_the_region(aws_env):
    aws_env.setenv("AWS_ENDPOINT_URL", "http://generic.test:9000")
    aws_env.setenv("AWS_ENDPOINT_URL_S3", "http://s3only.test:9000")
    aws_env.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    client = s3._client()
    assert client.meta.endpoint_url == "http://s3only.test:9000"
    assert client.meta.region_name == "eu-west-1"


def test_client_without_an_endpoint_falls_back_to_the_default_aws_endpoint(aws_env):
    client = s3._client()
    # botocore's own regional default, not anything taken from the environment
    assert client.meta.endpoint_url.startswith("https://s3.")
