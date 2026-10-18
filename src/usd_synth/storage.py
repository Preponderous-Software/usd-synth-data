"""Output sinks: where shard outputs and the merged dataset are stored.

Resume and merge use only this interface and rely only on "a scene's ``DONE`` object is written
after all of its files" — no atomic rename, no directory semantics — so the same logic works on
a local directory (M1–M6) and on an S3-compatible bucket (M7).

Keys are ``/``-separated paths relative to the sink root::

    shards/<k>/shard.json                  shard description (range, seed, config)
    shards/<k>/scenes/<iiiiii>/<file>      per-scene outputs, DONE written last
    <dataset prefix>annotations.json ...   merged dataset ("" locally, "dataset/" on S3)
"""

from __future__ import annotations

import os
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

DONE = "DONE"
_DONE_RE = re.compile(r"^scenes/(\d+)/DONE$")


def shard_prefix(shard: int) -> str:
    return f"shards/{shard}/"


def scene_prefix(shard: int, index: int) -> str:
    return f"{shard_prefix(shard)}scenes/{index:06d}/"


class Sink(ABC):
    """Minimal object-store-like interface."""

    #: where the merged dataset goes, relative to the sink root
    dataset_prefix = ""

    @abstractmethod
    def put_file(self, key: str, local_path: Path | str) -> None: ...

    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def list(self, prefix: str) -> list[str]:
        """All keys under ``prefix`` (recursive), sorted."""

    def get_file(self, key: str, local_path: Path | str) -> None:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_bytes(self.get_bytes(key))

    def copy(self, src: str, dst: str) -> None:
        self.put_bytes(dst, self.get_bytes(src))

    def describe(self, key: str = "") -> str:
        return key

    # scene-level helpers, used by resume and merge
    def mark_done(self, shard: int, index: int) -> None:
        self.put_bytes(scene_prefix(shard, index) + DONE, b"")

    def done_scenes(self, shard: int) -> set[int]:
        prefix = shard_prefix(shard)
        done = set()
        for key in self.list(prefix + "scenes/"):
            m = _DONE_RE.match(key[len(prefix) :])
            if m:
                done.add(int(m.group(1)))
        return done

    def shards(self) -> list[int]:
        found = set()
        for key in self.list("shards/"):
            parts = key.split("/")
            if len(parts) >= 3 and parts[1].isdigit() and parts[2] == "shard.json":
                found.add(int(parts[1]))
        return sorted(found)


class LocalSink(Sink):
    """A local directory. Files are written to a temporary name and renamed into place, which
    keeps readers from seeing torn files; completion still relies only on ``DONE``."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError(f"invalid key {key!r}")
        return self.root / key

    def put_file(self, key: str, local_path: Path | str) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        shutil.copyfile(local_path, tmp)
        os.replace(tmp, dest)

    def put_bytes(self, key: str, data: bytes) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, dest)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def get_file(self, key: str, local_path: Path | str) -> None:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(key), local_path)

    def copy(self, src: str, dst: str) -> None:
        self.put_file(dst, self._path(src))

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list(self, prefix: str) -> list[str]:
        base = self._path(prefix.rstrip("/")) if prefix else self.root
        if not base.exists():
            return []
        if base.is_file():
            return [prefix]
        keys = []
        for path in base.rglob("*"):
            if path.is_file() and not path.name.endswith(".part"):
                keys.append(path.relative_to(self.root).as_posix())
        return sorted(keys)

    def describe(self, key: str = "") -> str:
        return str(self.root / key)


def open_sink(out: str | Path) -> Sink:
    """A sink for ``--out``: a filesystem path, or (from M7) an ``s3://bucket/prefix`` URI."""
    text = str(out)
    if text.startswith("s3://"):
        raise ValueError("s3:// outputs need the S3-compatible sink, which arrives in M7")
    return LocalSink(text)
