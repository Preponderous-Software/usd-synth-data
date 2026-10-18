"""Deterministic seed tree: ``(dataset seed, scene index) -> scene seed -> per-stage streams``.

The scene stream is ``SeedSequence(entropy=dataset_seed, spawn_key=(scene_index,))``. Each stage
stream is that sequence's child with a fixed stage key appended to the spawn key, so:

* scene *i* depends only on ``(dataset_seed, i)``, never on generation order, worker or shard;
* consuming more numbers in one stage never shifts another stage's numbers.

Stage keys are the CRC-32 of the stage name, so adding a stage never renumbers existing ones.
"""

from __future__ import annotations

import zlib

import numpy as np

STAGES = (
    "support",
    "objects",
    "poses",
    "distractors",
    "materials",
    "lighting",
    "camera",
    "render",
)


def stage_key(stage: str) -> int:
    """Fixed integer key for a stage name."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {list(STAGES)}")
    return zlib.crc32(stage.encode("ascii"))


def _check(dataset_seed: int, scene_index: int) -> None:
    if dataset_seed < 0:
        raise ValueError(f"dataset seed must be >= 0 (got {dataset_seed})")
    if scene_index < 0:
        raise ValueError(f"scene index must be >= 0 (got {scene_index})")


def scene_sequence(dataset_seed: int, scene_index: int) -> np.random.SeedSequence:
    _check(dataset_seed, scene_index)
    return np.random.SeedSequence(entropy=dataset_seed, spawn_key=(scene_index,))


def stage_sequence(dataset_seed: int, scene_index: int, stage: str) -> np.random.SeedSequence:
    _check(dataset_seed, scene_index)
    return np.random.SeedSequence(entropy=dataset_seed, spawn_key=(scene_index, stage_key(stage)))


def stage_rng(dataset_seed: int, scene_index: int, stage: str) -> np.random.Generator:
    """An independent random generator for one stage of one scene."""
    return np.random.Generator(np.random.PCG64(stage_sequence(dataset_seed, scene_index, stage)))


def scene_seed(dataset_seed: int, scene_index: int) -> int:
    """A 64-bit summary of the scene stream, recorded in the manifest for provenance."""
    return int(scene_sequence(dataset_seed, scene_index).generate_state(1, np.uint64)[0])


def render_seed(dataset_seed: int, scene_index: int) -> int:
    """Cycles sampling seed (``scene.cycles.seed``), drawn from the ``render`` stream."""
    return int(stage_rng(dataset_seed, scene_index, "render").integers(0, 2**31 - 1))
