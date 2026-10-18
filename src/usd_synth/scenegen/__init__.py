"""Scene generation: config + seed + index -> one flattened ``.usda`` stage."""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from .tabletop import Scene, sample_scene


def scene_filename(index: int) -> str:
    return f"scene_{index:06d}.usda"


def generate_scene(
    cfg: Config, dataset_seed: int, index: int, config_dir: Path | str = "."
) -> tuple[Scene, str]:
    """Sample scene ``index`` and return it with its ``.usda`` text."""
    from .usd_writer import scene_to_usda  # pxr is imported lazily

    scene = sample_scene(cfg, dataset_seed, index, config_dir)
    return scene, scene_to_usda(scene)


def write_scene(
    cfg: Config, dataset_seed: int, index: int, out_dir: Path | str, config_dir: Path | str = "."
) -> Path:
    _, text = generate_scene(cfg, dataset_seed, index, config_dir)
    path = Path(out_dir) / scene_filename(index)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
