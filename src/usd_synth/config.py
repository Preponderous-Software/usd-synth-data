"""Declarative scene-randomization config: YAML in, validated Pydantic model out.

Every numeric range is written as ``[min, max]`` and sampled uniformly. Unknown keys are an
error, and every validation error names the offending field as a dotted path
(``objects.count``) so a broken config can be fixed without reading this module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

PRIMITIVES = ("box", "cylinder", "sphere", "capsule", "torus")


class ConfigError(ValueError):
    """Raised for an invalid config; the message names each offending field."""


def _ordered(value: tuple[float, float]) -> tuple[float, float]:
    lo, hi = value
    if lo > hi:
        raise ValueError(f"min ({lo}) must be <= max ({hi})")
    return value


def _bounded(lo: float | None = None, hi: float | None = None):
    def check(value: tuple[float, float]) -> tuple[float, float]:
        for v in value:
            if lo is not None and v < lo:
                raise ValueError(f"value {v} is below the allowed minimum {lo}")
            if hi is not None and v > hi:
                raise ValueError(f"value {v} is above the allowed maximum {hi}")
        return value

    return AfterValidator(check)


Range = Annotated[tuple[float, float], AfterValidator(_ordered)]
IntRange = Annotated[tuple[int, int], AfterValidator(_ordered)]
PositiveRange = Annotated[Range, _bounded(lo=1e-9)]
UnitRange = Annotated[Range, _bounded(0.0, 1.0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ImageConfig(_Model):
    width: int = Field(640, ge=16, le=8192)
    height: int = Field(480, ge=16, le=8192)


class RenderConfig(_Model):
    engine: Literal["cycles"] = "cycles"
    device: Literal["cpu", "gpu"] = "cpu"
    samples: int = Field(64, ge=1, le=65536)
    denoise: bool = True
    threads: int = Field(0, ge=0, le=1024, description="0 = let Blender decide")
    batch_size: int = Field(8, ge=1, le=10000, description="scenes per Blender process")


class LibraryEntry(_Model):
    name: str = Field(min_length=1)
    source: str = Field(min_length=1, description="primitive:<shape> or a USD/USDZ path")
    category: str = Field(min_length=1)
    size_m: PositiveRange | None = None

    @model_validator(mode="after")
    def _check_source(self) -> LibraryEntry:
        if self.source.startswith("primitive:"):
            shape = self.source.split(":", 1)[1]
            if shape not in PRIMITIVES:
                raise ValueError(
                    f"unknown primitive {shape!r} in source; expected one of {list(PRIMITIVES)}"
                )
            if self.size_m is None:
                raise ValueError("size_m is required for primitive sources")
        elif not self.source.endswith((".usd", ".usda", ".usdc", ".usdz")):
            raise ValueError("source must be primitive:<shape> or a .usd/.usda/.usdc/.usdz path")
        return self


class PlacementConfig(_Model):
    region: Literal["support"] = "support"
    min_separation_m: float = Field(0.005, ge=0.0, le=1.0)
    yaw_deg: Range = (0.0, 360.0)
    allow_tipped: Probability = 0.2


class ObjectsConfig(_Model):
    library: list[LibraryEntry] = Field(min_length=1)
    count: Annotated[IntRange, _bounded(lo=1, hi=1000)] = (3, 12)
    placement: PlacementConfig = PlacementConfig()

    @model_validator(mode="after")
    def _unique_names(self) -> ObjectsConfig:
        names = [e.name for e in self.library]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"library entry names must be unique; duplicated: {dupes}")
        return self


class DistractorsConfig(_Model):
    count: Annotated[IntRange, _bounded(lo=0, hi=1000)] = (0, 8)
    label: Literal["ignore", "omit"] = "ignore"
    size_m: PositiveRange = (0.02, 0.06)


class SupportSize(_Model):
    x: PositiveRange = (0.6, 1.2)
    y: PositiveRange = (0.4, 0.8)


class SupportConfig(_Model):
    kind: list[Literal["table", "bin"]] = Field(default=["table", "bin"], min_length=1)
    size_m: SupportSize = SupportSize()


class HSVConfig(_Model):
    h: UnitRange = (0.0, 1.0)
    s: UnitRange = (0.0, 0.9)
    v: UnitRange = (0.1, 1.0)


class ChoiceConfig(_Model):
    choice: list[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(min_length=1)
    p: list[Probability]

    @model_validator(mode="after")
    def _check_p(self) -> ChoiceConfig:
        if len(self.p) != len(self.choice):
            raise ValueError("p must have one probability per choice")
        if abs(sum(self.p) - 1.0) > 1e-6:
            raise ValueError(f"p must sum to 1 (got {sum(self.p)})")
        return self


class MaterialsConfig(_Model):
    base_color: HSVConfig = HSVConfig()
    roughness: UnitRange = (0.1, 0.9)
    metallic: ChoiceConfig = ChoiceConfig(choice=[0.0, 1.0], p=[0.85, 0.15])
    textures: list[str] = []


class DomeConfig(_Model):
    enabled: bool = True
    intensity: Annotated[Range, _bounded(lo=0.0)] = (0.1, 1.0)
    hdris: list[str] = []


class LightingConfig(_Model):
    key_lights: Annotated[IntRange, _bounded(lo=1, hi=16)] = (1, 3)
    intensity: PositiveRange = (30.0, 300.0)
    color_temp_k: Annotated[Range, _bounded(1000.0, 40000.0)] = (2700.0, 6500.0)
    dome: DomeConfig = DomeConfig()


class CameraConfig(_Model):
    look_at: Literal["support_center"] = "support_center"
    distance_m: PositiveRange = (0.6, 1.5)
    elevation_deg: Annotated[Range, _bounded(1.0, 90.0)] = (25.0, 80.0)
    azimuth_deg: Range = (0.0, 360.0)
    focal_length_mm: PositiveRange = (24.0, 50.0)
    sensor_width_mm: float = Field(36.0, gt=0.0, le=100.0)
    jitter_target_m: float = Field(0.05, ge=0.0, le=10.0)


class DepthConfig(_Model):
    format: Literal["png16_mm"] = "png16_mm"
    max_m: float = Field(10.0, gt=0.0, le=65.535)
    keep_exr: bool = False


class LabelsConfig(_Model):
    min_visible_pixels: int = Field(50, ge=0)
    min_visible_fraction: Probability = 0.1
    depth: DepthConfig = DepthConfig()


class Config(_Model):
    version: Literal[1]
    scenario: Literal["tabletop"]
    image: ImageConfig = ImageConfig()
    render: RenderConfig = RenderConfig()
    objects: ObjectsConfig
    distractors: DistractorsConfig = DistractorsConfig()
    support: SupportConfig = SupportConfig()
    materials: MaterialsConfig = MaterialsConfig()
    lighting: LightingConfig = LightingConfig()
    camera: CameraConfig = CameraConfig()
    labels: LabelsConfig = LabelsConfig()

    @model_validator(mode="after")
    def _reserved_category(self) -> Config:
        if self.distractors.label == "ignore" and "distractor" in self.categories():
            raise ValueError(
                "objects.library: category 'distractor' is reserved for distractors when "
                "distractors.label is 'ignore'; rename the category"
            )
        return self

    def categories(self) -> list[str]:
        """Labelled category names in config order (first appearance); COCO id = index + 1."""
        seen: list[str] = []
        for entry in self.objects.library:
            if entry.category not in seen:
                seen.append(entry.category)
        return seen

    def resolved(self) -> dict:
        """The config with every default filled in, as plain JSON-compatible data."""
        return self.model_dump(mode="json")

    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON of the resolved config."""
        canonical = json.dumps(self.resolved(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _format_errors(err: ValidationError) -> str:
    lines = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"] if not str(p).startswith("function-after"))
        msg = e["msg"].removeprefix("Value error, ")
        lines.append(f"{loc or '<root>'}: {msg}")
    return "invalid config:\n  " + "\n  ".join(lines)


def parse_config(data: object) -> Config:
    """Validate already-parsed config data, raising ConfigError that names each bad field."""
    if not isinstance(data, dict):
        raise ConfigError("invalid config:\n  <root>: expected a mapping at the top level")
    try:
        return Config.model_validate(data)
    except ValidationError as err:
        raise ConfigError(_format_errors(err)) from None


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    with open(path, encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)
        except yaml.YAMLError as err:
            raise ConfigError(f"invalid config: {path} is not valid YAML: {err}") from None
    return parse_config(data)
