from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from routee.powertrain.core.year import Year, format_year, parse_year, serialize_year

_VERSION_RE = re.compile(r"^v(\d+)$")


@dataclass
class ModelId:
    """Uniquely identifies a model in the registry."""

    make: str
    model: str
    year: Year
    variant: str
    feature_set_id: str
    version: int

    def __post_init__(self):
        self.make = self.make.lower()
        self.model = self.model.lower()
        self.variant = self.variant.lower()
        self.feature_set_id = self.feature_set_id.lower()
        self.year = parse_year(self.year)

    @classmethod
    def from_path(cls, path: str) -> ModelId:
        """Parse a ModelId from a path string.

        Expected format: ``make/model/year/variant/feature_set/v<N>``

        Args:
            path: a ``/``-separated path string

        Returns: a ModelId instance

        Raises:
            ValueError: if the path cannot be parsed as a valid model id
        """
        parts = [p for p in path.strip("/").split("/") if p]

        if len(parts) != 6:
            raise ValueError(
                f"Cannot parse model id from path '{path}'. "
                f"Expected <make>/<model>/<year>/<variant>/<feature_set_id>/v<N>, "
                f"got {len(parts)} segments."
            )

        make, model, year_str, variant, feature_set_id, version_dir = parts

        match = _VERSION_RE.match(version_dir)
        if not match:
            raise ValueError(
                f"Version segment '{version_dir}' does not match expected pattern v<N>"
            )

        return cls(
            make=make,
            model=model,
            year=parse_year(year_str),
            variant=variant,
            feature_set_id=feature_set_id,
            version=int(match.group(1)),
        )

    def to_path(self) -> str:
        """
        Build the registry path for this model.

        Returns: e.g. "toyota/camry_4cyl_fwd/2016/default/speed_grade/v1"
        """
        return (
            f"{self.make}/{self.model}/"
            f"{format_year(self.year)}/{self.variant}/"
            f"{self.feature_set_id}/v{self.version}"
        )

    def to_dict(self) -> dict:
        return {
            "make": self.make,
            "model": self.model,
            "year": serialize_year(self.year),
            "variant": self.variant,
            "feature_set_id": self.feature_set_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelId:
        d = d.copy()
        if "model_name" in d and "model" not in d:
            d["model"] = d.pop("model_name")
        return cls(**d)

    def __str__(self) -> str:
        return (
            f"{self.make}/{self.model}/{format_year(self.year)}/"
            f"{self.variant}/{self.feature_set_id}/v{self.version}"
        )


@dataclass
class ModelInfo:
    """Lightweight model summary returned from registry queries (no binary data)."""

    model_id: ModelId
    estimator_type: str
    feature_names: List[str]
    target_names: List[str]
    powertrain_type: str
    vehicle_description: str
    path: Optional[str] = None
    mass_lbs: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id.to_dict(),
            "estimator_type": self.estimator_type,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "powertrain_type": self.powertrain_type,
            "vehicle_description": self.vehicle_description,
            "path": self.path,
            "mass_lbs": self.mass_lbs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelInfo:
        d = d.copy()
        d["model_id"] = ModelId.from_dict(d["model_id"])
        d.setdefault("mass_lbs", None)
        return cls(**d)

    def __repr__(self) -> str:
        lines = [
            "\n\nModelInfo:",
            f"  model_id:       {self.model_id.to_path()}",
            f"  description:    {self.vehicle_description}",
            f"  year:           {format_year(self.model_id.year)}",
            f"  make:           {self.model_id.make}",
            f"  model:          {self.model_id.model}",
            f"  powertrain:     {self.powertrain_type}",
            f"  variant:        {self.model_id.variant}",
            f"  estimator:      {self.estimator_type}",
            f"  features:       {self.feature_names}",
            f"  targets:        {self.target_names}",
            f"  mass_lbs:       {self.mass_lbs}",
        ]
        return "\n".join(lines)
