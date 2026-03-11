from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from routee.powertrain.core.year import Year, format_year, parse_year, serialize_year

_VERSION_RE = re.compile(r"^v(\d+)$")


@dataclass
class ModelId:
    """Uniquely identifies a model in the registry."""

    make: str
    model_name: str
    year: Year
    variant: str
    feature_set_id: str
    version: int

    def __post_init__(self):
        self.make = self.make.lower()
        self.model_name = self.model_name.lower()
        self.variant = self.variant.lower()
        self.feature_set_id = self.feature_set_id.lower()
        self.year = parse_year(self.year)

    @classmethod
    def from_path(cls, path: str) -> ModelId:
        """Parse a ModelId from a path string.

        Accepts both formats:
        - With schema version: ``v2/make/model/year/variant/feature_set/v1``
        - Without schema version: ``make/model/year/variant/feature_set/v1``

        Args:
            path: a ``/``-separated path string

        Returns: a ModelId instance

        Raises:
            ValueError: if the path cannot be parsed as a valid model id
        """
        parts = [p for p in path.strip("/").split("/") if p]

        # With schema version prefix (e.g. "v2/make/model/...")
        if len(parts) == 7:
            _, make, model_name, year_str, variant, feature_set_id, version_dir = parts
        # Without schema version prefix
        elif len(parts) == 6:
            make, model_name, year_str, variant, feature_set_id, version_dir = parts
        else:
            raise ValueError(
                f"Cannot parse model id from path '{path}'. "
                f"Expected <make>/<model>/<year>/<variant>/<feature_set_id>/v<N> "
                f"(optionally prefixed with a schema version), got {len(parts)} segments."
            )

        match = _VERSION_RE.match(version_dir)
        if not match:
            raise ValueError(
                f"Version segment '{version_dir}' does not match expected pattern v<N>"
            )

        return cls(
            make=make,
            model_name=model_name,
            year=parse_year(year_str),
            variant=variant,
            feature_set_id=feature_set_id,
            version=int(match.group(1)),
        )

    def to_path(self, schema_version: str = "v2") -> str:
        """
        Build the registry path for this model.

        Returns: e.g. "v2/toyota/camry_4cyl_fwd/2016/default/speed_grade/v1"
        """
        return (
            f"{schema_version}/{self.make}/{self.model_name}/"
            f"{format_year(self.year)}/{self.variant}/"
            f"{self.feature_set_id}/v{self.version}"
        )

    def to_dict(self) -> dict:
        return {
            "make": self.make,
            "model_name": self.model_name,
            "year": serialize_year(self.year),
            "variant": self.variant,
            "feature_set_id": self.feature_set_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelId:
        return cls(**d)

    def __str__(self) -> str:
        return (
            f"{self.make}/{self.model_name}/{format_year(self.year)}/"
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
    errors: Dict[str, Dict[str, float]]
    vehicle_description: str
    path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id.to_dict(),
            "estimator_type": self.estimator_type,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "powertrain_type": self.powertrain_type,
            "errors": self.errors,
            "vehicle_description": self.vehicle_description,
            "path": self.path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelInfo:
        d = d.copy()
        d["model_id"] = ModelId.from_dict(d["model_id"])
        return cls(**d)

    def __repr__(self) -> str:
        lines = [
            "ModelInfo:",
            f"  model_id:       {self.model_id.to_path()}",
            f"  description:    {self.vehicle_description}",
            f"  year:           {format_year(self.model_id.year)}",
            f"  make:           {self.model_id.make}",
            f"  model:          {self.model_id.model_name}",
            f"  powertrain:     {self.powertrain_type}",
            f"  variant:        {self.model_id.variant}",
            f"  estimator:      {self.estimator_type}",
            f"  features:       {self.feature_names}",
            f"  targets:        {self.target_names}",
        ]
        return "\n".join(lines)
