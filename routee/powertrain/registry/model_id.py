from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from routee.powertrain.core.year import Year, format_year, parse_year, serialize_year


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
            f"ModelInfo({self.model_id})",
            f"  make and model: {self.model_id.make} {self.model_id.model_name}",
            f"  year:           {format_year(self.model_id.year)}",
            f"  variant:        {self.model_id.variant}",
            f"  description:    {self.vehicle_description}",
            f"  estimator:      {self.estimator_type}",
            f"  features:       {self.feature_names}",
            f"  targets:        {self.target_names}",
            f"  powertrain:     {self.powertrain_type}",
        ]
        return "\n".join(lines)
