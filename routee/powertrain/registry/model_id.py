from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelId:
    """Uniquely identifies a model in the registry."""

    make: str
    model_name: str
    year: int
    trim: str
    variant: str
    version: int

    def __post_init__(self):
        self.make = self.make.lower()
        self.model_name = self.model_name.lower()
        self.trim = self.trim.lower()
        self.variant = self.variant.lower()

    def to_path(self, schema_version: str = "v2") -> str:
        """
        Build the registry path for this model.

        Returns: e.g. "v2/toyota/camry/2016/4cyl_fwd/default/v1.zip"
        """
        return (
            f"{schema_version}/{self.make}/{self.model_name}/"
            f"{self.year}/{self.trim}/{self.variant}/v{self.version}.zip"
        )

    def to_dict(self) -> dict:
        return {
            "make": self.make,
            "model_name": self.model_name,
            "year": self.year,
            "trim": self.trim,
            "variant": self.variant,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelId:
        return cls(**d)

    def __str__(self) -> str:
        return (
            f"{self.make}/{self.model_name}/{self.year}/"
            f"{self.trim}/{self.variant}/v{self.version}"
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
            f"  description: {self.vehicle_description}",
            f"  estimator:   {self.estimator_type}",
            f"  features:    {self.feature_names}",
            f"  targets:     {self.target_names}",
            f"  powertrain:  {self.powertrain_type}",
        ]
        return "\n".join(lines)
