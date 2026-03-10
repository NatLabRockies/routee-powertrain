from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from routee.powertrain.registry.model_id import ModelInfo


@dataclass
class Catalog:
    """
    Central catalog index listing all models in a registry.

    This is the sole source for model discovery — a single file that can be
    fetched with one HTTP request, then filtered in-memory.
    """

    schema_version: str
    models: List[ModelInfo] = field(default_factory=list)

    def query(
        self,
        make: Optional[str] = None,
        model_name: Optional[str] = None,
        year: Optional[int] = None,
        trim: Optional[str] = None,
        variant: Optional[str] = None,
    ) -> List[ModelInfo]:
        """
        Filter models by the given criteria.  All filters are optional;
        passing none returns all models.
        """
        results = self.models
        if make is not None:
            make_lower = make.lower()
            results = [m for m in results if m.model_id.make == make_lower]
        if model_name is not None:
            model_name_lower = model_name.lower()
            results = [m for m in results if m.model_id.model_name == model_name_lower]
        if year is not None:
            results = [m for m in results if m.model_id.year == year]
        if trim is not None:
            trim_lower = trim.lower()
            results = [m for m in results if m.model_id.trim == trim_lower]
        if variant is not None:
            variant_lower = variant.lower()
            results = [m for m in results if m.model_id.variant == variant_lower]
        return results

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "models": [m.to_dict() for m in self.models],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Catalog:
        models = [ModelInfo.from_dict(m) for m in d.get("models", [])]
        return cls(schema_version=d["schema_version"], models=models)

    def to_json(self, path: Union[str, Path]) -> None:
        """Write the catalog to a JSON file."""
        path = Path(path)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> Catalog:
        """Read a catalog from a JSON file."""
        path = Path(path)
        with path.open("r") as f:
            return cls.from_dict(json.load(f))
