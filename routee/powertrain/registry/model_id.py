from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, field_validator

from routee.powertrain.core.pydantic_fields import YearField
from routee.powertrain.core.year import format_year, parse_year

_VERSION_RE = re.compile(r"^v(\d+)$")


class ModelId(BaseModel):
    """Uniquely identifies a model in the registry.

    A ``config_slug`` disambiguates multiple models for the same vehicle/year —
    e.g. ``rf_default``, ``cnn_5link``, ``rf_speed_grade``. The slug is a
    user-chosen identifier; the full feature composition and estimator
    architecture live in the archive's ``metadata.json`` (and in ``index.json``
    for registry-level search).
    """

    make: str
    model: str
    year: YearField
    config_slug: str
    version: int

    def __init__(
        self,
        make: object = None,
        model: object = None,
        year: object = None,
        config_slug: object = None,
        version: object = None,
        **data: object,
    ) -> None:
        # Preserve positional construction (make, model, year, config_slug, version)
        # while still routing through pydantic validation.
        if make is not None:
            data["make"] = make
        if model is not None:
            data["model"] = model
        if year is not None:
            data["year"] = year
        if config_slug is not None:
            data["config_slug"] = config_slug
        if version is not None:
            data["version"] = version
        super().__init__(**data)

    @field_validator("make", "model", "config_slug", mode="after")
    @classmethod
    def _lowercase(cls, v: str) -> str:
        return v.lower()

    @classmethod
    def from_path(cls, path: str) -> ModelId:
        """Parse a ModelId from a path string.

        Expected format: ``make/model/year/config_slug/v<N>``

        Args:
            path: a ``/``-separated path string

        Returns: a ModelId instance

        Raises:
            ValueError: if the path cannot be parsed as a valid model id
        """
        parts = [p for p in path.strip("/").split("/") if p]

        if len(parts) != 5:
            raise ValueError(
                f"Cannot parse model id from path '{path}'. "
                f"Expected <make>/<model>/<year>/<config_slug>/v<N>, "
                f"got {len(parts)} segments."
            )

        make, model, year_str, config_slug, version_dir = parts

        match = _VERSION_RE.match(version_dir)
        if not match:
            raise ValueError(
                f"Version segment '{version_dir}' does not match expected pattern v<N>"
            )

        return cls(
            make=make,
            model=model,
            year=parse_year(year_str),
            config_slug=config_slug,
            version=int(match.group(1)),
        )

    def to_path(self) -> str:
        """
        Build the registry path for this model.

        Returns: e.g. "toyota/camry_4cyl_fwd/2016/rf_default/v1"
        """
        return (
            f"{self.make}/{self.model}/"
            f"{format_year(self.year)}/{self.config_slug}/v{self.version}"
        )

    def __str__(self) -> str:
        return (
            f"{self.make}/{self.model}/{format_year(self.year)}/"
            f"{self.config_slug}/v{self.version}"
        )


class ModelInfo(BaseModel):
    """Lightweight model summary returned from registry queries (no binary data)."""

    model_id: ModelId
    estimator_type: str
    feature_names: List[str]
    target_names: List[str]
    powertrain_type: str
    vehicle_description: str
    architecture_tag: str = "unknown"
    input_spec: Optional[dict] = None
    path: Optional[str] = None
    mass_lbs: Optional[float] = None
    fuel_type: Optional[str] = None
    drivetrain: Optional[str] = None
    engine: Optional[str] = None
    trim: Optional[str] = None

    def __repr__(self) -> str:
        lines = [
            "\n\nModelInfo:",
            f"  model_id:       {self.model_id.to_path()}",
            f"  description:    {self.vehicle_description}",
            f"  year:           {format_year(self.model_id.year)}",
            f"  make:           {self.model_id.make}",
            f"  model:          {self.model_id.model}",
            f"  powertrain:     {self.powertrain_type}",
            f"  config_slug:    {self.model_id.config_slug}",
            f"  architecture:   {self.architecture_tag}",
            f"  estimator:      {self.estimator_type}",
            f"  features:       {self.feature_names}",
            f"  targets:        {self.target_names}",
            f"  mass_lbs:       {self.mass_lbs}",
            f"  fuel_type:      {self.fuel_type}",
            f"  drivetrain:     {self.drivetrain}",
            f"  engine:         {self.engine}",
            f"  trim:           {self.trim}",
        ]
        return "\n".join(lines)
