from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, ConfigDict, field_validator

from routee.powertrain.core.pydantic_fields import YearField
from routee.powertrain.core.year import format_year, parse_year

if TYPE_CHECKING:
    from routee.powertrain.core.metadata import Metadata

_VERSION_RE = re.compile(r"^v(\d+)$")


class ModelKey(BaseModel):
    """A model's intrinsic, version-less identity.

    ``make``/``model``/``year``/``config_slug`` are all pure functions of a
    model's metadata, so a ``ModelKey`` is fully determined the moment a model
    is trained — unlike ``version``, which is a registry coordinate assigned
    when the model is placed into a registry. ``Model.key`` exposes this, so
    every model self-describes its identity without needing a registry.

    Frozen (and therefore hashable) so it can be used as a grouping key that
    collapses the versions of one model.
    """

    model_config = ConfigDict(frozen=True)

    make: str
    model: str
    year: YearField
    config_slug: str

    @field_validator("make", "model", "config_slug", mode="after")
    @classmethod
    def _lowercase(cls, v: str) -> str:
        return v.lower()

    @classmethod
    def from_metadata(cls, metadata: Metadata) -> ModelKey:
        """Derive the version-less identity from a model's metadata."""
        from routee.powertrain.registry.slug import derive_config_slug

        return cls(
            make=metadata.vehicle.make,
            model=metadata.vehicle.model,
            year=metadata.vehicle.year,
            config_slug=derive_config_slug(metadata),
        )

    @classmethod
    def from_path(cls, path: str) -> ModelKey:
        """Parse a ModelKey from a ``make/model/year/config_slug`` path string."""
        parts = [p for p in path.strip("/").split("/") if p]
        if len(parts) != 4:
            raise ValueError(
                f"Cannot parse model key from path '{path}'. "
                f"Expected <make>/<model>/<year>/<config_slug>, "
                f"got {len(parts)} segments."
            )
        make, model, year_str, config_slug = parts
        return cls(
            make=make,
            model=model,
            year=parse_year(year_str),
            config_slug=config_slug,
        )

    def to_path(self) -> str:
        """Build the version-less registry path prefix for this model."""
        return f"{self.make}/{self.model}/{format_year(self.year)}/{self.config_slug}"

    def __str__(self) -> str:
        return self.to_path()


class ModelId(BaseModel):
    """Uniquely identifies a model in the registry.

    A ``config_slug`` disambiguates multiple models for the same vehicle/year —
    e.g. ``rf_steady_a1b2c3d4``, ``ngb_96224f1f``. The slug is *derived* from the
    model's metadata (architecture + optional ``config.variant`` + feature-set
    hash) via ``derive_config_slug``; it is not stored separately. The full
    feature composition and estimator architecture live in the archive's
    ``metadata.json`` (and in ``index.json`` for registry-level search).

    A ``ModelId`` is a :class:`ModelKey` (the intrinsic, version-less identity,
    derivable from metadata) plus a registry ``version`` (the only coordinate a
    registry assigns). Use ``id.key`` to get the version-less identity, or
    ``ModelId.from_key(key, version)`` / ``from_metadata(metadata, version)`` to
    attach a version. ``from_path`` reconstructs one from the frozen registry
    path.
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

    @property
    def key(self) -> ModelKey:
        """The version-less identity — everything except the registry version."""
        return ModelKey(
            make=self.make,
            model=self.model,
            year=self.year,
            config_slug=self.config_slug,
        )

    @classmethod
    def from_key(cls, key: ModelKey, version: int) -> ModelId:
        """Attach a registry ``version`` to a version-less ``ModelKey``."""
        return cls(
            make=key.make,
            model=key.model,
            year=key.year,
            config_slug=key.config_slug,
            version=version,
        )

    @classmethod
    def from_metadata(cls, metadata: Metadata, version: int) -> ModelId:
        """Mint a ModelId from a model's metadata plus a registry version.

        This is the canonical constructor: the version-less identity is derived
        from ``metadata`` (via ``ModelKey.from_metadata``), so the only
        registry-assigned coordinate is ``version``.

        Args:
            metadata: the model metadata
            version: the registry version (positive integer)

        Returns: a ModelId instance
        """
        return cls.from_key(ModelKey.from_metadata(metadata), version)

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
        return f"{self.key.to_path()}/v{self.version}"

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
    #: The model's registry-independent instance identity (``sha256:<hex>``),
    #: read from ``metadata.json``. ``None`` for models published before
    #: digests existed.
    model_digest: Optional[str] = None

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
        if self.model_digest is not None:
            from routee.powertrain.core.digest import short_digest

            lines.append(f"  digest:         {short_digest(self.model_digest)}")
        return "\n".join(lines)
