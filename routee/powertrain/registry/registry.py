from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence, Union

from routee.powertrain.core.model import Model
from routee.powertrain.registry.filtering import VersionStrategy
from routee.powertrain.registry.model_id import ModelId, ModelInfo


def _resolve_model_id(model_id: Union[str, ModelId]) -> ModelId:
    """Coerce a string or ModelId into a ModelId."""
    if isinstance(model_id, str):
        return ModelId.from_path(model_id)
    return model_id


class ModelRegistry(ABC):
    """
    Abstract interface for a model registry backend.

    Implementations provide model discovery (query) and retrieval (load).
    The registry is read-only from the package's perspective; publishing
    models is handled out-of-band by CI/scripts.
    """

    @abstractmethod
    def query(
        self,
        make: Optional[str] = None,
        model: Optional[str] = None,
        year: Optional[int] = None,
        config_slug: Optional[str] = None,
        feature_names: Optional[Sequence[str]] = None,
        powertrain_type: Optional[str] = None,
        fuel_type: Optional[str] = None,
        drivetrain: Optional[str] = None,
        engine: Optional[str] = None,
        trim: Optional[str] = None,
        version: Optional[int] = None,
        version_strategy: VersionStrategy = "latest",
        custom_filters: Optional[Sequence[Callable[[ModelInfo], bool]]] = None,
        fuzzy: bool = True,
        fuzzy_threshold: int = 80,
    ) -> List[ModelInfo]:
        """
        List models matching the given filters.

        All parameters are optional; passing none returns all models.
        Returns lightweight metadata — no model binaries are downloaded.

        Args:
            make: filter by vehicle make
            model: filter by model name
            year: filter by model year
            config_slug: filter by config slug (e.g. "rf_default", "cnn_5link")
            feature_names: filter to models whose feature set contains every
                listed feature column (subset match, exact names)
            powertrain_type: filter by powertrain type (e.g. "ICE", "BEV", "HEV")
            fuel_type: filter by fuel type (e.g. "GASOLINE", "DIESEL", "ELECTRICITY")
            drivetrain: filter by drivetrain (e.g. "FWD", "RWD", "AWD")
            engine: filter by engine specification (e.g. "4cyl", "2.0tdi")
            trim: filter by trim level (e.g. "sport", "active")
            version: pin results to an exact version (e.g. 2). When set,
                ``version_strategy`` is ignored.
            version_strategy: how to collapse multiple versions of the same
                model. ``"latest"`` (default) keeps only the highest version
                per (make, model, year, config_slug) group; ``"all"`` returns
                every version. Ignored when ``version`` is specified.
            custom_filters: optional list of callables that accept a ModelInfo
                and return True to keep the model or False to exclude it
            fuzzy: if True, use fuzzy string matching for string
                fields (default True)
            fuzzy_threshold: minimum score (0–100) for a fuzzy match
                to be accepted (default 80)
        """

    @abstractmethod
    def load(self, model_id: Union[str, ModelId]) -> Model:
        """
        Download and deserialize a specific model.

        Args:
            model_id: unique identifier for the model to load.
                Can be a ModelId instance or a string path that will
                be parsed via ``ModelId.from_path()``.

        Returns: a fully deserialized Model instance
        """

    @abstractmethod
    def list_models(
        self,
        version_strategy: VersionStrategy = "latest",
    ) -> List[ModelId]:
        """
        List model identifiers in the registry.

        This is a lightweight operation that returns only the model
        paths/identifiers without fetching any metadata or binaries.

        Args:
            version_strategy: ``"latest"`` (default) returns only the highest
                version per (make, model, year, config_slug) group; ``"all"``
                returns every versioned identifier.

        Returns: list of ModelId matching the strategy
        """

    @abstractmethod
    def get_metadata(self, model_id: Union[str, ModelId]) -> dict:
        """
        Fetch only the metadata for a model (without downloading the binary).

        Args:
            model_id: unique identifier for the model.
                Can be a ModelId instance or a string path that will
                be parsed via ``ModelId.from_path()``.

        Returns: the parsed metadata dictionary from the archive
        """
