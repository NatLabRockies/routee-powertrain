from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Union

from routee.powertrain.core.model import Model
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
        model_name: Optional[str] = None,
        year: Optional[int] = None,
        variant: Optional[str] = None,
        feature_set_id: Optional[str] = None,
        fuzzy: bool = True,
        fuzzy_threshold: int = 80,
    ) -> List[ModelInfo]:
        """
        List models matching the given filters.

        All parameters are optional; passing none returns all models.
        Returns lightweight metadata — no model binaries are downloaded.

        Args:
            make: filter by vehicle make
            model_name: filter by model name
            year: filter by model year
            variant: filter by variant
            feature_set_id: filter by feature set id
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
    def get_metadata(self, model_id: Union[str, ModelId]) -> dict:
        """
        Fetch only the metadata for a model (without downloading the binary).

        Args:
            model_id: unique identifier for the model.
                Can be a ModelId instance or a string path that will
                be parsed via ``ModelId.from_path()``.

        Returns: the parsed metadata dictionary from the archive
        """
