from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from routee.powertrain.core.model import Model
from routee.powertrain.registry.model_id import ModelId, ModelInfo


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
        trim: Optional[str] = None,
        variant: Optional[str] = None,
        feature_set_id: Optional[str] = None,
    ) -> List[ModelInfo]:
        """
        List models matching the given filters.

        All parameters are optional; passing none returns all models.
        Returns lightweight metadata — no model binaries are downloaded.
        """

    @abstractmethod
    def load(self, model_id: ModelId) -> Model:
        """
        Download and deserialize a specific model.

        Args:
            model_id: unique identifier for the model to load

        Returns: a fully deserialized Model instance
        """

    @abstractmethod
    def get_metadata(self, model_id: ModelId) -> dict:
        """
        Fetch only the metadata for a model (without downloading the binary).

        Args:
            model_id: unique identifier for the model

        Returns: the parsed metadata dictionary from the archive
        """
