from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from routee.powertrain.core.features import DataColumn, FeatureSet, TargetSet
from routee.powertrain.core.model_config import PredictMethod


class Estimator(ABC):
    """Abstract base class for all estimator backends."""

    #: File extension used when serializing this estimator's binary in a ZIP archive.
    file_extension: str

    @classmethod
    @abstractmethod
    def from_file(cls, filepath: str | Path) -> Estimator:
        """
        Load an estimator from a file
        """

    @abstractmethod
    def to_file(self, filepath: str | Path):
        """
        Save an estimator to a file
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, in_dict: dict) -> Estimator:
        """
        Load an estimator from a bytes object in memory
        """

    @abstractmethod
    def to_dict(self) -> dict:
        """
        Serialize an estimator to a python dictionary
        """

    @abstractmethod
    def to_bytes(self) -> bytes:
        """
        Serialize the estimator to raw bytes (native binary format).
        """

    @classmethod
    @abstractmethod
    def from_bytes(cls, data: bytes) -> Estimator:
        """
        Deserialize an estimator from raw bytes.
        """

    @abstractmethod
    def predict(
        self,
        links_df: pd.DataFrame,
        feature_set: FeatureSet,
        distance: DataColumn,
        target_set: TargetSet,
        predict_method: PredictMethod = PredictMethod.RATE,
    ) -> pd.DataFrame:
        """
        Predict absolute energy consumption for each link
        """
