from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from routee.powertrain.core.model_config import ModelConfig

PadStrategy = Literal["zero", "repeat_first"]


@dataclass(frozen=True)
class InputSpec:
    """Declares what an estimator needs from the caller's dataframe beyond plain feature columns."""

    #: rows of prior context required per prediction. 0 = pointwise (classic tabular).
    lookback: int = 0
    #: column used to bucket rows into independent sequences (e.g. "route_id").
    #: Required whenever ``lookback > 0`` so windows don't cross sequence boundaries.
    grouping_column: Optional[str] = None
    #: how to pad the lookback window at the start of a group (first rows lack prior context).
    pad_strategy: PadStrategy = "repeat_first"

    def to_dict(self) -> dict:
        return {
            "lookback": self.lookback,
            "grouping_column": self.grouping_column,
            "pad_strategy": self.pad_strategy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> InputSpec:
        return cls(
            lookback=int(d.get("lookback", 0)),
            grouping_column=d.get("grouping_column"),
            pad_strategy=d.get("pad_strategy", "zero"),
        )


class Estimator(ABC):
    """Abstract base class for all estimator backends."""

    #: File extension used when serializing this estimator's binary in a ZIP archive.
    file_extension: str

    @property
    def input_spec(self) -> InputSpec:
        """Describe the shape of input this estimator consumes.

        Override in subclasses that need lookback or grouping.
        """
        return InputSpec()

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
        config: ModelConfig,
    ) -> pd.DataFrame:
        """
        Predict absolute energy consumption for each link.

        Args:
            links_df: the input dataframe. Must contain every column in
                ``config.feature_set`` plus (if ``predict_method == RAW``) the
                distance column, and (if ``input_spec.grouping_column`` is set)
                the grouping column.
            config: the model's ``ModelConfig``. Estimators read ``feature_set``,
                ``distance``, ``target`` and ``predict_method`` from here.
        """
