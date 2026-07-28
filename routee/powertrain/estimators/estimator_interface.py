from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict

from routee.powertrain.core.features import DataColumn
from routee.powertrain.core.model_config import ModelConfig

PadStrategy = Literal["zero", "repeat_first"]


class ColumnSpec(BaseModel):
    """Identity of one positional column in an estimator's input or output tensor.

    Carries the column ``name`` plus its ``units`` and ``dtype`` so a consumer
    holding only the serialized binary can both order its inputs correctly and
    interpret/convert their values.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    units: Optional[str] = None
    dtype: Optional[str] = None

    @classmethod
    def from_data_column(cls, column: DataColumn) -> "ColumnSpec":
        return cls(name=column.name, units=column.units, dtype=column.dtype)


class InputSpec(BaseModel):
    """The full input/output contract an estimator's serialized binary implements.

    Beyond the windowing fields (``lookback``/``grouping_column``/``pad_strategy``)
    it pins the *positional order* of the estimator's input and output tensors —
    the piece a downstream consumer (e.g. routee-compass) needs to feed columns
    in the right slots. The contract fields are ``Optional`` and default to
    ``None`` so legacy artifacts minted before the contract existed still parse.
    """

    model_config = ConfigDict(frozen=True)

    #: rows of prior context required per prediction. 0 = pointwise (classic tabular).
    lookback: int = 0
    #: column used to bucket rows into independent sequences (e.g. "route_id").
    #: Required whenever ``lookback > 0`` so windows don't cross sequence boundaries.
    grouping_column: Optional[str] = None
    #: how to pad the lookback window at the start of a group (first rows lack prior context).
    pad_strategy: PadStrategy = "repeat_first"

    #: Ordered positional columns of the input tensor: the feature columns, plus
    #: the distance column appended when ``predict_method == "raw"``. ``None`` on
    #: legacy artifacts that predate the contract.
    input_columns: Optional[List[ColumnSpec]] = None
    #: Ordered positional columns of the output tensor: the energy target(s), in
    #: order. Estimators that emit uncertainty append their std columns.
    output_columns: Optional[List[ColumnSpec]] = None
    #: How the raw estimator output maps to energy: ``"rate"`` (multiply by
    #: distance) or ``"raw"`` (already absolute energy).
    predict_method: Optional[str] = None
    #: The distance column — the RATE multiplier / the RAW input position.
    distance_column: Optional[str] = None


class Estimator(ABC):
    """Abstract base class for all estimator backends."""

    #: File extension used when serializing this estimator's binary in a ZIP archive.
    file_extension: str

    #: Backing store for :attr:`input_spec`. A class-level default (the frozen,
    #: hence shareable, empty contract) so estimators that never set one still
    #: read cleanly.
    _input_spec: InputSpec = InputSpec()

    @property
    def input_spec(self) -> InputSpec:
        """The input/output contract this estimator implements.

        Subclasses set the windowing fields at construction; the trainer stamps
        the ordered input/output columns via :meth:`bind_io_contract`.
        """
        return self._input_spec

    @input_spec.setter
    def input_spec(self, spec: InputSpec) -> None:
        self._input_spec = spec

    def output_column_specs(self, config: ModelConfig) -> List[ColumnSpec]:
        """Positional output tensor columns this estimator emits.

        Default: one column per energy target, in order. Estimators that emit
        extra columns (e.g. per-target uncertainty) override this.
        """
        return [ColumnSpec.from_data_column(t) for t in config.target.targets]

    def bind_io_contract(self, config: ModelConfig) -> None:
        """Stamp the input/output contract derived from ``config`` onto ``input_spec``.

        Preserves any windowing fields already set by the trainer and records the
        ordered input columns (features, plus distance for RAW), the ordered
        output columns, the predict method, and the distance column — so the
        serialized binary and metadata are self-describing and a consumer never
        has to guess the positional order.
        """
        self.input_spec = self.input_spec.model_copy(
            update={
                "input_columns": [
                    ColumnSpec.from_data_column(c) for c in config.all_features
                ],
                "output_columns": self.output_column_specs(config),
                "predict_method": config.predict_method.value,
                "distance_column": config.distance.name,
            }
        )

    @abstractmethod
    def to_bytes(self) -> bytes:
        """
        Serialize the estimator to raw bytes (native binary format).

        This is the estimator's *only* serialization primitive. Estimators are
        not independently persistable artifacts: a model on disk is always an
        estimator binary paired with its ``metadata.json`` sidecar, and that
        pairing — along with the required input/output contract and the instance
        digest — is enforced exclusively at the ``Model`` save/load choke points
        (see ``routee.powertrain.io.archive``). Persist via ``Model.to_file`` /
        ``Model.from_file``, never by writing these bytes directly.
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
