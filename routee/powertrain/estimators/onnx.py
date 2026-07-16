from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Literal, Optional, cast

import numpy as np
import onnx
import onnxruntime as rt
import pandas as pd

from routee.powertrain.core.model_config import ModelConfig, PredictMethod
from routee.powertrain.estimators.estimator_interface import (
    ColumnSpec,
    Estimator,
    InputSpec,
    PadStrategy,
)
from routee.powertrain.utils.threading import get_restricted_threads

ONNX_INPUT_NAME = "input"
ONNX_DTYPE = "float32"

_META_LOOKBACK = "routee_lookback"
_META_GROUPING = "routee_grouping_column"
_META_PAD = "routee_pad_strategy"
_META_INPUT_COLUMNS = "routee_input_columns"
_META_OUTPUT_COLUMNS = "routee_output_columns"
_META_PREDICT_METHOD = "routee_predict_method"
_META_DISTANCE_COLUMN = "routee_distance_column"

#: Every metadata_props key this module manages, stripped before re-embedding so
#: a re-save never leaves stale contract entries behind.
_ROUTEE_META_KEYS = (
    _META_LOOKBACK,
    _META_GROUPING,
    _META_PAD,
    _META_INPUT_COLUMNS,
    _META_OUTPUT_COLUMNS,
    _META_PREDICT_METHOD,
    _META_DISTANCE_COLUMN,
)


def _read_input_spec(onnx_model: onnx.ModelProto) -> InputSpec:
    """Read an InputSpec from ONNX metadata_props, or return the default if absent."""
    lookback: int = 0
    grouping: Optional[str] = None
    pad: PadStrategy = "repeat_first"
    input_columns: Optional[list[ColumnSpec]] = None
    output_columns: Optional[list[ColumnSpec]] = None
    predict_method: Optional[str] = None
    distance_column: Optional[str] = None
    for mp in onnx_model.metadata_props:
        if mp.key == _META_LOOKBACK:
            lookback = int(mp.value)
        elif mp.key == _META_GROUPING:
            grouping = mp.value or None
        elif mp.key == _META_PAD:
            if mp.value not in ("zero", "repeat_first"):
                raise ValueError(f"Unknown pad strategy in ONNX metadata: {mp.value}")
            pad = cast(PadStrategy, mp.value)
        elif mp.key == _META_INPUT_COLUMNS:
            input_columns = [ColumnSpec(**d) for d in json.loads(mp.value)]
        elif mp.key == _META_OUTPUT_COLUMNS:
            output_columns = [ColumnSpec(**d) for d in json.loads(mp.value)]
        elif mp.key == _META_PREDICT_METHOD:
            predict_method = mp.value or None
        elif mp.key == _META_DISTANCE_COLUMN:
            distance_column = mp.value or None
    return InputSpec(
        lookback=lookback,
        grouping_column=grouping,
        pad_strategy=pad,
        input_columns=input_columns,
        output_columns=output_columns,
        predict_method=predict_method,
        distance_column=distance_column,
    )


def _embed_input_spec(
    onnx_model: onnx.ModelProto, input_spec: InputSpec
) -> onnx.ModelProto:
    """Return a copy of ``onnx_model`` with ``input_spec`` embedded in metadata_props.

    The input/output contract (ordered columns, predict method, distance column)
    is written whenever present — including the common ``lookback == 0`` tabular
    case — so a consumer holding only the ``.onnx`` can reconstruct the exact
    positional input order. The windowing keys are written only when
    ``lookback > 0``.
    """
    model = onnx.ModelProto()
    model.CopyFrom(onnx_model)
    keep = [mp for mp in model.metadata_props if mp.key not in _ROUTEE_META_KEYS]
    del model.metadata_props[:]
    model.metadata_props.extend(keep)

    def _add(key: str, value: str) -> None:
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value

    if input_spec.input_columns is not None:
        _add(
            _META_INPUT_COLUMNS,
            json.dumps([c.model_dump() for c in input_spec.input_columns]),
        )
    if input_spec.output_columns is not None:
        _add(
            _META_OUTPUT_COLUMNS,
            json.dumps([c.model_dump() for c in input_spec.output_columns]),
        )
    if input_spec.predict_method is not None:
        _add(_META_PREDICT_METHOD, input_spec.predict_method)
    if input_spec.distance_column is not None:
        _add(_META_DISTANCE_COLUMN, input_spec.distance_column)
    if input_spec.lookback > 0:
        _add(_META_LOOKBACK, str(input_spec.lookback))
        _add(_META_GROUPING, input_spec.grouping_column or "")
        _add(_META_PAD, input_spec.pad_strategy)
    return model


def _build_windows(
    feature_matrix: np.ndarray,
    links_df: pd.DataFrame,
    lookback: int,
    grouping_column: str,
    pad_strategy: Literal["zero", "repeat_first"],
) -> np.ndarray:
    """Assemble a (N, lookback, F) tensor of per-row sliding windows within each group."""
    n_rows, n_features = feature_matrix.shape
    windowed = np.zeros((n_rows, lookback, n_features), dtype=ONNX_DTYPE)
    groups = links_df.groupby(grouping_column, sort=False).indices
    for idx_array in groups.values():
        idx_array = np.asarray(idx_array)
        group_features = feature_matrix[idx_array]
        if pad_strategy == "repeat_first":
            pad_row = group_features[:1]
        else:
            pad_row = np.zeros((1, n_features), dtype=ONNX_DTYPE)
        padded = np.concatenate(
            [np.repeat(pad_row, lookback - 1, axis=0), group_features],
            axis=0,
        )
        for i, row_pos in enumerate(idx_array):
            windowed[row_pos] = padded[i : i + lookback]
    return windowed


class ONNXEstimator(Estimator):
    """Runs any ONNX model via ``onnxruntime``.

    When ``input_spec.lookback > 0`` the estimator wraps feature rows into a
    ``(N, lookback, F)`` windowed tensor grouped by ``input_spec.grouping_column``
    (with padding at sequence starts per ``input_spec.pad_strategy``) before
    inference. When ``lookback == 0`` the estimator feeds a plain ``(N, F)``
    tabular tensor — the common case for tree ensembles converted via ``skl2onnx``.
    """

    onnx_model: onnx.ModelProto
    session: rt.InferenceSession
    file_extension: str = ".onnx"

    def __init__(
        self,
        onnx_model: onnx.ModelProto,
        input_spec: InputSpec = InputSpec(),
    ) -> None:
        self.onnx_model = onnx_model
        sess_options = rt.SessionOptions()
        restricted_threads = get_restricted_threads()
        if restricted_threads is not None:
            sess_options.intra_op_num_threads = restricted_threads
        self.session = rt.InferenceSession(
            onnx_model.SerializeToString(),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_spec = input_spec

    @classmethod
    def from_dict(cls, in_dict: dict) -> ONNXEstimator:
        onnx_model_raw = in_dict.get("onnx_model")
        if onnx_model_raw is None:
            raise ValueError("Model file must contain onnx model at key: 'onnx_model'")
        in_bytes = base64.b64decode(onnx_model_raw)
        onnx_model = onnx.load_from_string(in_bytes)
        return cls(onnx_model, input_spec=_read_input_spec(onnx_model))

    def to_dict(self) -> dict:
        model = _embed_input_spec(self.onnx_model, self._input_spec)
        return {
            "onnx_model": base64.b64encode(model.SerializeToString()).decode("utf-8"),
        }

    @classmethod
    def from_file(cls, filepath: str | Path) -> ONNXEstimator:
        filepath = Path(filepath)
        if filepath.suffix != ".onnx":
            raise ValueError("ONNX model must be saved as a .onnx file")
        with filepath.open("rb") as f:
            onnx_model = onnx.load_from_string(f.read())
        return cls(onnx_model, input_spec=_read_input_spec(onnx_model))

    def to_file(self, filepath: str | Path):
        filepath = Path(filepath)
        if filepath.suffix != ".onnx":
            raise ValueError("ONNX model must be saved as a .onnx file")
        model = _embed_input_spec(self.onnx_model, self._input_spec)
        with filepath.open("wb") as f:
            f.write(model.SerializeToString())

    def to_bytes(self) -> bytes:
        model = _embed_input_spec(self.onnx_model, self._input_spec)
        return model.SerializeToString()

    @classmethod
    def from_bytes(cls, data: bytes) -> ONNXEstimator:
        onnx_model = onnx.load_from_string(data)
        return cls(onnx_model, input_spec=_read_input_spec(onnx_model))

    def predict(
        self,
        links_df: pd.DataFrame,
        config: ModelConfig,
    ) -> pd.DataFrame:
        distance = config.distance
        target_set = config.target
        predict_method = config.predict_method

        if predict_method not in (PredictMethod.RATE, PredictMethod.RAW):
            raise ValueError(
                f"Predict method {predict_method} is not supported by ONNXEstimator"
            )

        # ``all_feature_names`` is the single source of truth for the positional
        # input order (features, plus distance appended for RAW) — the same list
        # that ``bind_io_contract`` embeds into the estimator binary.
        feature_name_list = config.all_feature_names

        feature_matrix = links_df[feature_name_list].to_numpy(dtype=ONNX_DTYPE)

        spec = self._input_spec
        if spec.lookback > 0:
            if spec.grouping_column is None:
                raise ValueError(
                    "ONNXEstimator with lookback > 0 requires input_spec.grouping_column"
                )
            x = _build_windows(
                feature_matrix,
                links_df,
                spec.lookback,
                spec.grouping_column,
                spec.pad_strategy,
            )
        else:
            x = feature_matrix

        pred = self.session.run(None, {ONNX_INPUT_NAME: x})[0]

        energy_df = pd.DataFrame(index=links_df.index)
        for i, energy in enumerate(target_set.targets):
            energy_pred_series = pd.Series(pred[:, i], index=links_df.index)
            if predict_method == PredictMethod.RATE:
                energy_pred = energy_pred_series * links_df[distance.name]
            else:
                energy_pred = energy_pred_series
            energy_df[energy.name] = energy_pred

        return energy_df
