from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Literal, Optional, cast

import numpy as np
import onnx
import onnxruntime as rt
import pandas as pd

from routee.powertrain.core.model_config import ModelConfig, PredictMethod
from routee.powertrain.estimators.estimator_interface import (
    Estimator,
    InputSpec,
    PadStrategy,
)

ONNX_INPUT_NAME = "input"
ONNX_DTYPE = "float32"

_META_LOOKBACK = "routee_lookback"
_META_GROUPING = "routee_grouping_column"
_META_PAD = "routee_pad_strategy"


def _read_input_spec(onnx_model: onnx.ModelProto) -> InputSpec:
    """Read an InputSpec from ONNX metadata_props, or return the default if absent."""
    lookback: int = 0
    grouping: Optional[str] = None
    pad: PadStrategy = "repeat_first"
    for mp in onnx_model.metadata_props:
        if mp.key == _META_LOOKBACK:
            lookback = int(mp.value)
        elif mp.key == _META_GROUPING:
            grouping = mp.value or None
        elif mp.key == _META_PAD:
            if mp.value not in ("zero", "repeat_first"):
                raise ValueError(f"Unknown pad strategy in ONNX metadata: {mp.value}")
            pad = cast(PadStrategy, mp.value)
    return InputSpec(lookback=lookback, grouping_column=grouping, pad_strategy=pad)


def _embed_input_spec(
    onnx_model: onnx.ModelProto, input_spec: InputSpec
) -> onnx.ModelProto:
    """Return a copy of ``onnx_model`` with ``input_spec`` embedded in metadata_props.

    When ``input_spec.lookback == 0`` the model is tabular and no metadata is
    written (keeping the serialized artifact clean for the common case).
    """
    model = onnx.ModelProto()
    model.CopyFrom(onnx_model)
    keep = [
        mp
        for mp in model.metadata_props
        if mp.key not in (_META_LOOKBACK, _META_GROUPING, _META_PAD)
    ]
    del model.metadata_props[:]
    model.metadata_props.extend(keep)
    if input_spec.lookback > 0:
        for key, value in (
            (_META_LOOKBACK, str(input_spec.lookback)),
            (_META_GROUPING, input_spec.grouping_column or ""),
            (_META_PAD, input_spec.pad_strategy),
        ):
            prop = model.metadata_props.add()
            prop.key = key
            prop.value = value
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
        try:
            num_threads = len(os.sched_getaffinity(0))
        except AttributeError:
            num_threads = os.cpu_count() or 1
        sess_options.intra_op_num_threads = num_threads
        self.session = rt.InferenceSession(
            onnx_model.SerializeToString(),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_spec = input_spec

    @property
    def input_spec(self) -> InputSpec:
        return self._input_spec

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
        feature_set = config.feature_set
        distance = config.distance
        target_set = config.target
        predict_method = config.predict_method

        if predict_method == PredictMethod.RATE:
            feature_name_list = feature_set.feature_name_list
        elif predict_method == PredictMethod.RAW:
            feature_name_list = feature_set.feature_name_list + [distance.name]
        else:
            raise ValueError(
                f"Predict method {predict_method} is not supported by ONNXEstimator"
            )

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
