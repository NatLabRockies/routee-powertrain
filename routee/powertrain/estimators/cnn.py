from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal, cast

import numpy as np
import onnx
import onnxruntime as rt
import pandas as pd

from routee.powertrain.core.model_config import ModelConfig, PredictMethod
from routee.powertrain.estimators.estimator_interface import Estimator, InputSpec

CNN_INPUT_NAME = "input"
CNN_DTYPE = "float32"
DEFAULT_LOOKBACK = 5
DEFAULT_GROUPING_COLUMN = "route_id"

_META_LOOKBACK = "routee_lookback"
_META_GROUPING = "routee_grouping_column"
_META_PAD = "routee_pad_strategy"


def _read_hparams(
    onnx_model: onnx.ModelProto,
) -> tuple[int, str, Literal["zero", "repeat_first"]]:
    lookback = DEFAULT_LOOKBACK
    grouping = DEFAULT_GROUPING_COLUMN
    pad: Literal["zero", "repeat_first"] = "zero"
    for mp in onnx_model.metadata_props:
        if mp.key == _META_LOOKBACK:
            lookback = int(mp.value)
        elif mp.key == _META_GROUPING:
            grouping = mp.value
        elif mp.key == _META_PAD:
            if mp.value not in ("zero", "repeat_first"):
                raise ValueError(f"Unknown pad strategy in ONNX metadata: {mp.value}")
            pad = cast(Literal["zero", "repeat_first"], mp.value)
    return lookback, grouping, pad


def _embed_hparams(
    onnx_model: onnx.ModelProto,
    lookback: int,
    grouping_column: str,
    pad_strategy: str,
) -> onnx.ModelProto:
    """Return a copy of the ONNX proto with the CNN hparams embedded in metadata_props."""
    model = onnx.ModelProto()
    model.CopyFrom(onnx_model)
    keep = [
        mp
        for mp in model.metadata_props
        if mp.key not in (_META_LOOKBACK, _META_GROUPING, _META_PAD)
    ]
    del model.metadata_props[:]
    model.metadata_props.extend(keep)
    for key, value in (
        (_META_LOOKBACK, str(lookback)),
        (_META_GROUPING, grouping_column),
        (_META_PAD, pad_strategy),
    ):
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value
    return model


class CNNEstimator(Estimator):
    """CNN estimator with a fixed lookback window across ordered links in a route.

    Wraps an ONNX-exported 1D CNN. For each prediction the model sees the
    current link's features plus the features of the prior ``lookback - 1``
    links from the same route (padded per ``pad_strategy`` at the route's
    start). Input tensor shape: ``(N, lookback, F)``.
    """

    file_extension: str = ".onnx"

    def __init__(
        self,
        onnx_model: onnx.ModelProto,
        lookback: int = DEFAULT_LOOKBACK,
        grouping_column: str = DEFAULT_GROUPING_COLUMN,
        pad_strategy: Literal["zero", "repeat_first"] = "zero",
    ) -> None:
        self.onnx_model = onnx_model
        self.session = rt.InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        self._lookback = lookback
        self._grouping_column = grouping_column
        self._pad_strategy = pad_strategy

    @property
    def input_spec(self) -> InputSpec:
        return InputSpec(
            lookback=self._lookback,
            grouping_column=self._grouping_column,
            pad_strategy=cast(Literal["zero", "repeat_first"], self._pad_strategy),
        )

    @classmethod
    def from_dict(cls, in_dict: dict) -> CNNEstimator:
        onnx_raw = in_dict.get("onnx_model")
        if onnx_raw is None:
            raise ValueError("Model file must contain onnx model at key 'onnx_model'")
        data = base64.b64decode(onnx_raw)
        model = onnx.load_from_string(data)
        lookback, grouping, pad = _read_hparams(model)
        return cls(
            model,
            lookback=int(in_dict.get("lookback", lookback)),
            grouping_column=in_dict.get("grouping_column", grouping),
            pad_strategy=in_dict.get("pad_strategy", pad),
        )

    def to_dict(self) -> dict:
        model = _embed_hparams(
            self.onnx_model, self._lookback, self._grouping_column, self._pad_strategy
        )
        return {
            "onnx_model": base64.b64encode(model.SerializeToString()).decode("utf-8"),
            "lookback": self._lookback,
            "grouping_column": self._grouping_column,
            "pad_strategy": self._pad_strategy,
        }

    @classmethod
    def from_file(cls, filepath: str | Path) -> CNNEstimator:
        filepath = Path(filepath)
        if filepath.suffix != ".onnx":
            raise ValueError("CNN model must be saved as a .onnx file")
        with filepath.open("rb") as f:
            model = onnx.load_from_string(f.read())
        lookback, grouping, pad = _read_hparams(model)
        return cls(model, lookback=lookback, grouping_column=grouping, pad_strategy=pad)

    def to_file(self, filepath: str | Path):
        filepath = Path(filepath)
        if filepath.suffix != ".onnx":
            raise ValueError("CNN model must be saved as a .onnx file")
        model = _embed_hparams(
            self.onnx_model, self._lookback, self._grouping_column, self._pad_strategy
        )
        with filepath.open("wb") as f:
            f.write(model.SerializeToString())

    def to_bytes(self) -> bytes:
        model = _embed_hparams(
            self.onnx_model, self._lookback, self._grouping_column, self._pad_strategy
        )
        return model.SerializeToString()

    @classmethod
    def from_bytes(cls, data: bytes) -> CNNEstimator:
        model = onnx.load_from_string(data)
        lookback, grouping, pad = _read_hparams(model)
        return cls(model, lookback=lookback, grouping_column=grouping, pad_strategy=pad)

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
                f"Predict method {predict_method} is not supported by CNNEstimator"
            )

        n_rows = len(links_df)
        n_features = len(feature_name_list)
        feature_matrix = links_df[feature_name_list].to_numpy(dtype=CNN_DTYPE)

        windowed = np.zeros((n_rows, self._lookback, n_features), dtype=CNN_DTYPE)

        groups = links_df.groupby(self._grouping_column, sort=False).indices
        for idx_array in groups.values():
            idx_array = np.asarray(idx_array)
            group_features = feature_matrix[idx_array]
            if self._pad_strategy == "repeat_first":
                pad_row = group_features[:1]
            else:
                pad_row = np.zeros((1, n_features), dtype=CNN_DTYPE)
            padded = np.concatenate(
                [np.repeat(pad_row, self._lookback - 1, axis=0), group_features],
                axis=0,
            )
            for i, row_pos in enumerate(idx_array):
                windowed[row_pos] = padded[i : i + self._lookback]

        pred = self.session.run(None, {CNN_INPUT_NAME: windowed})[0]

        energy_df = pd.DataFrame(index=links_df.index)
        for i, energy in enumerate(target_set.targets):
            energy_pred_series = pd.Series(pred[:, i], index=links_df.index)
            if predict_method == PredictMethod.RATE:
                energy_pred = energy_pred_series * links_df[distance.name]
            else:
                energy_pred = energy_pred_series
            energy_df[energy.name] = energy_pred

        return energy_df
