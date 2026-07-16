from __future__ import annotations

from pathlib import Path
import base64
import io
import json
import pandas as pd

from importlib.util import find_spec

from routee.powertrain.core.model_config import ModelConfig, PredictMethod
from routee.powertrain.estimators.estimator_interface import (
    ColumnSpec,
    Estimator,
    InputSpec,
)
from typing import List


class NGBoostEstimator(Estimator):
    file_extension: str = ".joblib"

    def __init__(self, ngboost) -> None:
        self.model = ngboost

    def output_column_specs(self, config: ModelConfig) -> List[ColumnSpec]:
        """NGBoost emits a point prediction plus a per-target standard deviation."""
        target = config.target.targets[0]
        return [
            ColumnSpec.from_data_column(target),
            ColumnSpec(
                name=f"{target.name}_std", units=target.units, dtype=target.dtype
            ),
        ]

    @classmethod
    def from_file(cls, filepath: str | Path) -> Estimator:
        """
        Load an estimator from a file
        """
        filepath = Path(filepath)

        with filepath.open("rb") as f:
            loaded_dict = json.load(f)

        return cls.from_dict(loaded_dict)

    def to_file(self, filepath: str | Path):
        """
        Save an estimator to a file
        """
        filepath = Path(filepath)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, in_dict: dict) -> NGBoostEstimator:
        """
        Load an estimator from a bytes object in memory
        """
        if find_spec("ngboost") is None:
            raise ImportError(
                "The NGBoostEstimator estimator requires extra dependencies like joblib and ngboost. "
                "To install, you can do pip install routee.powertrain[ngboost]"
            )

        if find_spec("joblib") is None:
            raise ImportError(
                "The NGBoostEstimator estimator requires extra dependencies like joblib and ngboost. "
                "To install, you can do pip install routee.powertrain[ngboost]"
            )
        else:
            import joblib

        model_base64 = in_dict.get("ngboost_model")

        if model_base64 is None:
            raise ValueError(
                "Model file must contain ngboost model at key: 'ngboost_model'"
            )
        byte_stream = io.BytesIO(base64.b64decode(model_base64))
        ngboost_model = joblib.load(byte_stream)
        estimator = cls(ngboost_model)
        in_spec = in_dict.get("input_spec")
        if in_spec is not None:
            estimator.input_spec = InputSpec.model_validate(in_spec)
        return estimator

    def to_dict(self) -> dict:
        """
        Serialize an estimator to a python dictionary
        """
        try:
            import joblib
        except ImportError:
            raise ImportError(
                "The NGBoostEstimator estimator requires extra dependencies like joblib and ngboost. "
                "To install, you can do pip install routee.powertrain[ngboost]"
            )
        byte_stream = io.BytesIO()
        joblib.dump(self.model, byte_stream)
        byte_stream.seek(0)
        model_base64 = base64.b64encode(byte_stream.read()).decode("utf-8")
        out_dict: dict = {"ngboost_model": model_base64}
        out_dict["input_spec"] = self.input_spec.model_dump(mode="json")

        return out_dict

    def to_bytes(self) -> bytes:
        try:
            import joblib
        except ImportError:
            raise ImportError(
                "The NGBoostEstimator estimator requires extra dependencies like joblib and ngboost. "
                "To install, you can do pip install routee.powertrain[ngboost]"
            )
        byte_stream = io.BytesIO()
        joblib.dump(self.model, byte_stream)
        byte_stream.seek(0)
        return byte_stream.read()

    @classmethod
    def from_bytes(cls, data: bytes) -> NGBoostEstimator:
        if find_spec("joblib") is None:
            raise ImportError(
                "The NGBoostEstimator estimator requires extra dependencies like joblib and ngboost. "
                "To install, you can do pip install routee.powertrain[ngboost]"
            )
        import joblib

        byte_stream = io.BytesIO(data)
        ngboost_model = joblib.load(byte_stream)
        return cls(ngboost_model)

    def predict(
        self,
        links_df: pd.DataFrame,
        config: ModelConfig,
    ) -> pd.DataFrame:
        distance = config.distance
        target_set = config.target
        predict_method = config.predict_method

        if len(target_set.targets) != 1:
            raise ValueError(
                "NGBoost only supports a single energy target. "
                "Please use a different estimator for multiple energy targets."
            )
        energy = target_set.targets[0]

        distance_col = distance.name
        if predict_method not in (PredictMethod.RATE, PredictMethod.RAW):
            raise ValueError(
                f"Predict method {predict_method} is not supported by NGBoostEstimator"
            )
        # Single source of truth for the positional input order (features, plus
        # distance appended for RAW) — matches the embedded input contract.
        x = links_df[config.all_feature_names].values

        energy_pred_series = self.model.pred_dist(x.tolist())
        energy_pred_mean = energy_pred_series.loc
        energy_pred_std = energy_pred_series.scale

        energy_df = pd.DataFrame(index=links_df.index)

        if predict_method == PredictMethod.RAW:
            energy_pred_mean = energy_pred_mean
            energy_pred_std = energy_pred_std

        elif predict_method == PredictMethod.RATE:
            energy_pred_mean = energy_pred_mean * links_df[distance_col]
            energy_pred_std = energy_pred_std * links_df[distance_col]

        else:
            raise ValueError(
                f"Predict method {predict_method} is not supported by NGBoostEstimator"
            )

        energy_df[energy.name] = energy_pred_mean
        energy_df[energy.name + "_std"] = energy_pred_std

        return energy_df
