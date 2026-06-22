from enum import Enum

import numpy as np
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestRegressor

from routee.powertrain.core.model_config import ModelConfig
from routee.powertrain.estimators.estimator_interface import Estimator
from routee.powertrain.estimators.onnx import ONNX_INPUT_NAME, ONNXEstimator
from routee.powertrain.trainers.trainer import Trainer


class RandomForestTrainerOutput(Enum):
    ONNX = 1


class SklearnRandomForestTrainer(Trainer):
    architecture_tag: str = "random_forest"

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 10,
        n_estimators: int = 20,
        random_state: int = 52,
        cores: int = 4,
        output_type=RandomForestTrainerOutput.ONNX,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.cores = cores
        self.output_type = output_type

    def inner_train(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
        config: ModelConfig,
        **kwargs: object,
    ) -> Estimator:
        """
        Uses a random forest to predict the energy rate values
        """
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_jobs=self.cores,
            random_state=self.random_state,
        )
        X = features.values
        y: np.ndarray = target.values

        if y.shape[1] == 1:
            y = y.ravel()

        rf.fit(X, y)

        if self.output_type == RandomForestTrainerOutput.ONNX:
            # convert to ONNX
            n_features = len(features.columns)
            n_targets = len(target.columns)

            # explicity specify the output shape since skl2onnx was not able to infer it
            def custom_transform_shape_calculator(operator):
                operator.outputs[0].type = FloatTensorType([None, n_targets])

            initial_type = [(ONNX_INPUT_NAME, FloatTensorType([None, n_features]))]
            onnx_model = convert_sklearn(
                rf,
                initial_types=initial_type,
                custom_shape_calculators={
                    rf.__class__: custom_transform_shape_calculator
                },
            )

            estimator = ONNXEstimator(onnx_model)
        else:
            # extension point here for adding other estimator output types
            raise ValueError(f"Unknown output type: {self.output_type}")

        return estimator
