import numpy as np
import pandas as pd

from routee.powertrain.core.model_config import ModelConfig
from routee.powertrain.estimators.estimator_interface import Estimator
from routee.powertrain.estimators.smart_core import SmartCoreEstimator
from routee.powertrain.trainers.trainer import Trainer


class SmartcoreRandomForestTrainer(Trainer):
    def inner_train(
        self, features: pd.DataFrame, target: pd.DataFrame, config: ModelConfig
    ) -> Estimator:
        """
        Uses a random forest to predict the energy rate values
        """
        try:
            from powertrain_rust import RustRandomForest
        except ImportError:
            raise ImportError(
                "Please install powertrain_rust to use "
                "the SmartCoreRandomForest estimator."
            )

        x = features.values.tolist()
        y: np.ndarray = target.values

        if y.shape[1] != 1:
            raise ValueError(
                "SmartCore only supports a single energy rate. "
                "Please use a different estimator."
            )
        y = y.ravel()

        model = RustRandomForest()
        model.train(x, y.tolist())

        estimator = SmartCoreEstimator(model)

        return estimator
