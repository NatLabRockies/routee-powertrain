import logging
from abc import ABC, abstractmethod

import pandas as pd

from routee.powertrain.core.metadata import Metadata
from routee.powertrain.core.model import Model
from routee.powertrain.core.model_config import ModelConfig, PredictMethod
from routee.powertrain.estimators.estimator_interface import Estimator
from routee.powertrain.trainers.utils import test_train_split
from routee.powertrain.validation.errors import compute_errors

ENERGY_RATE_NAME = "energy_rate"

log = logging.getLogger(__name__)


class Trainer(ABC):
    def train(self, data: pd.DataFrame, config: ModelConfig) -> Model:
        """
        A wrapper for inner train that does some pre and post processing.
        """
        distance_name = config.distance.name

        if config.predict_method == PredictMethod.RATE:
            for energy_target in config.target.targets:
                energy_rate_name = f"{energy_target.name}_rate"
                data[energy_rate_name] = data[energy_target.name] / data[distance_name]

        train, test = test_train_split(
            data, test_size=config.test_size, seed=config.random_seed
        )

        all_features = train[config.all_feature_names]
        if all_features.isnull().values.any():
            raise ValueError("Features contain null values")

        if config.predict_method == PredictMethod.RATE:
            target = train[config.target.target_rate_name_list]
        else:
            target = train[config.target.target_name_list]

        if target.isnull().values.any():
            raise ValueError(
                "Energy target contains null values. The predict method is "
                f" set to {config.predict_method} and the target is {config.target}."
            )

        # train the estimator for the feature set
        feature_set = config.feature_set
        name_list = list(feature_set.feature_name_list)
        if config.predict_method == PredictMethod.RAW:
            name_list.append(distance_name)
        sub_features = all_features[name_list]
        estimator = self.inner_train(
            features=sub_features, target=target, config=config
        )

        metadata = Metadata(config=config)

        model_errors = compute_errors(test, estimator, feature_set, config)

        vehicle_model = Model(estimator, metadata, model_errors)

        return vehicle_model

    @abstractmethod
    def inner_train(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
        config: ModelConfig,
    ) -> Estimator:
        """
        Builds an estimator from the given data.
        """
        pass
