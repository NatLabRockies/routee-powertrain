from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from routee.powertrain.core.metadata import Metadata
from routee.powertrain.core.model import Model
from routee.powertrain.core.model_config import ModelConfig, PredictMethod
from routee.powertrain.estimators.estimator_interface import Estimator
from routee.powertrain.trainers.utils import test_train_validation_split
from routee.powertrain.validation.errors import compute_errors

ENERGY_RATE_NAME = "energy_rate"

log = logging.getLogger(__name__)


class Trainer(ABC):
    #: Coarse architecture family, used in Metadata for registry-level filtering.
    #: Subclasses override (e.g. ``"random_forest"``, ``"cnn"``, ``"ngboost"``).
    architecture_tag: str = "unknown"
    
    #: Default split sizes used when ModelConfig leaves them unspecified.
    default_test_size: float = 0.2
    default_validation_size: float = 0.0

    @property
    def required_extra_columns(self) -> List[str]:
        """Columns the trainer needs beyond the declared feature set.

        Example: a CNN trainer with lookback needs a grouping column
        (e.g. ``route_id``) so windows don't cross route boundaries.
        """
        return []

    @property
    def split_grouping_column(self) -> str | None:
        """If set, the train/test split keeps all rows of a given group together.

        Sequence-aware trainers (e.g. the 1D CNN) must set this so that a
        route's links stay contiguous within train or test — otherwise the
        per-group lookback windows built at both train and predict time stitch
        together non-consecutive rows and the temporal signal is lost.
        """
        return None

    def train(self, data: pd.DataFrame, config: ModelConfig) -> Model:
        """
        A wrapper for inner train that does some pre and post processing.
        """
        distance_name = config.distance.name

        if config.predict_method == PredictMethod.RATE:
            for energy_target in config.target.targets:
                energy_rate_name = f"{energy_target.name}_rate"
                data[energy_rate_name] = data[energy_target.name] / data[distance_name]

        effective_test_size = (
            config.test_size if config.test_size is not None else self.default_test_size
        )
        effective_validation_size = (
            config.validation_size
            if config.validation_size is not None
            else self.default_validation_size
        )
        use_validation_split = effective_validation_size > 0

        train, validation, test = test_train_validation_split(
            data,
            test_size=effective_test_size,
            validation_size=effective_validation_size,
            seed=config.random_seed,
            grouping_column=self.split_grouping_column,
        )

        feature_columns = list(config.all_feature_names)
        all_features = train[feature_columns]
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
        for extra in self.required_extra_columns:
            if extra not in train.columns:
                raise ValueError(
                    f"Trainer requires column '{extra}' which is not in the input data"
                )
            if extra not in name_list:
                name_list.append(extra)
        sub_features = train[name_list]
        validation_sub_features: pd.DataFrame | None = None
        validation_target: pd.DataFrame | None = None
        if use_validation_split and not validation.empty:
            validation_sub_features = validation[name_list]
            if config.predict_method == PredictMethod.RATE:
                validation_target = validation[config.target.target_rate_name_list]
            else:
                validation_target = validation[config.target.target_name_list]

        estimator = self.inner_train(
            features=sub_features,
            target=target,
            config=config,
            validation_features=validation_sub_features,
            validation_target=validation_target,
        )

        model_errors = compute_errors(test, estimator, config)

        metadata = Metadata(
            config=config,
            errors=model_errors,
            estimator_type=estimator.__class__.__name__,
            model_file="model" + estimator.file_extension,
            architecture_tag=self.architecture_tag,
            input_spec=estimator.input_spec.to_dict(),
        )

        vehicle_model = Model(estimator, metadata)

        return vehicle_model

    @abstractmethod
    def inner_train(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
        config: ModelConfig,
        validation_features: pd.DataFrame | None = None,
        validation_target: pd.DataFrame | None = None,
    ) -> Estimator:
        """
        Builds an estimator from the given data.
        """
        pass
