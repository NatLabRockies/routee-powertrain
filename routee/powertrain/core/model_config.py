from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

from routee.powertrain.core.features import (
    DataColumn,
    FeatureSet,
    TargetSet,
)
from routee.powertrain.core.powertrain_type import PowertrainType


class PredictMethod(Enum):
    # Predict the rate of energy consumption and then multiply it by the distance.
    RATE = "rate"
    # Predict the total energy consumption for the link (including distance as a feature).
    RAW = "raw"

    @classmethod
    def from_string(cls, string: str) -> PredictMethod:
        if string.lower() == "rate":
            return PredictMethod.RATE
        elif string.lower() == "raw":
            return PredictMethod.RAW
        else:
            raise ValueError("Unknown predict method: {}".format(string))


@dataclass
class ModelConfig:
    ## vehicle information
    vehicle_description: str
    powertrain_type: PowertrainType

    ## estimator information
    feature_set: FeatureSet
    distance: DataColumn
    target: TargetSet

    ## structured vehicle identification
    make: str
    model_name: str
    year: int
    trim: str

    predict_method: PredictMethod = PredictMethod.RATE

    test_size: float = 0.2
    random_seed: int = 42

    trip_column: str = "trip_id"

    apply_real_world_adjustment: bool = True

    def __post_init__(self):
        # normalize vehicle id fields to lowercase
        self.make = self.make.lower()
        self.model_name = self.model_name.lower()
        self.trim = self.trim.lower()
        # convert feature_set to the correct type
        if isinstance(self.feature_set, dict):
            self.feature_set = FeatureSet.from_dict(self.feature_set)
        elif isinstance(self.feature_set, list):
            self.feature_set = FeatureSet(features=self.feature_set)

        if isinstance(self.distance, dict):
            self.distance = DataColumn.from_dict(self.distance)

        if isinstance(self.target, dict):
            self.target = TargetSet.from_dict(self.target)
        elif isinstance(self.target, DataColumn):
            self.target = TargetSet([self.target])
        elif isinstance(self.target, list):
            self.target = TargetSet(self.target)

        if isinstance(self.powertrain_type, str):
            self.powertrain_type = PowertrainType.from_string(self.powertrain_type)

        if isinstance(self.predict_method, str):
            self.predict_method = PredictMethod.from_string(self.predict_method)

        # now check all the types
        if not isinstance(self.feature_set, FeatureSet):
            raise ValueError("feature_set must be a FeatureSet")
        if not isinstance(self.distance, DataColumn):
            raise ValueError("Distance must be a DataColumn")
        if not isinstance(self.target, TargetSet):
            raise ValueError("Target set must be a TargetSet")
        if not isinstance(self.powertrain_type, PowertrainType):
            raise ValueError("Powertrain type must be a PowertrainType")
        if not isinstance(self.predict_method, PredictMethod):
            raise ValueError("Predict method must be a PredictMethod")

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        # provide defaults for legacy model files that lack vehicle id fields
        d = d.copy()
        d.setdefault("make", "unknown")
        d.setdefault("model_name", "unknown")
        d.setdefault("year", 0)
        d.setdefault("trim", "unknown")
        return cls(**d)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["powertrain_type"] = self.powertrain_type.name
        d["feature_set"] = self.feature_set.to_dict()
        d["distance"] = self.distance.to_dict()
        d["target"] = self.target.to_dict()
        d["predict_method"] = self.predict_method.value

        return d

    @property
    def feature_names(self) -> List[str]:
        """
        Returns the list of feature names from the feature set.
        """
        return self.feature_set.feature_name_list

    @property
    def all_feature_names(self) -> List[str]:
        """
        Returns the list of feature names, including distance if predict method is RAW.
        """
        names = list(self.feature_set.feature_name_list)
        if self.predict_method == PredictMethod.RAW:
            names.append(self.distance.name)
        return names

    @property
    def all_features(self) -> List[DataColumn]:
        """
        Returns the list of features, including distance if predict method is RAW.
        """
        features = list(self.feature_set.features)
        if self.predict_method == PredictMethod.RAW:
            features.append(self.distance)
        return features
