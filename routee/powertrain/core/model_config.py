from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, field_validator

from routee.powertrain.core.features import (
    DataColumn,
    FeatureSet,
    TargetSet,
)
from routee.powertrain.core.predict_method import PredictMethod
from routee.powertrain.core.pydantic_fields import (
    DrivetrainField,
    FuelTypeField,
    PowertrainTypeField,
    PredictMethodField,
    YearField,
)

# Re-exported for backwards compatibility: ``PredictMethod`` used to live here.
__all__ = ["ModelConfig", "PredictMethod"]


class ModelConfig(BaseModel):
    ## vehicle information
    vehicle_description: str
    powertrain_type: PowertrainTypeField

    ## estimator information
    feature_set: FeatureSet
    distance: DataColumn
    target: TargetSet

    ## structured vehicle identification
    make: str
    model: str
    year: YearField

    predict_method: PredictMethodField = PredictMethod.RATE

    test_size: float = 0.2
    random_seed: int = 42

    trip_column: str = "trip_id"

    apply_real_world_adjustment: bool = True

    mass_lbs: Optional[float] = None

    fuel_type: Optional[FuelTypeField] = None
    drivetrain: Optional[DrivetrainField] = None
    engine: Optional[str] = None
    trim: Optional[str] = None

    @field_validator("make", "model", mode="after")
    @classmethod
    def _lowercase(cls, v: str) -> str:
        return v.lower()

    @field_validator("feature_set", mode="before")
    @classmethod
    def _coerce_feature_set(cls, v: object) -> object:
        # accept a bare list of columns; dicts/FeatureSet handled natively
        if isinstance(v, list):
            return {"features": v}
        return v

    @field_validator("target", mode="before")
    @classmethod
    def _coerce_target(cls, v: object) -> object:
        # accept a single column or a bare list; dicts/TargetSet handled natively
        if isinstance(v, DataColumn):
            return {"targets": [v]}
        if isinstance(v, list):
            return {"targets": v}
        return v

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
