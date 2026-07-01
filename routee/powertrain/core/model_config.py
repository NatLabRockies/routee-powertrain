from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, field_validator, model_validator

from routee.powertrain.core.features import (
    DataColumn,
    FeatureSet,
    TargetSet,
)
from routee.powertrain.core.powertrain_type import PowertrainType
from routee.powertrain.core.predict_method import PredictMethod
from routee.powertrain.core.real_world_adjustments import ADJUSTMENT_FACTORS
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

    #: Short label distinguishing configs that share the same architecture and
    #: feature set (e.g. ``"steady"`` vs ``"warmup"`` thermal regimes). Feeds the
    #: derived ``config_slug``; leave ``None`` when no such distinction is needed.
    variant: Optional[str] = None

    predict_method: PredictMethodField = PredictMethod.RATE

    test_size: float = 0.2
    random_seed: int = 42

    trip_column: str = "trip_id"

    #: Multiplicative factor applied to predicted energy to correct for
    #: real-world conditions (e.g. temperature). Defaults to the
    #: powertrain-type factor in ``ADJUSTMENT_FACTORS``; set to ``1.0`` to
    #: apply no adjustment.
    real_world_adjustment_factor: float = 1.0

    mass_lbs: Optional[float] = None

    fuel_type: Optional[FuelTypeField] = None
    drivetrain: Optional[DrivetrainField] = None
    engine: Optional[str] = None
    trim: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _default_adjustment_factor(cls, data: object) -> object:
        # When no factor is supplied, derive it from the powertrain type so the
        # real-world adjustment matches the vehicle's default behavior.
        if isinstance(data, dict) and data.get("real_world_adjustment_factor") is None:
            pt_val = data.get("powertrain_type")
            try:
                pt = (
                    pt_val
                    if isinstance(pt_val, PowertrainType)
                    else PowertrainType.from_string(pt_val)
                )
            except Exception:
                pt = PowertrainType.UNDEFINED
            data = dict(data)
            data["real_world_adjustment_factor"] = float(
                ADJUSTMENT_FACTORS.get(pt, 1.0)
            )
        return data

    @field_validator("make", "model", mode="after")
    @classmethod
    def _lowercase(cls, v: str) -> str:
        return v.lower()

    @field_validator("variant", mode="after")
    @classmethod
    def _slug_safe_variant(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().lower()
        if not v:
            return None
        if "/" in v or any(c.isspace() for c in v):
            raise ValueError(
                f"variant '{v}' must not contain '/' or whitespace; "
                "use a short slug-safe label like 'steady' or 'warmup'"
            )
        return v

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
