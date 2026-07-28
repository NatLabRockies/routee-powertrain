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
__all__ = [
    "ModelConfig",
    "PredictMethod",
    "Vehicle",
    "Contract",
    "TrainingConfig",
]


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

    test_size: Optional[float] = None
    validation_size: Optional[float] = None
    random_seed: int = 42

    trip_column: str = "trip_id"

    #: Optional human-readable identifier of the training dataset (e.g. a file
    #: name or dataset release label). Feeds the model digest, so two models
    #: trained on differently-named data get distinct identities.
    dataset_name: Optional[str] = None
    #: Optional fingerprint of the training data — use
    #: ``routee.powertrain.hash_dataframe(df)`` to compute one. Feeds the model
    #: digest.
    dataset_hash: Optional[str] = None

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


# ---------------------------------------------------------------------------
# Grouped metadata sections
#
# ``ModelConfig`` is the flat object a user builds to train a model. When the
# model is persisted, its fields are stored decomposed into the sections below,
# grouped by the job a reader needs them for: identity (``Vehicle``),
# input/output contract (``Contract``), and build-time reproduction
# (``TrainingConfig``). ``Metadata.config`` reconstructs a flat ``ModelConfig``
# from these on demand, so nothing is stored twice.
# ---------------------------------------------------------------------------


class Vehicle(BaseModel):
    """The vehicle a model describes — identity plus descriptive attributes.

    ``make``/``model``/``year``/``variant`` feed the derived ``config_slug`` and
    ``ModelKey``; the remaining fields are descriptive and registry-filterable.
    """

    vehicle_description: str
    powertrain_type: PowertrainTypeField

    make: str
    model: str
    year: YearField
    variant: Optional[str] = None

    mass_lbs: Optional[float] = None
    fuel_type: Optional[FuelTypeField] = None
    drivetrain: Optional[DrivetrainField] = None
    engine: Optional[str] = None
    trim: Optional[str] = None

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

    @classmethod
    def from_config(cls, config: ModelConfig) -> Vehicle:
        return cls(
            vehicle_description=config.vehicle_description,
            powertrain_type=config.powertrain_type,
            make=config.make,
            model=config.model,
            year=config.year,
            variant=config.variant,
            mass_lbs=config.mass_lbs,
            fuel_type=config.fuel_type,
            drivetrain=config.drivetrain,
            engine=config.engine,
            trim=config.trim,
        )


class Contract(BaseModel):
    """A model's input/output contract — everything needed to interpret a
    prediction: the feature columns it consumes, the distance column, the energy
    target(s) it emits, how the raw estimator output maps to energy
    (``predict_method``), and the real-world correction applied afterward.
    """

    feature_set: FeatureSet
    distance: DataColumn
    target: TargetSet

    predict_method: PredictMethodField = PredictMethod.RATE

    #: Multiplicative factor applied to predicted energy to correct for
    #: real-world conditions. Resolved on the source ``ModelConfig`` (defaulting
    #: from the powertrain type) and stored concretely here.
    real_world_adjustment_factor: float = 1.0

    @field_validator("feature_set", mode="before")
    @classmethod
    def _coerce_feature_set(cls, v: object) -> object:
        if isinstance(v, list):
            return {"features": v}
        return v

    @field_validator("target", mode="before")
    @classmethod
    def _coerce_target(cls, v: object) -> object:
        if isinstance(v, DataColumn):
            return {"targets": [v]}
        if isinstance(v, list):
            return {"targets": v}
        return v

    @classmethod
    def from_config(cls, config: ModelConfig) -> Contract:
        return cls(
            feature_set=config.feature_set,
            distance=config.distance,
            target=config.target,
            predict_method=config.predict_method,
            real_world_adjustment_factor=config.real_world_adjustment_factor,
        )


class TrainingConfig(BaseModel):
    """Build-time hyperparameters — needed to reproduce training, not to use the
    model. Safe to drop from a shipped artifact without affecting prediction.
    """

    test_size: Optional[float] = None
    validation_size: Optional[float] = None
    random_seed: int = 42
    trip_column: str = "trip_id"

    #: Calendar date the model was trained, as an ISO ``YYYY-MM-DD`` string.
    #: Stamped at training time by ``Trainer.train``; ``None`` when unknown
    #: (e.g. legacy models converted from the v1 format).
    trained_date: Optional[str] = None

    #: Optional identifier of the training dataset (see ``ModelConfig``).
    dataset_name: Optional[str] = None
    #: Optional fingerprint of the training data (see ``ModelConfig``).
    dataset_hash: Optional[str] = None

    @classmethod
    def from_config(cls, config: ModelConfig) -> TrainingConfig:
        return cls(
            test_size=config.test_size,
            validation_size=config.validation_size,
            random_seed=config.random_seed,
            trip_column=config.trip_column,
            dataset_name=config.dataset_name,
            dataset_hash=config.dataset_hash,
        )
