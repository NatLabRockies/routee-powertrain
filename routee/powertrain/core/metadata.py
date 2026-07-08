from __future__ import annotations

import warnings
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from routee.powertrain.core.model_config import (
    Contract,
    ModelConfig,
    TrainingConfig,
    Vehicle,
)
from routee.powertrain.utils.fs import get_version
from routee.powertrain.validation.errors import ModelErrors

SCHEMA_VERSION = 2
SCHEMA_VERSION_STRING = f"v{SCHEMA_VERSION}"


class EstimatorInfo(BaseModel):
    """Describes the serialized estimator artifact: what to load and how to
    shape inputs. Everything a consumer needs to instantiate and run the binary,
    without cracking it open.
    """

    estimator_type: str
    model_file: str
    #: Coarse architecture family (``"random_forest"``, ``"cnn"``, ``"ngboost"`` …).
    #: Used for registry-level filtering without parsing ``estimator_type`` strings.
    architecture_tag: str = "unknown"
    #: Serialized ``Estimator.input_spec`` (lookback, grouping_column, pad_strategy).
    #: Allows a registry consumer to see lookback requirements before loading the binary.
    input_spec: Optional[dict] = None


class Metadata(BaseModel):
    """
    Carries all model metadata that gets persisted alongside the estimator binary.

    Serializes 1:1 with the ``metadata.json`` file inside a model archive. Fields
    are grouped by the job a reader needs them for:

    - ``vehicle`` — the model's identity and descriptive attributes
    - ``contract`` — the input/output contract needed to interpret a prediction
    - ``estimator`` — how to load and run the serialized binary
    - ``training`` — build-time hyperparameters (reproduction only)
    - ``errors`` — validation metrics
    """

    vehicle: Vehicle
    contract: Contract
    estimator: EstimatorInfo
    training: TrainingConfig
    errors: ModelErrors
    routee_version: str = Field(default_factory=get_version)
    schema_version: int = SCHEMA_VERSION

    @property
    def config(self) -> ModelConfig:
        """A flat ``ModelConfig`` view reconstructed from the grouped sections.

        The identity/contract/training fields are stored decomposed, but many
        runtime consumers (estimators, error computation, ``Model.predict``)
        want the single flat object the model was trained from. This derives it
        on demand — nothing is stored twice.
        """
        return ModelConfig(
            vehicle_description=self.vehicle.vehicle_description,
            powertrain_type=self.vehicle.powertrain_type,
            make=self.vehicle.make,
            model=self.vehicle.model,
            year=self.vehicle.year,
            variant=self.vehicle.variant,
            mass_lbs=self.vehicle.mass_lbs,
            fuel_type=self.vehicle.fuel_type,
            drivetrain=self.vehicle.drivetrain,
            engine=self.vehicle.engine,
            trim=self.vehicle.trim,
            feature_set=self.contract.feature_set,
            distance=self.contract.distance,
            target=self.contract.target,
            predict_method=self.contract.predict_method,
            real_world_adjustment_factor=self.contract.real_world_adjustment_factor,
            test_size=self.training.test_size,
            validation_size=self.training.validation_size,
            random_seed=self.training.random_seed,
            trip_column=self.training.trip_column,
        )

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        errors: ModelErrors,
        estimator_type: str,
        model_file: str,
        architecture_tag: str = "unknown",
        input_spec: Optional[dict] = None,
        routee_version: Optional[str] = None,
    ) -> Metadata:
        """Build grouped metadata from a flat ``ModelConfig`` and estimator facts.

        The inverse of the ``config`` property: decomposes the flat training
        config into the ``vehicle`` / ``contract`` / ``training`` sections and
        pairs them with the ``estimator`` descriptor. ``routee_version`` defaults
        to the running package version; pass it explicitly to preserve the
        provenance of a model trained under an older version (e.g. when
        converting legacy archives).
        """
        fields: dict = dict(
            vehicle=Vehicle.from_config(config),
            contract=Contract.from_config(config),
            training=TrainingConfig.from_config(config),
            estimator=EstimatorInfo(
                estimator_type=estimator_type,
                model_file=model_file,
                architecture_tag=architecture_tag,
                input_spec=input_spec,
            ),
            errors=errors,
        )
        if routee_version is not None:
            fields["routee_version"] = routee_version
        return cls(**fields)

    @model_validator(mode="after")
    def _warn_version_mismatch(self) -> Metadata:
        current = get_version()
        if self.routee_version.split(".")[0] != current.split(".")[0]:
            warnings.warn(
                "this model was trained using routee-powertrain version "
                f"{self.routee_version} but you're using version {current}"
            )
        return self
