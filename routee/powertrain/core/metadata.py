from __future__ import annotations

import re
import warnings
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

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

_MODEL_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ESTIMATOR_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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
    #: Bare lowercase-hex sha256 of the exact serialized estimator bytes (the
    #: file named by ``model_file``). A pure content address, stamped at train
    #: time and verified against the raw bytes on load. ``None`` for legacy
    #: models saved before digests existed.
    estimator_sha256: Optional[str] = None

    @field_validator("estimator_sha256", mode="after")
    @classmethod
    def _valid_estimator_sha256(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().lower()
        if not _ESTIMATOR_SHA256_RE.match(v):
            raise ValueError(
                f"estimator_sha256 must be 64 lowercase hex characters, got '{v}'"
            )
        return v


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
    #: Registry-independent instance identity, minted at train time:
    #: ``sha256:<64 hex>`` over the frozen spec-1 identity payload (see
    #: ``core.digest``), which embeds ``estimator.estimator_sha256`` — so the
    #: digest pins the binary transitively while remaining recomputable from
    #: metadata alone. Registry versions (``v<N>``) are coordinates that map to
    #: this identity, never the reverse. ``None`` for legacy models.
    model_digest: Optional[str] = None

    @field_validator("model_digest", mode="after")
    @classmethod
    def _valid_model_digest(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().lower()
        if not _MODEL_DIGEST_RE.match(v):
            raise ValueError(
                f"model_digest must have the form 'sha256:<64 hex chars>', got '{v}'"
            )
        return v

    @property
    def short_digest(self) -> Optional[str]:
        """Truncated display form of ``model_digest`` (``sha256:<12 hex>``)."""
        from routee.powertrain.core.digest import short_digest

        return short_digest(self.model_digest)

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
            dataset_name=self.training.dataset_name,
            dataset_hash=self.training.dataset_hash,
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
        trained_date: Optional[str] = None,
    ) -> Metadata:
        """Build grouped metadata from a flat ``ModelConfig`` and estimator facts.

        The inverse of the ``config`` property: decomposes the flat training
        config into the ``vehicle`` / ``contract`` / ``training`` sections and
        pairs them with the ``estimator`` descriptor. ``routee_version`` defaults
        to the running package version; pass it explicitly to preserve the
        provenance of a model trained under an older version (e.g. when
        converting legacy archives). ``trained_date`` (ISO ``YYYY-MM-DD``) is
        stamped onto the ``training`` section; leave it ``None`` when the
        training date is unknown (e.g. converting legacy archives).
        """
        training = TrainingConfig.from_config(config)
        training.trained_date = trained_date
        fields: dict = dict(
            vehicle=Vehicle.from_config(config),
            contract=Contract.from_config(config),
            training=training,
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
        """Warn only about models built by a *newer* major version.

        Older models are the normal case — the v2 registry is full of artifacts
        converted from v1 that still record the version that trained them — and
        real format drift is caught hard by the ``schema_version`` check in
        ``io/archive.py``. Warning on every backward-compatible load would fire
        on essentially every model in the library.
        """
        current = get_version()
        try:
            model_major = int(self.routee_version.split(".")[0])
            current_major = int(current.split(".")[0])
        except ValueError:
            return self
        if model_major > current_major:
            warnings.warn(
                "this model was trained using routee-powertrain version "
                f"{self.routee_version} but you're using version {current}; "
                "upgrade routee-powertrain if you hit unexpected behavior"
            )
        return self
