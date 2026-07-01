from __future__ import annotations

import warnings
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from routee.powertrain.core.model_config import ModelConfig
from routee.powertrain.utils.fs import get_version
from routee.powertrain.validation.errors import ModelErrors

SCHEMA_VERSION = 2
SCHEMA_VERSION_STRING = f"v{SCHEMA_VERSION}"


class Metadata(BaseModel):
    """
    Carries all model metadata that gets persisted alongside the estimator binary.

    Serializes 1:1 with the ``metadata.json`` file inside a model archive.
    """

    config: ModelConfig
    errors: ModelErrors
    estimator_type: str
    model_file: str
    #: Coarse architecture family (``"random_forest"``, ``"cnn"``, ``"ngboost"`` …).
    #: Used for registry-level filtering without parsing ``estimator_type`` strings.
    architecture_tag: str = "unknown"
    #: Serialized ``Estimator.input_spec`` (lookback, grouping_column, pad_strategy).
    #: Allows a registry consumer to see lookback requirements before loading the binary.
    input_spec: Optional[dict] = None
    routee_version: str = Field(default_factory=get_version)
    schema_version: int = SCHEMA_VERSION

    @model_validator(mode="after")
    def _warn_version_mismatch(self) -> Metadata:
        current = get_version()
        if self.routee_version.split(".")[0] != current.split(".")[0]:
            warnings.warn(
                "this model was trained using routee-powertrain version "
                f"{self.routee_version} but you're using version {current}"
            )
        return self
