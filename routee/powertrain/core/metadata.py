from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

from routee.powertrain.core.model_config import ModelConfig
from routee.powertrain.utils.fs import get_version
from routee.powertrain.validation.errors import ModelErrors

SCHEMA_VERSION = 2
SCHEMA_VERSION_STRING = f"v{SCHEMA_VERSION}"


@dataclass
class Metadata:
    """
    Carries all model metadata that gets persisted alongside the estimator binary.

    Serializes 1:1 with the ``metadata.json`` file inside a model archive.
    """

    config: ModelConfig
    errors: ModelErrors
    estimator_type: str
    model_file: str
    routee_version: str = field(default_factory=get_version)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "estimator_type": self.estimator_type,
            "model_file": self.model_file,
            "config": self.config.to_dict(),
            "routee_version": self.routee_version,
            "errors": self.errors.to_dict(),
        }

    def to_json(self) -> str:
        """
        Convert metadata to json string
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> Metadata:
        v = get_version()
        major_v = v.split(".")[0]

        incoming_v = d["routee_version"]
        incoming_major_v = incoming_v.split(".")[0]
        if incoming_major_v != major_v:
            warnings.warn(
                "this model was trained using routee-powertrain version "
                f"{d['routee_version']} but you're using version {v}"
            )

        return Metadata(
            config=ModelConfig.from_dict(d["config"]),
            errors=ModelErrors.from_dict(d["errors"]),
            estimator_type=d["estimator_type"],
            model_file=d["model_file"],
            routee_version=d["routee_version"],
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )

    @classmethod
    def from_json(cls, j: str) -> Metadata:
        return cls.from_dict(json.loads(j))
