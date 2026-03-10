from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

from routee.powertrain.core.model_config import ModelConfig
from routee.powertrain.utils.fs import get_version


@dataclass
class Metadata:
    """
    A named tuple carrying model metadata information that gets set post training
    """

    config: ModelConfig

    routee_version: str = get_version()

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["config"] = self.config.to_dict()

        return d

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

        d["config"] = ModelConfig.from_dict(d["config"])

        return Metadata(**d)

    @classmethod
    def from_json(cls, j: str) -> Metadata:
        return cls.from_dict(json.loads(j))
