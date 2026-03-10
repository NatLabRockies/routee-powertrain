from __future__ import annotations
from dataclasses import dataclass, field

from typing import Dict, List

import numpy as np


@dataclass
class Constraints:
    lower: float = -np.inf
    upper: float = np.inf

    def __post_init__(self):
        if self.lower >= self.upper:
            raise ValueError("lower bound must be less than upper bound")

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> Constraints:
        lower = float(d.get("lower", -np.inf))
        upper = float(d.get("upper", np.inf))
        return Constraints(lower=lower, upper=upper)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class DataColumn:
    name: str
    units: str

    dtype: str = "float32"

    constraints: Constraints = field(default_factory=Constraints)

    def __post_init__(self):
        if "&" in self.name:
            raise ValueError("feature name cannot contain '&'")

    @classmethod
    def from_dict(cls, d: dict) -> DataColumn:
        if "name" not in d:
            raise ValueError("must provide feature name when building from dictionary")
        elif "units" not in d:
            raise ValueError("must provide feature units when building from dictionary")
        elif "constraints" not in d:
            raise ValueError("must provide constraints when building from dictionary")

        constraints = Constraints.from_dict(d["constraints"])

        return DataColumn(name=d["name"], units=d["units"], constraints=constraints)

    def to_dict(self) -> dict:
        out = self.__dict__.copy()
        out["constraints"] = self.constraints.to_dict()

        return out


FeatureSetId = str


def feature_names_to_id(feature_names: List[str]) -> FeatureSetId:
    """
    Returns a string that uniquely identifies a feature set.
    The names are sorted to provide a consistent id.
    """
    sorted_names = sorted(feature_names)
    return "&".join(sorted_names)


def feature_id_to_names(feature_id: FeatureSetId) -> List[str]:
    """
    Returns a list of feature names from a feature set id.
    """
    return feature_id.split("&")


@dataclass
class FeatureSet:
    features: List[DataColumn]

    def __post_init__(self):
        if isinstance(self.features, DataColumn):
            self.features = [self.features]

    def __repr__(self) -> str:
        summary_lines = []
        for feature in self.features:
            summary_lines.append(f"{feature.name} ({feature.units})")
        return "\n".join(summary_lines)

    @property
    def features_id(self) -> FeatureSetId:
        """
        Returns a string that uniquely identifies this feature set.
        The names are sorted to provide a consistent id.
        """
        return feature_names_to_id(self.feature_name_list)

    @property
    def feature_map(self) -> Dict[str, DataColumn]:
        return {f.name: f for f in self.features}

    @property
    def feature_name_list(self) -> List[str]:
        """
        Returns a list of feature names in the order they
        appear in the feature set.

        Order is important since the underlying estimator might
        expect it.
        """
        return [f.name for f in self.features]

    def to_dict(self) -> dict:
        return {
            "features": [f.to_dict() for f in self.features],
        }

    @classmethod
    def from_dict(cls, json: dict) -> FeatureSet:
        features = [DataColumn.from_dict(d) for d in json["features"]]
        return FeatureSet(features)


@dataclass
class TargetSet:
    targets: List[DataColumn]

    def __post_init__(self):
        if isinstance(self.targets, DataColumn):
            self.targets = [self.targets]
        elif isinstance(self.targets, dict):
            self.targets = TargetSet.from_dict(self.targets)

    @property
    def target_map(self) -> Dict[str, DataColumn]:
        return {t.name: t for t in self.targets}

    @property
    def target_name_list(self) -> List[str]:
        return [t.name for t in self.targets]

    @property
    def target_rate_name_list(self) -> List[str]:
        return [f"{t.name}_rate" for t in self.targets]

    def to_dict(self) -> dict:
        return {
            "targets": [t.to_dict() for t in self.targets],
        }

    @classmethod
    def from_dict(cls, json: dict) -> TargetSet:
        targets = [DataColumn.from_dict(d) for d in json["targets"]]
        return TargetSet(targets)
