from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import (
    BaseModel,
    Field,
    SerializationInfo,
    field_validator,
    model_serializer,
    model_validator,
)


class Constraints(BaseModel):
    lower: Optional[float] = None
    upper: Optional[float] = None

    @model_validator(mode="after")
    def _check_bounds(self) -> Constraints:
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower >= self.upper
        ):
            raise ValueError("lower bound must be less than upper bound")
        return self


class DataColumn(BaseModel):
    name: str
    units: str

    dtype: str = "float32"

    constraints: Constraints = Field(default_factory=Constraints)

    @field_validator("name")
    @classmethod
    def _name_no_ampersand(cls, v: str) -> str:
        if "&" in v:
            raise ValueError("feature name cannot contain '&'")
        return v


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


class FeatureSet(BaseModel):
    features: List[DataColumn]

    @field_validator("features", mode="before")
    @classmethod
    def _wrap_single(cls, v: object) -> object:
        if isinstance(v, DataColumn):
            return [v]
        return v

    @model_serializer
    def _serialize(self, info: SerializationInfo) -> list:
        # Serialize as a bare list of feature columns (no nested "features" key);
        # a model holds a single feature set, so the wrapper adds no information.
        return [f.model_dump(mode=info.mode) for f in self.features]

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


class TargetSet(BaseModel):
    targets: List[DataColumn]

    @field_validator("targets", mode="before")
    @classmethod
    def _wrap_single(cls, v: object) -> object:
        if isinstance(v, DataColumn):
            return [v]
        return v

    @model_serializer
    def _serialize(self, info: SerializationInfo) -> list:
        # Serialize as a bare list of target columns (no nested "targets" key).
        return [t.model_dump(mode=info.mode) for t in self.targets]

    @property
    def target_map(self) -> Dict[str, DataColumn]:
        return {t.name: t for t in self.targets}

    @property
    def target_name_list(self) -> List[str]:
        return [t.name for t in self.targets]

    @property
    def target_rate_name_list(self) -> List[str]:
        return [f"{t.name}_rate" for t in self.targets]
