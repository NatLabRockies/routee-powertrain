"""Reusable ``Annotated`` field types for pydantic models.

These centralize the coercion (on load) and readable serialization (on dump) of
the enum and ``Year`` fields shared across ``ModelConfig`` and ``ModelId``:

- enums deserialize via their ``from_string`` classmethod (case-insensitive,
  and accepting an already-constructed enum member) and serialize to a readable
  string (``.name`` for the int-valued enums, ``.value`` for ``PredictMethod``);
- ``Year`` deserializes via :func:`parse_year` (accepting int / list /
  ``"YYYY"`` / ``"YYYY-YYYY"``) and serializes natively — an int stays an int and
  a range becomes a JSON list ``[start, end]``.
"""

from __future__ import annotations

from typing import Annotated, Union

from pydantic import BeforeValidator, PlainSerializer

from routee.powertrain.core.drivetrain import Drivetrain
from routee.powertrain.core.fuel_type import FuelType
from routee.powertrain.core.powertrain_type import PowertrainType
from routee.powertrain.core.predict_method import PredictMethod
from routee.powertrain.core.year import parse_year


def _coerce_powertrain(v: object) -> PowertrainType:
    return v if isinstance(v, PowertrainType) else PowertrainType.from_string(v)  # type: ignore[arg-type]


def _coerce_fuel_type(v: object) -> FuelType:
    return v if isinstance(v, FuelType) else FuelType.from_string(v)  # type: ignore[arg-type]


def _coerce_drivetrain(v: object) -> Drivetrain:
    return v if isinstance(v, Drivetrain) else Drivetrain.from_string(v)  # type: ignore[arg-type]


def _coerce_predict_method(v: object) -> PredictMethod:
    return v if isinstance(v, PredictMethod) else PredictMethod.from_string(v)  # type: ignore[arg-type]


PowertrainTypeField = Annotated[
    PowertrainType,
    BeforeValidator(_coerce_powertrain),
    PlainSerializer(lambda v: v.name, return_type=str),
]

FuelTypeField = Annotated[
    FuelType,
    BeforeValidator(_coerce_fuel_type),
    PlainSerializer(lambda v: v.name, return_type=str),
]

DrivetrainField = Annotated[
    Drivetrain,
    BeforeValidator(_coerce_drivetrain),
    PlainSerializer(lambda v: v.name, return_type=str),
]

PredictMethodField = Annotated[
    PredictMethod,
    BeforeValidator(_coerce_predict_method),
    PlainSerializer(lambda v: v.value, return_type=str),
]

#: ``int`` for a single year, ``tuple[int, int]`` for a range (dumps to a JSON list).
YearField = Annotated[
    Union[int, tuple[int, int]],
    BeforeValidator(parse_year),
]
