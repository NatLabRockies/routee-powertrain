from __future__ import annotations

from enum import Enum
from typing import Optional


class FuelType(Enum):
    UNDEFINED = 0
    GASOLINE = 1
    DIESEL = 2
    ELECTRICITY = 3
    HYDROGEN = 4

    @classmethod
    def from_string(cls, s: Optional[str]) -> FuelType:
        if not s:
            return FuelType.UNDEFINED
        e = cls.__members__.get(s.upper())
        if not e:
            raise TypeError(
                f"{s} is not a recognized fuel type. "
                f"Try one of these: {list(FuelType.__members__.keys())}"
            )
        return e
