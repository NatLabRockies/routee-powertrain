from __future__ import annotations

from enum import Enum
from typing import Optional


class PowertrainType(Enum):
    UNDEFINED = 0
    ICE = 1
    HEV = 2
    BEV = 3
    PHEV_EV_MODE = 4
    PHEV_HEV_MODE = 5
    HEAVY_DUTY = 6

    @classmethod
    def from_string(cls, s: Optional[str]) -> PowertrainType:
        if not s:
            return PowertrainType.UNDEFINED
        e = cls.__members__.get(s.upper())
        if not e:
            raise TypeError(
                f"{s} is not a recognized powertrain type"
                f"Try one of these: {PowertrainType.__members__.keys()}"
            )
        return e
