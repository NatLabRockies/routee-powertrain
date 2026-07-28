from __future__ import annotations

from enum import Enum
from typing import Optional


class Drivetrain(Enum):
    UNDEFINED = 0
    FWD = 1
    RWD = 2
    AWD = 3
    FOURWD = 4

    @classmethod
    def from_string(cls, s: Optional[str]) -> Drivetrain:
        if not s:
            return Drivetrain.UNDEFINED
        e = cls.__members__.get(s.upper())
        if not e:
            raise TypeError(
                f"{s} is not a recognized drivetrain type. "
                f"Try one of these: {list(Drivetrain.__members__.keys())}"
            )
        return e
