from __future__ import annotations

from enum import Enum


class PredictMethod(Enum):
    # Predict the rate of energy consumption and then multiply it by the distance.
    RATE = "rate"
    # Predict the total energy consumption for the link (including distance as a feature).
    RAW = "raw"

    @classmethod
    def from_string(cls, string: str) -> PredictMethod:
        if string.lower() == "rate":
            return PredictMethod.RATE
        elif string.lower() == "raw":
            return PredictMethod.RAW
        else:
            raise ValueError("Unknown predict method: {}".format(string))
