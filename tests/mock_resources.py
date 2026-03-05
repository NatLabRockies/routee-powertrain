import uuid

import pandas as pd
import numpy as np

from nrel.routee.powertrain import load_model, FeatureSet, DataColumn
from nrel.routee.powertrain.core.model import Model


def mock_model() -> Model:
    return load_model("2016_TOYOTA_Camry_4cyl_2WD")


def mock_ev_data(n_links: int = 100):
    speeds = np.random.uniform(1, 100, n_links)
    grades = np.random.uniform(-0.2, 0.2, n_links)
    kwh = np.random.uniform(0.2, 0.5, n_links)
    trip_id = str(uuid.uuid4())
    data = [
        {"miles": 1, "speed": s, "grade": g, "kwh": e, "trip_id": trip_id}
        for s, g, e in zip(speeds, grades, kwh)
    ]
    return pd.DataFrame(data)


def mock_ice_data(n_links: int = 100):
    speeds = np.random.uniform(1, 100, n_links)
    grades = np.random.uniform(-0.2, 0.2, n_links)
    gges = np.random.uniform(0.2, 0.5, n_links)
    trip_id = str(uuid.uuid4())
    data = [
        {"miles": 1, "speed": s, "grade": g, "gge": e, "trip_id": trip_id}
        for s, g, e in zip(speeds, grades, gges)
    ]
    return pd.DataFrame(data)


def mock_data_single_feature():
    data = [
        {"miles": 1, "speed": 1, "energy": 1},
        {"miles": 1, "speed": 2, "energy": 1},
        {"miles": 1, "speed": 3, "energy": 1},
        {"miles": 1, "speed": 4, "energy": 1},
        {"miles": 1, "speed": 5, "energy": 1},
        {"miles": 1, "speed": 6, "energy": 1},
        {"miles": 1, "speed": 7, "energy": 1},
        {"miles": 1, "speed": 8, "energy": 1},
        {"miles": 1, "speed": 9, "energy": 1},
        {"miles": 1, "speed": 10, "energy": 1},
        {"miles": 1, "speed": 12, "energy": 1},
    ]

    feature_set = FeatureSet(
        features=[DataColumn(name="speed", units="")],
    )
    return pd.DataFrame(data), feature_set
