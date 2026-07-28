import pandas as pd
import numpy as np
import glob
import re
import multiprocessing
import random

from pathlib import Path

from joblib import Parallel, delayed

import shapely

import routee.powertrain as pt

from routee.powertrain.trainers.sklearn_random_forest import (
    SklearnRandomForestTrainer,
)

MODEL_BASE_PATH = "/projects/mbap/data/fastsim-results/wm1-fastsim-results/"
ALL_FILES = glob.glob(f"{MODEL_BASE_PATH}**/**/*.parquet")

OUTPUT_BASE_PATH = Path("/projects/mbap/data/routee-results/model-catalog/")
if not OUTPUT_BASE_PATH.exists():
    OUTPUT_BASE_PATH.mkdir()

TRAIN_CACHE_DIR = OUTPUT_BASE_PATH / "train_data"


LOW_MILES_THRESHOLD = 50 / 5280  # 50 feet
HIGH_MILES_THRESHOLD = 0.5
HIGH_GGE_PER_MILE_THRESHOLD = 0.5  # 2 mpg
HIGH_KWH_PER_MILE_THRESHOLD = 5
LOW_KWH_PER_MILE_THRESHOLD = -5
KWH_TO_GGE = 0.029931063
HIGH_GRADE_THRESHOLD = 0.4
LOW_GRADE_THRESHOLD = -0.4

FILE_LIMIT = 2
N_CORES = 80

models = [
    "2012_Ford_Focus",
    "2012_Ford_Fusion",
    "2016_AUDI_A3_4cyl_2WD",
    "2016_BMW_328d_4cyl_2WD",
    # "2016_BMW_i3_REx_PHEV_Charge_Depleting",
    # "2016_BMW_i3_REx_PHEV_Charge_Sustaining",
    "2016_CHEVROLET_Malibu_4cyl_2WD",
    "2016_CHEVROLET_Spark_EV",
    # "2016_CHEVROLET_Volt_Charge_Depleting",
    # "2016_CHEVROLET_Volt_Charge_Sustaining",
    "2016_FORD_C-MAX_HEV",
    # "2016_FORD_C-MAX_(PHEV)_Charge_Depleting",
    # "2016_FORD_C-MAX_(PHEV)_Charge_Sustaining",
    "2016_FORD_Escape_4cyl_2WD",
    "2016_FORD_Explorer_4cyl_2WD",
    "2016_HYUNDAI_Elantra_4cyl_2WD",
    # "2016_HYUNDAI_Sonata_PHEV_Charge_Depleting",
    # "2016_HYUNDAI_Sonata_PHEV_Charge_Sustaining",
    "2016_Hyundai_Tucson_Fuel_Cell",
    "2016_KIA_Optima_Hybrid",
    "2016_Leaf_24_kWh",
    "2016_MITSUBISHI_i-MiEV",
    "2016_Nissan_Leaf_30_kWh",
    "2016_TESLA_Model_S60_2WD",
    "2016_TOYOTA_Camry_4cyl_2WD",
    "2016_TOYOTA_Corolla_4cyl_2WD",
    "2016_TOYOTA_Highlander_Hybrid",
    "2016_Toyota_Prius_Two_FWD",
    "2017_CHEVROLET_Bolt",
    "2017_Maruti_Dzire_VDI",
    # "2017_Prius_Prime_Charge_Depleting",
    # "2017_Prius_Prime_Charge_Sustaining",
    "2017_Toyota_Highlander_3.5_L",
    "2020_Chevrolet_Colorado_2WD_Diesel",
    "2020_VW_Golf_1.5TSI",
    "2020_VW_Golf_2.0TDI",
    "2021_Fiat_Panda_Mild_Hybrid",
    "2021_Peugot_3008",
    "2022_Ford_F-150_Lightning_4WD",
    "2022_Renault_Zoe_ZE50_R135",
    "2022_Tesla_Model_3_RWD",
    "2022_Tesla_Model_Y_RWD",
    "2022_Toyota_Yaris_Hybrid_Mid",
    "2022_Volvo_XC40_Recharge_twin",
    "2023_Mitsubishi_Pajero_Sport",
    # "Maruti_Swift_4cyl_2WD",
    # "Nissan_Navara",
    # "Renault_Clio_IV_diesel",
    # "Renault_Megane_1.5_dCi_Authentique",
    # "Toyota_Hilux_Double_Cab_4WD",
    # "Toyota_Mirai",
]


def load_geom_df():
    geom_df = pd.read_table(
        "/projects/mbap/data/routee-compass-tomtom/data/tomtom_multinet_r_2021/edges-geometries-enumerated.txt.gz",
        header=None,
    ).rename(columns={0: "geometry"})
    id_df = pd.read_csv(
        "/projects/mbap/data/routee-compass-tomtom/data/tomtom_multinet_r_2021/edges-uuid-enumerated.txt.gz",
        header=None,
    ).rename(columns={0: "link_id"})
    geom_df.index = id_df.link_id

    # a link id can be represent twice if it's bi-directional but we don't have any directionality information in the training data so let's just drop the duplicates
    geom_df = geom_df[~geom_df.index.duplicated()]
    return geom_df


GEOM_DF = load_geom_df()


def apply_parallel(df_grouped, func):
    result = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(func)(group) for name, group in df_grouped
    )
    return pd.concat(result)


def get_link_id(road_id_str):
    if "RoadId" in road_id_str:
        pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        matches = re.findall(pattern, road_id_str)

        return matches[2]
    else:
        return road_id_str


def match_link_geoms(link1, link2):
    """
    compares link start and end points to determine the directionality between the two links
    """
    if not link1 or not link2:
        return None, None
    elif link1 == -1:
        return None, None
    elif link2 == -1:
        return None, None
    coords1 = list(link1.coords)
    coords2 = list(link2.coords)

    if coords1[-1] == coords2[0]:
        # link1: TF, link2: TF
        return (coords1[-2], coords1[-1]), (coords2[0], coords2[1])
    elif coords1[0] == coords2[0]:
        # link1: FT, link2: TF
        return (coords1[1], coords1[0]), (coords2[0], coords2[1])
    elif coords1[0] == coords2[-1]:
        # link1: FT, link2: FT
        return (coords1[1], coords1[0]), (coords2[-1], coords2[-2])
    elif coords1[-1] == coords2[-1]:
        # link1: FT, link2: FT
        return (coords1[-2], coords2[-1]), (coords2[-1], coords2[-2])
    else:
        # none of the coords match; might be a result of a map match error;
        # we'll return None
        return None, None


def compute_angle(line1, line2) -> float:
    """
    helper function to compute the angle between two lines.
    """
    if not line1:
        return 0.0
    if not line2:
        return 0.0

    def _azimuth(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        angle = np.arctan2(x2 - x1, y2 - y1)
        return np.degrees(angle)

    a1 = _azimuth(line1[0], line1[1])
    a2 = _azimuth(line2[1], line2[0])

    return 180 - abs((a2 - a1))


def add_entry_link_angles(group):
    angles = [0]  # can't determine first link entry angle so we set to 0
    for i in range(1, len(group)):
        link1 = group.iloc[i - 1]
        link2 = group.iloc[i]
        line1, line2 = match_link_geoms(link1, link2)
        angle = compute_angle(line1, line2)
        angles.append(angle)
    s = pd.Series(angles, index=group.index)
    return s


def add_exit_link_angles(group):
    angles = []
    for i in range(0, len(group) - 1):
        link1 = group.iloc[i]
        link2 = group.iloc[i + 1]
        line1, line2 = match_link_geoms(link1, link2)
        angle = compute_angle(line1, line2)
        angles.append(angle)
    angles.append(0)  # can't determine last link exit angle so we set to 0
    s = pd.Series(angles, index=group.index)
    return s


def add_features(t):
    t = t.sort_values(by="start_time")
    geom = t.geometry.apply(lambda g: shapely.from_wkt(g))
    t["entry_angle"] = add_entry_link_angles(geom)
    t["exit_angle"] = add_exit_link_angles(geom)
    t["previous_grade_dec"] = t.grade_dec.shift(1)
    t["previous_speed_mph"] = t.speed_mph.shift(1)

    return t


def load_df(file, trip_start_id):
    df = pd.read_parquet(
        file,
        columns=[
            "start_time",
            "journey_id",
            "road_id",
            "speed_mph",
            "grade_dec",
            "fs_kwh_out_ach",
            "ess_kwh_out_ach",
            "miles",
        ],
    )

    df = df[(df.miles > LOW_MILES_THRESHOLD) & (df.miles < HIGH_MILES_THRESHOLD)]
    df = df[
        (df.grade_dec > LOW_GRADE_THRESHOLD) & (df.grade_dec < HIGH_GRADE_THRESHOLD)
    ]
    df["gge"] = df.fs_kwh_out_ach * KWH_TO_GGE
    df["kwh"] = df.ess_kwh_out_ach

    df["gal_per_miles"] = df.gge / df.miles
    df = df[df.gal_per_miles < HIGH_GGE_PER_MILE_THRESHOLD]

    df["kwh_per_mile"] = df.ess_kwh_out_ach / df.miles
    df = df[
        (df.kwh_per_mile > LOW_KWH_PER_MILE_THRESHOLD)
        & (df.kwh_per_mile < HIGH_KWH_PER_MILE_THRESHOLD)
    ]

    df["road_id"] = df.road_id.apply(get_link_id)
    df = df.set_index("road_id").join(GEOM_DF, how="left")

    df = apply_parallel(df.groupby("journey_id"), add_features)
    df = df.dropna()

    df["trip_id"] = pd.factorize(df.journey_id)[0]
    df["trip_id"] = df.trip_id + (trip_start_id + 1)

    df = df.drop(
        columns=[
            "gal_per_miles",
            "fs_kwh_out_ach",
            "ess_kwh_out_ach",
            "journey_id",
            "kwh_per_mile",
            "geometry",
            "start_time",
        ]
    )

    df = df.astype(
        {
            "speed_mph": "float32",
            "previous_speed_mph": "float32",
            "grade_dec": "float32",
            "previous_grade_dec": "float32",
            "miles": "float32",
            "gge": "float32",
            "kwh": "float32",
            "entry_angle": "float32",
            "exit_angle": "float32",
        }
    )

    max_trip_id = df.trip_id.iloc[-1]

    return df.reset_index(drop=True), max_trip_id


def load_all_files(files, file_limit=FILE_LIMIT):
    files_to_load = random.choices(files, k=file_limit)
    dfs = []
    trip_start_id = 0
    for file in files_to_load:
        df, trip_start_id = load_df(file, trip_start_id)
        dfs.append(df)
    df = pd.concat(dfs, copy=False)
    return df


feature_set_1 = [pt.DataColumn(name="speed_mph", units="mph")]
feature_set_2 = [
    pt.DataColumn(name="speed_mph", units="mph"),
    pt.DataColumn(name="grade_dec", units="decimal"),
]
feature_set_3 = [
    pt.DataColumn(name="previous_speed_mph", units="mph"),
    pt.DataColumn(name="speed_mph", units="mph"),
    pt.DataColumn(name="previous_grade_dec", units="decimal"),
    pt.DataColumn(name="grade_dec", units="decimal"),
    pt.DataColumn(name="entry_angle", units="degrees"),
]

# Default feature set used for training (most comprehensive)
default_features = [
    pt.DataColumn(name="previous_speed_mph", units="mph"),
    pt.DataColumn(name="speed_mph", units="mph"),
    pt.DataColumn(name="previous_grade_dec", units="decimal"),
    pt.DataColumn(name="grade_dec", units="decimal"),
    pt.DataColumn(name="entry_angle", units="degrees"),
    pt.DataColumn(name="exit_angle", units="degrees"),
]

distance = pt.DataColumn(name="miles", units="miles")


def train_model(model_name):
    model_files = list(filter(lambda f: model_name in f, ALL_FILES))

    train_file = TRAIN_CACHE_DIR / Path(f"{model_name}.csv")
    if train_file.exists():
        df = pd.read_csv(train_file)
    else:
        df = load_all_files(model_files)
        df.to_csv(train_file)

    print(f"loaded {len(df)} samples")

    if df.gge.sum() > 0 and df.kwh.sum() < 0.001:
        if "HEV" in model_name or "Hybrid" in model_name:
            powertrain_type = pt.PowertrainType.HEV
        else:
            powertrain_type = pt.PowertrainType.ICE
        energy_target = pt.DataColumn(
            name="gge",
            units="gallons_gasoline",
        )
    elif df.gge.sum() < 0.001 and df.kwh.sum() > 0:
        powertrain_type = pt.PowertrainType.BEV
        energy_target = pt.DataColumn(
            name="kwh",
            units="kwh",
        )
    else:
        raise ValueError(f"Dual fuel powertrains not yet supported for {model_name}")

    # parse structured fields from model_name like "2016_TOYOTA_Camry_4cyl_2WD"
    parts = model_name.split("_")
    year_str = parts[0] if parts else "0"
    make_str = parts[1] if len(parts) > 1 else "unknown"
    name_str = parts[2] if len(parts) > 2 else "unknown"
    trim_str = "_".join(parts[3:]) if len(parts) > 3 else "default"

    config = pt.ModelConfig(
        vehicle_description=model_name,
        powertrain_type=powertrain_type,
        feature_set=default_features,
        distance=distance,
        target=energy_target,
        make=make_str,
        model=name_str,
        year=int(year_str),
        trim=trim_str if trim_str else "default",
        test_size=0.2,
    )

    trainer = SklearnRandomForestTrainer(cores=80)

    vehicle = trainer.train(df, config)

    return vehicle


if __name__ == "__main__":
    for m_name in models:
        print(f"working on {m_name}")
        outfile = OUTPUT_BASE_PATH / Path(f"{m_name}.json")
        if outfile.exists():
            print("file already exists, skipping")
            continue
        try:
            vehicle = train_model(m_name)
        except ValueError as e:
            print(f"error for {m_name}, skipping")
            print(e)
            continue

        vehicle.to_file(outfile)
