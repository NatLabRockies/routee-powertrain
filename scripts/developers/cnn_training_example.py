"""Train a CNN energy model end-to-end — feature engineering, device
selection, batching, save/load, and inference.

What the CNN trainer adds over ``SklearnRandomForestTrainer``:

* It is **sequence-aware**. For each row it builds a lookback window of
  the previous ``lookback - 1`` rows in the same route and feeds the
  whole stack to a 1D CNN. This means:
  - You must supply a ``grouping_column`` (default ``"route_id"``) so
    windows do not stitch together rows from different routes.
  - Rows within each route must already be **sorted by time** before
    the trainer sees them — the trainer trusts the row order it gets.
  - The train/test split is forced to be group-aware (every link of a
    route lands entirely in train or entirely in test). This is
    handled automatically by ``Trainer.train``.
* It exports to ONNX with an ``InputSpec`` describing the window, so
  the runtime stays torch-free. The same ``grouping_column`` must be
  present in the dataframe you pass to ``Model.predict``.

"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkb

import routee.powertrain as pt
from routee.powertrain.core.model_config import PredictMethod
from routee.powertrain.trainers.cnn import CNNTrainer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column conventions
# ---------------------------------------------------------------------------
# These names are what the trainer sees in the dataframe. The CNN cares about
# three classes of columns:
#   1. Feature columns (declared on the ModelConfig.feature_set)
#   2. Distance column + energy target column (also on ModelConfig)
#   3. The grouping column — required by CNNTrainer so per-route lookback
#      windows do not bleed across routes. It must also be present at
#      predict time (see Model.predict in routee/powertrain/core/model.py).
TARGET_COL = "energy_gge"
DISTANCE_COL = "miles"
TRIP_COL = "journey_id"
GROUPING_COL = "route_id"  # what CNNTrainer windows over; alias of TRIP_COL


# ---------------------------------------------------------------------------
# 1. Feature engineering for a sequence model
# ---------------------------------------------------------------------------
# A tree model can ignore order: each row is judged on its own values. A CNN
# can use the previous N-1 rows, so features that *describe the trajectory*
# pay off here even when they would be redundant for an RF. Two cheap wins:
#   - link_sinuosity:        how curvy the link is (road length / chord)
#   - link_abs_bearing_delta: how sharp the turn was vs. the previous link
# Both are computed from the link's WKB geometry. Bearing delta is a
# per-route diff, which is why we sort by (trip, time) before differencing.


def _calc_geom_features(geom_hex: str) -> tuple[float, float]:
    g = wkb.loads(geom_hex, hex=True)
    coords = list(g.coords)
    start, end = coords[0], coords[-1]
    dx, dy = end[0] - start[0], end[1] - start[1]
    straight = math.sqrt(dx * dx + dy * dy)
    if straight < 1e-10:
        sinuosity = 1.0
    else:
        road_len = sum(
            math.sqrt(
                (coords[i + 1][0] - coords[i][0]) ** 2
                + (coords[i + 1][1] - coords[i][1]) ** 2
            )
            for i in range(len(coords) - 1)
        )
        sinuosity = road_len / straight
    bearing = math.atan2(dx, dy) * 180.0 / math.pi
    return sinuosity, bearing


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the sequence features the CNN consumes.

    The trainer trusts the row order it gets, so this function:
      1. Sorts by (trip, link_start_time) so each route is contiguous and
         chronological. Without this, the lookback window for a given
         row is meaningless.
      2. Computes per-link geometry features (sinuosity, bearing).
      3. Differences bearing within each route to get a turn-magnitude
         feature. The first link of each route has no prior bearing — we
         fill it with 0.0 so the CNN's pad-zero strategy keeps working.
      4. Aliases ``journey_id`` to ``route_id`` so it matches the
         trainer's default ``grouping_column`` (you can also pass a
         different ``grouping_column`` to ``CNNTrainer``, but matching
         the default is the path of least surprise).
      5. Drops rows with nulls in any column we'll feed to the trainer.
    """
    df = df.sort_values([TRIP_COL, "link_start_time"]).reset_index(drop=True)

    geom = df["geometry"].apply(_calc_geom_features)
    df["link_sinuosity"] = geom.apply(lambda t: t[0]).astype(np.float32)
    df["_bearing"] = geom.apply(lambda t: t[1]).astype(np.float32)

    prev_bearing = df.groupby(TRIP_COL)["_bearing"].shift(1)
    raw_delta = df["_bearing"] - prev_bearing
    bearing_delta = ((raw_delta + 180.0) % 360.0) - 180.0
    df["link_abs_bearing_delta"] = bearing_delta.abs().fillna(0.0).astype(np.float32)
    df = df.drop(columns=["_bearing"])

    df["link_time"] = df["time_seconds"].astype(np.float32)
    df[GROUPING_COL] = df[TRIP_COL]

    needed = [
        "speed_mph",
        "grade_percent",
        "link_time",
        "link_sinuosity",
        "link_abs_bearing_delta",
        DISTANCE_COL,
        TARGET_COL,
        TRIP_COL,
        GROUPING_COL,
    ]
    df = df.dropna(subset=needed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. ModelConfig — same shape as any other trainer
# ---------------------------------------------------------------------------
#   - the grouping column is declared on the *trainer* (not the config),
#     because it is a trainer concern (windowing), not a model concern.
#     At inference time the runtime reads it from the estimator's
#     InputSpec (set when CNNTrainer exports the ONNX file).


def cnn_config() -> pt.ModelConfig:
    return pt.ModelConfig(
        vehicle_description="Example CNN(5-link lookback, 6 features)",
        powertrain_type=pt.PowertrainType.BEV,
        feature_set=pt.FeatureSet(
            features=[
                pt.DataColumn(
                    name="speed_mph",
                    units="mph",
                    constraints=pt.Constraints(lower=0.0, upper=100.0),
                ),
                pt.DataColumn(
                    name="grade_percent",
                    units="percent",
                    constraints=pt.Constraints(lower=-20.0, upper=20.0),
                ),
                pt.DataColumn(name=DISTANCE_COL, units="miles"),
                pt.DataColumn(name="link_time", units="seconds"),
                pt.DataColumn(name="link_sinuosity", units="ratio"),
                pt.DataColumn(name="link_abs_bearing_delta", units="degrees"),
            ],
        ),
        distance=pt.DataColumn(name=DISTANCE_COL, units="miles"),
        target=pt.TargetSet(
            targets=[pt.DataColumn(name=TARGET_COL, units="gge")],
        ),
        make="example",
        model="bev",
        year=2024,
        predict_method=PredictMethod.RATE,
        trip_column=TRIP_COL,
        apply_real_world_adjustment=True,
    )


# ---------------------------------------------------------------------------
# 3. Device selection — GPU when present, CPU fallback for tests
# ---------------------------------------------------------------------------
# CNNTrainer's own ``_pick_device`` already does CUDA → MPS → CPU auto-detect
# when you pass ``device=None`` (the default). You only need to override it
# when:
#   - you want to force CPU for a smoke test or CI run (set device="cpu")
#   - you have multiple GPUs and want a specific one (e.g. device="cuda:1")
#
# Two safety nets are already built in:
#   1. CNNTrainer checks the trained parameters for non-finite values and
#      retrains on CPU if the GPU run produced NaNs/Infs.
#   2. ONNX export always happens on CPU and a numerical-parity check runs
#      against onnxruntime before the estimator is returned.
#


def pick_device(prefer: str | None = None) -> str:
    """Return the device string CNNTrainer should use.

    Used here mostly so callers can log which device they got and so the
    ``--smoke-test`` flag has a clean way to force CPU. In real pipelines
    you'd usually pass ``prefer=None`` and let the trainer auto-pick.
    """
    if prefer is not None:
        return prefer
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# 4. Batch sizing
# ---------------------------------------------------------------------------
# CNNTrainer stages the **entire windowed training tensor** on the target
# device once and indexes into it per batch (see ``_train_on`` in
# routee/powertrain/trainers/cnn.py). This is intentional — small CNNs
# bottleneck on PCIe transfer when you DataLoader-feed them, and the
# windowed tensor is small enough to fit on a single GPU for realistic
# routee.powertrain datasets.
#
# Memory footprint of the windowed tensor:
#     bytes = n_rows * lookback * n_features * 4   (float32)
# e.g. 5M rows * 5 lookback * 6 features * 4 = ~600 MB. A 12 GB GPU has
# plenty of headroom even at 50M rows.
#
# Batch size then trades wall-clock vs. gradient quality:
#   - GPU/MPS:  2048–8192 is a good band; throughput-bound, so push up
#               until you stop seeing speed gains. Larger batches also
#               make the cosine LR schedule (CosineAnnealingLR over total
#               steps) more stable.
#   - CPU:      256–512 — Adam's per-step cost dominates and giant
#               batches just waste cycles. Useful for smoke tests.
#
# Epoch count: the autoresearch sweep landed around 200 epochs for the
# Bolt dataset; tune this on a held-out journey set, not on metadata
# errors (the trainer's internal split is small).


def cnn_trainer(
    device: str,
    lookback: int = 5,
    epochs: int = 200,
    batch_size: int = 2048,
) -> CNNTrainer:
    return CNNTrainer(
        lookback=lookback,
        grouping_column=GROUPING_COL,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        # The remaining knobs default to the autoresearch-tuned values:
        #   hidden_channels=64, kernel_size=3, n_conv_layers=auto,
        #   learning_rate=1e-3, min_learning_rate=1e-5,
        #   weight_decay=1e-4, grad_clip_norm=1.0,
        #   normalize_features=True, pad_strategy="zero".
    )


# ---------------------------------------------------------------------------
# 5. Train, save, load, predict
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to a parquet file with one row per link and the columns "
        f"speed_mph, grade_percent, time_seconds, geometry, {DISTANCE_COL}, "
        f"{TARGET_COL}, {TRIP_COL}, link_start_time.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("cnn_training_output"),
        help="Directory to write the trained model into.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Force CPU, 2 epochs, batch_size=256. Useful for CI or for "
        "validating wiring without waiting on a full training run.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (cuda, cuda:0, mps, cpu). Default: auto-pick.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---------- load + feature-engineer
    log.info("loading %s", args.data_path)
    df = pd.read_parquet(args.data_path)
    df = build_features(df)
    log.info(
        "after feature build: rows=%d  routes=%d",
        len(df),
        df[GROUPING_COL].nunique(),
    )

    # ---------- pick device + batch profile
    if args.smoke_test:
        device = "cpu"
        epochs = 2
        batch_size = 256
        log.info("smoke-test mode: device=cpu epochs=2 batch_size=256")
    else:
        device = pick_device(args.device)
        # Conservative defaults that work on a single mid-range GPU.
        epochs = 200
        batch_size = 2048 if device != "cpu" else 512
        log.info(
            "training mode: device=%s epochs=%d batch_size=%d",
            device,
            epochs,
            batch_size,
        )

    # ---------- train
    config = cnn_config()
    trainer = cnn_trainer(
        device=device, epochs=epochs, batch_size=batch_size, lookback=5
    )
    t0 = time.time()
    model = trainer.train(df.copy(), config)
    log.info("cnn train time: %.1fs", time.time() - t0)

    # ---------- save (directory / .zip / .tar.gz — suffix decides)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "cnn_model"
    model.to_file(model_path)
    log.info("wrote model to %s", model_path)

    # ---------- load round-trip + predict
    # The grouping column (``route_id`` here) MUST be present in the
    # dataframe you hand to Model.predict. The estimator reads its
    # InputSpec from the ONNX metadata and uses the grouping column to
    # rebuild lookback windows on the fly — exactly like training did.
    loaded = pt.load_model(model_path)
    sample = df.head(10_000).copy()
    pred = loaded.predict(sample)
    log.info(
        "predicted on %d links; pred[%s] sum=%.4f  actual sum=%.4f",
        len(sample),
        TARGET_COL,
        float(pred[TARGET_COL].sum()),
        float(sample[TARGET_COL].sum()),
    )

if __name__ == "__main__":
    main()
