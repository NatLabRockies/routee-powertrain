from __future__ import annotations

import shutil
from importlib.util import find_spec
from pathlib import Path
from unittest import TestCase, skipUnless

import numpy as np
import pandas as pd

import routee.powertrain as pt
from routee.powertrain.estimators.onnx import ONNXEstimator

this_dir = Path(__file__).parent

_HAS_TORCH = find_spec("torch") is not None


class TestCNNOptionalDependency(TestCase):
    def test_importing_cnn_trainer_does_not_import_torch(self):
        """The trainer module stays importable without the PyTorch extra."""
        import subprocess
        import sys

        code = (
            "import sys;"
            "from routee.powertrain.trainers.cnn import CNNTrainer;"
            "assert 'torch' not in sys.modules, 'torch was imported eagerly';"
            "assert CNNTrainer.__name__ == 'CNNTrainer'"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)


@skipUnless(_HAS_TORCH, "torch is required for CNN training")
class TestCNNPipeline(TestCase):
    def setUp(self) -> None:
        data_path = (
            this_dir
            / Path("routee-powertrain-test-data")
            / Path("sample_train_data.csv")
        )
        df = pd.read_csv(data_path)
        # The sample data is a single trip; synthesize five pseudo-trips so
        # the group-aware train/test split in ``Trainer.train`` produces a
        # non-empty test set for the CNN pipeline round-trip.
        n = len(df)
        df["trip_id"] = (np.arange(n) // max(1, n // 5)).astype(np.int64)
        df["route_id"] = df["trip_id"]
        self.df = df

        self.out_path = Path("tmp")
        self.out_path.mkdir(exist_ok=True)

        self.config = pt.ModelConfig(
            vehicle_description="CNN test vehicle",
            powertrain_type=pt.PowertrainType.ICE,
            feature_set=pt.FeatureSet(
                features=[
                    pt.DataColumn(name="speed_mph", units="mph"),
                    pt.DataColumn(name="grade_dec", units="decimal"),
                ],
            ),
            distance=pt.DataColumn(name="miles", units="miles"),
            target=pt.TargetSet(
                targets=[
                    pt.DataColumn(
                        name="gallons_fastsim",
                        units="gallons_gasoline",
                        constraints=pt.Constraints(lower=0.0, upper=100.0),
                    )
                ],
            ),
            make="test",
            model="model",
            year=2024,
        )

    def tearDown(self) -> None:
        if self.out_path.exists():
            shutil.rmtree(self.out_path)

    def test_train_predict_roundtrip(self):
        from routee.powertrain.trainers.cnn import CNNTrainer

        trainer = CNNTrainer(
            lookback=5,
            grouping_column="route_id",
            epochs=2,
            batch_size=256,
            hidden_channels=8,
        )
        model = trainer.train(self.df, self.config)

        # sanity: the estimator declares its lookback/grouping
        spec = model.estimator.input_spec
        self.assertEqual(spec.lookback, 5)
        self.assertEqual(spec.grouping_column, "route_id")

        # the windowed model also carries the positional input/output contract
        self.assertIsNotNone(spec.input_columns)
        self.assertEqual(
            [c.name for c in spec.input_columns],
            model.metadata.config.all_feature_names,
        )
        self.assertEqual(spec.predict_method, "rate")

        # Metadata carries architecture_tag and input_spec
        self.assertEqual(model.metadata.estimator.architecture_tag, "cnn")
        self.assertEqual(model.metadata.estimator.input_spec["lookback"], 5)

        r1 = model.predict(self.df)
        self.assertEqual(len(r1), len(self.df))
        self.assertIn("gallons_fastsim", r1.columns)
        self.assertTrue(np.isfinite(r1["gallons_fastsim"]).all())

        # round-trip via flat directory
        outdir = self.out_path / "cnn_model"
        model.to_file(outdir)
        loaded = pt.load_model(outdir)

        r2 = loaded.predict(self.df)
        # lookback hparams should have been preserved
        self.assertEqual(loaded.estimator.input_spec.lookback, 5)
        self.assertEqual(loaded.estimator.input_spec.grouping_column, "route_id")

        # Predictions should be byte-identical since the ONNX model is the same
        np.testing.assert_array_almost_equal(
            r1["gallons_fastsim"].to_numpy(),
            r2["gallons_fastsim"].to_numpy(),
            decimal=5,
        )

    def test_predict_rejects_missing_grouping_column(self):
        from routee.powertrain.trainers.cnn import CNNTrainer

        trainer = CNNTrainer(
            lookback=3,
            grouping_column="route_id",
            epochs=1,
            batch_size=256,
            hidden_channels=4,
        )
        model = trainer.train(self.df, self.config)

        df_no_group = self.df.drop(columns=["route_id"])
        with self.assertRaises(ValueError) as ctx:
            model.predict(df_no_group)
        self.assertIn("route_id", str(ctx.exception))

    def test_estimator_bytes_roundtrip_preserves_hparams(self):
        from routee.powertrain.trainers.cnn import CNNTrainer

        trainer = CNNTrainer(
            lookback=4,
            grouping_column="route_id",
            pad_strategy="repeat_first",
            epochs=1,
            batch_size=256,
            hidden_channels=4,
        )
        model = trainer.train(self.df, self.config)

        restored = ONNXEstimator.from_bytes(model.estimator.to_bytes())
        self.assertEqual(restored.input_spec.lookback, 4)
        self.assertEqual(restored.input_spec.grouping_column, "route_id")
        self.assertEqual(restored.input_spec.pad_strategy, "repeat_first")
