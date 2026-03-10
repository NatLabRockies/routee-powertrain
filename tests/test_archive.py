import logging as log
import math
from pathlib import Path
from unittest import TestCase

import pandas as pd

import routee.powertrain as pt
from routee.powertrain.estimators.onnx import ONNXEstimator
from routee.powertrain.io.archive import load_archive, save_archive

from routee.powertrain.trainers.sklearn_random_forest import (
    SklearnRandomForestTrainer,
)
from routee.powertrain.trainers.ngboost_trainer import (
    NGBoostTrainer,
)

this_dir = Path(__file__).parent
log.basicConfig(level=log.INFO)


class TestArchiveRoundTrip(TestCase):
    """Test the new .zip archive serialization format."""

    def setUp(self) -> None:
        data_path = (
            this_dir
            / Path("routee-powertrain-test-data")
            / Path("sample_train_data.csv")
        )
        self.df = pd.read_csv(data_path)
        self.out_path = Path("tmp")
        self.out_path.mkdir(exist_ok=True)
        feature_set = pt.FeatureSet(
            features=[
                pt.DataColumn(name="speed_mph", units="mph"),
                pt.DataColumn(name="grade_dec", units="decimal"),
            ],
        )
        distance = pt.DataColumn(name="miles", units="miles")
        targets = pt.TargetSet(
            targets=[
                pt.DataColumn(
                    name="gallons_fastsim",
                    units="gallons_gasoline",
                    constraints=pt.Constraints(lower=0.0, upper=100.0),
                )
            ],
        )
        self.rate_config = pt.ModelConfig(
            vehicle_description="Test Model",
            powertrain_type=pt.PowertrainType.ICE,
            feature_set=feature_set,
            distance=distance,
            target=targets,
            make="test",
            model_name="model",
            year=2024,
            trim="base",
        )

    def tearDown(self) -> None:
        # cleanup tmp files
        for f in self.out_path.iterdir():
            f.unlink()
        self.out_path.rmdir()

    def test_sklearn_zip_roundtrip(self):
        """Train an ONNX model, save as .zip, reload, and verify predictions match."""
        trainer = SklearnRandomForestTrainer()
        model = trainer.train(self.df, self.rate_config)

        r1 = model.predict(self.df)
        energy1 = round(r1.gallons_fastsim.sum(), 2)

        outfile = self.out_path / "model.zip"
        model.to_file(outfile)
        self.assertTrue(outfile.exists())

        loaded = pt.load_model(outfile)
        r2 = loaded.predict(self.df)
        energy2 = round(r2.gallons_fastsim.sum(), 2)

        self.assertTrue(math.isclose(energy1, energy2))

    def test_ngboost_zip_roundtrip(self):
        """Train an NGBoost model, save as .zip, reload, and verify predictions match."""
        trainer = NGBoostTrainer()
        model = trainer.train(self.df, self.rate_config)

        r1 = model.predict(self.df)
        energy1 = round(r1.gallons_fastsim.sum(), 2)

        outfile = self.out_path / "model_ngboost.zip"
        model.to_file(outfile)
        self.assertTrue(outfile.exists())

        loaded = pt.load_model(outfile)
        r2 = loaded.predict(self.df)
        energy2 = round(r2.gallons_fastsim.sum(), 2)

        self.assertTrue(math.isclose(energy1, energy2))

    def test_save_archive_load_archive(self):
        """Test the low-level save_archive / load_archive functions directly."""
        trainer = SklearnRandomForestTrainer()
        model = trainer.train(self.df, self.rate_config)

        outfile = self.out_path / "direct_archive.zip"
        save_archive(model, outfile)

        loaded = load_archive(outfile)
        self.assertEqual(loaded.metadata.config.vehicle_description, "Test Model")
        self.assertIsInstance(loaded.estimator, ONNXEstimator)

        r1 = model.predict(self.df)
        r2 = loaded.predict(self.df)
        self.assertTrue(
            math.isclose(r1.gallons_fastsim.sum(), r2.gallons_fastsim.sum())
        )

    def test_json_still_works(self):
        """Legacy .json format should still work for both read and write."""
        trainer = SklearnRandomForestTrainer()
        model = trainer.train(self.df, self.rate_config)

        outfile = self.out_path / "legacy.json"
        model.to_file(outfile)
        self.assertTrue(outfile.exists())

        loaded = pt.load_model(outfile)
        r1 = model.predict(self.df)
        r2 = loaded.predict(self.df)
        self.assertTrue(
            math.isclose(r1.gallons_fastsim.sum(), r2.gallons_fastsim.sum())
        )

    def test_structured_vehicle_fields(self):
        """ModelConfig with structured vehicle fields should round-trip through archive."""
        config = pt.ModelConfig(
            vehicle_description="2016 Toyota Camry 4cyl FWD",
            powertrain_type=pt.PowertrainType.ICE,
            feature_set=self.rate_config.feature_set,
            distance=self.rate_config.distance,
            target=self.rate_config.target,
            make="Toyota",
            model_name="Camry",
            year=2016,
            trim="4cyl_FWD",
        )
        trainer = SklearnRandomForestTrainer()
        model = trainer.train(self.df, config)

        outfile = self.out_path / "structured.zip"
        model.to_file(outfile)

        loaded = pt.load_model(outfile)
        self.assertEqual(loaded.metadata.config.make, "toyota")
        self.assertEqual(loaded.metadata.config.model_name, "camry")
        self.assertEqual(loaded.metadata.config.year, 2016)
        self.assertEqual(loaded.metadata.config.trim, "4cyl_fwd")
