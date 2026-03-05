import logging as log
import math
from pathlib import Path
from unittest import TestCase, skip

import pandas as pd

import nrel.routee.powertrain as pt
from nrel.routee.powertrain.core.model_config import PredictMethod
from nrel.routee.powertrain.estimators.onnx import ONNXEstimator
from nrel.routee.powertrain.estimators.smart_core import SmartCoreEstimator
from nrel.routee.powertrain.estimators.ngboost_estimator import NGBoostEstimator

from nrel.routee.powertrain.trainers.sklearn_random_forest import (
    SklearnRandomForestTrainer,
)
from nrel.routee.powertrain.trainers.smartcore_random_forest import (
    SmartcoreRandomForestTrainer,
)
from nrel.routee.powertrain.trainers.ngboost_trainer import (
    NGBoostTrainer,
)

this_dir = Path(__file__).parent

log.basicConfig(level=log.INFO)


class TestTrainEstimatePipeline(TestCase):
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
        )
        self.raw_config = pt.ModelConfig(
            vehicle_description="Test Model",
            powertrain_type=pt.PowertrainType.ICE,
            feature_set=feature_set,
            distance=distance,
            target=targets,
            predict_method=PredictMethod.RAW,
        )

    def tearDown(self) -> None:
        pass
        self.out_path.rmdir()

    def test_sklearn_random_forest_rate(self):
        trainer = SklearnRandomForestTrainer()

        vehicle_model = trainer.train(self.df, self.rate_config)

        r1 = vehicle_model.predict(self.df)
        energy1 = round(r1.gallons_fastsim.sum(), 2)

        # test out writing and reading to file
        outfile = self.out_path / "model_rate.json"
        vehicle_model.to_file(outfile)
        new_vehicle_model = pt.load_model(outfile)
        outfile.unlink()

        # test writing inner estimator to file
        outfile = self.out_path / "estimator_rate.onnx"
        vehicle_model.estimator.to_file(outfile)
        _ = ONNXEstimator.from_file(outfile)
        outfile.unlink()

        r2 = new_vehicle_model.predict(self.df)
        energy2 = round(r2.gallons_fastsim.sum(), 2)

        self.assertTrue(math.isclose(energy1, energy2))

    def test_sklearn_random_forest_raw(self):
        trainer = SklearnRandomForestTrainer()

        vehicle_model = trainer.train(self.df, self.raw_config)

        r1 = vehicle_model.predict(self.df)
        energy1 = round(r1.gallons_fastsim.sum(), 2)

        # test out writing and reading to file
        outfile = self.out_path / "model_raw.json"
        vehicle_model.to_file(outfile)
        new_vehicle_model = pt.load_model(outfile)
        outfile.unlink()

        # test writing inner estimator to file
        outfile = self.out_path / "estimator_raw.onnx"
        vehicle_model.estimator.to_file(outfile)
        _ = ONNXEstimator.from_file(outfile)
        outfile.unlink()

        r2 = new_vehicle_model.predict(self.df)
        energy2 = round(r2.gallons_fastsim.sum(), 2)

        self.assertTrue(math.isclose(energy1, energy2))

    @skip("This requires rust to be installed")
    def test_smartcore_random_forest(self):
        trainer = SmartcoreRandomForestTrainer()

        vehicle_model = trainer.train(self.df, self.rate_config)

        r1 = vehicle_model.predict(self.df)
        energy1 = round(r1.gallons_fastsim.sum(), 2)

        # test out writing and reading to file
        outfile = self.out_path / "model.json"
        vehicle_model.to_file(outfile)
        new_vehicle_model = pt.load_model(outfile)
        outfile.unlink()

        # test writing inner estimator to file
        outfile = self.out_path / "estimator.json"
        vehicle_model.estimator.to_file(outfile)
        _ = SmartCoreEstimator.from_file(outfile)
        outfile.unlink()

        outfile = self.out_path / "estimator.bin"
        vehicle_model.estimator.to_file(outfile)
        _ = SmartCoreEstimator.from_file(outfile)
        outfile.unlink()

        r2 = new_vehicle_model.predict(self.df)
        energy2 = round(r2.gallons_fastsim.sum(), 2)

        self.assertTrue(math.isclose(energy1, energy2))

    def test_ngboost_rate(self):
        trainer = NGBoostTrainer()

        vehicle_model = trainer.train(self.df, self.rate_config)

        r1 = vehicle_model.predict(self.df)
        energy1 = round(r1.gallons_fastsim.sum(), 2)

        # test out writing and reading to file
        outfile = self.out_path / "model_rate.json"
        vehicle_model.to_file(outfile)
        new_vehicle_model = pt.load_model(outfile)
        outfile.unlink()

        # test writing inner estimator to file
        outfile = self.out_path / "estimator_rate.json"
        vehicle_model.estimator.to_file(outfile)
        _ = NGBoostEstimator.from_file(outfile)
        outfile.unlink()

        r2 = new_vehicle_model.predict(self.df)
        energy2 = round(r2.gallons_fastsim.sum(), 2)

        self.assertTrue(math.isclose(energy1, energy2))

    def test_ngboost_raw(self):
        trainer = NGBoostTrainer()

        vehicle_model = trainer.train(self.df, self.raw_config)

        r1 = vehicle_model.predict(self.df)
        energy1 = round(r1.gallons_fastsim.sum(), 2)

        # test out writing and reading to file
        outfile = self.out_path / "model_raw.json"
        vehicle_model.to_file(outfile)
        new_vehicle_model = pt.load_model(outfile)
        outfile.unlink()

        # test writing inner estimator to file
        outfile = self.out_path / "estimator_raw.json"
        vehicle_model.estimator.to_file(outfile)
        _ = NGBoostEstimator.from_file(outfile)
        outfile.unlink()

        r2 = new_vehicle_model.predict(self.df)
        energy2 = round(r2.gallons_fastsim.sum(), 2)

        self.assertTrue(math.isclose(energy1, energy2))
