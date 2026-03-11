import math
from pathlib import Path
from unittest import TestCase

import pandas as pd

import routee.powertrain as pt
from routee.powertrain.io.archive import save_model_directory
from routee.powertrain.registry.local import LocalRegistry
from routee.powertrain.registry.model_id import ModelId, ModelInfo
from routee.powertrain.trainers.sklearn_random_forest import (
    SklearnRandomForestTrainer,
)

this_dir = Path(__file__).parent


class TestModelId(TestCase):
    def test_to_path(self):
        mid = ModelId("toyota", "camry", 2016, "4cyl_fwd", "default", 1)
        path = mid.to_path("v2")
        self.assertEqual(path, "v2/toyota/camry/2016/4cyl_fwd/default/v1")

    def test_lowercase_normalization(self):
        mid = ModelId("Toyota", "Camry", 2016, "4Cyl_FWD", "Default", 1)
        self.assertEqual(mid.make, "toyota")
        self.assertEqual(mid.model_name, "camry")
        self.assertEqual(mid.trim, "4cyl_fwd")
        self.assertEqual(mid.variant, "default")

    def test_roundtrip_dict(self):
        mid = ModelId("toyota", "camry", 2016, "4cyl_fwd", "default", 1)
        d = mid.to_dict()
        mid2 = ModelId.from_dict(d)
        self.assertEqual(mid, mid2)


class TestLocalRegistry(TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp)
        self.schema_version = "v2"

        # Train a model
        data_path = (
            this_dir
            / Path("routee-powertrain-test-data")
            / Path("sample_train_data.csv")
        )
        df = pd.read_csv(data_path)
        config = pt.ModelConfig(
            vehicle_description="2016 Toyota Camry 4cyl FWD",
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
            make="Toyota",
            model_name="Camry",
            year=2016,
            trim="4cyl_fwd",
        )
        trainer = SklearnRandomForestTrainer()
        self.model = trainer.train(df, config)
        self.df = df

        # Save to registry path as a flat directory
        model_id = ModelId("toyota", "camry", 2016, "4cyl_fwd", "default", 1)
        rel_path = model_id.to_path(self.schema_version)
        full_path = self.root / rel_path
        save_model_directory(self.model, full_path)

        self.registry = LocalRegistry(
            root=self.root, schema_version=self.schema_version
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp)

    def test_query(self):
        results = self.registry.query(make="toyota")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.model_name, "camry")

    def test_query_all(self):
        results = self.registry.query()
        self.assertEqual(len(results), 1)

    def test_query_no_match(self):
        results = self.registry.query(make="ford")
        self.assertEqual(len(results), 0)

    def test_load(self):
        model_id = ModelId("toyota", "camry", 2016, "4cyl_fwd", "default", 1)
        loaded = self.registry.load(model_id)

        r1 = self.model.predict(self.df)
        r2 = loaded.predict(self.df)
        self.assertTrue(
            math.isclose(r1.gallons_fastsim.sum(), r2.gallons_fastsim.sum())
        )

    def test_get_metadata(self):
        model_id = ModelId("toyota", "camry", 2016, "4cyl_fwd", "default", 1)
        meta = self.registry.get_metadata(model_id)
        self.assertIn("metadata", meta)
        self.assertIn("estimator_type", meta)

    def test_query_returns_model_info(self):
        results = self.registry.query(make="toyota")
        info = results[0]
        self.assertIsInstance(info, ModelInfo)
        self.assertEqual(info.estimator_type, "ONNXEstimator")
        self.assertIn("speed_mph", info.feature_names)
        self.assertIn("gallons_fastsim", info.target_names)
