from __future__ import annotations

import math
import shutil
from pathlib import Path
from unittest import TestCase

import pandas as pd

import routee.powertrain as pt
from routee.powertrain.registry.local import LocalRegistry
from routee.powertrain.registry.slug import derive_config_slug
from routee.powertrain.trainers.sklearn_random_forest import (
    SklearnRandomForestTrainer,
)

this_dir = Path(__file__).parent


class TestSaveToRegistry(TestCase):
    def setUp(self) -> None:
        data_path = this_dir / "routee-powertrain-test-data" / "sample_train_data.csv"
        df = pd.read_csv(data_path)
        feature_set = pt.FeatureSet(
            features=[
                pt.DataColumn(name="speed_mph", units="mph"),
                pt.DataColumn(name="grade_dec", units="decimal"),
            ],
        )
        config = pt.ModelConfig(
            vehicle_description="Test Model",
            powertrain_type=pt.PowertrainType.ICE,
            feature_set=feature_set,
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
            make="Test",
            model="Sedan",
            year=2024,
        )
        self.df = df
        self.model = SklearnRandomForestTrainer().train(df, config)
        self.slug = derive_config_slug(self.model.metadata)
        self.registry_root = Path("tmp_registry")

    def tearDown(self) -> None:
        if self.registry_root.exists():
            shutil.rmtree(self.registry_root)

    def test_save_and_load_round_trip(self) -> None:
        model_id = self.model.save_to_registry(
            registry_root=self.registry_root,
            version=1,
        )

        expected_dir = (
            self.registry_root / "v2" / "test" / "sedan_ice" / "2024" / self.slug / "v1"
        )
        self.assertTrue(expected_dir.is_dir())
        self.assertTrue((expected_dir / "metadata.json").exists())
        self.assertTrue((expected_dir / "model.onnx").exists())

        self.assertEqual(model_id.make, "test")
        self.assertEqual(model_id.vehicle_slug, "sedan_ice")
        self.assertEqual(model_id.year, 2024)
        self.assertEqual(model_id.config_slug, self.slug)
        self.assertEqual(model_id.version, 1)

        loaded = LocalRegistry(self.registry_root).load(model_id)
        before = round(self.model.predict(self.df).gallons_fastsim.sum(), 2)
        after = round(loaded.predict(self.df).gallons_fastsim.sum(), 2)
        self.assertTrue(math.isclose(before, after))

    def test_existing_slot_raises_without_overwrite(self) -> None:
        self.model.save_to_registry(
            registry_root=self.registry_root,
            version=1,
        )
        with self.assertRaises(FileExistsError):
            self.model.save_to_registry(
                registry_root=self.registry_root,
                version=1,
            )

    def test_overwrite_replaces_existing(self) -> None:
        self.model.save_to_registry(
            registry_root=self.registry_root,
            version=1,
        )
        # Second call with overwrite=True must succeed.
        model_id = self.model.save_to_registry(
            registry_root=self.registry_root,
            version=1,
            overwrite=True,
        )
        loaded = LocalRegistry(self.registry_root).load(model_id)
        self.assertIsNotNone(loaded)

    def test_invalid_version_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.model.save_to_registry(
                registry_root=self.registry_root,
                version=0,
            )

    def test_query_finds_published_model(self) -> None:
        self.model.save_to_registry(
            registry_root=self.registry_root,
            version=1,
        )
        infos = LocalRegistry(self.registry_root).query(make="test", model="sedan")
        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0].model_id.config_slug, self.slug)
