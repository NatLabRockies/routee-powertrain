import math
from pathlib import Path
from unittest import TestCase

import pandas as pd

import routee.powertrain as pt
from routee.powertrain.core.drivetrain import Drivetrain
from routee.powertrain.core.fuel_type import FuelType
from routee.powertrain.io.archive import save_model_directory
from routee.powertrain.registry.local import LocalRegistry
from routee.powertrain.registry.model_id import ModelId, ModelInfo
from routee.powertrain.trainers.sklearn_random_forest import (
    SklearnRandomForestTrainer,
)

this_dir = Path(__file__).parent


class TestModelId(TestCase):
    def test_to_path(self):
        mid = ModelId("toyota", "camry_4cyl_fwd", 2016, "rf_default", 1)
        path = mid.to_path()
        self.assertEqual(path, "toyota/camry_4cyl_fwd/2016/rf_default/v1")

    def test_lowercase_normalization(self):
        mid = ModelId("Toyota", "Camry_4Cyl_FWD", 2016, "RF_Default", 1)
        self.assertEqual(mid.make, "toyota")
        self.assertEqual(mid.model, "camry_4cyl_fwd")
        self.assertEqual(mid.config_slug, "rf_default")

    def test_roundtrip_dict(self):
        mid = ModelId("toyota", "camry_4cyl_fwd", 2016, "rf_default", 1)
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
            model="Camry_4cyl_fwd",
            year=2016,
        )
        trainer = SklearnRandomForestTrainer()
        self.model = trainer.train(df, config)
        self.df = df

        # Save to registry path as a flat directory
        model_id = ModelId("toyota", "camry_4cyl_fwd", 2016, "rf_default", 1)
        rel_path = f"{self.schema_version}/{model_id.to_path()}"
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
        self.assertEqual(results[0].model_id.model, "camry_4cyl_fwd")

    def test_query_all(self):
        results = self.registry.query()
        self.assertEqual(len(results), 1)

    def test_query_no_match(self):
        results = self.registry.query(make="ford")
        self.assertEqual(len(results), 0)

    def test_load(self):
        model_id = ModelId("toyota", "camry_4cyl_fwd", 2016, "rf_default", 1)
        loaded = self.registry.load(model_id)

        r1 = self.model.predict(self.df)
        r2 = loaded.predict(self.df)
        self.assertTrue(
            math.isclose(r1.gallons_fastsim.sum(), r2.gallons_fastsim.sum())
        )

    def test_get_metadata(self):
        model_id = ModelId("toyota", "camry_4cyl_fwd", 2016, "rf_default", 1)
        meta = self.registry.get_metadata(model_id)
        self.assertIn("config", meta)
        self.assertIn("estimator_type", meta)

    def test_query_returns_model_info(self):
        results = self.registry.query(make="toyota")
        info = results[0]
        self.assertIsInstance(info, ModelInfo)
        self.assertEqual(info.estimator_type, "ONNXEstimator")
        self.assertIn("speed_mph", info.feature_names)
        self.assertIn("gallons_fastsim", info.target_names)

    def test_fuzzy_partial_make(self):
        """Partial make like 'toy' should match 'toyota'."""
        results = self.registry.query(make="toy")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.make, "toyota")

    def test_fuzzy_partial_model(self):
        """Partial model name like 'camry' should match 'camry_4cyl_fwd'."""
        results = self.registry.query(model="camry")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.model, "camry_4cyl_fwd")

    def test_fuzzy_disabled_exact_match(self):
        """With fuzzy=False, exact match is required."""
        results = self.registry.query(make="toyota", fuzzy=False)
        self.assertEqual(len(results), 1)

    def test_fuzzy_disabled_no_partial(self):
        """With fuzzy=False, partial match should not work."""
        results = self.registry.query(make="toy", fuzzy=False)
        self.assertEqual(len(results), 0)

    def test_fuzzy_no_match(self):
        """Completely unrelated query should not match even with fuzzy."""
        results = self.registry.query(make="zzzzz")
        self.assertEqual(len(results), 0)

    def test_fuzzy_threshold_controls_sensitivity(self):
        """High threshold rejects weak matches; low threshold accepts them."""
        # With a very high threshold, a weak partial match is rejected
        results_strict = self.registry.query(make="toyta", fuzzy_threshold=100)
        self.assertEqual(len(results_strict), 0)

        # With a lower threshold, the partial match is accepted
        results_relaxed = self.registry.query(make="toyta", fuzzy_threshold=80)
        self.assertEqual(len(results_relaxed), 1)

    def test_query_by_powertrain_type_exact(self):
        """Exact powertrain type match should work."""
        results = self.registry.query(powertrain_type="ICE")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].powertrain_type, "ICE")

    def test_query_by_powertrain_type_case_insensitive(self):
        """Powertrain type filtering should be case insensitive."""
        results = self.registry.query(powertrain_type="ice")
        self.assertEqual(len(results), 1)

    def test_query_by_powertrain_type_no_match(self):
        """Non-matching powertrain type should return empty results."""
        results = self.registry.query(powertrain_type="BEV")
        self.assertEqual(len(results), 0)

    def test_query_by_powertrain_type_fuzzy(self):
        """Fuzzy matching should work for powertrain type."""
        results = self.registry.query(powertrain_type="IC")
        self.assertEqual(len(results), 1)

    def test_query_by_powertrain_type_fuzzy_disabled(self):
        """With fuzzy=False, partial powertrain type should not match."""
        results = self.registry.query(powertrain_type="IC", fuzzy=False)
        self.assertEqual(len(results), 0)

    def test_query_by_powertrain_type_combined_with_make(self):
        """Powertrain type filter should combine with other filters."""
        results = self.registry.query(make="toyota", powertrain_type="ICE")
        self.assertEqual(len(results), 1)

        results = self.registry.query(make="toyota", powertrain_type="BEV")
        self.assertEqual(len(results), 0)

    def test_mass_lbs_in_model_info(self):
        """ModelInfo should include mass_lbs from the config."""
        results = self.registry.query(make="toyota")
        info = results[0]
        # The test model config doesn't set mass_lbs, so it should be None
        self.assertIsNone(info.mass_lbs)

    def test_custom_filter_single(self):
        """A single custom filter function should be applied."""
        # Filter that accepts everything
        results = self.registry.query(custom_filters=[lambda m: True])
        self.assertEqual(len(results), 1)

        # Filter that rejects everything
        results = self.registry.query(custom_filters=[lambda m: False])
        self.assertEqual(len(results), 0)

    def test_custom_filter_on_feature_names(self):
        """Custom filter can inspect ModelInfo fields like feature_names."""
        results = self.registry.query(
            custom_filters=[lambda m: "speed_mph" in m.feature_names]
        )
        self.assertEqual(len(results), 1)

        results = self.registry.query(
            custom_filters=[lambda m: "nonexistent_feature" in m.feature_names]
        )
        self.assertEqual(len(results), 0)

    def test_custom_filter_multiple(self):
        """Multiple custom filters are all applied (AND logic)."""
        results = self.registry.query(
            custom_filters=[
                lambda m: m.powertrain_type == "ICE",
                lambda m: "speed_mph" in m.feature_names,
            ]
        )
        self.assertEqual(len(results), 1)

        # Second filter rejects
        results = self.registry.query(
            custom_filters=[
                lambda m: m.powertrain_type == "ICE",
                lambda m: False,
            ]
        )
        self.assertEqual(len(results), 0)

    def test_custom_filter_combined_with_named_filters(self):
        """Custom filters work alongside named filters like make."""
        results = self.registry.query(
            make="toyota",
            custom_filters=[lambda m: m.powertrain_type == "ICE"],
        )
        self.assertEqual(len(results), 1)

        results = self.registry.query(
            make="ford",
            custom_filters=[lambda m: m.powertrain_type == "ICE"],
        )
        self.assertEqual(len(results), 0)

    def test_custom_filter_on_mass_lbs(self):
        """Custom filter can filter by mass_lbs (None-safe)."""
        # Our test model has no mass_lbs, so filtering for heavy vehicles returns nothing
        results = self.registry.query(
            custom_filters=[lambda m: m.mass_lbs is not None and m.mass_lbs > 10000]
        )
        self.assertEqual(len(results), 0)

        # Filtering for None mass should return our model
        results = self.registry.query(custom_filters=[lambda m: m.mass_lbs is None])
        self.assertEqual(len(results), 1)


class TestMassLbsModelConfig(TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp)
        self.schema_version = "v2"

        data_path = (
            this_dir
            / Path("routee-powertrain-test-data")
            / Path("sample_train_data.csv")
        )
        df = pd.read_csv(data_path)
        config = pt.ModelConfig(
            vehicle_description="Heavy Duty Truck",
            powertrain_type=pt.PowertrainType.HEAVY_DUTY,
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
            make="Freightliner",
            model="Cascadia",
            year=2022,
            mass_lbs=33000.0,
        )
        trainer = SklearnRandomForestTrainer()
        model = trainer.train(df, config)

        model_id = ModelId("freightliner", "cascadia", 2022, "rf_default", 1)
        rel_path = f"{self.schema_version}/{model_id.to_path()}"
        full_path = self.root / rel_path
        save_model_directory(model, full_path)

        self.registry = LocalRegistry(
            root=self.root, schema_version=self.schema_version
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp)

    def test_mass_lbs_populated(self):
        """ModelInfo should have mass_lbs when set in ModelConfig."""
        results = self.registry.query()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].mass_lbs, 33000.0)

    def test_custom_filter_mass_lbs_heavy(self):
        """Custom filter for mass > 10000 should match heavy vehicle."""
        results = self.registry.query(
            custom_filters=[lambda m: m.mass_lbs is not None and m.mass_lbs > 10000]
        )
        self.assertEqual(len(results), 1)

    def test_custom_filter_mass_lbs_light(self):
        """Custom filter for mass < 5000 should not match heavy vehicle."""
        results = self.registry.query(
            custom_filters=[lambda m: m.mass_lbs is not None and m.mass_lbs < 5000]
        )
        self.assertEqual(len(results), 0)

    def test_mass_lbs_in_model_info_dict(self):
        """mass_lbs should roundtrip through to_dict/from_dict."""
        results = self.registry.query()
        info = results[0]
        d = info.to_dict()
        self.assertEqual(d["mass_lbs"], 33000.0)
        restored = ModelInfo.from_dict(d)
        self.assertEqual(restored.mass_lbs, 33000.0)


class TestFuelTypeEnum(TestCase):
    def test_from_string_roundtrip(self):
        for member in FuelType:
            self.assertEqual(FuelType.from_string(member.name), member)

    def test_from_string_case_insensitive(self):
        self.assertEqual(FuelType.from_string("diesel"), FuelType.DIESEL)
        self.assertEqual(FuelType.from_string("Gasoline"), FuelType.GASOLINE)

    def test_from_string_empty_returns_undefined(self):
        self.assertEqual(FuelType.from_string(""), FuelType.UNDEFINED)
        self.assertEqual(FuelType.from_string(None), FuelType.UNDEFINED)

    def test_from_string_invalid_raises(self):
        with self.assertRaises(TypeError):
            FuelType.from_string("propane")


class TestDrivetrainEnum(TestCase):
    def test_from_string_roundtrip(self):
        for member in Drivetrain:
            self.assertEqual(Drivetrain.from_string(member.name), member)

    def test_from_string_case_insensitive(self):
        self.assertEqual(Drivetrain.from_string("fwd"), Drivetrain.FWD)
        self.assertEqual(Drivetrain.from_string("Awd"), Drivetrain.AWD)

    def test_from_string_empty_returns_undefined(self):
        self.assertEqual(Drivetrain.from_string(""), Drivetrain.UNDEFINED)
        self.assertEqual(Drivetrain.from_string(None), Drivetrain.UNDEFINED)

    def test_from_string_invalid_raises(self):
        with self.assertRaises(TypeError):
            Drivetrain.from_string("6wd")


class TestVehicleAttributeFields(TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp)
        self.schema_version = "v2"

        data_path = (
            this_dir
            / Path("routee-powertrain-test-data")
            / Path("sample_train_data.csv")
        )
        df = pd.read_csv(data_path)
        config = pt.ModelConfig(
            vehicle_description="2020 Chevrolet Colorado 2WD Diesel",
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
            make="Chevrolet",
            model="colorado_2wd_diesel",
            year=2020,
            fuel_type=FuelType.DIESEL,
            drivetrain=Drivetrain.FOURWD,
            engine="4cyl",
            trim="lt",
        )
        trainer = SklearnRandomForestTrainer()
        model = trainer.train(df, config)

        model_id = ModelId(
            "chevrolet",
            "colorado_2wd_diesel",
            2020,
            "rf_default",
            1,
        )
        rel_path = f"{self.schema_version}/{model_id.to_path()}"
        full_path = self.root / rel_path
        save_model_directory(model, full_path)

        self.registry = LocalRegistry(
            root=self.root, schema_version=self.schema_version
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp)

    def test_fuel_type_in_model_info(self):
        results = self.registry.query()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].fuel_type, "DIESEL")

    def test_drivetrain_in_model_info(self):
        results = self.registry.query()
        self.assertEqual(results[0].drivetrain, "FOURWD")

    def test_engine_in_model_info(self):
        results = self.registry.query()
        self.assertEqual(results[0].engine, "4cyl")

    def test_trim_in_model_info(self):
        results = self.registry.query()
        self.assertEqual(results[0].trim, "lt")

    def test_query_by_fuel_type(self):
        results = self.registry.query(fuel_type="DIESEL")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].fuel_type, "DIESEL")

    def test_query_by_fuel_type_no_match(self):
        results = self.registry.query(fuel_type="GASOLINE")
        self.assertEqual(len(results), 0)

    def test_query_by_drivetrain(self):
        results = self.registry.query(drivetrain="FOURWD")
        self.assertEqual(len(results), 1)

    def test_query_by_drivetrain_no_match(self):
        results = self.registry.query(drivetrain="FWD")
        self.assertEqual(len(results), 0)

    def test_query_by_engine(self):
        results = self.registry.query(engine="4cyl")
        self.assertEqual(len(results), 1)

    def test_query_by_trim(self):
        results = self.registry.query(trim="lt")
        self.assertEqual(len(results), 1)

    def test_query_combined_filters(self):
        results = self.registry.query(fuel_type="DIESEL", drivetrain="FOURWD")
        self.assertEqual(len(results), 1)

        results = self.registry.query(fuel_type="DIESEL", drivetrain="FWD")
        self.assertEqual(len(results), 0)

    def test_model_info_roundtrip_dict(self):
        results = self.registry.query()
        info = results[0]
        d = info.to_dict()
        self.assertEqual(d["fuel_type"], "DIESEL")
        self.assertEqual(d["drivetrain"], "FOURWD")
        self.assertEqual(d["engine"], "4cyl")
        self.assertEqual(d["trim"], "lt")
        restored = ModelInfo.from_dict(d)
        self.assertEqual(restored.fuel_type, "DIESEL")
        self.assertEqual(restored.drivetrain, "FOURWD")
        self.assertEqual(restored.engine, "4cyl")
        self.assertEqual(restored.trim, "lt")

    def test_model_config_string_coercion(self):
        """ModelConfig should coerce string fuel_type/drivetrain to enums."""
        config = pt.ModelConfig(
            vehicle_description="test",
            powertrain_type=pt.PowertrainType.ICE,
            feature_set=pt.FeatureSet(
                features=[pt.DataColumn(name="speed_mph", units="mph")]
            ),
            distance=pt.DataColumn(name="miles", units="miles"),
            target=pt.TargetSet(
                targets=[pt.DataColumn(name="gallons", units="gallons")]
            ),
            make="test",
            model="test",
            year=2020,
            fuel_type="DIESEL",
            drivetrain="AWD",
        )
        self.assertEqual(config.fuel_type, FuelType.DIESEL)
        self.assertEqual(config.drivetrain, Drivetrain.AWD)

    def test_model_config_backwards_compat(self):
        """ModelConfig.from_dict should handle missing new fields gracefully."""
        d = {
            "vehicle_description": "test",
            "powertrain_type": "ICE",
            "feature_set": {
                "features": [
                    {
                        "name": "speed_mph",
                        "units": "mph",
                        "dtype": "float32",
                        "constraints": {"lower": None, "upper": None},
                    }
                ]
            },
            "distance": {
                "name": "miles",
                "units": "miles",
                "dtype": "float32",
                "constraints": {"lower": None, "upper": None},
            },
            "target": {
                "targets": [
                    {
                        "name": "gallons",
                        "units": "gallons",
                        "dtype": "float32",
                        "constraints": {"lower": None, "upper": None},
                    }
                ]
            },
            "make": "test",
            "model": "test",
            "year": 2020,
        }
        config = pt.ModelConfig.from_dict(d)
        self.assertIsNone(config.fuel_type)
        self.assertIsNone(config.drivetrain)
        self.assertIsNone(config.engine)
        self.assertIsNone(config.trim)

    def test_model_info_from_dict_backwards_compat(self):
        """ModelInfo.from_dict should handle missing new fields gracefully."""
        d = {
            "model_id": {
                "make": "test",
                "model": "test",
                "year": 2020,
                "config_slug": "rf_default",
                "version": 1,
            },
            "estimator_type": "ONNXEstimator",
            "feature_names": ["speed_mph"],
            "target_names": ["gallons"],
            "powertrain_type": "ICE",
            "vehicle_description": "test",
        }
        info = ModelInfo.from_dict(d)
        self.assertIsNone(info.fuel_type)
        self.assertIsNone(info.drivetrain)
        self.assertIsNone(info.engine)
        self.assertIsNone(info.trim)

    def test_none_fields_excluded_from_filter(self):
        """Models with None fuel_type should not match a fuel_type filter."""
        from routee.powertrain.registry.filtering import filter_models

        info_with = ModelInfo(
            model_id=ModelId("a", "b", 2020, "rf_default", 1),
            estimator_type="ONNXEstimator",
            feature_names=["speed"],
            target_names=["gal"],
            powertrain_type="ICE",
            vehicle_description="test",
            fuel_type="DIESEL",
        )
        info_without = ModelInfo(
            model_id=ModelId("a", "c", 2020, "rf_default", 1),
            estimator_type="ONNXEstimator",
            feature_names=["speed"],
            target_names=["gal"],
            powertrain_type="ICE",
            vehicle_description="test",
        )
        results = filter_models([info_with, info_without], fuel_type="DIESEL")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].fuel_type, "DIESEL")
