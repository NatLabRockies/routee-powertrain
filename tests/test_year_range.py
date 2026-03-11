from __future__ import annotations

import math
import shutil
from pathlib import Path
from unittest import TestCase

import pandas as pd

import routee.powertrain as pt
from routee.powertrain.core.year import (
    format_year,
    parse_year,
    serialize_year,
    year_contains,
)
from routee.powertrain.io.archive import load_archive, save_archive
from routee.powertrain.registry.local import LocalRegistry
from routee.powertrain.registry.model_id import ModelId
from routee.powertrain.io.archive import save_model_directory
from routee.powertrain.trainers.sklearn_random_forest import (
    SklearnRandomForestTrainer,
)


class TestParseYear(TestCase):
    """Tests for the parse_year utility function."""

    def test_parse_int(self):
        self.assertEqual(parse_year(2020), 2020)

    def test_parse_float(self):
        self.assertEqual(parse_year(2020.0), 2020)

    def test_parse_string_single(self):
        self.assertEqual(parse_year("2020"), 2020)

    def test_parse_string_range(self):
        self.assertEqual(parse_year("2020-2026"), (2020, 2026))

    def test_parse_tuple(self):
        self.assertEqual(parse_year((2020, 2026)), (2020, 2026))

    def test_parse_list(self):
        self.assertEqual(parse_year([2020, 2026]), (2020, 2026))

    def test_parse_tuple_wrong_length(self):
        with self.assertRaises(ValueError):
            parse_year((2020,))

    def test_parse_range_inverted(self):
        with self.assertRaises(ValueError):
            parse_year("2026-2020")

    def test_parse_bad_type(self):
        with self.assertRaises(ValueError):
            parse_year(None)


class TestFormatYear(TestCase):
    def test_format_single(self):
        self.assertEqual(format_year(2020), "2020")

    def test_format_range(self):
        self.assertEqual(format_year((2020, 2026)), "2020-2026")


class TestSerializeYear(TestCase):
    def test_serialize_single(self):
        self.assertEqual(serialize_year(2020), 2020)

    def test_serialize_range(self):
        self.assertEqual(serialize_year((2020, 2026)), "2020-2026")


class TestYearContains(TestCase):
    def test_single_match(self):
        self.assertTrue(year_contains(2020, 2020))

    def test_single_no_match(self):
        self.assertFalse(year_contains(2020, 2021))

    def test_range_contains(self):
        self.assertTrue(year_contains((2020, 2026), 2023))

    def test_range_start_boundary(self):
        self.assertTrue(year_contains((2020, 2026), 2020))

    def test_range_end_boundary(self):
        self.assertTrue(year_contains((2020, 2026), 2026))

    def test_range_before(self):
        self.assertFalse(year_contains((2020, 2026), 2019))

    def test_range_after(self):
        self.assertFalse(year_contains((2020, 2026), 2027))


class TestModelConfigYearRange(TestCase):
    """Test that ModelConfig accepts and round-trips year ranges."""

    def test_single_year(self):
        config = pt.ModelConfig(
            vehicle_description="Test",
            powertrain_type=pt.PowertrainType.ICE,
            feature_set=pt.FeatureSet(
                features=[pt.DataColumn(name="speed_mph", units="mph")]
            ),
            distance=pt.DataColumn(name="miles", units="miles"),
            target=pt.TargetSet(targets=[pt.DataColumn(name="energy", units="kwh")]),
            make="test",
            model_name="model",
            year=2020,
            trim="base",
        )
        self.assertEqual(config.year, 2020)
        d = config.to_dict()
        self.assertEqual(d["year"], 2020)

    def test_year_range_tuple(self):
        config = pt.ModelConfig(
            vehicle_description="Test",
            powertrain_type=pt.PowertrainType.ICE,
            feature_set=pt.FeatureSet(
                features=[pt.DataColumn(name="speed_mph", units="mph")]
            ),
            distance=pt.DataColumn(name="miles", units="miles"),
            target=pt.TargetSet(targets=[pt.DataColumn(name="energy", units="kwh")]),
            make="test",
            model_name="model",
            year=(2020, 2026),
            trim="base",
        )
        self.assertEqual(config.year, (2020, 2026))
        d = config.to_dict()
        self.assertEqual(d["year"], "2020-2026")

    def test_year_range_string(self):
        config = pt.ModelConfig(
            vehicle_description="Test",
            powertrain_type=pt.PowertrainType.ICE,
            feature_set=pt.FeatureSet(
                features=[pt.DataColumn(name="speed_mph", units="mph")]
            ),
            distance=pt.DataColumn(name="miles", units="miles"),
            target=pt.TargetSet(targets=[pt.DataColumn(name="energy", units="kwh")]),
            make="test",
            model_name="model",
            year="2020-2026",
            trim="base",
        )
        self.assertEqual(config.year, (2020, 2026))

    def test_year_range_roundtrip_via_dict(self):
        config = pt.ModelConfig(
            vehicle_description="Test",
            powertrain_type=pt.PowertrainType.ICE,
            feature_set=pt.FeatureSet(
                features=[pt.DataColumn(name="speed_mph", units="mph")]
            ),
            distance=pt.DataColumn(name="miles", units="miles"),
            target=pt.TargetSet(targets=[pt.DataColumn(name="energy", units="kwh")]),
            make="test",
            model_name="model",
            year=(2020, 2026),
            trim="base",
        )
        d = config.to_dict()
        restored = pt.ModelConfig.from_dict(d)
        self.assertEqual(restored.year, (2020, 2026))


class TestModelIdYearRange(TestCase):
    """Test that ModelId works with year ranges."""

    def test_single_year_path(self):
        mid = ModelId("toyota", "camry", 2016, "base", "default", "grade_speed", 1)
        self.assertIn("/2016/", mid.to_path())

    def test_year_range_path(self):
        mid = ModelId(
            "generic", "sedan", (2020, 2026), "base", "default", "grade_speed", 1
        )
        self.assertIn("/2020-2026/", mid.to_path())

    def test_year_range_str(self):
        mid = ModelId(
            "generic", "sedan", (2020, 2026), "base", "default", "grade_speed", 1
        )
        self.assertIn("2020-2026", str(mid))

    def test_year_range_dict_roundtrip(self):
        mid = ModelId(
            "generic", "sedan", (2020, 2026), "base", "default", "grade_speed", 1
        )
        d = mid.to_dict()
        restored = ModelId.from_dict(d)
        self.assertEqual(restored.year, (2020, 2026))

    def test_single_year_dict_roundtrip(self):
        mid = ModelId("toyota", "camry", 2016, "base", "default", "grade_speed", 1)
        d = mid.to_dict()
        restored = ModelId.from_dict(d)
        self.assertEqual(restored.year, 2016)


class TestLocalRegistryYearRange(TestCase):
    """Test that the local registry discovers and queries year-range models."""

    schema_version = "v2"

    def setUp(self):
        self.root = Path("tmp/test_registry_year_range")
        self.root.mkdir(parents=True, exist_ok=True)

        data_path = (
            Path(__file__).parent
            / Path("routee-powertrain-test-data")
            / Path("sample_train_data.csv")
        )
        df = pd.read_csv(data_path)

        config = pt.ModelConfig(
            vehicle_description="Generic Sedan 2020-2026",
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
            make="Generic",
            model_name="Sedan",
            year=(2020, 2026),
            trim="base",
        )
        trainer = SklearnRandomForestTrainer()
        self.model = trainer.train(df, config)

        model_id = ModelId(
            "generic",
            "sedan",
            (2020, 2026),
            "base",
            "default",
            "grade_dec_speed_mph",
            1,
        )
        rel_path = model_id.to_path(self.schema_version)
        full_path = self.root / rel_path
        save_model_directory(self.model, full_path)

        self.registry = LocalRegistry(
            root=self.root, schema_version=self.schema_version
        )

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_query_all_finds_range_model(self):
        results = self.registry.query()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.year, (2020, 2026))

    def test_query_year_within_range(self):
        results = self.registry.query(year=2023)
        self.assertEqual(len(results), 1)

    def test_query_year_at_boundaries(self):
        self.assertEqual(len(self.registry.query(year=2020)), 1)
        self.assertEqual(len(self.registry.query(year=2026)), 1)

    def test_query_year_outside_range(self):
        self.assertEqual(len(self.registry.query(year=2019)), 0)
        self.assertEqual(len(self.registry.query(year=2027)), 0)


class TestYearRangeArchiveRoundtrip(TestCase):
    """Test that a model with a year range survives archive serialization."""

    def setUp(self):
        self.out_path = Path("tmp/test_year_range_archive")
        self.out_path.mkdir(parents=True, exist_ok=True)

        data_path = (
            Path(__file__).parent
            / Path("routee-powertrain-test-data")
            / Path("sample_train_data.csv")
        )
        self.df = pd.read_csv(data_path)

    def tearDown(self):
        if self.out_path.exists():
            shutil.rmtree(self.out_path)

    def test_zip_roundtrip_with_year_range(self):
        config = pt.ModelConfig(
            vehicle_description="Generic Sedan 2020-2026",
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
            make="Generic",
            model_name="Sedan",
            year=(2020, 2026),
            trim="base",
        )
        trainer = SklearnRandomForestTrainer()
        model = trainer.train(self.df, config)

        outfile = self.out_path / "year_range_model.zip"
        save_archive(model, outfile)

        loaded = load_archive(outfile)
        self.assertEqual(loaded.metadata.config.year, (2020, 2026))

        r1 = model.predict(self.df)
        r2 = loaded.predict(self.df)
        self.assertTrue(
            math.isclose(r1.gallons_fastsim.sum(), r2.gallons_fastsim.sum())
        )
