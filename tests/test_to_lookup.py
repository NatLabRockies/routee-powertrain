from pathlib import Path
from unittest import TestCase

import pandas as pd
import numpy as np

import routee.powertrain as pt
from routee.powertrain.io.to_lookup_table import (
    to_lookup_table,
    LookupTableFeatureParameter,
)

this_dir = Path(__file__).parent


class TestToLookup(TestCase):
    def setUp(self) -> None:
        self.mock_model = pt.load_model("2016_TOYOTA_Camry_4cyl_2WD")

    def test_to_lookup_single_feature(self):
        """Test lookup table generation with all features, varying speed only."""
        feature_parameters = [
            {
                "feature_name": "speed_mph",
                "lower_bound": 10.0,
                "upper_bound": 60.0,
                "n_samples": 6,
            },
            {
                "feature_name": "grade_percent",
                "lower_bound": -5.0,
                "upper_bound": 5.0,
                "n_samples": 1,
            },
        ]

        result = to_lookup_table(
            model=self.mock_model,
            feature_parameters=feature_parameters,
            energy_target="gge",
        )

        # Check basic structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 6)  # 6 * 1 = 6 rows
        self.assertIn("speed_mph", result.columns)
        self.assertIn("grade_percent", result.columns)
        self.assertIn("gge_per_distance", result.columns)

        # Check speed values are correctly distributed
        expected_speeds = np.linspace(10.0, 60.0, 6)
        np.testing.assert_array_almost_equal(
            result["speed_mph"].values, expected_speeds
        )

        # Check that gge predictions are numeric and positive
        self.assertTrue(result["gge_per_distance"].dtype in [np.float64, np.float32])
        self.assertTrue((result["gge_per_distance"] > 0).all())

    def test_to_lookup_two_features(self):
        """Test lookup table generation with all features varied."""
        feature_parameters = [
            {
                "feature_name": "speed_mph",
                "lower_bound": 20.0,
                "upper_bound": 40.0,
                "n_samples": 3,
            },
            {
                "feature_name": "grade_percent",
                "lower_bound": -5.0,
                "upper_bound": 5.0,
                "n_samples": 3,
            },
        ]

        result = to_lookup_table(
            model=self.mock_model,
            feature_parameters=feature_parameters,
            energy_target="gge",
        )

        # Check basic structure - should have 3 * 3 = 9 combinations
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 9)
        self.assertIn("speed_mph", result.columns)
        self.assertIn("grade_percent", result.columns)
        self.assertIn("gge_per_distance", result.columns)

        # Check that we have all combinations
        expected_speeds = [20.0, 30.0, 40.0]
        expected_grades = [-5.0, 0.0, 5.0]

        unique_speeds = sorted(result["speed_mph"].unique())
        unique_grades = sorted(result["grade_percent"].unique())

        np.testing.assert_array_almost_equal(unique_speeds, expected_speeds)
        np.testing.assert_array_almost_equal(unique_grades, expected_grades)

        # Check that all combinations exist
        for speed in expected_speeds:
            for grade in expected_grades:
                combo_exists = (
                    (result["speed_mph"] == speed) & (result["grade_percent"] == grade)
                ).any()
                self.assertTrue(
                    combo_exists, f"Missing combination: speed={speed}, grade={grade}"
                )

    def test_invalid_energy_target(self):
        """Test error handling for invalid energy target."""
        feature_parameters = [
            {
                "feature_name": "speed_mph",
                "lower_bound": 10.0,
                "upper_bound": 60.0,
                "n_samples": 5,
            }
        ]

        with self.assertRaises(KeyError) as context:
            to_lookup_table(
                model=self.mock_model,
                feature_parameters=feature_parameters,
                energy_target="invalid_target",
            )

        self.assertIn(
            "Model does not have a target named invalid_target", str(context.exception)
        )

    def test_invalid_feature_set(self):
        """Test error handling for unknown features."""
        feature_parameters = [
            {
                "feature_name": "invalid_feature",
                "lower_bound": 0.0,
                "upper_bound": 100.0,
                "n_samples": 5,
            },
            {
                "feature_name": "speed_mph",
                "lower_bound": 10.0,
                "upper_bound": 60.0,
                "n_samples": 5,
            },
            {
                "feature_name": "grade_percent",
                "lower_bound": -5.0,
                "upper_bound": 5.0,
                "n_samples": 5,
            },
        ]

        with self.assertRaises(KeyError) as context:
            to_lookup_table(
                model=self.mock_model,
                feature_parameters=feature_parameters,
                energy_target="gge",
            )

        self.assertIn(
            "Unknown features",
            str(context.exception),
        )

    def test_missing_feature_parameters(self):
        """Test error handling when not all model features are provided."""
        feature_parameters = [
            {
                "feature_name": "speed_mph",
                "lower_bound": 10.0,
                "upper_bound": 60.0,
                "n_samples": 5,
            },
        ]

        with self.assertRaises(KeyError) as context:
            to_lookup_table(
                model=self.mock_model,
                feature_parameters=feature_parameters,
                energy_target="gge",
            )

        self.assertIn(
            "Missing required feature parameters",
            str(context.exception),
        )

    def test_single_sample(self):
        """Test lookup table generation with n_samples=1 for all features."""
        feature_parameters = [
            {
                "feature_name": "speed_mph",
                "lower_bound": 30.0,
                "upper_bound": 30.1,  # Slightly different from lower bound
                "n_samples": 1,
            },
            {
                "feature_name": "grade_percent",
                "lower_bound": -5.0,
                "upper_bound": 5.0,
                "n_samples": 1,
            },
        ]

        result = to_lookup_table(
            model=self.mock_model,
            feature_parameters=feature_parameters,
            energy_target="gge",
        )

        # Should have exactly 1 row (1 * 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result["speed_mph"].iloc[0], 30.0
        )  # linspace with 1 sample returns the start
        self.assertIn("gge_per_distance", result.columns)

    def test_large_number_of_samples(self):
        """Test lookup table generation with a larger number of samples."""
        feature_parameters = [
            {
                "feature_name": "speed_mph",
                "lower_bound": 0.0,
                "upper_bound": 100.0,
                "n_samples": 101,  # 101 samples from 0 to 100
            },
            {
                "feature_name": "grade_percent",
                "lower_bound": -5.0,
                "upper_bound": 5.0,
                "n_samples": 1,
            },
        ]

        result = to_lookup_table(
            model=self.mock_model,
            feature_parameters=feature_parameters,
            energy_target="gge",
        )

        # Should have exactly 101 rows (101 * 1)
        self.assertEqual(len(result), 101)

        # Check first and last values
        self.assertEqual(result["speed_mph"].iloc[0], 0.0)
        self.assertEqual(result["speed_mph"].iloc[-1], 100.0)

        # Check that all values are unique and correctly spaced
        speeds = result["speed_mph"].values
        expected_speeds = np.linspace(0.0, 100.0, 101)
        np.testing.assert_array_almost_equal(speeds, expected_speeds)

    def test_empty_feature_parameters(self):
        """Test error handling for empty feature parameters list."""
        feature_parameters = []

        with self.assertRaises(Exception):  # This might raise various exceptions
            to_lookup_table(
                model=self.mock_model,
                feature_parameters=feature_parameters,
                energy_target="gge",
            )


class TestLookupTableFeatureParameter(TestCase):
    """Test the LookupTableFeatureParameter dataclass."""

    def test_from_dict_valid(self):
        """Test creating LookupTableFeatureParameter from valid dictionary."""
        data = {
            "feature_name": "speed_mph",
            "lower_bound": 10.0,
            "upper_bound": 60.0,
            "n_samples": 10,
        }

        param = LookupTableFeatureParameter.from_dict(data)

        self.assertEqual(param.feature_name, "speed_mph")
        self.assertEqual(param.lower_bound, 10.0)
        self.assertEqual(param.upper_bound, 60.0)
        self.assertEqual(param.n_samples, 10)

    def test_from_dict_missing_feature_name(self):
        """Test error when feature_name is missing."""
        data = {
            "lower_bound": 10.0,
            "upper_bound": 60.0,
            "n_samples": 10,
        }

        with self.assertRaises(ValueError) as context:
            LookupTableFeatureParameter.from_dict(data)

        self.assertIn("must provide feature name", str(context.exception))

    def test_from_dict_missing_lower_bound(self):
        """Test error when lower_bound is missing."""
        data = {
            "feature_name": "speed_mph",
            "upper_bound": 60.0,
            "n_samples": 10,
        }

        with self.assertRaises(ValueError) as context:
            LookupTableFeatureParameter.from_dict(data)

        self.assertIn("must provide lower bound", str(context.exception))

    def test_from_dict_missing_upper_bound(self):
        """Test error when upper_bound is missing."""
        data = {
            "feature_name": "speed_mph",
            "lower_bound": 10.0,
            "n_samples": 10,
        }

        with self.assertRaises(ValueError) as context:
            LookupTableFeatureParameter.from_dict(data)

        self.assertIn("must provide upper bound", str(context.exception))

    def test_from_dict_missing_n_samples(self):
        """Test error when n_samples is missing."""
        data = {
            "feature_name": "speed_mph",
            "lower_bound": 10.0,
            "upper_bound": 60.0,
        }

        with self.assertRaises(ValueError) as context:
            LookupTableFeatureParameter.from_dict(data)

        self.assertIn("must provide n_samples", str(context.exception))

    def test_from_dict_invalid_bounds(self):
        """Test error when lower_bound >= upper_bound."""
        data = {
            "feature_name": "speed_mph",
            "lower_bound": 60.0,
            "upper_bound": 10.0,
            "n_samples": 10,
        }

        with self.assertRaises(ValueError) as context:
            LookupTableFeatureParameter.from_dict(data)

        self.assertIn(
            "lower bound must be less than upper bound", str(context.exception)
        )

    def test_from_dict_equal_bounds(self):
        """Test error when lower_bound == upper_bound."""
        data = {
            "feature_name": "speed_mph",
            "lower_bound": 30.0,
            "upper_bound": 30.0,
            "n_samples": 10,
        }

        with self.assertRaises(ValueError) as context:
            LookupTableFeatureParameter.from_dict(data)

        self.assertIn(
            "lower bound must be less than upper bound", str(context.exception)
        )

    def test_from_dict_type_conversion(self):
        """Test that numeric values are properly converted to correct types."""
        data = {
            "feature_name": "speed_mph",
            "lower_bound": "10.5",  # String that can be converted to float
            "upper_bound": "60.7",  # String that can be converted to float
            "n_samples": "15",  # String that can be converted to int
        }

        param = LookupTableFeatureParameter.from_dict(data)

        self.assertEqual(param.feature_name, "speed_mph")
        self.assertEqual(param.lower_bound, 10.5)
        self.assertEqual(param.upper_bound, 60.7)
        self.assertEqual(param.n_samples, 15)
        self.assertIsInstance(param.lower_bound, float)
        self.assertIsInstance(param.upper_bound, float)
        self.assertIsInstance(param.n_samples, int)
