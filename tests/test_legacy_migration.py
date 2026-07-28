"""v1 -> v2 migration affordances.

A user coming from routee-powertrain v1 hits the new package at four points:
a flat model name, a v1 ``.json`` file, the conversion API, and the CLI. Each
should fail (or succeed) with a message that names the way forward, rather than
a generic parse or format error.
"""

import base64
import json
import shutil
from pathlib import Path
from unittest import TestCase

import pandas as pd

import routee.powertrain as pt
from routee.powertrain.cli import main as cli_main
from routee.powertrain.estimators.onnx import ONNXEstimator
from routee.powertrain.io.legacy import convert_legacy_model
from routee.powertrain.trainers.sklearn_random_forest import (
    SklearnRandomForestTrainer,
)

this_dir = Path(__file__).parent

LEGACY_NAME = "2016_TOYOTA_Camry_4cyl_2WD"


def _column(name: str, units: str) -> dict:
    return {
        "name": name,
        "units": units,
        "dtype": "float32",
        "constraints": {"lower": None, "upper": None},
    }


class TestLegacyLoadErrors(TestCase):
    """load_model() explains the v1 -> v2 change instead of a bare parse error."""

    def test_flat_v1_model_name_points_at_search_and_converter(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            pt.load_model(LEGACY_NAME)
        msg = str(ctx.exception)

        self.assertIn(LEGACY_NAME, msg)
        self.assertIn("v1", msg)
        # Both exits are offered: find the published equivalent, or convert.
        self.assertIn("query_available_models", msg)
        self.assertIn("convert-v1", msg)
        self.assertIn("migrating_from_v1", msg)

    def test_malformed_id_still_gets_the_precise_grammar_message(self) -> None:
        # A string with slashes is a malformed id, not a legacy name -- the
        # legacy heuristic must not swallow it.
        with self.assertRaises(ValueError) as ctx:
            pt.load_model("toyota/camry_ice")
        msg = str(ctx.exception)

        self.assertIn("Could not parse", msg)
        self.assertIn("<make>/<vehicle_slug>/<year>/<config_slug>", msg)
        self.assertNotIn("convert-v1", msg)

    def test_missing_json_path_explains_the_format_change(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            pt.load_model("DoesNotExist.json")
        msg = str(ctx.exception)

        self.assertIn("v1 model file", msg)
        self.assertIn("convert-v1", msg)
        self.assertIn("migrating_from_v1", msg)


class TestLegacyConversion(TestCase):
    """The converter is reachable as public API and as a console script."""

    def setUp(self) -> None:
        self.df = pd.read_csv(
            this_dir / "routee-powertrain-test-data" / "sample_train_data.csv"
        )
        self.out_path = Path("tmp_legacy_migration")
        self.out_path.mkdir(exist_ok=True)
        self.json_path = self._write_legacy_json()

    def tearDown(self) -> None:
        if self.out_path.exists():
            shutil.rmtree(self.out_path)

    def _write_legacy_json(self, estimator_type: str = "ONNXEstimator") -> Path:
        """Build a v1-shaped .json holding two feature sets.

        Two feature sets, because the one-v1-file-becomes-many-v2-models fan-out
        is the behavior most likely to surprise a migrating user.
        """
        config = pt.ModelConfig(
            vehicle_description="Legacy Camry",
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
            make="toyota",
            model="camry",
            year=2016,
        )
        two_feature = SklearnRandomForestTrainer().train(self.df, config)
        assert isinstance(two_feature.estimator, ONNXEstimator)

        one_feature_config = config.model_copy(
            update={
                "feature_set": pt.FeatureSet(
                    features=[pt.DataColumn(name="speed_mph", units="mph")]
                )
            }
        )
        one_feature = SklearnRandomForestTrainer().train(self.df, one_feature_config)
        assert isinstance(one_feature.estimator, ONNXEstimator)

        def _entry(model: pt.Model) -> dict:
            assert isinstance(model.estimator, ONNXEstimator)
            raw = model.estimator.onnx_model.SerializeToString()
            return {
                "estimator_constructor_type": estimator_type,
                "estimator": {"onnx_model": base64.b64encode(raw).decode("utf-8")},
            }

        both = "grade_dec&speed_mph"
        speed_only = "speed_mph"
        legacy = {
            "metadata": {
                "config": {
                    "vehicle_description": "Legacy Camry",
                    "powertrain_type": "ICE",
                    "feature_sets": [
                        {
                            "features": [
                                _column("speed_mph", "mph"),
                                _column("grade_dec", "decimal"),
                            ]
                        },
                        {"features": [_column("speed_mph", "mph")]},
                    ],
                    "distance": _column("miles", "miles"),
                    "target": {"targets": [_column("gallons_fastsim", "gallons")]},
                    "predict_method": "rate",
                },
                "routee_version": "1.0.0",
            },
            "errors": {
                "estimator_errors": {
                    both: {"error_by_target": {}},
                    speed_only: {"error_by_target": {}},
                }
            },
            "all_estimators": {
                both: _entry(two_feature),
                speed_only: _entry(one_feature),
            },
        }
        path = self.out_path / "legacy.json"
        path.write_text(json.dumps(legacy))
        return path

    def test_convert_legacy_model_fans_out_one_model_per_feature_set(self) -> None:
        created = convert_legacy_model(
            self.json_path,
            self.out_path / "converted",
            make="toyota",
            model="camry",
            year=2016,
        )
        self.assertEqual(len(created), 2)

        # Each converted directory is a loadable v2 model, and the two differ
        # only by feature set -- which is what makes them separate models.
        loaded = [pt.load_model(p) for p in created]
        feature_sets = {frozenset(m.feature_names) for m in loaded}
        self.assertEqual(
            feature_sets,
            {frozenset({"speed_mph"}), frozenset({"speed_mph", "grade_dec"})},
        )

        # Every converted model gets a minted identity.
        for model in loaded:
            self.assertIsNotNone(model.digest)
            self.assertTrue(str(model.digest).startswith("sha256:"))

        # The derived config slugs are distinct, so the models don't collide.
        self.assertEqual(len({m.key.config_slug for m in loaded}), 2)

    def test_exported_at_package_top_level(self) -> None:
        self.assertIs(pt.convert_legacy_model, convert_legacy_model)

    def test_cli_converts(self) -> None:
        out = self.out_path / "cli_converted"
        exit_code = cli_main(
            [
                "convert-v1",
                str(self.json_path),
                str(out),
                "--make",
                "toyota",
                "--model",
                "camry",
                "--year",
                "2016",
            ]
        )
        self.assertEqual(exit_code, 0)
        created = sorted(out.rglob("metadata.json"))
        self.assertEqual(len(created), 2)

    def test_non_legacy_json_is_rejected(self) -> None:
        not_legacy = self.out_path / "random.json"
        not_legacy.write_text(json.dumps({"hello": "world"}))

        with self.assertRaises(ValueError) as ctx:
            convert_legacy_model(
                not_legacy, self.out_path / "x", make="a", model="b", year=2016
            )
        self.assertIn("does not look like", str(ctx.exception))

    def test_unconvertible_estimator_type_raises_with_reason(self) -> None:
        # A v1 model saved with the removed smartcore estimator has nothing to
        # convert; say why rather than silently producing zero models.
        legacy = json.loads(self.json_path.read_text())
        for entry in legacy["all_estimators"].values():
            entry["estimator_constructor_type"] = "SmartCoreEstimator"
        smartcore_path = self.out_path / "smartcore.json"
        smartcore_path.write_text(json.dumps(legacy))

        with self.assertRaises(ValueError) as ctx:
            convert_legacy_model(
                smartcore_path,
                self.out_path / "y",
                make="toyota",
                model="camry",
                year=2016,
            )
        msg = str(ctx.exception)
        self.assertIn("SmartCoreEstimator", msg)
        self.assertIn("removed in v2", msg)
