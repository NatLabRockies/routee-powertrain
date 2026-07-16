"""The legacy-v1 → v2 conversion script fills in the input/output contract.

Legacy v1 binaries predate the self-describing contract; the conversion must
re-embed it (into the ONNX binary) and record it in metadata.json so every
converted model is required-contract-complete and loads cleanly.
"""

import base64
import json
import shutil
import sys
from pathlib import Path
from unittest import TestCase

import onnx
import pandas as pd

import routee.powertrain as pt
from routee.powertrain.estimators.onnx import ONNXEstimator
from routee.powertrain.trainers.sklearn_random_forest import (
    SklearnRandomForestTrainer,
)

this_dir = Path(__file__).parent
sys.path.insert(0, str(this_dir.parent / "scripts"))


def _column(name: str, units: str) -> dict:
    return {
        "name": name,
        "units": units,
        "dtype": "float32",
        "constraints": {"lower": None, "upper": None},
    }


class TestConvertLegacyContract(TestCase):
    def setUp(self) -> None:
        self.df = pd.read_csv(
            this_dir / "routee-powertrain-test-data" / "sample_train_data.csv"
        )
        self.out_path = Path("tmp_convert")
        self.out_path.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        if self.out_path.exists():
            shutil.rmtree(self.out_path)

    def _legacy_json(self) -> Path:
        # Train a real RF, then take its *bare* ONNX graph (no embedded contract)
        # to stand in for a legacy v1 binary.
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
        model = SklearnRandomForestTrainer().train(self.df, config)
        assert isinstance(model.estimator, ONNXEstimator)
        bare_onnx = model.estimator.onnx_model.SerializeToString()

        fs_id = "grade_dec&speed_mph"  # sorted feature names, '&'-joined
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
                        }
                    ],
                    "distance": _column("miles", "miles"),
                    "target": {"targets": [_column("gallons_fastsim", "gallons")]},
                    "predict_method": "rate",
                },
                "routee_version": "1.0.0",
            },
            "errors": {"estimator_errors": {fs_id: {"error_by_target": {}}}},
            "all_estimators": {
                fs_id: {
                    "estimator_constructor_type": "ONNXEstimator",
                    "estimator": {
                        "onnx_model": base64.b64encode(bare_onnx).decode("utf-8")
                    },
                }
            },
        }
        json_path = self.out_path / "legacy.json"
        json_path.write_text(json.dumps(legacy))
        return json_path

    def test_converted_model_is_contract_complete(self) -> None:
        from convert_legacy_models import VehicleIdentity, convert_legacy_json

        json_path = self._legacy_json()
        created = convert_legacy_json(
            json_path,
            self.out_path / "converted",
            VehicleIdentity(make="toyota", model="camry", year=2016),
            version=1,
        )
        self.assertEqual(len(created), 1)
        model_dir = created[0]

        # The converted model loads (load-time contract cross-check passes) and
        # carries the full positional contract.
        loaded = pt.load_model(model_dir)
        spec = loaded.estimator.input_spec
        assert spec.input_columns is not None
        self.assertEqual(
            [c.name for c in spec.input_columns], ["speed_mph", "grade_dec"]
        )
        self.assertEqual(spec.predict_method, "rate")
        self.assertEqual(spec.distance_column, "miles")

        # The binary itself is self-describing (contract embedded in metadata_props).
        props = {
            p.key: p.value
            for p in onnx.load(str(model_dir / "model.onnx")).metadata_props
        }
        self.assertIn("routee_input_columns", props)
        embedded = [c["name"] for c in json.loads(props["routee_input_columns"])]
        self.assertEqual(embedded, ["speed_mph", "grade_dec"])

        # metadata.json records the ordered features once, in ``contract``;
        # ``estimator.input_spec`` carries only the mechanics (no column echo).
        meta = json.loads((model_dir / "metadata.json").read_text())
        self.assertEqual(
            [c["name"] for c in meta["contract"]["feature_set"]],
            ["speed_mph", "grade_dec"],
        )
        self.assertNotIn("input_columns", meta["estimator"]["input_spec"])


if __name__ == "__main__":
    import unittest

    unittest.main()
