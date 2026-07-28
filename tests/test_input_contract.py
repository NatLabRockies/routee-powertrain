"""Tests for the self-describing estimator input/output contract.

The contract pins the positional order of an estimator's input tensor (and its
outputs) so a downstream consumer holding only the serialized binary — e.g.
routee-compass feeding an ONNX inference engine — can order its columns
correctly instead of relying on an out-of-band assumption.
"""

import json
import shutil
from pathlib import Path
from unittest import TestCase

import onnx
import pandas as pd

import routee.powertrain as pt
from routee.powertrain.core.model_config import PredictMethod
from routee.powertrain.estimators.estimator_interface import InputSpec
from routee.powertrain.estimators.onnx import (
    _META_INPUT_COLUMNS,
    _META_OUTPUT_COLUMNS,
    _META_PREDICT_METHOD,
)
from routee.powertrain.trainers.sklearn_random_forest import (
    SklearnRandomForestTrainer,
)

this_dir = Path(__file__).parent


def _config(predict_method: PredictMethod = PredictMethod.RATE) -> pt.ModelConfig:
    return pt.ModelConfig(
        vehicle_description="Contract Test Model",
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
        predict_method=predict_method,
    )


class TestInputContract(TestCase):
    def setUp(self) -> None:
        data_path = (
            this_dir
            / Path("routee-powertrain-test-data")
            / Path("sample_train_data.csv")
        )
        self.df = pd.read_csv(data_path)
        self.out_path = Path("tmp")
        self.out_path.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        if self.out_path.exists():
            shutil.rmtree(self.out_path)

    def test_rate_contract_order_and_embedding(self) -> None:
        model = SklearnRandomForestTrainer().train(self.df, _config())
        spec = model.estimator.input_spec
        assert spec.input_columns is not None
        assert spec.output_columns is not None

        # RATE: input tensor is exactly the feature columns, in order.
        self.assertEqual(
            [c.name for c in spec.input_columns], ["speed_mph", "grade_dec"]
        )
        self.assertEqual([c.name for c in spec.output_columns], ["gallons_fastsim"])
        self.assertEqual(spec.predict_method, "rate")
        self.assertEqual(spec.distance_column, "miles")
        # units travel with the contract for a consumer that converts inputs
        self.assertEqual(spec.input_columns[0].units, "mph")

        # The contract is embedded in the .onnx binary itself — a consumer with
        # only the bytes can reconstruct the input order.
        props = {
            p.key: p.value
            for p in onnx.load_from_string(model.estimator.to_bytes()).metadata_props
        }
        self.assertIn(_META_INPUT_COLUMNS, props)
        self.assertIn(_META_OUTPUT_COLUMNS, props)
        self.assertEqual(props[_META_PREDICT_METHOD], "rate")
        embedded = [c["name"] for c in json.loads(props[_META_INPUT_COLUMNS])]
        self.assertEqual(embedded, ["speed_mph", "grade_dec"])

    def test_raw_contract_appends_distance(self) -> None:
        model = SklearnRandomForestTrainer().train(self.df, _config(PredictMethod.RAW))
        spec = model.estimator.input_spec
        assert spec.input_columns is not None
        # RAW: distance is the final input column.
        self.assertEqual(
            [c.name for c in spec.input_columns], ["speed_mph", "grade_dec", "miles"]
        )
        self.assertEqual(spec.predict_method, "raw")

    def test_contract_round_trips_through_all_formats(self) -> None:
        model = SklearnRandomForestTrainer().train(self.df, _config())
        in_cols = model.estimator.input_spec.input_columns
        assert in_cols is not None
        expected = [c.name for c in in_cols]
        for name in ("model_dir", "model.zip", "model.tar.gz"):
            with self.subTest(format=name):
                target = self.out_path / name
                model.to_file(target)
                loaded = pt.load_model(target)
                loaded_cols = loaded.estimator.input_spec.input_columns
                assert loaded_cols is not None
                self.assertEqual([c.name for c in loaded_cols], expected)

    def test_metadata_json_orders_via_contract_not_input_spec(self) -> None:
        # metadata.json keeps the ordered feature list once, in ``contract``;
        # ``estimator.input_spec`` carries only the estimator mechanics
        # (lookback/grouping/pad) and does NOT duplicate the columns.
        model = SklearnRandomForestTrainer().train(self.df, _config())
        outdir = self.out_path / "meta_model"
        model.to_file(outdir)
        meta = json.loads((outdir / "metadata.json").read_text())

        self.assertEqual(
            [c["name"] for c in meta["contract"]["feature_set"]],
            ["speed_mph", "grade_dec"],
        )
        self.assertEqual(meta["contract"]["predict_method"], "rate")

        spec = meta["estimator"]["input_spec"]
        self.assertEqual(set(spec), {"lookback", "grouping_column", "pad_strategy"})
        self.assertNotIn("input_columns", spec)

    def test_contract_mismatch_raises_on_load(self) -> None:
        # Edit metadata's feature order so it disagrees with the (unchanged)
        # embedded binary contract — loading must raise, not silently transpose.
        model = SklearnRandomForestTrainer().train(self.df, _config())
        outdir = self.out_path / "tamper_model"
        model.to_file(outdir)
        meta_path = outdir / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["contract"]["feature_set"] = list(
            reversed(meta["contract"]["feature_set"])
        )
        meta_path.write_text(json.dumps(meta))
        with self.assertRaises(ValueError) as ctx:
            pt.load_model(outdir)
        self.assertIn("contract", str(ctx.exception))

    def test_ngboost_contract_survives_load_and_resave(self) -> None:
        # NGBoost's binary (joblib) doesn't embed the contract, but metadata
        # carries it and the loader injects it — so a loaded NGBoost model is
        # contract-complete and can be re-saved (required-on-persist) without
        # re-binding.
        from routee.powertrain.trainers.ngboost_trainer import NGBoostTrainer

        model = NGBoostTrainer().train(self.df, _config())
        spec = model.estimator.input_spec
        assert spec.output_columns is not None
        # NGBoost declares the point target plus its std column.
        self.assertEqual(
            [c.name for c in spec.output_columns],
            ["gallons_fastsim", "gallons_fastsim_std"],
        )

        outdir = self.out_path / "ngb_model"
        model.to_file(outdir)
        loaded = pt.load_model(outdir)
        loaded_spec = loaded.estimator.input_spec
        assert loaded_spec.input_columns is not None
        self.assertEqual(
            [c.name for c in loaded_spec.input_columns], ["speed_mph", "grade_dec"]
        )
        # re-saving the loaded model succeeds (contract present in memory)
        loaded.to_file(self.out_path / "ngb_model_resave")

    def test_save_requires_contract(self) -> None:
        # A model persisted without a complete contract must be rejected — the
        # contract is required on save. (Trained models bind it automatically;
        # here we simulate an estimator that was never bound.)
        model = SklearnRandomForestTrainer().train(self.df, _config())
        model.estimator.input_spec = InputSpec()
        with self.assertRaises(ValueError) as ctx:
            model.to_file(self.out_path / "no_contract")
        self.assertIn("contract", str(ctx.exception))


if __name__ == "__main__":
    import unittest

    unittest.main()
