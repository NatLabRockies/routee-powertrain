"""Behavioral tests for the pydantic (schema v3) serialization layer.

Covers the wire shape produced by ``model_dump(mode="json")``, validation of
bad input, and the hard break against older schema versions.
"""

from typing import Any
from unittest import TestCase

from pydantic import ValidationError

import routee.powertrain as pt
from routee.powertrain.core.metadata import SCHEMA_VERSION
from routee.powertrain.core.model_config import PredictMethod
from routee.powertrain.io.archive import _model_from_metadata_and_bytes
from routee.powertrain.registry.model_id import ModelId, ModelInfo


def _make_config(**overrides: Any) -> pt.ModelConfig:
    kwargs: dict[str, Any] = dict(
        vehicle_description="test",
        powertrain_type=pt.PowertrainType.ICE,
        feature_set=[pt.DataColumn(name="speed_mph", units="mph")],
        distance=pt.DataColumn(name="miles", units="miles"),
        target=pt.DataColumn(name="gge", units="gallons"),
        make="Toyota",
        model="Camry",
        year=2016,
    )
    kwargs.update(overrides)
    return pt.ModelConfig(**kwargs)


class TestSerializationShape(TestCase):
    def test_enum_and_optional_wire_shape(self):
        cfg = _make_config(
            predict_method=PredictMethod.RAW,
            fuel_type="gasoline",
        )
        d = cfg.model_dump(mode="json")
        # int-valued enums serialize by name, PredictMethod by value
        self.assertEqual(d["powertrain_type"], "ICE")
        self.assertEqual(d["predict_method"], "raw")
        self.assertEqual(d["fuel_type"], "GASOLINE")
        # unset optional enum -> null (not dropped)
        self.assertIn("drivetrain", d)
        self.assertIsNone(d["drivetrain"])
        # make/model lowercased
        self.assertEqual(d["make"], "toyota")
        self.assertEqual(d["model"], "camry")

    def test_year_single_and_range_shape(self):
        self.assertEqual(_make_config(year=2016).model_dump(mode="json")["year"], 2016)
        ranged = _make_config(year=(2020, 2026)).model_dump(mode="json")
        self.assertEqual(ranged["year"], [2020, 2026])

    def test_config_roundtrips(self):
        cfg = _make_config(year="2020-2026", drivetrain="awd")
        restored = pt.ModelConfig.model_validate(cfg.model_dump(mode="json"))
        self.assertEqual(restored, cfg)
        self.assertEqual(restored.year, (2020, 2026))
        self.assertEqual(restored.drivetrain, pt.Drivetrain.AWD)

    def test_modelid_roundtrips(self):
        mid = ModelId("Toyota", "Camry", 2016, "rf_default", 1)
        self.assertEqual(ModelId.model_validate(mid.model_dump(mode="json")), mid)

    def test_feature_set_and_target_serialize_as_bare_lists(self):
        cfg = _make_config(
            feature_set=[
                pt.DataColumn(name="speed_mph", units="mph"),
                pt.DataColumn(name="grade", units="percent"),
            ],
        )
        d = cfg.model_dump(mode="json")
        # no nested "features"/"targets" wrapper — just a list of columns
        self.assertIsInstance(d["feature_set"], list)
        self.assertEqual([f["name"] for f in d["feature_set"]], ["speed_mph", "grade"])
        self.assertIsInstance(d["target"], list)
        self.assertEqual([t["name"] for t in d["target"]], ["gge"])
        # round-trips from the flat shape
        self.assertEqual(pt.ModelConfig.model_validate(d), cfg)


class TestRealWorldAdjustmentFactor(TestCase):
    def test_factor_defaults_from_powertrain_type(self):
        ice = _make_config(powertrain_type=pt.PowertrainType.ICE)
        bev = _make_config(powertrain_type=pt.PowertrainType.BEV)
        self.assertAlmostEqual(ice.real_world_adjustment_factor, 1.166)
        self.assertAlmostEqual(bev.real_world_adjustment_factor, 1.3958)

    def test_explicit_factor_respected(self):
        cfg = _make_config(real_world_adjustment_factor=1.0)
        self.assertEqual(cfg.real_world_adjustment_factor, 1.0)

    def test_no_apply_flag_in_dump(self):
        d = _make_config().model_dump(mode="json")
        self.assertNotIn("apply_real_world_adjustment", d)
        self.assertIn("real_world_adjustment_factor", d)


class TestValidation(TestCase):
    def test_constraints_bounds(self):
        with self.assertRaises(ValidationError):
            pt.Constraints(lower=5.0, upper=1.0)

    def test_datacolumn_name_rejects_ampersand(self):
        with self.assertRaises(ValidationError):
            pt.DataColumn(name="a&b", units="mph")

    def test_missing_required_field(self):
        with self.assertRaises(ValidationError):
            pt.ModelConfig(
                vehicle_description="x",
                powertrain_type=pt.PowertrainType.ICE,
                feature_set=[pt.DataColumn(name="speed_mph", units="mph")],
                distance=pt.DataColumn(name="miles", units="miles"),
                target=pt.DataColumn(name="gge", units="gallons"),
                # make/model/year intentionally omitted
            )


class TestModelInfoDefaults(TestCase):
    def test_optional_fields_default_none(self):
        info = ModelInfo(
            model_id=ModelId("a", "b", 2020, "rf_default", 1),
            estimator_type="ONNXEstimator",
            feature_names=["speed_mph"],
            target_names=["gge"],
            powertrain_type="ICE",
            vehicle_description="x",
        )
        self.assertEqual(info.architecture_tag, "unknown")
        self.assertIsNone(info.fuel_type)
        # nested model_id also accepts a dict
        info2 = ModelInfo.model_validate(info.model_dump(mode="json"))
        self.assertEqual(info2, info)


class TestSchemaHardBreak(TestCase):
    def test_old_schema_version_rejected(self):
        stale = {
            "schema_version": SCHEMA_VERSION - 1,
            "estimator_type": "ONNXEstimator",
        }
        with self.assertRaises(ValueError) as ctx:
            _model_from_metadata_and_bytes(stale, b"")
        self.assertIn("schema version", str(ctx.exception).lower())
