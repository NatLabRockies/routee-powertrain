"""The output guardrail: ``Model.predict`` clips into the physical envelope.

Every model here is a ``StubEstimator`` over a known energy function, so what
the guardrail changes is exactly what the test says it changes.
"""

from __future__ import annotations

import unittest
from typing import Callable

import numpy as np
import pandas as pd

import routee.powertrain as pt
from routee.powertrain.core.digest import compute_model_digest
from routee.powertrain.core.metadata import Metadata
from routee.powertrain.core.model import Model
from routee.powertrain.core.model_config import Contract
from routee.powertrain.io.to_lookup_table import to_lookup_table
from routee.powertrain.validation.errors import EstimatorErrors, ModelErrors
from routee.powertrain.validation.physics import (
    KM_TO_MI,
    KPH_TO_MPH,
    check_physics,
    physical_bounds,
)
from tests.mock_resources import mock_model
from tests.test_physics import (
    MASS_LBS,
    StubEstimator,
    correct_energy,
    failing,
    make_config,
    over_refunding_energy,
)

EnergyFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def runaway_energy(speed: np.ndarray, grade: np.ndarray, distance: np.ndarray):
    """A model that has left the data: fifty times the true energy."""
    return 50.0 * correct_energy(speed, grade, distance)


def make_model(fn: EnergyFn, config: pt.ModelConfig) -> Model:
    errors = ModelErrors(estimator_errors=EstimatorErrors(error_by_target={}))
    metadata = Metadata.from_config(
        config, errors=errors, estimator_type="StubEstimator", model_file="model.stub"
    )
    return Model(StubEstimator(fn), metadata)


def links(**overrides) -> pd.DataFrame:
    """A small frame of links spanning the sweep, in the reference model's units."""
    speed, grade, distance = (
        a.ravel()
        for a in np.meshgrid(
            [5.0, 30.0, 70.0], [-8.0, -2.0, 0.0, 4.0], [0.05, 0.5, 1.0], indexing="ij"
        )
    )
    frame = pd.DataFrame(
        {"speed_mph": speed, "grade_percent": grade, "distance": distance},
        index=np.arange(100, 100 + speed.size),
    )
    for name, value in overrides.items():
        frame[name] = value
    return frame


def raw(model: Model, frame: pd.DataFrame) -> np.ndarray:
    return model.estimator.predict(frame, model.metadata.config)["kwh"].to_numpy()


class TestEnvelopeClamp(unittest.TestCase):
    def test_correct_model_is_untouched(self):
        model = make_model(correct_energy, make_config(mass_lbs=MASS_LBS))
        frame = links()
        out = model.predict(frame)
        np.testing.assert_array_equal(out["kwh"].to_numpy(), raw(model, frame))
        self.assertTrue(out.index.equals(frame.index))

    def test_electric_floor_is_reported_but_not_enforced(self):
        """A link average hides the speed a slow link really braked from, so
        the floor would clip correct regeneration; it stays a check only."""
        model = make_model(over_refunding_energy, make_config(mass_lbs=MASS_LBS))
        frame = links()
        before = raw(model, frame)
        floor, _ = physical_bounds(frame, model.metadata.config)["kwh"]
        self.assertTrue((before < floor).any(), "the defect must show in the frame")
        np.testing.assert_array_equal(model.predict(frame)["kwh"].to_numpy(), before)

    def test_runaway_is_cut_to_the_ceiling(self):
        model = make_model(runaway_energy, make_config(mass_lbs=MASS_LBS))
        frame = links()
        _, ceiling = physical_bounds(frame, model.metadata.config)["kwh"]
        after = model.predict(frame)["kwh"].to_numpy()
        self.assertTrue((raw(model, frame) > ceiling).any())
        np.testing.assert_allclose(after, np.minimum(raw(model, frame), ceiling))

    def test_none_disables_the_clamp(self):
        config = make_config(mass_lbs=MASS_LBS).model_copy(
            update={"output_guardrail": "none"}
        )
        model = make_model(runaway_energy, config)
        frame = links()
        self.assertEqual(model.metadata.contract.output_guardrail, "none")
        np.testing.assert_array_equal(
            model.predict(frame)["kwh"].to_numpy(), raw(model, frame)
        )

    def test_runtime_disable_through_metadata(self):
        model = make_model(runaway_energy, make_config(mass_lbs=MASS_LBS))
        frame = links()
        model.metadata.contract.output_guardrail = "none"
        np.testing.assert_array_equal(
            model.predict(frame)["kwh"].to_numpy(), raw(model, frame)
        )

    def test_combustion_floor_is_zero(self):
        config = make_config(mass_lbs=MASS_LBS).model_copy(
            update={
                "powertrain_type": pt.PowertrainType.ICE,
                "target": pt.TargetSet(
                    targets=[pt.DataColumn(name="kwh", units="gallons gasoline")]
                ),
            }
        )

        def negative_on_descents(speed, grade, distance):
            gallons = correct_energy(speed, grade, distance) * 3.6e6 / 1.2132e8
            return np.where(grade < 0, -0.01, gallons)

        model = make_model(negative_on_descents, config)
        frame = links()
        out = model.predict(frame)["kwh"].to_numpy()
        descent = frame["grade_percent"].to_numpy() < 0
        self.assertTrue((out[descent] == 0.0).all())
        # Exactly +0.0, not -0.0, so it prints as zero.
        self.assertTrue(all(np.copysign(1.0, v) > 0 for v in out[descent]))
        np.testing.assert_array_equal(out[~descent], raw(model, frame)[~descent])

    def test_bev_in_gallon_equivalents_keeps_its_regeneration(self):
        """A BEV target in GGE is electricity: descents may go negative."""
        config = make_config(mass_lbs=MASS_LBS).model_copy(
            update={
                "target": pt.TargetSet(
                    targets=[pt.DataColumn(name="kwh", units="gallons_gasoline")]
                ),
            }
        )
        self.assertEqual(config.powertrain_type, pt.PowertrainType.BEV)

        def correct_energy_gge(speed, grade, distance):
            return correct_energy(speed, grade, distance) * 3.6e6 / 1.2132e8

        model = make_model(correct_energy_gge, config)
        frame = links()
        before = raw(model, frame)
        self.assertTrue((before < 0).any(), "the reference model regenerates")
        np.testing.assert_array_equal(model.predict(frame)["kwh"].to_numpy(), before)
        floor, _ = physical_bounds(frame, config)["kwh"]
        self.assertTrue((floor < 0).any())

    def test_no_mass_means_no_clamp(self):
        model = make_model(runaway_energy, make_config())
        frame = links()
        self.assertEqual(physical_bounds(frame, model.metadata.config), {})
        np.testing.assert_array_equal(
            model.predict(frame)["kwh"].to_numpy(), raw(model, frame)
        )

    def test_distance_in_kilometers_gives_the_same_band(self):
        mph_config = make_config(mass_lbs=MASS_LBS)
        km_config = mph_config.model_copy(
            update={"distance": pt.DataColumn(name="distance", units="kilometers")}
        )
        frame = links()
        km_frame = frame.copy()
        km_frame["distance"] = frame["distance"] / KM_TO_MI
        for side in (0, 1):
            np.testing.assert_allclose(
                physical_bounds(km_frame, km_config)["kwh"][side],
                physical_bounds(frame, mph_config)["kwh"][side],
            )

    def test_per_row_mass_feature_beats_metadata_mass(self):
        config = make_config(
            mass_lbs=MASS_LBS,
            extra_features=[pt.DataColumn(name="mass_lbs", units="pounds")],
        )
        frame = links(mass_lbs=2.0 * MASS_LBS)
        heavy_floor, heavy_ceiling = physical_bounds(frame, config)["kwh"]
        light_floor, light_ceiling = physical_bounds(links(mass_lbs=MASS_LBS), config)[
            "kwh"
        ]
        self.assertTrue((heavy_ceiling > light_ceiling).all())
        self.assertTrue((heavy_floor <= light_floor).all())

    def test_bad_rows_pass_through(self):
        model = make_model(runaway_energy, make_config(mass_lbs=MASS_LBS))
        frame = links()
        frame.loc[frame.index[0], "speed_mph"] = np.nan
        frame.loc[frame.index[1], "distance"] = -1.0
        before = raw(model, frame)
        after = model.predict(frame)["kwh"].to_numpy()
        floor, ceiling = physical_bounds(frame, model.metadata.config)["kwh"]
        self.assertTrue(np.isnan(floor[:2]).all() and np.isnan(ceiling[:2]).all())
        # A row the band cannot be evaluated on keeps whatever the model said.
        np.testing.assert_array_equal(after[:2], before[:2])
        self.assertTrue((after[2:] <= ceiling[2:]).all())

    def test_nan_prediction_stays_nan(self):
        def sometimes_nan(speed, grade, distance):
            out = correct_energy(speed, grade, distance)
            out[0] = np.nan
            return out

        model = make_model(sometimes_nan, make_config(mass_lbs=MASS_LBS))
        out = model.predict(links())["kwh"].to_numpy()
        self.assertTrue(np.isnan(out[0]))
        self.assertTrue(np.isfinite(out[1:]).all())

    def test_extra_columns_and_factor(self):
        class WithStd(StubEstimator):
            def predict(self, links_df, config):
                out = super().predict(links_df, config)
                out["kwh_std"] = 123.0
                return out

        config = make_config(mass_lbs=MASS_LBS).model_copy(
            update={"real_world_adjustment_factor": 1.3958}
        )
        errors = ModelErrors(estimator_errors=EstimatorErrors(error_by_target={}))
        metadata = Metadata.from_config(
            config,
            errors=errors,
            estimator_type="StubEstimator",
            model_file="model.stub",
        )
        model = Model(WithStd(runaway_energy), metadata)
        frame = links()
        out = model.predict(frame)
        _, ceiling = physical_bounds(frame, config)["kwh"]
        self.assertEqual(list(out.columns), ["kwh", "kwh_std"])
        self.assertTrue((out["kwh_std"] == 123.0).all())
        # Clipped first, then the factor, so the delivered value may exceed the
        # band by exactly that factor and no more.
        np.testing.assert_allclose(
            out["kwh"].to_numpy(),
            np.minimum(raw(model, frame), ceiling) * 1.3958,
        )

    def test_bundled_model_stays_within_its_band(self):
        model = mock_model()
        config = model.metadata.config
        self.assertEqual(config.output_guardrail, "envelope")
        frame = pt.load_sample_route().rename(
            columns={"grade_percent": "grade_pct", "distance": "distance_mi"}
        )
        out = model.predict(frame)
        _, ceiling = physical_bounds(frame, config)["fuel_gge"]
        factor = config.real_world_adjustment_factor
        self.assertTrue((out["fuel_gge"] >= 0.0).all())
        self.assertTrue((out["fuel_gge"] <= ceiling * factor).all())
        self.assertTrue(out.index.equals(frame.index))


class TestKphSpeed(unittest.TestCase):
    """A model taking kilometres per hour is judged and bounded as its mph twin."""

    def test_kph_twin_matches_mph(self):
        mph_config = make_config(mass_lbs=MASS_LBS)
        kph_config = mph_config.model_copy(
            update={
                "feature_set": pt.FeatureSet(
                    features=[
                        pt.DataColumn(name="speed_kph", units="kph"),
                        mph_config.feature_set.features[1],
                    ]
                )
            }
        )

        def kph_energy(speed_kph, grade, distance):
            return correct_energy(speed_kph * KPH_TO_MPH, grade, distance)

        self.assertEqual(
            failing(check_physics(StubEstimator(kph_energy), kph_config)), []
        )

        frame = links()
        kph_frame = frame.rename(columns={"speed_mph": "speed_kph"})
        kph_frame["speed_kph"] = frame["speed_mph"] / KPH_TO_MPH
        for side in (0, 1):
            np.testing.assert_allclose(
                physical_bounds(kph_frame, kph_config)["kwh"][side],
                physical_bounds(frame, mph_config)["kwh"][side],
            )


class TestLookupTable(unittest.TestCase):
    def test_table_is_bounded(self):
        model = make_model(runaway_energy, make_config(mass_lbs=MASS_LBS))
        parameters = [
            {
                "feature_name": "speed_mph",
                "lower_bound": 10.0,
                "upper_bound": 60.0,
                "n_samples": 6,
            },
            {
                "feature_name": "grade_percent",
                "lower_bound": -8.0,
                "upper_bound": 8.0,
                "n_samples": 5,
            },
        ]
        table = to_lookup_table(model, parameters, energy_target="kwh")
        grid = table[["speed_mph", "grade_percent"]].copy()
        grid["distance"] = 1.0
        _, ceiling = physical_bounds(grid, model.metadata.config)["kwh"]
        rate = table["kwh_per_distance"].to_numpy()
        self.assertTrue((rate <= ceiling + 1e-12).all())
        self.assertTrue(
            (
                runaway_energy(
                    grid["speed_mph"].to_numpy(),
                    grid["grade_percent"].to_numpy(),
                    grid["distance"].to_numpy(),
                )
                > ceiling
            ).any()
        )


class TestMetadataField(unittest.TestCase):
    def test_old_metadata_loads_as_envelope(self):
        config = make_config(mass_lbs=MASS_LBS)
        dumped = Contract.from_config(config).model_dump(mode="json")
        self.assertEqual(dumped["output_guardrail"], "envelope")
        del dumped["output_guardrail"]
        self.assertEqual(Contract.model_validate(dumped).output_guardrail, "envelope")

    def test_field_round_trips_and_does_not_touch_the_digest(self):
        errors = ModelErrors(estimator_errors=EstimatorErrors(error_by_target={}))
        on = Metadata.from_config(
            make_config(mass_lbs=MASS_LBS),
            errors=errors,
            estimator_type="ONNXEstimator",
            model_file="model.onnx",
        )
        on.estimator.estimator_sha256 = "0" * 64
        off = on.model_copy(deep=True)
        off.contract.output_guardrail = "none"
        self.assertEqual(off.config.output_guardrail, "none")
        self.assertEqual(
            Metadata.model_validate(
                off.model_dump(mode="json")
            ).contract.output_guardrail,
            "none",
        )
        self.assertEqual(compute_model_digest(on), compute_model_digest(off))


if __name__ == "__main__":
    unittest.main()
