from __future__ import annotations

import contextlib
import csv
import io
import json
import shutil
import unittest
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd

import routee.powertrain as pt
from routee.powertrain.core.model_config import PredictMethod
from routee.powertrain.estimators.estimator_interface import Estimator
from routee.powertrain.cli import main
from routee.powertrain.validation.physics import (
    check_model,
    G,
    LB_TO_KG,
    MI_TO_M,
    MPH_TO_MS,
    PhysicsReport,
    check_physics,
    normalize_units,
)

this_dir = Path(__file__).parent

MASS_LBS = 3800.0
MASS_KG = MASS_LBS * LB_TO_KG
J_TO_KWH = 1.0 / 3.6e6

# A plausible driveline for the reference vehicle. These are the *true* values
# of the synthetic model below, and are deliberately tighter than the generous
# constants the checks assume — a correct model must pass against loose bounds.
ETA_DRIVE = 0.85
ETA_REGEN = 0.70
CRR = 0.008
CDA = 0.60
AIR_DENSITY = 1.225
ACCESSORY_W = 500.0


class StubEstimator(Estimator):
    """An estimator backed by a plain Python function instead of a binary.

    Lets a test state exactly what energy function is being checked, so a check
    failure is unambiguously about the check rather than about a trained model.
    """

    file_extension = ".stub"

    def __init__(self, fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]):
        self.fn = fn

    def to_bytes(self) -> bytes:
        return b""

    @classmethod
    def from_bytes(cls, data: bytes) -> Estimator:
        raise NotImplementedError

    def predict(self, links_df: pd.DataFrame, config: pt.ModelConfig) -> pd.DataFrame:
        names = config.feature_set.feature_name_list
        speed = np.asarray(links_df[names[0]], dtype=float)
        grade_column = config.feature_set.features[1]
        grade = np.asarray(links_df[grade_column.name], dtype=float)
        if normalize_units(grade_column.units) != "percent":
            grade = grade * 100.0
        distance = np.asarray(links_df[config.distance.name], dtype=float)
        target = config.target.targets[0].name
        return pd.DataFrame(
            {target: self.fn(speed, grade, distance)}, index=links_df.index
        )


def correct_energy(
    speed: np.ndarray, grade: np.ndarray, distance: np.ndarray
) -> np.ndarray:
    """A physically consistent energy function, in kWh.

    Road load and grade are paid through the driveline when the net demand is
    positive and recovered at the regeneration efficiency when it is negative;
    accessories are billed per unit time.
    """
    velocity = np.maximum(speed * MPH_TO_MS, 1e-6)
    meters = distance * MI_TO_M
    road = (CRR * MASS_KG * G + 0.5 * AIR_DENSITY * CDA * velocity**2) * meters
    potential = MASS_KG * G * meters * grade / 100.0
    demand = road + potential
    tractive = np.where(demand >= 0, demand / ETA_DRIVE, demand * ETA_REGEN)
    accessory = ACCESSORY_W * meters / velocity
    return (tractive + accessory) * J_TO_KWH


def over_refunding_energy(
    speed: np.ndarray, grade: np.ndarray, distance: np.ndarray
) -> np.ndarray:
    """The reported defect: descents return more than the hill ever held."""
    base = correct_energy(speed, grade, distance)
    meters = distance * MI_TO_M
    potential = MASS_KG * G * meters * grade / 100.0
    extra = np.where(grade < 0, 2.5 * potential * J_TO_KWH, 0.0)
    return base + extra


def make_config(
    grade_units: str = "percent",
    predict_method: PredictMethod = PredictMethod.RATE,
    with_grade: bool = True,
    mass_lbs: float | None = None,
    extra_features: List[pt.DataColumn] | None = None,
) -> pt.ModelConfig:
    features = [pt.DataColumn(name="speed_mph", units="mph")]
    if with_grade:
        name = "grade_percent" if grade_units == "percent" else "grade_dec"
        features.append(
            pt.DataColumn(
                name=name,
                units=grade_units,
                constraints=pt.Constraints(lower=-20.0, upper=20.0),
            )
        )
    features.extend(extra_features or [])
    return pt.ModelConfig(
        vehicle_description="Reference vehicle",
        powertrain_type=pt.PowertrainType.BEV,
        feature_set=pt.FeatureSet(features=features),
        distance=pt.DataColumn(name="distance", units="miles"),
        target=pt.TargetSet(targets=[pt.DataColumn(name="kwh", units="kilowatt-hour")]),
        make="test",
        model="reference",
        year=2024,
        mass_lbs=mass_lbs,
        predict_method=predict_method,
        real_world_adjustment_factor=1.0,
    )


def failing(report: PhysicsReport) -> List[str]:
    return [c.name for c in report.checks if c.status == "fail"]


class TestPhysicsChecks(unittest.TestCase):
    def test_correct_model_passes_every_check(self):
        """The check must not fire on a model that is already right.

        This is the property that decides whether the report is worth reading:
        a validation with false positives gets ignored.
        """
        report = check_physics(
            StubEstimator(correct_energy), make_config(mass_lbs=MASS_LBS)
        )
        self.assertTrue(report.passed, f"unexpected failures: {failing(report)}")
        self.assertEqual(report.mass_source, "metadata")

    def test_over_refunding_model_fails_the_right_checks(self):
        report = check_physics(
            StubEstimator(over_refunding_energy), make_config(mass_lbs=MASS_LBS)
        )
        self.assertIn("round_trip_convexity", failing(report))
        self.assertIn("regen_ceiling", failing(report))
        self.assertGreater(report.diagnostics.implied_eta_regen, 1.0)
        self.assertTrue(report.diagnostics.regen_exceeds_unity)

    def test_decimal_and_percent_grade_agree(self):
        """A model expressing grade as a decimal must be read the same way.

        Missing the conversion would scale every grade by 100 and turn correct
        models into failures.
        """
        percent = check_physics(
            StubEstimator(correct_energy),
            make_config(grade_units="percent", mass_lbs=MASS_LBS),
        )
        decimal = check_physics(
            StubEstimator(correct_energy),
            make_config(grade_units="decimal", mass_lbs=MASS_LBS),
        )
        self.assertTrue(percent.passed)
        self.assertTrue(decimal.passed)
        self.assertAlmostEqual(
            percent.diagnostics.implied_eta_drive,
            decimal.diagnostics.implied_eta_drive,
            places=6,
        )

    def test_raw_method_does_not_hold_out_the_distance_column(self):
        """Under RAW the distance column is also a feature, and must be swept."""
        config = make_config(predict_method=PredictMethod.RAW, mass_lbs=MASS_LBS)
        report = check_physics(StubEstimator(correct_energy), config)
        self.assertNotIn("distance", report.diagnostics.held_features)
        self.assertTrue(report.passed, f"unexpected failures: {failing(report)}")

    def test_no_grade_feature_reports_not_applicable(self):
        config = make_config(with_grade=False)

        def speed_only(speed, grade, distance):
            return correct_energy(speed, np.zeros_like(speed), distance)

        class SpeedOnlyStub(StubEstimator):
            def predict(self, links_df, config):
                speed = np.asarray(links_df["speed_mph"], dtype=float)
                distance = np.asarray(links_df[config.distance.name], dtype=float)
                return pd.DataFrame(
                    {"kwh": speed_only(speed, None, distance)}, index=links_df.index
                )

        report = check_physics(SpeedOnlyStub(correct_energy), config)
        statuses = {c.name: c.status for c in report.checks}
        self.assertEqual(statuses["round_trip_convexity"], "not_applicable")
        self.assertEqual(statuses["monotone_in_grade"], "not_applicable")
        self.assertEqual(statuses["flat_energy_positive"], "pass")

    def test_missing_mass_skips_rather_than_guesses(self):
        report = check_physics(StubEstimator(correct_energy), make_config())
        statuses = {c.name: c.status for c in report.checks}
        for name in (
            "climb_floor",
            "regen_ceiling",
            "absolute_ceiling",
            "absolute_floor",
        ):
            self.assertEqual(statuses[name], "not_applicable")
            self.assertEqual(report.check_map[name].reason, "no vehicle mass available")
        self.assertEqual(report.mass_source, "unavailable")
        # The constant-free checks still run without any vehicle knowledge.
        self.assertEqual(statuses["round_trip_convexity"], "pass")

    def test_clamped_output_still_reports_a_bind_rate(self):
        """Once guardrails ship, the violation rate goes to zero by construction.

        The bind rate is what stays informative, so a clamped model must still
        report a nonzero one.
        """
        config = make_config(mass_lbs=MASS_LBS)

        def clamped(speed, grade, distance):
            energy = over_refunding_energy(speed, grade, distance)
            flat = correct_energy(speed, np.zeros_like(grade), distance)
            potential = MASS_KG * G * distance * MI_TO_M * grade / 100.0 * J_TO_KWH
            return np.maximum(energy, flat - 0.75 * np.abs(potential))

        raw = check_physics(StubEstimator(over_refunding_energy), config)
        self.assertGreater(raw.check_map["regen_ceiling"].violation_rate, 0.0)

        clamped_report = check_physics(StubEstimator(clamped), config)
        check = clamped_report.check_map["regen_ceiling"]
        self.assertEqual(check.violation_rate, 0.0)
        self.assertIsNotNone(check.would_bind_rate)

    def test_report_round_trips_through_json(self):
        report = check_physics(
            StubEstimator(over_refunding_energy), make_config(mass_lbs=MASS_LBS)
        )
        restored = PhysicsReport.model_validate(report.model_dump(mode="json"))
        self.assertEqual(len(restored.checks), len(report.checks))
        self.assertEqual(restored.passed, report.passed)
        self.assertIn("round_trip_convexity", failing(restored))


class TestAuxiliaryFeatureHandling(unittest.TestCase):
    """Features beyond speed and grade — turn angle, sinuosity, link time.

    These decide whether the sweep asks the model a coherent question. A link
    whose traversal time contradicts its speed and distance, or whose sinuosity
    is below the straight-line minimum, is not a link any vehicle could drive.
    """

    def _capture(self, extra_features: List[pt.DataColumn], **kwargs):
        """Run the checks and return the input frames the estimator was given."""
        seen: List[pd.DataFrame] = []

        class CapturingStub(StubEstimator):
            def predict(self, links_df, config):
                seen.append(links_df.copy())
                return super().predict(links_df, config)

        config = make_config(mass_lbs=MASS_LBS, extra_features=extra_features)
        check_physics(CapturingStub(correct_energy), config, **kwargs)
        return pd.concat(seen, ignore_index=True)

    def test_bearing_delta_is_held_straight_not_at_the_midpoint(self):
        """A 0-180 range midpoint is a 90-degree turn on every link."""
        frame = self._capture(
            [
                pt.DataColumn(
                    name="link_abs_bearing_delta",
                    units="degrees",
                    constraints=pt.Constraints(lower=0.0, upper=180.0),
                )
            ]
        )
        self.assertTrue((frame["link_abs_bearing_delta"] == 0.0).all())

    def test_sinuosity_is_held_at_the_straight_line_minimum(self):
        """Sinuosity below 1.0 describes a link shorter than the straight line."""
        frame = self._capture(
            [
                pt.DataColumn(
                    name="link_sinuosity",
                    units="ratio",
                    constraints=pt.Constraints(lower=1.0, upper=None),
                )
            ]
        )
        self.assertTrue((frame["link_sinuosity"] == 1.0).all())

    def test_link_time_is_derived_from_the_sweep_not_held(self):
        """Traversal time is distance over speed, so holding it is incoherent."""
        config = make_config(
            mass_lbs=MASS_LBS,
            extra_features=[
                pt.DataColumn(
                    name="link_time",
                    units="seconds",
                    constraints=pt.Constraints(lower=0.0, upper=None),
                )
            ],
        )
        report = check_physics(StubEstimator(correct_energy), config)
        self.assertEqual(report.diagnostics.derived_features, ["link_time"])
        self.assertNotIn("link_time", report.diagnostics.held_features)

        frame = self._capture(
            [
                pt.DataColumn(
                    name="link_time",
                    units="seconds",
                    constraints=pt.Constraints(lower=0.0, upper=None),
                )
            ]
        )
        expected = frame["distance"] / frame["speed_mph"] * 3600.0
        np.testing.assert_allclose(frame["link_time"], expected, rtol=1e-9)

    def test_minutes_are_scaled_not_taken_as_seconds(self):
        frame = self._capture([pt.DataColumn(name="link_time", units="minutes")])
        expected = frame["distance"] / frame["speed_mph"] * 60.0
        np.testing.assert_allclose(frame["link_time"], expected, rtol=1e-9)

    def test_reference_data_beats_the_neutral_table_but_is_clamped(self):
        column = pt.DataColumn(
            name="link_sinuosity",
            units="ratio",
            constraints=pt.Constraints(lower=1.0, upper=None),
        )
        reference = pd.DataFrame({"link_sinuosity": [1.4, 1.6, 1.5]})
        frame = self._capture([column], reference_df=reference)
        self.assertTrue((frame["link_sinuosity"] == 1.5).all())

        # A median outside the declared range is pulled back inside it.
        below = pd.DataFrame({"link_sinuosity": [0.1, 0.2, 0.3]})
        frame = self._capture([column], reference_df=below)
        self.assertTrue((frame["link_sinuosity"] == 1.0).all())

    def test_an_uninterpretable_feature_is_held_and_reported(self):
        report = check_physics(
            StubEstimator(correct_energy),
            make_config(
                mass_lbs=MASS_LBS,
                extra_features=[pt.DataColumn(name="mystery", units="widgets")],
            ),
        )
        self.assertEqual(report.diagnostics.held_features["mystery"], 0.0)
        self.assertTrue(any("mystery" in note for note in report.notes))
        # An unknown feature must not stop the physical checks from running.
        self.assertEqual(report.check_map["round_trip_convexity"].status, "pass")


class TestPhysicsCLI(unittest.TestCase):
    """The report is produced by the standalone tool, not by training."""

    def setUp(self):
        self.out_path = Path("tmp")
        self.out_path.mkdir(exist_ok=True)
        self.bundled = (
            "routee/powertrain/resources/bundled_registry/v2"
            "/toyota/rav4_xle_ice/2022/rf_fe510e40/v1"
        )

    def tearDown(self):
        shutil.rmtree(self.out_path, ignore_errors=True)

    def _run(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = main(["validate-physics", *argv])
        return code, buf.getvalue()

    def test_reports_on_a_model_path(self):
        code, out = self._run(self.bundled)
        self.assertEqual(code, 0)
        self.assertIn("Physics validation", out)
        self.assertIn("round_trip_convexity", out)
        self.assertIn("1 model(s) checked", out)

    def test_requires_a_target(self):
        code, out = self._run()
        self.assertEqual(code, 1)
        self.assertIn("--all", out)

    def test_writes_json_and_csv(self):
        json_path = self.out_path / "report.json"
        csv_path = self.out_path / "report.csv"
        code, _ = self._run(
            self.bundled,
            "--summary-only",
            "--json",
            str(json_path),
            "--csv",
            str(csv_path),
        )
        self.assertEqual(code, 0)

        payload = json.loads(json_path.read_text())
        report = PhysicsReport.model_validate(next(iter(payload.values())))
        self.assertTrue(report.checks)

        rows = list(csv.DictReader(csv_path.read_text().splitlines()))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model_key"], "toyota/rav4_xle_ice/2022/rf_fe510e40")

    def test_fail_on_violation_gates_only_when_asked(self):
        # The bundled model violates round-trip convexity, so this is a real
        # failure rather than a contrived one.
        self.assertEqual(self._run(self.bundled)[0], 0)
        self.assertEqual(self._run(self.bundled, "--fail-on-violation")[0], 2)

    def test_an_unloadable_target_does_not_abandon_the_sweep(self):
        code, out = self._run(self.bundled, "no/such/model/2024/rf_nope/v1")
        self.assertEqual(code, 0)
        self.assertIn("could not load", out)
        self.assertIn("1 model(s) checked", out)

    def test_a_model_carries_no_physics_field(self):
        """The report lives in the tool, not in the artifact."""
        model = pt.load_model(self.bundled)
        self.assertFalse(hasattr(model.metadata, "physics"))
        report = check_model(model)
        self.assertTrue(report.checks)


class TestTrainingIsUnaffected(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv(
            this_dir / "routee-powertrain-test-data" / "sample_train_data.csv"
        )

    def test_training_does_not_run_the_physical_checks(self):
        from routee.powertrain.trainers.sklearn_random_forest import (
            SklearnRandomForestTrainer,
        )

        config = pt.ModelConfig(
            vehicle_description="Test Model",
            powertrain_type=pt.PowertrainType.ICE,
            feature_set=pt.FeatureSet(
                features=[
                    pt.DataColumn(name="speed_mph", units="mph"),
                    pt.DataColumn(name="grade_dec", units="decimal"),
                ]
            ),
            distance=pt.DataColumn(name="miles", units="miles"),
            target=pt.TargetSet(
                targets=[
                    pt.DataColumn(name="gallons_fastsim", units="gallons_gasoline")
                ]
            ),
            make="test",
            model="model",
            year=2024,
        )
        model = SklearnRandomForestTrainer().train(self.df, config)
        self.assertFalse(hasattr(model.metadata, "physics"))
        # The trained model is still checkable on demand.
        self.assertTrue(check_model(model).checks)


if __name__ == "__main__":
    unittest.main()
