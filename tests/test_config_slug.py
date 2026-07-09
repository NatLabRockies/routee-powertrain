from __future__ import annotations

import shutil
from pathlib import Path
from unittest import TestCase

import pandas as pd

import routee.powertrain as pt
from routee.powertrain.core.metadata import SCHEMA_VERSION_STRING
from routee.powertrain.registry.local import LocalRegistry
from routee.powertrain.registry.model_id import ModelId, ModelKey
from routee.powertrain.registry.slug import (
    architecture_short_code,
    assert_metadata_matches_id,
    derive_config_slug,
    derive_vehicle_slug,
)
from routee.powertrain.validation.errors import EstimatorErrors, ModelErrors
from routee.powertrain.trainers.sklearn_random_forest import (
    SklearnRandomForestTrainer,
)

this_dir = Path(__file__).parent


def _config(variant: str | None = None) -> pt.ModelConfig:
    return pt.ModelConfig(
        vehicle_description="Test Model",
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
        make="Test",
        model="Sedan",
        year=2024,
        variant=variant,
    )


def _metadata(**config_overrides) -> pt.Metadata:
    """Build metadata from a config without training — slug derivation only
    reads identity fields, so a placeholder errors object suffices."""
    config = _config().model_copy(update=config_overrides)
    return pt.Metadata.from_config(
        config,
        errors=ModelErrors(estimator_errors=EstimatorErrors(error_by_target={})),
        estimator_type="ONNXEstimator",
        model_file="model.onnx",
        architecture_tag="random_forest",
    )


class TestDeriveVehicleSlug(TestCase):
    def test_model_plus_family(self) -> None:
        # The default test config is ICE.
        self.assertEqual(derive_vehicle_slug(_metadata()), "sedan_ice")
        self.assertEqual(
            derive_vehicle_slug(_metadata(powertrain_type=pt.PowertrainType.BEV)),
            "sedan_bev",
        )

    def test_phev_modes_collapse_to_one_family(self) -> None:
        # Charge-depleting/charge-sustaining models describe the same vehicle;
        # that split lives in config.variant, not the vehicle identity.
        cd = _metadata(powertrain_type=pt.PowertrainType.PHEV_EV_MODE)
        cs = _metadata(powertrain_type=pt.PowertrainType.PHEV_HEV_MODE)
        self.assertEqual(derive_vehicle_slug(cd), "sedan_phev")
        self.assertEqual(derive_vehicle_slug(cs), "sedan_phev")

    def test_undefined_powertrain_treated_as_unset(self) -> None:
        metadata = _metadata(powertrain_type=pt.PowertrainType.UNDEFINED)
        self.assertEqual(derive_vehicle_slug(metadata), "sedan")

    def test_descriptive_fields_do_not_affect_slug(self) -> None:
        metadata = _metadata(
            engine="4cyl", drivetrain="FWD", trim="LE", fuel_type="GASOLINE"
        )
        self.assertEqual(derive_vehicle_slug(metadata), "sedan_ice")

    def test_model_token_sanitization(self) -> None:
        metadata = _metadata(model="Golf 1.5 TSI")
        self.assertEqual(derive_vehicle_slug(metadata), "golf-1.5-tsi_ice")

    def test_model_key_uses_vehicle_slug(self) -> None:
        metadata = _metadata(powertrain_type=pt.PowertrainType.HEV)
        key = ModelKey.from_metadata(metadata)
        self.assertEqual(key.vehicle_slug, "sedan_hev")
        self.assertEqual(
            key.to_path(),
            f"test/sedan_hev/2024/{derive_config_slug(metadata)}",
        )


class TestDeriveConfigSlug(TestCase):
    def setUp(self) -> None:
        data_path = this_dir / "routee-powertrain-test-data" / "sample_train_data.csv"
        self.df = pd.read_csv(data_path)
        self.model = SklearnRandomForestTrainer().train(self.df, _config())

    def test_slug_shape(self) -> None:
        slug = derive_config_slug(self.model.metadata)
        # {arch}_{hash8}; random forest short code is "rf".
        parts = slug.split("_")
        self.assertEqual(parts[0], "rf")
        self.assertEqual(len(parts), 2)
        self.assertEqual(len(parts[-1]), 8)

    def test_slug_is_stable(self) -> None:
        again = SklearnRandomForestTrainer().train(self.df, _config())
        self.assertEqual(
            derive_config_slug(self.model.metadata),
            derive_config_slug(again.metadata),
        )

    def test_variant_changes_slug(self) -> None:
        variant_model = SklearnRandomForestTrainer().train(
            self.df, _config(variant="steady")
        )
        base = derive_config_slug(self.model.metadata)
        with_variant = derive_config_slug(variant_model.metadata)
        self.assertNotEqual(base, with_variant)
        self.assertEqual(with_variant, f"rf_steady_{base.split('_')[-1]}")

    def test_architecture_short_code_fallback(self) -> None:
        self.assertEqual(architecture_short_code("random_forest"), "rf")
        self.assertEqual(architecture_short_code("something_new"), "something_new")


class TestModelIdFromMetadata(TestCase):
    def setUp(self) -> None:
        data_path = this_dir / "routee-powertrain-test-data" / "sample_train_data.csv"
        df = pd.read_csv(data_path)
        self.model = SklearnRandomForestTrainer().train(df, _config())

    def test_round_trips_through_path(self) -> None:
        mid = ModelId.from_metadata(self.model.metadata, version=3)
        self.assertEqual(mid.make, "test")
        self.assertEqual(mid.vehicle_slug, "sedan_ice")
        self.assertEqual(mid.year, 2024)
        self.assertEqual(mid.config_slug, derive_config_slug(self.model.metadata))
        self.assertEqual(mid.version, 3)
        self.assertEqual(ModelId.from_path(mid.to_path()), mid)

    def test_key_is_version_less(self) -> None:
        key = ModelKey.from_metadata(self.model.metadata)
        slug = derive_config_slug(self.model.metadata)
        self.assertEqual(key.to_path(), f"test/sedan_ice/2024/{slug}")
        # The same key yields different ModelIds per version.
        v1 = ModelId.from_key(key, 1)
        v2 = ModelId.from_key(key, 2)
        self.assertEqual(v1.key, v2.key)
        self.assertNotEqual(v1, v2)
        self.assertEqual(v1.key, key)

    def test_key_round_trips_through_path(self) -> None:
        key = ModelKey.from_metadata(self.model.metadata)
        self.assertEqual(ModelKey.from_path(key.to_path()), key)


class TestLoadDriftValidation(TestCase):
    def setUp(self) -> None:
        data_path = this_dir / "routee-powertrain-test-data" / "sample_train_data.csv"
        df = pd.read_csv(data_path)
        self.model = SklearnRandomForestTrainer().train(df, _config())
        self.registry_root = Path("tmp_drift_registry")
        self.model_id = self.model.save_to_registry(
            registry_root=self.registry_root, version=1
        )

    def tearDown(self) -> None:
        if self.registry_root.exists():
            shutil.rmtree(self.registry_root)

    def test_loaded_model_carries_key(self) -> None:
        loaded = LocalRegistry(self.registry_root).load(self.model_id)
        self.assertEqual(loaded.key, self.model_id.key)

    def test_freshly_trained_model_has_key_but_no_version(self) -> None:
        # The version-less identity is always available, even before any registry
        # placement — version is not part of the model's identity.
        self.assertEqual(self.model.key, self.model_id.key)
        self.assertEqual(self.model.key, ModelKey.from_metadata(self.model.metadata))

    def test_republish_same_model_is_idempotent(self) -> None:
        # Re-publishing an identical model (same model_digest) resolves to the
        # existing version instead of minting a duplicate.
        second = self.model.save_to_registry(registry_root=self.registry_root)
        self.assertEqual(second, self.model_id)

    def test_next_version_auto_increments(self) -> None:
        # A genuinely different model under the same key gets the next version.
        data_path = this_dir / "routee-powertrain-test-data" / "sample_train_data.csv"
        df = pd.read_csv(data_path)
        retrained = SklearnRandomForestTrainer().train(
            df.sample(frac=0.8, random_state=7).reset_index(drop=True), _config()
        )
        self.assertNotEqual(retrained.digest, self.model.digest)
        second = retrained.save_to_registry(registry_root=self.registry_root)
        self.assertEqual(second.version, 2)

    def test_drift_raises_on_load(self) -> None:
        # Hand-edit the on-disk metadata so its feature set no longer matches the
        # slug frozen into the path — loading must surface the drift loudly.
        model_dir = self.registry_root / SCHEMA_VERSION_STRING / self.model_id.to_path()
        meta_path = model_dir / "metadata.json"
        meta = meta_path.read_text().replace("speed_mph", "velocity_mph")
        meta_path.write_text(meta)

        with self.assertRaises(ValueError) as ctx:
            LocalRegistry(self.registry_root).load(self.model_id)
        self.assertIn("config_slug", str(ctx.exception))

    def test_vehicle_slug_drift_raises_on_load(self) -> None:
        # Editing an identity field that feeds the vehicle_slug (the powertrain
        # family) makes the derived slug disagree with the path — loading must
        # raise. Descriptive fields like engine are correctable without this.
        import json

        model_dir = self.registry_root / SCHEMA_VERSION_STRING / self.model_id.to_path()
        meta_path = model_dir / "metadata.json"
        metadata_dict = json.loads(meta_path.read_text())
        metadata_dict["vehicle"]["powertrain_type"] = "BEV"
        meta_path.write_text(json.dumps(metadata_dict))

        with self.assertRaises(ValueError) as ctx:
            LocalRegistry(self.registry_root).load(self.model_id)
        self.assertIn("vehicle_slug", str(ctx.exception))

    def test_assert_helper_passes_for_consistent_pair(self) -> None:
        # Should not raise.
        assert_metadata_matches_id(self.model.metadata, self.model_id)
