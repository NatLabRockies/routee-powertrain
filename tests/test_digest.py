import json
import shutil
from pathlib import Path
from typing import Optional
from unittest import TestCase

import pandas as pd

import routee.powertrain as pt
from routee.powertrain.core.digest import (
    compute_model_digest,
    estimator_sha256,
    hash_dataframe,
    normalize_digest,
    stamp_digest,
)
from routee.powertrain.core.metadata import Metadata
from routee.powertrain.io.archive import save_to_registry
from routee.powertrain.registry.local import LocalRegistry
from routee.powertrain.trainers.sklearn_random_forest import (
    SklearnRandomForestTrainer,
)
from routee.powertrain.validation.errors import EstimatorErrors, ModelErrors

this_dir = Path(__file__).parent

GOLDEN_FAKE_BYTES = b"golden fake estimator bytes"
#: Frozen expected digest for ``_golden_metadata()`` + ``GOLDEN_FAKE_BYTES``
#: under digest spec 1. If this test ever fails, the spec-1 payload or its
#: canonicalization drifted — that breaks every digest already published, so
#: fix the drift; do NOT update this constant. A deliberate payload change
#: must ship as digest spec 2 with its own builder.
GOLDEN_MODEL_DIGEST = (
    "sha256:18cf52ec36604fbc003b7d79c3580a4752912475feec5359f6228147200d5c9c"
)


def _golden_config(**overrides) -> pt.ModelConfig:
    fields = dict(
        vehicle_description="Golden Test Vehicle",
        powertrain_type=pt.PowertrainType.ICE,
        feature_set=pt.FeatureSet(
            features=[
                pt.DataColumn(name="speed_mph", units="mph"),
                pt.DataColumn(name="grade_dec", units="decimal"),
            ],
        ),
        distance=pt.DataColumn(name="miles", units="miles"),
        target=pt.TargetSet(targets=[pt.DataColumn(name="gge", units="gallons")]),
        make="testmake",
        model="testmodel",
        year=2024,
        test_size=0.2,
        random_seed=42,
        real_world_adjustment_factor=1.0,
        dataset_name="golden-dataset",
        dataset_hash="ab" * 32,
    )
    fields.update(overrides)
    return pt.ModelConfig.model_validate(fields)


def _golden_metadata(config: Optional[pt.ModelConfig] = None) -> Metadata:
    errors = ModelErrors(estimator_errors=EstimatorErrors(error_by_target={}))
    return Metadata.from_config(
        config if config is not None else _golden_config(),
        errors=errors,
        estimator_type="ONNXEstimator",
        model_file="model.onnx",
        architecture_tag="random_forest",
        trained_date="2026-07-09",
    )


class TestDigestSpec(TestCase):
    """The digest primitives: canonicalization, determinism, sensitivity."""

    def test_golden_digest(self):
        """Fixed metadata + fixed bytes must hash to the frozen spec-1 value."""
        metadata = _golden_metadata()
        stamp_digest(metadata, GOLDEN_FAKE_BYTES)
        self.assertEqual(
            metadata.estimator.estimator_sha256, estimator_sha256(GOLDEN_FAKE_BYTES)
        )
        self.assertEqual(metadata.model_digest, GOLDEN_MODEL_DIGEST)

    def test_stamp_deterministic(self):
        m1 = _golden_metadata()
        m2 = _golden_metadata()
        stamp_digest(m1, GOLDEN_FAKE_BYTES)
        stamp_digest(m2, GOLDEN_FAKE_BYTES)
        self.assertEqual(m1.model_digest, m2.model_digest)
        # re-stamping the same metadata is a no-op
        before = m1.model_digest
        stamp_digest(m1, GOLDEN_FAKE_BYTES)
        self.assertEqual(m1.model_digest, before)

    def test_identity_fields_change_digest(self):
        base = _golden_metadata()
        stamp_digest(base, GOLDEN_FAKE_BYTES)

        changed_configs = {
            "variant": _golden_config(variant="steady"),
            "dataset_hash": _golden_config(dataset_hash="cd" * 32),
            "dataset_name": _golden_config(dataset_name="other-dataset"),
        }
        for label, config in changed_configs.items():
            metadata = _golden_metadata(config)
            stamp_digest(metadata, GOLDEN_FAKE_BYTES)
            self.assertNotEqual(
                metadata.model_digest,
                base.model_digest,
                f"changing {label} should change the digest",
            )

        # different estimator bytes -> different digest (the same-day /
        # different-training-data collision case)
        other_bytes = _golden_metadata()
        stamp_digest(other_bytes, b"different estimator bytes")
        self.assertNotEqual(other_bytes.model_digest, base.model_digest)

        # different trained_date -> different digest
        errors = ModelErrors(estimator_errors=EstimatorErrors(error_by_target={}))
        other_date = Metadata.from_config(
            _golden_config(),
            errors=errors,
            estimator_type="ONNXEstimator",
            model_file="model.onnx",
            architecture_tag="random_forest",
            trained_date="2026-07-10",
        )
        stamp_digest(other_date, GOLDEN_FAKE_BYTES)
        self.assertNotEqual(other_date.model_digest, base.model_digest)

    def test_descriptive_fields_do_not_change_digest(self):
        base = _golden_metadata()
        stamp_digest(base, GOLDEN_FAKE_BYTES)

        config = _golden_config(
            vehicle_description="A different description",
            mass_lbs=3500.0,
            engine="4cyl",
            trim="sport",
        )
        metadata = _golden_metadata(config)
        stamp_digest(metadata, GOLDEN_FAKE_BYTES)
        self.assertEqual(metadata.model_digest, base.model_digest)

    def test_normalize_digest(self):
        bare = "ab" * 32
        self.assertEqual(normalize_digest(bare), f"sha256:{bare}")
        self.assertEqual(normalize_digest(f"sha256:{bare}"), f"sha256:{bare}")
        self.assertEqual(normalize_digest(f"SHA256:{'AB' * 32}"), f"sha256:{bare}")

    def test_digest_validators(self):
        base = _golden_metadata().model_dump(mode="json")
        with self.assertRaises(ValueError):
            Metadata.model_validate({**base, "model_digest": "not-a-digest"})
        bad_estimator = {**base["estimator"], "estimator_sha256": "sha256:" + "ab" * 32}
        with self.assertRaises(ValueError):
            # estimator_sha256 is bare hex; the sha256: prefix is invalid here
            Metadata.model_validate({**base, "estimator": bad_estimator})

    def test_hash_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        self.assertEqual(hash_dataframe(df), hash_dataframe(df.copy()))
        shuffled = df.iloc[::-1].reset_index(drop=True)
        self.assertNotEqual(hash_dataframe(df), hash_dataframe(shuffled))


class TestDigestLifecycle(TestCase):
    """Digest behavior through the train -> save -> load -> publish lifecycle."""

    def setUp(self) -> None:
        data_path = (
            this_dir
            / Path("routee-powertrain-test-data")
            / Path("sample_train_data.csv")
        )
        self.df = pd.read_csv(data_path)
        self.out_path = Path("tmp")
        self.out_path.mkdir(exist_ok=True)
        self.config = pt.ModelConfig(
            vehicle_description="Digest Test Model",
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
        )
        self.model = SklearnRandomForestTrainer().train(self.df, self.config)

    def tearDown(self) -> None:
        if self.out_path.exists():
            shutil.rmtree(self.out_path)

    def test_trained_model_has_digest(self):
        metadata = self.model.metadata
        self.assertIsNotNone(metadata.estimator.estimator_sha256)
        self.assertIsNotNone(metadata.model_digest)
        self.assertEqual(self.model.digest, metadata.model_digest)
        # the stamped fields are consistent with the artifact and metadata
        self.assertEqual(
            metadata.estimator.estimator_sha256,
            estimator_sha256(self.model.estimator.to_bytes()),
        )
        self.assertEqual(metadata.model_digest, compute_model_digest(metadata))

    def test_to_bytes_is_stable(self):
        self.assertEqual(
            self.model.estimator.to_bytes(), self.model.estimator.to_bytes()
        )

    def test_digest_round_trip_all_formats(self):
        for name in ("model_dir", "model.zip", "model.tar.gz"):
            with self.subTest(format=name):
                target = self.out_path / name
                self.model.to_file(target)
                loaded = pt.load_model(target)
                self.assertEqual(loaded.digest, self.model.digest)
                self.assertEqual(
                    loaded.metadata.estimator.estimator_sha256,
                    self.model.metadata.estimator.estimator_sha256,
                )

    def test_corrupt_binary_raises_on_load(self):
        outdir = self.out_path / "corrupt_dir"
        self.model.to_file(outdir)
        binary = outdir / "model.onnx"
        binary.write_bytes(binary.read_bytes() + b"tampered")
        with self.assertRaises(ValueError):
            pt.load_model(outdir)

    def test_edited_metadata_warns_on_load(self):
        outdir = self.out_path / "edited_dir"
        self.model.to_file(outdir)
        meta_path = outdir / "metadata.json"
        metadata_dict = json.loads(meta_path.read_text())
        metadata_dict["training"]["trained_date"] = "1999-01-01"
        meta_path.write_text(json.dumps(metadata_dict))
        with self.assertWarns(UserWarning):
            pt.load_model(outdir)

    def test_legacy_metadata_without_digest_loads_clean(self):
        outdir = self.out_path / "legacy_dir"
        self.model.to_file(outdir)
        meta_path = outdir / "metadata.json"
        metadata_dict = json.loads(meta_path.read_text())
        del metadata_dict["model_digest"]
        del metadata_dict["estimator"]["estimator_sha256"]
        meta_path.write_text(json.dumps(metadata_dict))
        loaded = pt.load_model(outdir)
        self.assertIsNone(loaded.digest)

    def test_idempotent_publish(self):
        registry_root = self.out_path / "registry"

        first = save_to_registry(self.model, registry_root)
        second = save_to_registry(self.model, registry_root)
        self.assertEqual(first, second)

        config_dir = (registry_root / "v2" / first.to_path()).parent
        versions = sorted(p.name for p in config_dir.iterdir() if p.is_dir())
        self.assertEqual(versions, ["v1"])

        # a genuinely different model (retrained on different data) gets v2
        retrained = SklearnRandomForestTrainer().train(
            self.df.sample(frac=0.8, random_state=7).reset_index(drop=True),
            self.config,
        )
        self.assertNotEqual(retrained.digest, self.model.digest)
        third = save_to_registry(retrained, registry_root)
        self.assertEqual(third.version, 2)

        # an explicit version bypasses the idempotency check
        with self.assertRaises(FileExistsError):
            save_to_registry(self.model, registry_root, version=1)

    def test_find_by_digest(self):
        registry_root = self.out_path / "registry"
        model_id = save_to_registry(self.model, registry_root)
        registry = LocalRegistry(registry_root)

        hits = registry.find_by_digest(self.model.digest)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].model_id, model_id)
        self.assertEqual(hits[0].model_digest, self.model.digest)

        # accepted without the sha256: prefix too
        bare = self.model.digest.split(":", 1)[1]
        self.assertEqual(len(registry.find_by_digest(bare)), 1)

        # query() plumbing
        hits = registry.query(model_digest=self.model.digest)
        self.assertEqual(len(hits), 1)

        # unknown digest -> no hits
        self.assertEqual(len(registry.find_by_digest("00" * 32)), 0)


if __name__ == "__main__":
    import unittest

    unittest.main()
