import json
import shutil
import warnings
from pathlib import Path
from unittest import TestCase

import pandas as pd
import pydantic

import routee.powertrain as pt
from routee.powertrain.core.digest import compute_model_digest
from routee.powertrain.core.metadata import Metadata
from routee.powertrain.io.archive import save_to_registry
from routee.powertrain.registry.local import LocalRegistry
from routee.powertrain.registry.model_id import ModelKey
from routee.powertrain.resources.bundled_registry import bundled_registry_root
from routee.powertrain.trainers.sklearn_random_forest import (
    SklearnRandomForestTrainer,
)

this_dir = Path(__file__).parent

ARCHIVE_FORMATS = ("model_dir", "model.zip", "model.tar.gz")


class TestModelKey(TestCase):
    """The version-less identity cached in ``metadata.model_key``."""

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
            vehicle_description="Model Key Test Model",
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

    def _saved_metadata(self, name: str = "model_dir") -> dict:
        outdir = self.out_path / name
        self.model.to_file(outdir)
        return json.loads((outdir / "metadata.json").read_text())

    def test_saving_stamps_the_model_key(self):
        metadata_dict = self._saved_metadata()
        self.assertEqual(metadata_dict["model_key"], self.model.key.to_path())
        self.assertEqual(metadata_dict["model_key"], "test/model_ice/2024/rf_aaa9554f")

    def test_stamped_on_every_format(self):
        expected = self.model.key.to_path()
        for name in ARCHIVE_FORMATS:
            with self.subTest(format=name):
                target = self.out_path / name
                self.model.to_file(target)
                loaded = pt.load_model(target)
                self.assertEqual(loaded.metadata.model_key, expected)
                self.assertEqual(loaded.key, self.model.key)

    def test_model_key_has_no_version_segment(self):
        """The registry assigns a version, so an artifact must not claim one."""
        model_key = self._saved_metadata()["model_key"]
        self.assertEqual(len(model_key.split("/")), 4)

        model_id = save_to_registry(self.model, self.out_path / "registry")
        self.assertEqual(model_id.key.to_path(), model_key)
        self.assertEqual(model_id.to_path(), f"{model_key}/v1")

    def test_model_key_matches_registry_path(self):
        registry_root = self.out_path / "registry"
        model_id = save_to_registry(self.model, registry_root)
        entry = registry_root / "v2" / model_id.to_path()
        metadata_dict = json.loads((entry / "metadata.json").read_text())
        self.assertEqual(metadata_dict["model_key"], model_id.key.to_path())

        loaded = LocalRegistry(registry_root).load(model_id)
        self.assertEqual(loaded.metadata.model_key, model_id.key.to_path())

    def test_model_key_is_outside_the_digest(self):
        """The key derives from fields the digest already covers, so stamping it
        must leave ``model_digest`` untouched."""
        metadata = self.model.metadata
        before = compute_model_digest(metadata)
        metadata.model_key = "some/other_ice/1999/rf_00000000"
        self.assertEqual(compute_model_digest(metadata), before)

        saved = self._saved_metadata()
        self.assertEqual(saved["model_digest"], self.model.digest)

    def test_round_trip_re_derives_rather_than_copies(self):
        """A load then save re-derives the key from the identity fields."""
        outdir = self.out_path / "edited_dir"
        self.model.to_file(outdir)
        meta_path = outdir / "metadata.json"
        metadata_dict = json.loads(meta_path.read_text())
        metadata_dict["vehicle"]["model"] = "renamed"
        meta_path.write_text(json.dumps(metadata_dict))

        with warnings.catch_warnings():
            # the edit also invalidates the digest, which warns separately
            warnings.simplefilter("ignore")
            loaded = pt.load_model(outdir)

        resaved = self.out_path / "resaved_dir"
        loaded.to_file(resaved)
        resaved_dict = json.loads((resaved / "metadata.json").read_text())
        self.assertEqual(resaved_dict["model_key"], "test/renamed_ice/2024/rf_aaa9554f")

    def test_edited_model_key_warns_on_load(self):
        outdir = self.out_path / "tampered_dir"
        self.model.to_file(outdir)
        meta_path = outdir / "metadata.json"
        metadata_dict = json.loads(meta_path.read_text())
        metadata_dict["model_key"] = "wrong/key_ice/2024/rf_aaa9554f"
        meta_path.write_text(json.dumps(metadata_dict))

        with self.assertWarns(UserWarning):
            loaded = pt.load_model(outdir)
        # derivation stays authoritative
        self.assertEqual(loaded.key.to_path(), "test/model_ice/2024/rf_aaa9554f")

    def test_malformed_model_key_rejected(self):
        for bad in ("test/model_ice/2024", "test/model_ice/2024/rf/extra", ""):
            with self.subTest(model_key=bad):
                with self.assertRaises(pydantic.ValidationError):
                    Metadata.model_validate(
                        {
                            **self.model.metadata.model_dump(mode="json"),
                            "model_key": bad,
                        }
                    )

    def test_model_key_round_trips_through_model_key_type(self):
        stamped = self._saved_metadata()["model_key"]
        self.assertEqual(ModelKey.from_path(stamped), self.model.key)


class TestExistingArtifactsWithoutModelKey(TestCase):
    """Artifacts published before the field exists must load unchanged."""

    def setUp(self) -> None:
        self.out_path = Path("tmp")
        self.out_path.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        if self.out_path.exists():
            shutil.rmtree(self.out_path)

    def test_bundled_models_load(self):
        """The bundled registry holds artifacts written without a model_key —
        the same shape as everything already published to HuggingFace."""
        registry = LocalRegistry(root=bundled_registry_root(), schema_version="v2")
        model_ids = registry.list_models()
        self.assertGreater(len(model_ids), 0)

        for model_id in model_ids:
            with self.subTest(model_id=str(model_id)):
                metadata_dict = registry.get_metadata(model_id)
                self.assertNotIn("model_key", metadata_dict)

                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    model = registry.load(model_id)

                self.assertIsNone(model.metadata.model_key)
                self.assertEqual(model.key, model_id.key)

    def test_missing_model_key_survives_a_save(self):
        """Loading an artifact without the field and saving it stamps one."""
        registry = LocalRegistry(root=bundled_registry_root(), schema_version="v2")
        model_id = registry.list_models()[0]
        model = registry.load(model_id)
        self.assertIsNone(model.metadata.model_key)

        outdir = self.out_path / "restamped_dir"
        model.to_file(outdir)
        metadata_dict = json.loads((outdir / "metadata.json").read_text())
        self.assertEqual(metadata_dict["model_key"], model_id.key.to_path())

        reloaded = pt.load_model(outdir)
        self.assertEqual(reloaded.metadata.model_key, model_id.key.to_path())
        self.assertEqual(reloaded.digest, model.digest)

    def test_explicit_null_model_key_loads(self):
        """A metadata.json carrying an explicit null is equivalent to omitting it."""
        registry = LocalRegistry(root=bundled_registry_root(), schema_version="v2")
        model_id = registry.list_models()[0]
        model = registry.load(model_id)

        outdir = self.out_path / "null_key_dir"
        model.to_file(outdir)
        meta_path = outdir / "metadata.json"
        metadata_dict = json.loads(meta_path.read_text())
        metadata_dict["model_key"] = None
        meta_path.write_text(json.dumps(metadata_dict))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded = pt.load_model(outdir)
        self.assertIsNone(loaded.metadata.model_key)
