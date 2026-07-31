"""The ``provenance`` metadata section: where a model came from.

Covers the tagged-union round trip through ``metadata.json``, the flat
``ModelConfig`` ↔ grouped ``Metadata`` translation, and the hard break on
archives written before the section existed.
"""

import json
import shutil
from pathlib import Path
from unittest import TestCase

import pandas as pd

import routee.powertrain as pt
from routee.powertrain.core.metadata import Metadata
from routee.powertrain.trainers.sklearn_random_forest import (
    SklearnRandomForestTrainer,
)
from routee.powertrain.validation.errors import EstimatorErrors, ModelErrors

this_dir = Path(__file__).parent

FASTSIM_SOURCE = pt.FastSimSource(
    fastsim_vehicle_id="v1/fastsim-3/conv/toyota/camry-4cyl-2wd/2016/base/r1",
    fastsim_vehicles_ref="v1.2.0",
    fastsim_version="3.1.0",
    pipeline_version="0.4.1",
    pipeline_run_id="gha-2026-07-14-8871",
    pipeline_repo_ref="9f3c1ab",
    dataset_name="camry-2016-cycles",
    dataset_hash="ab" * 32,
)
REAL_WORLD_SOURCE = pt.RealWorldSource(
    data_source="fleet_dna",
    fleet="delivery_vans",
    collection_start="2023-01-01",
    collection_end="2023-12-31",
    n_vehicles=42,
    n_trips=9001,
    dataset_name="fleet-dna-2023",
    dataset_hash="cd" * 32,
)
LEGACY_SOURCE = pt.LegacySource(
    original_source="nrel_v1_json_library",
    converted_from="v1",
)


def _config(**overrides) -> pt.ModelConfig:
    fields = dict(
        vehicle_description="Provenance Test Vehicle",
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
        trip_column="journey_id",
    )
    fields.update(overrides)
    return pt.ModelConfig.model_validate(fields)


def _metadata(config: pt.ModelConfig) -> Metadata:
    return Metadata.from_config(
        config,
        errors=ModelErrors(estimator_errors=EstimatorErrors(error_by_target={})),
        estimator_type="ONNXEstimator",
        model_file="model.onnx",
        architecture_tag="random_forest",
        trained_date="2026-07-31",
    )


class TestProvenanceSection(TestCase):
    def test_source_variants_round_trip(self):
        """Each variant survives a JSON round trip as its own concrete type —
        the ``method`` discriminator is what picks the class back out."""
        for source in (FASTSIM_SOURCE, REAL_WORLD_SOURCE, LEGACY_SOURCE):
            with self.subTest(method=source.method.value):
                metadata = _metadata(_config(training_source=source))
                as_json = json.loads(metadata.model_dump_json())
                restored = Metadata.model_validate(as_json)
                self.assertIsInstance(restored.provenance.source, type(source))
                self.assertEqual(restored.provenance.source, source)
                self.assertEqual(restored.provenance.method, source.method)

    def test_fastsim_fields_persist(self):
        metadata = _metadata(_config(training_source=FASTSIM_SOURCE))
        source = json.loads(metadata.model_dump_json())["provenance"]["source"]
        self.assertEqual(source["method"], "fastsim_simulation")
        self.assertEqual(
            source["fastsim_vehicle_id"],
            "v1/fastsim-3/conv/toyota/camry-4cyl-2wd/2016/base/r1",
        )
        self.assertEqual(source["fastsim_version"], "3.1.0")
        self.assertEqual(source["pipeline_run_id"], "gha-2026-07-14-8871")

    def test_dataset_labels_live_on_the_source(self):
        """``dataset_name`` / ``dataset_hash`` are part of describing the
        source, not a section of their own."""
        source = json.loads(
            _metadata(_config(training_source=FASTSIM_SOURCE)).model_dump_json()
        )["provenance"]["source"]
        self.assertEqual(source["dataset_name"], "camry-2016-cycles")
        self.assertEqual(source["dataset_hash"], "ab" * 32)

    def test_source_is_optional(self):
        metadata = _metadata(_config())
        self.assertIsNone(metadata.provenance.source)
        self.assertIsNone(metadata.provenance.method)
        self.assertIsNone(metadata.provenance.dataset_name)
        self.assertIsNone(metadata.provenance.dataset_hash)

    def test_dataset_accessors_read_through_to_the_source(self):
        provenance = _metadata(_config(training_source=REAL_WORLD_SOURCE)).provenance
        self.assertEqual(provenance.dataset_name, "fleet-dna-2023")
        self.assertEqual(provenance.dataset_hash, "cd" * 32)

    def test_training_config_lands_in_provenance(self):
        config = _config(training_source=FASTSIM_SOURCE, validation_size=0.1)
        provenance = _metadata(config).provenance
        self.assertEqual(provenance.training.test_size, 0.2)
        self.assertEqual(provenance.training.validation_size, 0.1)
        self.assertEqual(provenance.training.random_seed, 42)
        self.assertEqual(provenance.training.trip_column, "journey_id")
        self.assertEqual(provenance.training.trained_date, "2026-07-31")

    def test_flat_config_round_trip(self):
        """``Metadata.config`` is the inverse of ``Metadata.from_config`` — the
        provenance fields must survive the trip back to the flat object."""
        restored = _metadata(_config(training_source=REAL_WORLD_SOURCE)).config
        self.assertEqual(restored.training_source, REAL_WORLD_SOURCE)
        self.assertEqual(restored.trip_column, "journey_id")
        self.assertEqual(restored.random_seed, 42)

    def test_unknown_method_rejected(self):
        as_json = json.loads(_metadata(_config()).model_dump_json())
        as_json["provenance"]["source"] = {"method": "vibes"}
        with self.assertRaises(ValueError):
            Metadata.model_validate(as_json)

    def test_pre_provenance_metadata_rejected(self):
        """Archives written before 2.0.1 carry a flat ``training`` block and no
        ``provenance``. They are not translated — the whole library was
        regenerated for 2.0.1, so the fix is to re-download. ``provenance`` is a
        required field, so such a file fails validation rather than loading with
        its provenance silently empty."""
        as_json = json.loads(_metadata(_config()).model_dump_json())
        del as_json["provenance"]
        as_json["training"] = {
            "test_size": 0.2,
            "validation_size": None,
            "random_seed": 42,
            "trip_column": "journey_id",
            "trained_date": None,
            "dataset_name": None,
            "dataset_hash": None,
        }
        with self.assertRaises(ValueError) as ctx:
            Metadata.model_validate(as_json)
        self.assertIn("provenance", str(ctx.exception))


class TestProvenanceOnDisk(TestCase):
    """Provenance survives a real train → save → load cycle."""

    model: pt.Model

    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv(
            this_dir / "routee-powertrain-test-data" / "sample_train_data.csv"
        )
        config = _config(
            target=pt.TargetSet(
                targets=[pt.DataColumn(name="gallons_fastsim", units="gallons")]
            ),
            training_source=FASTSIM_SOURCE.model_copy(
                update={
                    "dataset_name": "sample-train-data",
                    "dataset_hash": pt.hash_dataframe(df),
                }
            ),
        )
        cls.model = SklearnRandomForestTrainer(max_depth=5).train(df, config)

    def setUp(self) -> None:
        self.out_path = Path("tmp_provenance")
        self.out_path.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        if self.out_path.exists():
            shutil.rmtree(self.out_path)

    def test_saved_metadata_has_provenance_and_no_training_key(self):
        outdir = self.out_path / "model_dir"
        self.model.to_file(outdir)
        metadata_dict = json.loads((outdir / "metadata.json").read_text())
        self.assertIn("provenance", metadata_dict)
        self.assertNotIn("training", metadata_dict)
        self.assertEqual(
            metadata_dict["provenance"]["source"]["method"], "fastsim_simulation"
        )

    def test_provenance_round_trips_through_all_formats(self):
        for name in ("model_dir", "model.zip", "model.tar.gz"):
            with self.subTest(format=name):
                target = self.out_path / name
                self.model.to_file(target)
                loaded = pt.load_model(target)
                self.assertEqual(
                    loaded.metadata.provenance, self.model.metadata.provenance
                )
                self.assertIsInstance(
                    loaded.metadata.provenance.source, pt.FastSimSource
                )

    def test_trained_date_is_stamped(self):
        self.assertIsNotNone(self.model.metadata.provenance.training.trained_date)
