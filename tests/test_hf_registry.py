from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

from routee.powertrain.registry.hf import HFRegistry, build_index
from routee.powertrain.registry.registry import IndexMissingError
from routee.powertrain.resources.bundled_registry import bundled_registry_root

REPO_ID = "test-org/test-model-library"
SCHEMA = "v2"
BASE = SCHEMA  # root_prefix defaults to "", so the schema dir sits at the repo root


def _fake_metadata(
    make: str = "toyota",
    model: str = "camry_ice",
    year: int = 2016,
    powertrain: str = "ICE",
) -> dict:
    """Return a minimal metadata dict that model_info_from_metadata can parse."""
    return {
        "vehicle": {
            "model": model,
            "vehicle_description": f"{year} {make} {model}",
            "powertrain_type": powertrain,
        },
        "contract": {
            "feature_set": [
                {"name": "speed_mph", "units": "mph"},
                {"name": "grade_dec", "units": "decimal"},
            ],
            "target": [{"name": "gallons_fastsim", "units": "gallons_gasoline"}],
        },
        "estimator": {
            "estimator_type": "ONNXEstimator",
            "model_file": "model.onnx",
        },
    }


class _MockHfApi:
    """Minimal stand-in for HfApi backed by a preloaded {path: bytes} map.

    ``hf_hub_download`` really does return a *path* to a cached file rather
    than bytes, so the mock writes to a temp dir and returns that path — the
    same shape the production code unwraps.
    """

    def __init__(self, files: dict | None = None):
        self._files = files or {}
        self._tmp = Path(tempfile.mkdtemp())
        self.uploaded: list[dict] = []

    def hf_hub_download(self, repo_id, filename, repo_type=None, revision=None):
        if filename not in self._files:
            raise FileNotFoundError(f"{filename} not found in {repo_id}")
        local = self._tmp / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(self._files[filename])
        return str(local)

    def list_repo_files(self, repo_id, repo_type=None, revision=None):
        return sorted(self._files)

    def upload_file(self, path_or_fileobj, path_in_repo, **kwargs):
        self.uploaded.append({"path_in_repo": path_in_repo, "body": path_or_fileobj})
        self._files[path_in_repo] = path_or_fileobj

    def cleanup(self):
        shutil.rmtree(self._tmp, ignore_errors=True)


def _index_entry(
    make: str,
    model: str,
    year: int,
    config_slug: str,
    version: int,
    powertrain: str = "ICE",
    feature_names: list | None = None,
    fuel_type: str | None = None,
) -> dict:
    return {
        "model_id": {
            "make": make,
            "vehicle_slug": model,
            "year": year,
            "config_slug": config_slug,
            "version": version,
        },
        "vehicle_model": model,
        "estimator_type": "ONNXEstimator",
        "feature_names": feature_names or ["speed_mph", "grade_dec"],
        "target_names": ["gallons_fastsim"],
        "powertrain_type": powertrain,
        "vehicle_description": f"{year} {make} {model}",
        "path": f"{BASE}/{make}/{model}/{year}/{config_slug}/v{version}",
        "fuel_type": fuel_type,
    }


def _registry_with_files(files: dict) -> HFRegistry:
    registry = HFRegistry(repo_id=REPO_ID, schema_version=SCHEMA)
    registry._client = _MockHfApi(files)  # type: ignore[assignment]
    return registry


def _build_registry(entries: list[dict]) -> HFRegistry:
    """Create an HFRegistry pre-loaded with an index containing the given entries."""
    index = {"schema_version": SCHEMA, "models": entries}
    return _registry_with_files({f"{BASE}/index.json": json.dumps(index).encode()})


class TestQueryViaIndex(TestCase):
    """End-to-end tests for query() served entirely from index.json."""

    def test_query_by_make(self):
        registry = _build_registry(
            [
                _index_entry("toyota", "camry", 2016, "rf_default", 1),
                _index_entry("chevrolet", "bolt_ev", 2020, "rf_transient", 1),
            ]
        )
        results = registry.query(make="toyota")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.make, "toyota")

    def test_query_by_make_and_model(self):
        registry = _build_registry(
            [
                _index_entry("toyota", "camry", 2016, "rf_default", 1),
                _index_entry("chevrolet", "bolt_ev", 2020, "rf_transient", 1),
            ]
        )
        results = registry.query(make="chevrolet", model="bolt_ev")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.vehicle_slug, "bolt_ev")

    def test_query_by_year(self):
        registry = _build_registry(
            [
                _index_entry("toyota", "camry", 2016, "rf_default", 1),
                _index_entry("toyota", "camry", 2020, "rf_default", 1),
            ]
        )
        results = registry.query(year=2020)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.year, 2020)

    def test_query_by_feature_names(self):
        registry = _build_registry(
            [
                _index_entry("toyota", "camry", 2016, "rf_default", 1),
                _index_entry(
                    "toyota",
                    "camry",
                    2016,
                    "rf_mass",
                    1,
                    feature_names=["speed_mph", "grade_dec", "mass_lbs"],
                ),
            ]
        )
        results = registry.query(feature_names=["mass_lbs"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.config_slug, "rf_mass")

    def test_query_no_filters_returns_all(self):
        registry = _build_registry(
            [
                _index_entry("toyota", "camry", 2016, "rf_default", 1),
                _index_entry("chevrolet", "bolt_ev", 2020, "rf_transient", 1),
            ]
        )
        self.assertEqual(len(registry.query()), 2)

    def test_query_no_match_returns_empty(self):
        registry = _build_registry(
            [_index_entry("toyota", "camry", 2016, "rf_default", 1)]
        )
        self.assertEqual(registry.query(make="ford", fuzzy=False), [])


class TestVersionStrategy(TestCase):
    """Version strategy and exact-version filter served from the index."""

    def _versioned_registry(self):
        return _build_registry(
            [
                _index_entry("toyota", "camry", 2016, "rf_default", 1),
                _index_entry("toyota", "camry", 2016, "rf_default", 2),
                _index_entry("toyota", "camry", 2016, "rf_default", 3),
            ]
        )

    def test_default_strategy_returns_latest_only(self):
        results = self._versioned_registry().query(make="toyota")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.version, 3)

    def test_strategy_all_returns_every_version(self):
        results = self._versioned_registry().query(
            make="toyota", version_strategy="all"
        )
        self.assertEqual(len(results), 3)

    def test_exact_version_filter(self):
        results = self._versioned_registry().query(make="toyota", version=2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.version, 2)

    def test_list_models_default_latest(self):
        ids = self._versioned_registry().list_models()
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids[0].version, 3)

    def test_list_models_all(self):
        ids = self._versioned_registry().list_models(version_strategy="all")
        self.assertEqual(len(ids), 3)


class TestIndexRequired(TestCase):
    """When index.json is missing or unreadable, query/list_models must fail clearly."""

    def test_query_raises_when_index_missing(self):
        with self.assertRaises(IndexMissingError) as ctx:
            _registry_with_files({}).query(make="toyota")
        self.assertIn("build_index", str(ctx.exception))

    def test_list_models_raises_when_index_missing(self):
        with self.assertRaises(IndexMissingError):
            _registry_with_files({}).list_models()

    def test_query_raises_when_index_unparseable(self):
        registry = _registry_with_files({f"{BASE}/index.json": b"not valid json"})
        with self.assertRaises(IndexMissingError):
            registry.query()


class TestLoadAndMetadata(TestCase):
    """load() and get_metadata() against a real bundled model served over the mock."""

    MODEL_PATH = "toyota/rav4_xle_ice/2022/rf_fe510e40/v1"

    def setUp(self):
        source = bundled_registry_root() / SCHEMA / self.MODEL_PATH
        self.files = {
            f"{BASE}/{self.MODEL_PATH}/{f.name}": f.read_bytes()
            for f in source.iterdir()
            if f.is_file()
        }
        self.registry = _registry_with_files(dict(self.files))

    def test_get_metadata_returns_parsed_dict(self):
        metadata = self.registry.get_metadata(self.MODEL_PATH)
        self.assertEqual(metadata["vehicle"]["make"], "toyota")
        self.assertEqual(metadata["estimator"]["model_file"], "model.onnx")

    def test_load_returns_a_usable_model(self):
        model = self.registry.load(self.MODEL_PATH)
        # The path must agree with the identity derived from the metadata.
        self.assertEqual(model.key.to_path(), "toyota/rav4_xle_ice/2022/rf_fe510e40")
        self.assertIn("speed_mph", model.metadata.config.feature_set.feature_name_list)

    def test_load_accepts_a_model_id_object(self):
        from routee.powertrain.registry.model_id import ModelId

        model = self.registry.load(ModelId.from_path(self.MODEL_PATH))
        self.assertEqual(model.metadata.vehicle.make, "toyota")

    def test_load_propagates_the_fetch_error_for_a_missing_model(self):
        # The concrete type comes from huggingface_hub in production (and from
        # the mock here); what matters is that load() raises rather than
        # silently returning something.
        with self.assertRaises(Exception):
            self.registry.load("toyota/rav4_xle_ice/2022/rf_fe510e40/v99")


class TestBuildIndex(TestCase):
    """build_index walks the repo listing and uploads a catalog."""

    def test_build_index_collects_entries_and_uploads(self):
        path = f"{BASE}/toyota/camry_ice/2016/rf_default/v1"
        registry = _registry_with_files(
            {f"{path}/metadata.json": json.dumps(_fake_metadata()).encode()}
        )
        client = registry._client

        # build_index constructs its own registry; hand it back our mocked one.
        import routee.powertrain.registry.hf as hf_module

        original = hf_module.HFRegistry
        hf_module.HFRegistry = lambda **kwargs: registry  # type: ignore[assignment]
        try:
            index = build_index(repo_id=REPO_ID, schema_version=SCHEMA)
        finally:
            hf_module.HFRegistry = original  # type: ignore[assignment]

        self.assertEqual(index["schema_version"], SCHEMA)
        self.assertEqual(len(index["models"]), 1)
        entry = index["models"][0]
        self.assertEqual(entry["model_id"]["make"], "toyota")
        self.assertEqual(entry["model_id"]["version"], 1)
        self.assertEqual(entry["path"], path)

        self.assertEqual(len(client.uploaded), 1)
        self.assertEqual(client.uploaded[0]["path_in_repo"], f"{BASE}/index.json")

    def test_build_index_dry_run_uploads_nothing(self):
        registry = _registry_with_files({})
        client = registry._client

        import routee.powertrain.registry.hf as hf_module

        original = hf_module.HFRegistry
        hf_module.HFRegistry = lambda **kwargs: registry  # type: ignore[assignment]
        try:
            index = build_index(repo_id=REPO_ID, schema_version=SCHEMA, dry_run=True)
        finally:
            hf_module.HFRegistry = original  # type: ignore[assignment]

        self.assertEqual(index["models"], [])
        self.assertEqual(client.uploaded, [])


class TestDefaultBackend(TestCase):
    """The factory defaults to HuggingFace, and every backend stays reachable."""

    def setUp(self):
        self._saved = os.environ.pop("ROUTEE_REGISTRY_BACKEND", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["ROUTEE_REGISTRY_BACKEND"] = self._saved
        else:
            os.environ.pop("ROUTEE_REGISTRY_BACKEND", None)

    def test_default_backend_is_huggingface(self):
        from routee.powertrain.registry.default import get_default_registry

        self.assertIsInstance(get_default_registry(), HFRegistry)

    def test_repo_id_is_configurable(self):
        from routee.powertrain.registry.default import get_default_registry

        os.environ["ROUTEE_HF_REPO_ID"] = "someone/else"
        try:
            registry = get_default_registry()
        finally:
            os.environ.pop("ROUTEE_HF_REPO_ID")
        self.assertEqual(registry.repo_id, "someone/else")

    def test_unknown_backend_names_every_option(self):
        from routee.powertrain.registry.default import get_default_registry

        os.environ["ROUTEE_REGISTRY_BACKEND"] = "gcs"
        with self.assertRaises(ValueError) as ctx:
            get_default_registry()
        message = str(ctx.exception)
        for backend in ("hf", "s3", "local"):
            self.assertIn(f"'{backend}'", message)

    def test_importing_the_registry_package_does_not_import_boto3(self):
        """boto3 is an optional extra now — importing must not require it."""
        import subprocess
        import sys

        code = (
            "import sys;"
            "import routee.powertrain;"
            "from routee.powertrain.registry import S3Registry, HFRegistry;"
            "assert 'boto3' not in sys.modules, 'boto3 was imported eagerly';"
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)
