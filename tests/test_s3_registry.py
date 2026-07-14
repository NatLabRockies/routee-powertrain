from __future__ import annotations

import json
from unittest import TestCase
from unittest.mock import MagicMock

from routee.powertrain.registry.s3 import IndexMissingError, S3Registry

ROOT = "routee-powertrain-model-library"
SCHEMA = "v2"
BASE = f"{ROOT}/{SCHEMA}"


def _fake_metadata(
    make: str = "toyota",
    model: str = "camry_ice",
    year: int = 2016,
    powertrain: str = "ICE",
) -> dict:
    """Return a minimal metadata dict that _model_info_from_metadata can parse."""
    return {
        "vehicle": {
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
        "errors": {
            "estimator_errors": {
                "error_by_target": {"gallons_fastsim": {"mape": 0.05, "rmse": 0.01}}
            }
        },
    }


class _MockClient:
    """Minimal mock S3 client supporting get_object with a preloaded object map."""

    def __init__(self, objects: dict | None = None):
        self._objects = objects or {}

    def get_object(self, Bucket, Key):
        if Key not in self._objects:
            error = KeyError(Key)
            raise error
        body = MagicMock()
        body.read.return_value = self._objects[Key]
        return {"Body": body}


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


def _build_registry(entries: list[dict]) -> S3Registry:
    """Create an S3Registry pre-loaded with an index containing the given entries."""
    index = {"schema_version": SCHEMA, "models": entries}
    objects = {f"{BASE}/index.json": json.dumps(index).encode()}
    registry = S3Registry(
        bucket="test-bucket",
        schema_version=SCHEMA,
        root_prefix=ROOT,
    )
    registry._client = _MockClient(objects)  # type: ignore[assignment]
    return registry


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
                _index_entry("chevrolet", "bolt_ev", 2020, "rf_transient", 1),
            ]
        )
        results = registry.query(year=2016)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.make, "toyota")

    def test_query_by_config_slug(self):
        registry = _build_registry(
            [
                _index_entry("toyota", "camry", 2016, "rf_default", 1),
                _index_entry("chevrolet", "bolt_ev", 2020, "rf_transient", 1),
            ]
        )
        results = registry.query(config_slug="rf_transient")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.make, "chevrolet")

    def test_query_by_feature_names(self):
        registry = _build_registry(
            [
                _index_entry("toyota", "camry", 2016, "rf_default", 1),
                _index_entry("chevrolet", "bolt_ev", 2020, "rf_transient", 1),
            ]
        )
        results = registry.query(feature_names=["speed_mph"])
        self.assertEqual(len(results), 2)
        results = registry.query(feature_names=["speed_mph", "nonexistent"])
        self.assertEqual(len(results), 0)

    def test_query_fuzzy_make(self):
        registry = _build_registry(
            [
                _index_entry("toyota", "camry", 2016, "rf_default", 1),
                _index_entry("chevrolet", "bolt_ev", 2020, "rf_transient", 1),
            ]
        )
        results = registry.query(make="chevy")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.make, "chevrolet")

    def test_query_no_match(self):
        registry = _build_registry(
            [_index_entry("toyota", "camry", 2016, "rf_default", 1)]
        )
        self.assertEqual(registry.query(make="zzzzz"), [])

    def test_query_multiple_filters(self):
        registry = _build_registry(
            [
                _index_entry("toyota", "camry", 2016, "rf_default", 1),
                _index_entry("chevrolet", "bolt_ev", 2020, "rf_transient", 1),
            ]
        )
        self.assertEqual(len(registry.query(make="toyota", year=2016)), 1)
        self.assertEqual(len(registry.query(make="toyota", year=2020)), 0)

    def test_query_no_filters_returns_all(self):
        registry = _build_registry(
            [
                _index_entry("toyota", "camry", 2016, "rf_default", 1),
                _index_entry("chevrolet", "bolt_ev", 2020, "rf_transient", 1),
            ]
        )
        results = registry.query(version_strategy="all")
        self.assertEqual(len(results), 2)


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

    def test_version_filter_overrides_strategy(self):
        results = self._versioned_registry().query(
            make="toyota", version=1, version_strategy="latest"
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.version, 1)

    def test_list_models_default_latest(self):
        ids = self._versioned_registry().list_models()
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids[0].version, 3)

    def test_list_models_all(self):
        ids = self._versioned_registry().list_models(version_strategy="all")
        self.assertEqual(len(ids), 3)


class TestIndexRequired(TestCase):
    """When index.json is missing or unreadable, query/list_models must fail clearly."""

    def _no_index_registry(self) -> S3Registry:
        registry = S3Registry(
            bucket="test-bucket",
            schema_version=SCHEMA,
            root_prefix=ROOT,
        )
        # empty objects map -> get_object raises for everything
        registry._client = _MockClient(objects={})  # type: ignore[assignment]
        return registry

    def test_query_raises_when_index_missing(self):
        with self.assertRaises(IndexMissingError) as ctx:
            self._no_index_registry().query(make="toyota")
        self.assertIn("build_index", str(ctx.exception))

    def test_list_models_raises_when_index_missing(self):
        with self.assertRaises(IndexMissingError):
            self._no_index_registry().list_models()

    def test_query_raises_when_index_unparseable(self):
        registry = S3Registry(
            bucket="test-bucket",
            schema_version=SCHEMA,
            root_prefix=ROOT,
        )
        registry._client = _MockClient(  # type: ignore[assignment]
            objects={f"{BASE}/index.json": b"not valid json"}
        )
        with self.assertRaises(IndexMissingError):
            registry.query()
