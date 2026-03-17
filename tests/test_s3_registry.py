from __future__ import annotations

import json
from unittest import TestCase
from unittest.mock import MagicMock

from routee.powertrain.registry.s3 import S3Registry

ROOT = "routee-powertrain-model-library"
SCHEMA = "v2"
BASE = f"{ROOT}/{SCHEMA}"


def _fake_metadata(
    make: str = "toyota",
    model: str = "camry_4cyl_fwd",
    year: int = 2016,
    variant: str = "default",
    powertrain: str = "ICE",
) -> dict:
    """Return a minimal metadata dict that _model_info_from_metadata can parse."""
    return {
        "estimator_type": "ONNXEstimator",
        "model_file": "model.onnx",
        "config": {
            "vehicle_description": f"{year} {make} {model}",
            "powertrain_type": powertrain,
            "feature_set": {
                "features": [
                    {"name": "speed_mph", "units": "mph"},
                    {"name": "grade_dec", "units": "decimal"},
                ]
            },
            "target": {
                "targets": [{"name": "gallons_fastsim", "units": "gallons_gasoline"}]
            },
        },
        "errors": {
            "estimator_errors": {
                "error_by_target": {"gallons_fastsim": {"mape": 0.05, "rmse": 0.01}}
            }
        },
    }


def _common_prefixes(*children: str, prefix: str = "") -> dict:
    """Build a single page of ListObjectsV2 response with CommonPrefixes."""
    return {
        "CommonPrefixes": [{"Prefix": f"{prefix}{c}/"} for c in children],
    }


class _MockPaginator:
    """Paginator that returns pre-configured pages keyed by (Prefix, Delimiter)."""

    def __init__(self, pages_by_prefix: dict):
        self.pages_by_prefix = pages_by_prefix

    def paginate(self, Bucket, Prefix, Delimiter=None):
        key = Prefix
        return self.pages_by_prefix.get(key, [])


class _MockClient:
    """Minimal mock S3 client supporting get_paginator and get_object."""

    def __init__(self, pages_by_prefix: dict, objects: dict | None = None):
        self._paginator = _MockPaginator(pages_by_prefix)
        self._objects = objects or {}

    def get_paginator(self, operation_name):
        return self._paginator

    def get_object(self, Bucket, Key):
        body = MagicMock()
        body.read.return_value = self._objects.get(Key, b"")
        return {"Body": body}


def _build_registry(pages_by_prefix, objects=None):
    """Create an S3Registry with a mock client pre-injected."""
    registry = S3Registry(
        bucket="test-bucket",
        schema_version=SCHEMA,
        root_prefix=ROOT,
    )
    registry._client = _MockClient(pages_by_prefix, objects)
    return registry


class TestListChildren(TestCase):
    def test_basic(self):
        prefix = f"{BASE}/"
        pages = {
            prefix: [_common_prefixes("toyota", "ford", "chevrolet", prefix=prefix)]
        }
        registry = _build_registry(pages)

        children = registry._list_children(prefix)
        self.assertEqual(sorted(children), ["chevrolet", "ford", "toyota"])

    def test_empty(self):
        prefix = f"{BASE}/"
        pages = {prefix: [{"CommonPrefixes": []}]}
        registry = _build_registry(pages)
        self.assertEqual(registry._list_children(prefix), [])

    def test_pagination(self):
        prefix = f"{BASE}/"
        pages = {
            prefix: [
                _common_prefixes("toyota", "ford", prefix=prefix),
                _common_prefixes("chevrolet", prefix=prefix),
            ]
        }
        registry = _build_registry(pages)
        children = registry._list_children(prefix)
        self.assertEqual(sorted(children), ["chevrolet", "ford", "toyota"])


class TestNarrowPrefixes(TestCase):
    def _make_registry_with_makes(self, makes):
        base_prefix = f"{BASE}/"
        pages = {base_prefix: [_common_prefixes(*makes, prefix=base_prefix)]}
        return _build_registry(pages)

    def test_no_filter_expands_all(self):
        registry = self._make_registry_with_makes(["toyota", "ford", "chevrolet"])
        result = registry._narrow_prefixes([f"{BASE}/"], None, fuzzy=True, threshold=80)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(r.endswith("/") for r in result))

    def test_exact_filter(self):
        registry = self._make_registry_with_makes(["toyota", "ford", "chevrolet"])
        result = registry._narrow_prefixes(
            [f"{BASE}/"], "toyota", fuzzy=False, threshold=80
        )
        self.assertEqual(result, [f"{BASE}/toyota/"])

    def test_fuzzy_filter(self):
        registry = self._make_registry_with_makes(["toyota", "ford", "chevrolet"])
        result = registry._narrow_prefixes(
            [f"{BASE}/"], "toyta", fuzzy=True, threshold=80
        )
        self.assertEqual(result, [f"{BASE}/toyota/"])

    def test_no_match(self):
        registry = self._make_registry_with_makes(["toyota", "ford", "chevrolet"])
        result = registry._narrow_prefixes(
            [f"{BASE}/"], "zzzzz", fuzzy=True, threshold=80
        )
        self.assertEqual(result, [])

    def test_year_filtering(self):
        """Year level should use year_contains instead of string matching."""
        year_prefix = f"{BASE}/toyota/camry/"
        pages = {
            year_prefix: [
                _common_prefixes("2016", "2020", "2020-2026", prefix=year_prefix)
            ]
        }
        registry = _build_registry(pages)

        # Query year=2016 should match only "2016"
        result = registry._narrow_prefixes(
            [year_prefix],
            None,
            fuzzy=True,
            threshold=80,
            is_year=True,
            year_query=2016,
        )
        self.assertEqual(result, [f"{year_prefix}2016/"])

        # Query year=2023 should match "2020-2026" range
        result = registry._narrow_prefixes(
            [year_prefix],
            None,
            fuzzy=True,
            threshold=80,
            is_year=True,
            year_query=2023,
        )
        self.assertEqual(result, [f"{year_prefix}2020-2026/"])

    def test_multiple_input_prefixes(self):
        """Handles multiple parent prefixes (e.g. fuzzy matched multiple makes)."""
        toyota_prefix = f"{BASE}/toyota/"
        ford_prefix = f"{BASE}/ford/"
        pages = {
            toyota_prefix: [_common_prefixes("camry", "corolla", prefix=toyota_prefix)],
            ford_prefix: [_common_prefixes("escape", "focus", prefix=ford_prefix)],
        }
        registry = _build_registry(pages)

        result = registry._narrow_prefixes(
            [toyota_prefix, ford_prefix], None, fuzzy=True, threshold=80
        )
        self.assertEqual(len(result), 4)

    def test_case_insensitive(self):
        """Matching should be case-insensitive."""
        registry = self._make_registry_with_makes(["toyota", "ford"])
        result = registry._narrow_prefixes(
            [f"{BASE}/"], "Toyota", fuzzy=False, threshold=80
        )
        self.assertEqual(result, [f"{BASE}/toyota/"])


class TestQueryHierarchical(TestCase):
    """End-to-end tests for the rewritten query() with mocked S3."""

    def _build_full_registry(self):
        """Build a registry with two models:
        - toyota/camry/2016/default/grade_dec_speed_mph/v1
        - chevrolet/bolt_ev/2020/transient/ambient_temp_f_speed_mph/v1
        """
        toyota_meta = _fake_metadata("toyota", "camry", 2016)
        chevy_meta = _fake_metadata("chevrolet", "bolt_ev", 2020, "transient")

        toyota_dir = f"{BASE}/toyota/camry/2016/default/grade_dec_speed_mph/v1"
        chevy_dir = (
            f"{BASE}/chevrolet/bolt_ev/2020/transient/ambient_temp_f_speed_mph/v1"
        )

        pages = {
            # Level 0: makes
            f"{BASE}/": [_common_prefixes("toyota", "chevrolet", prefix=f"{BASE}/")],
            # Level 1: models under each make
            f"{BASE}/toyota/": [_common_prefixes("camry", prefix=f"{BASE}/toyota/")],
            f"{BASE}/chevrolet/": [
                _common_prefixes("bolt_ev", prefix=f"{BASE}/chevrolet/")
            ],
            # Level 2: years
            f"{BASE}/toyota/camry/": [
                _common_prefixes("2016", prefix=f"{BASE}/toyota/camry/")
            ],
            f"{BASE}/chevrolet/bolt_ev/": [
                _common_prefixes("2020", prefix=f"{BASE}/chevrolet/bolt_ev/")
            ],
            # Level 3: variants
            f"{BASE}/toyota/camry/2016/": [
                _common_prefixes("default", prefix=f"{BASE}/toyota/camry/2016/")
            ],
            f"{BASE}/chevrolet/bolt_ev/2020/": [
                _common_prefixes("transient", prefix=f"{BASE}/chevrolet/bolt_ev/2020/")
            ],
            # Level 4: feature sets
            f"{BASE}/toyota/camry/2016/default/": [
                _common_prefixes(
                    "grade_dec_speed_mph",
                    prefix=f"{BASE}/toyota/camry/2016/default/",
                )
            ],
            f"{BASE}/chevrolet/bolt_ev/2020/transient/": [
                _common_prefixes(
                    "ambient_temp_f_speed_mph",
                    prefix=f"{BASE}/chevrolet/bolt_ev/2020/transient/",
                )
            ],
            # Level 5: versions
            f"{BASE}/toyota/camry/2016/default/grade_dec_speed_mph/": [
                _common_prefixes(
                    "v1",
                    prefix=f"{BASE}/toyota/camry/2016/default/grade_dec_speed_mph/",
                )
            ],
            f"{BASE}/chevrolet/bolt_ev/2020/transient/ambient_temp_f_speed_mph/": [
                _common_prefixes(
                    "v1",
                    prefix=f"{BASE}/chevrolet/bolt_ev/2020/transient/ambient_temp_f_speed_mph/",
                )
            ],
        }

        objects = {
            f"{toyota_dir}/metadata.json": json.dumps(toyota_meta).encode(),
            f"{chevy_dir}/metadata.json": json.dumps(chevy_meta).encode(),
        }

        return _build_registry(pages, objects)

    def test_query_by_make(self):
        registry = self._build_full_registry()
        results = registry.query(make="toyota")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.make, "toyota")

    def test_query_by_make_and_model(self):
        registry = self._build_full_registry()
        results = registry.query(make="chevrolet", model="bolt_ev")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.model, "bolt_ev")

    def test_query_by_year(self):
        registry = self._build_full_registry()
        results = registry.query(year=2016)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.make, "toyota")

    def test_query_by_variant(self):
        registry = self._build_full_registry()
        results = registry.query(variant="transient")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.make, "chevrolet")

    def test_query_by_feature_set(self):
        registry = self._build_full_registry()
        results = registry.query(feature_set_id="grade_dec_speed_mph")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.make, "toyota")

    def test_query_fuzzy_make(self):
        registry = self._build_full_registry()
        results = registry.query(make="chevy")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id.make, "chevrolet")

    def test_query_no_match(self):
        registry = self._build_full_registry()
        results = registry.query(make="zzzzz")
        self.assertEqual(len(results), 0)

    def test_query_multiple_filters(self):
        registry = self._build_full_registry()
        results = registry.query(make="toyota", year=2016)
        self.assertEqual(len(results), 1)
        # Wrong year should return empty
        results = registry.query(make="toyota", year=2020)
        self.assertEqual(len(results), 0)

    def test_query_no_filters_uses_scan(self):
        """When no filters are provided, falls back to _scan_models."""
        registry = self._build_full_registry()
        # Patch _scan_models to verify it's called
        original_scan = registry._scan_models
        scan_called = []

        def tracked_scan():
            scan_called.append(True)
            return original_scan()

        registry._scan_models = tracked_scan

        # _scan_models needs _list_metadata_keys which expects
        # a full listing without Delimiter. Add the needed pages.
        toyota_dir = f"{BASE}/toyota/camry/2016/default/grade_dec_speed_mph/v1"
        chevy_dir = (
            f"{BASE}/chevrolet/bolt_ev/2020/transient/ambient_temp_f_speed_mph/v1"
        )
        toyota_meta = _fake_metadata("toyota", "camry", 2016)
        chevy_meta = _fake_metadata("chevrolet", "bolt_ev", 2020, "transient")

        # Override page data for non-Delimiter listing
        registry._client._paginator.pages_by_prefix[f"{BASE}/"] = [
            {
                "Contents": [
                    {"Key": f"{toyota_dir}/metadata.json"},
                    {"Key": f"{chevy_dir}/metadata.json"},
                ]
            }
        ]
        registry._client._objects[f"{toyota_dir}/metadata.json"] = json.dumps(
            toyota_meta
        ).encode()
        registry._client._objects[f"{chevy_dir}/metadata.json"] = json.dumps(
            chevy_meta
        ).encode()

        results = registry.query()
        self.assertTrue(scan_called)
        self.assertEqual(len(results), 2)
