import json
from unittest import TestCase
from unittest.mock import MagicMock
from routee.powertrain.registry.s3 import S3Registry, build_index, INDEX_FILENAME
from routee.powertrain.registry.model_id import ModelId, ModelInfo

ROOT = "routee-powertrain-model-library"
SCHEMA = "v2"
BASE = f"{ROOT}/{SCHEMA}"


def _fake_metadata(
    make: str = "toyota",
    model: str = "camry",
    year: int = 2016,
) -> dict:
    return {
        "estimator_type": "some-type",
        "model_file": "model.onnx",
        "config": {
            "vehicle_description": f"{year} {make} {model}",
            "powertrain_type": "ICE",
            "feature_set": {"features": [{"name": "speed"}]},
            "target": {"targets": [{"name": "energy"}]},
        },
    }


class TestS3Index(TestCase):
    def test_query_uses_index(self):
        # Create a mock index
        model_id = ModelId("toyota", "camry", 2016, "rf_default", 1)
        info = ModelInfo(
            model_id=model_id,
            estimator_type="some-type",
            feature_names=["speed"],
            target_names=["energy"],
            powertrain_type="ICE",
            vehicle_description="2016 toyota camry",
            path=f"{BASE}/toyota/camry/2016/rf_default/v1",
        )
        index = {"schema_version": SCHEMA, "models": [info.model_dump(mode="json")]}

        registry = S3Registry(
            bucket="test-bucket", schema_version=SCHEMA, root_prefix=ROOT
        )

        # Mock _fetch_bytes to return the index when index.json is requested
        def mock_fetch_bytes(key):
            if key.endswith(INDEX_FILENAME):
                return json.dumps(index).encode()
            raise ValueError(f"Unexpected key: {key}")

        registry._fetch_bytes = mock_fetch_bytes

        # Querying should use the index and return the model
        results = registry.query(make="toyota")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_id, model_id)

    def test_build_index(self):
        # Mock S3Registry and client
        registry_mock = MagicMock()
        registry_mock._s3_prefix.return_value = BASE

        # Mock metadata listing
        metadata_key = f"{BASE}/toyota/camry/2016/rf_default/v1/metadata.json"
        registry_mock._list_metadata_keys.return_value = [metadata_key]

        # Mock metadata fetching
        meta_dict = _fake_metadata()
        registry_mock._fetch_bytes.return_value = json.dumps(meta_dict).encode()

        # Mock client
        client_mock = MagicMock()
        registry_mock._get_client.return_value = client_mock

        # We need to patch S3Registry to return our mock
        with MagicMock() as _:
            import routee.powertrain.registry.s3 as s3_module

            original_s3_registry = s3_module.S3Registry
            s3_module.S3Registry = lambda **kwargs: registry_mock

            try:
                idx = build_index(
                    bucket="test-bucket", schema_version=SCHEMA, root_prefix=ROOT
                )
                self.assertEqual(len(idx["models"]), 1)
                self.assertEqual(idx["models"][0]["model_id"]["make"], "toyota")

                # Verify it was uploaded
                client_mock.put_object.assert_called_once()
                args, kwargs = client_mock.put_object.call_args
                self.assertEqual(kwargs["Bucket"], "test-bucket")
                self.assertEqual(kwargs["Key"], f"{BASE}/{INDEX_FILENAME}")
            finally:
                s3_module.S3Registry = original_s3_registry


if __name__ == "__main__":
    import unittest

    unittest.main()
