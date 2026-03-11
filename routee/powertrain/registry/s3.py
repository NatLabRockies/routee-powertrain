from __future__ import annotations

import json
import re
from typing import List, Optional

from routee.powertrain.core.model import Model
from routee.powertrain.core.year import parse_year, year_contains
from routee.powertrain.io.archive import (
    _model_from_metadata_and_bytes,
    METADATA_FILENAME,
)
from routee.powertrain.registry.model_id import ModelId, ModelInfo
from routee.powertrain.registry.registry import ModelRegistry

# Pattern to extract version from path segment like "v1", "v2"
VERSION_RE = re.compile(r"^v(\d+)$")


def _parse_model_id_from_key(key: str, schema_version: str) -> ModelId:
    """
    Derive a ModelId from an S3 key.

    Expected key format:
        <schema_version>/<make>/<model>/<year>/<trim>/<variant>/v<N>/metadata.json
    """
    prefix = schema_version + "/"
    if not key.startswith(prefix):
        raise ValueError(f"Key {key} does not start with {prefix}")

    rel = key[len(prefix) :]
    parts = rel.split("/")
    # parts: [make, model, year, trim, variant, vN, metadata.json]
    if len(parts) != 7 or parts[-1] != METADATA_FILENAME:
        raise ValueError(
            f"Unexpected S3 key structure: {key}. "
            f"Expected <schema>/<make>/<model>/<year>/<trim>/<variant>/v<N>/{METADATA_FILENAME}"
        )

    make, model_name, year_str, trim, variant, version_dir, _ = parts
    match = VERSION_RE.match(version_dir)
    if not match:
        raise ValueError(f"Version directory '{version_dir}' does not match v<N>")

    return ModelId(
        make=make,
        model_name=model_name,
        year=parse_year(year_str),
        trim=trim,
        variant=variant,
        version=int(match.group(1)),
    )


def _model_info_from_metadata(
    metadata_dict: dict, model_id: ModelId, path: str
) -> ModelInfo:
    """Convert an archive metadata dict + ModelId into a ModelInfo."""
    config = metadata_dict["config"]

    est_errors = metadata_dict["errors"]["estimator_errors"]
    error_summary = {}
    for target_name, target_errors in est_errors["error_by_target"].items():
        error_summary[target_name] = {
            k: v for k, v in target_errors.items() if v is not None
        }

    feature_names = [f["name"] for f in config["feature_set"]["features"]]
    target_names = [t["name"] for t in config["target"]["targets"]]

    return ModelInfo(
        model_id=model_id,
        estimator_type=metadata_dict["estimator_type"],
        feature_names=feature_names,
        target_names=target_names,
        powertrain_type=config["powertrain_type"],
        errors=error_summary,
        vehicle_description=config["vehicle_description"],
        path=path,
    )


class S3Registry(ModelRegistry):
    """
    A model registry backed by a public S3 bucket.

    Models are stored as directories containing ``metadata.json`` and a
    binary model file.  Discovery uses ``ListObjectsV2`` to scan for
    ``metadata.json`` keys under the schema prefix.

    Bucket layout::

        s3://<bucket>/<schema_version>/<make>/<model>/<year>/<trim>/<variant>/v<N>/
            metadata.json
            model.onnx

    Requires ``boto3`` (install with ``pip install routee.powertrain[s3]``).

    Args:
        bucket: S3 bucket name
        schema_version: schema version to use (default "v2")
        region: AWS region for the bucket (default "us-west-2")
    """

    def __init__(
        self,
        bucket: str,
        schema_version: str = "v2",
        region: str = "us-west-2",
    ) -> None:
        self.bucket = bucket
        self.schema_version = schema_version
        self.region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
                from botocore import UNSIGNED
                from botocore.config import Config
            except ImportError:
                raise ImportError(
                    "S3Registry requires boto3. "
                    "Install with: pip install routee.powertrain[s3]"
                )
            self._client = boto3.client(
                "s3",
                region_name=self.region,
                config=Config(signature_version=UNSIGNED),
            )
        return self._client

    def _fetch_bytes(self, key: str) -> bytes:
        client = self._get_client()
        response = client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def _list_metadata_keys(self) -> List[str]:
        """List all metadata.json keys under the schema prefix using pagination."""
        client = self._get_client()
        prefix = f"{self.schema_version}/"
        keys: List[str] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(f"/{METADATA_FILENAME}"):
                    keys.append(key)
        return keys

    def _scan_models(self) -> List[ModelInfo]:
        """Scan the bucket for all models and return their metadata."""
        results: List[ModelInfo] = []
        for key in self._list_metadata_keys():
            try:
                model_id = _parse_model_id_from_key(key, self.schema_version)
                data = self._fetch_bytes(key)
                metadata_dict = json.loads(data)
                # path is the directory prefix (key without /metadata.json)
                dir_key = key[: -len(f"/{METADATA_FILENAME}")]
                info = _model_info_from_metadata(metadata_dict, model_id, dir_key)
                results.append(info)
            except Exception:
                continue
        return results

    def query(
        self,
        make: Optional[str] = None,
        model_name: Optional[str] = None,
        year: Optional[int] = None,
        trim: Optional[str] = None,
        variant: Optional[str] = None,
    ) -> List[ModelInfo]:
        results = self._scan_models()
        if make is not None:
            make_lower = make.lower()
            results = [m for m in results if m.model_id.make == make_lower]
        if model_name is not None:
            model_name_lower = model_name.lower()
            results = [m for m in results if m.model_id.model_name == model_name_lower]
        if year is not None:
            results = [m for m in results if year_contains(m.model_id.year, year)]
        if trim is not None:
            trim_lower = trim.lower()
            results = [m for m in results if m.model_id.trim == trim_lower]
        if variant is not None:
            variant_lower = variant.lower()
            results = [m for m in results if m.model_id.variant == variant_lower]
        return results

    def load(self, model_id: ModelId) -> Model:
        dir_key = model_id.to_path(self.schema_version)
        # Fetch metadata to learn the model filename
        meta_key = f"{dir_key}/{METADATA_FILENAME}"
        meta_bytes = self._fetch_bytes(meta_key)
        metadata_dict = json.loads(meta_bytes)

        model_filename = metadata_dict.get("model_file")
        if model_filename is None:
            raise ValueError("metadata.json must contain 'model_file'")

        model_key = f"{dir_key}/{model_filename}"
        model_bytes = self._fetch_bytes(model_key)
        return _model_from_metadata_and_bytes(metadata_dict, model_bytes)

    def get_metadata(self, model_id: ModelId) -> dict:
        dir_key = model_id.to_path(self.schema_version)
        meta_key = f"{dir_key}/{METADATA_FILENAME}"
        data = self._fetch_bytes(meta_key)
        return json.loads(data)
