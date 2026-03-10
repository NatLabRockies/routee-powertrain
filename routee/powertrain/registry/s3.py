from __future__ import annotations

import json
from typing import List, Optional

from routee.powertrain.core.model import Model
from routee.powertrain.io.archive import load_archive_bytes
from routee.powertrain.registry.catalog import Catalog
from routee.powertrain.registry.model_id import ModelId, ModelInfo
from routee.powertrain.registry.registry import ModelRegistry

CATALOG_FILENAME = "catalog.json"


class S3Registry(ModelRegistry):
    """
    A model registry backed by a public S3 bucket.

    The bucket follows the path convention:
        s3://<bucket>/<schema_version>/catalog.json
        s3://<bucket>/<schema_version>/<make>/<model>/<year>/<trim>/<variant>/v<N>.zip

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

    def _load_catalog(self) -> Catalog:
        key = f"{self.schema_version}/{CATALOG_FILENAME}"
        data = self._fetch_bytes(key)
        return Catalog.from_dict(json.loads(data))

    def query(
        self,
        make: Optional[str] = None,
        model_name: Optional[str] = None,
        year: Optional[int] = None,
        trim: Optional[str] = None,
        variant: Optional[str] = None,
    ) -> List[ModelInfo]:
        catalog = self._load_catalog()
        return catalog.query(
            make=make,
            model_name=model_name,
            year=year,
            trim=trim,
            variant=variant,
        )

    def load(self, model_id: ModelId) -> Model:
        rel_path = model_id.to_path(self.schema_version)
        data = self._fetch_bytes(rel_path)
        return load_archive_bytes(data)

    def get_metadata(self, model_id: ModelId) -> dict:
        # For S3, we can get metadata from the catalog without downloading the zip
        catalog = self._load_catalog()
        matches = catalog.query(
            make=model_id.make,
            model_name=model_id.model_name,
            year=model_id.year,
            trim=model_id.trim,
            variant=model_id.variant,
        )
        for m in matches:
            if m.model_id.version == model_id.version:
                return m.to_dict()
        raise ValueError(f"Model not found in catalog: {model_id}")
