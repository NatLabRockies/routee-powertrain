from __future__ import annotations

import json
import logging
import re
from typing import Callable, List, Optional, Sequence, Union

from routee.powertrain.core.metadata import SCHEMA_VERSION_STRING
from routee.powertrain.core.model import Model
from routee.powertrain.core.year import parse_year
from routee.powertrain.io.archive import (
    _model_from_metadata_and_bytes,
    METADATA_FILENAME,
)
from routee.powertrain.registry.filtering import (
    VersionStrategy,
    filter_models,
    latest_model_ids,
)
from routee.powertrain.registry.model_id import ModelId, ModelInfo
from routee.powertrain.registry.registry import ModelRegistry, _resolve_model_id
from routee.powertrain.registry.default import (
    DEFAULT_BUCKET,
    DEFAULT_REGION,
    DEFAULT_ROOT_PREFIX,
)

import boto3

from botocore import UNSIGNED
from botocore.config import Config

# Pattern to extract version from path segment like "v1", "v2"
VERSION_RE = re.compile(r"^v(\d+)$")
INDEX_FILENAME = "index.json"


def _parse_model_id_from_key(
    key: str, schema_version: str, root_prefix: str = ""
) -> ModelId:
    """
    Derive a ModelId from an S3 key.

    Expected key format:
        [<root_prefix>/]<schema_version>/<make>/<model>/<year>/<config_slug>/v<N>/metadata.json
    """
    if root_prefix:
        full_prefix = root_prefix + "/" + schema_version + "/"
    else:
        full_prefix = schema_version + "/"
    if not key.startswith(full_prefix):
        raise ValueError(f"Key {key} does not start with {full_prefix}")

    rel = key[len(full_prefix) :]
    parts = rel.split("/")
    # parts: [make, model, year, config_slug, vN, metadata.json]
    if len(parts) != 6 or parts[-1] != METADATA_FILENAME:
        raise ValueError(
            f"Unexpected S3 key structure: {key}. "
            f"Expected <schema>/<make>/<model>/<year>/<config_slug>/v<N>/{METADATA_FILENAME}"
        )

    make, model, year_str, config_slug, version_dir, _ = parts
    match = VERSION_RE.match(version_dir)
    if not match:
        raise ValueError(f"Version directory '{version_dir}' does not match v<N>")

    return ModelId(
        make=make,
        model=model,
        year=parse_year(year_str),
        config_slug=config_slug,
        version=int(match.group(1)),
    )


def _model_info_from_metadata(
    metadata_dict: dict, model_id: ModelId, path: str
) -> ModelInfo:
    """Convert an archive metadata dict + ModelId into a ModelInfo."""
    config = metadata_dict["config"]

    feature_names = [f["name"] for f in config["feature_set"]["features"]]
    target_names = [t["name"] for t in config["target"]["targets"]]

    return ModelInfo(
        model_id=model_id,
        estimator_type=metadata_dict["estimator_type"],
        architecture_tag=metadata_dict.get("architecture_tag", "unknown"),
        input_spec=metadata_dict.get("input_spec"),
        feature_names=feature_names,
        target_names=target_names,
        powertrain_type=config["powertrain_type"],
        vehicle_description=config["vehicle_description"],
        path=path,
        mass_lbs=config.get("mass_lbs"),
        fuel_type=config.get("fuel_type"),
        drivetrain=config.get("drivetrain"),
        engine=config.get("engine"),
        trim=config.get("trim"),
    )


class IndexMissingError(RuntimeError):
    """Raised when ``index.json`` is missing or unreadable at the schema root."""


class S3Registry(ModelRegistry):
    """
    A model registry backed by a public S3 bucket.

    Models are stored as directories containing ``metadata.json`` and a
    binary model file. Discovery requires an ``index.json`` at the schema
    root; build it with :func:`build_index`.

    Bucket layout::

        s3://<bucket>/<root_prefix>/<schema_version>/
            index.json
            <make>/<model>/<year>/<config_slug>/v<N>/
                metadata.json
                model.onnx

    Args:
        bucket: S3 bucket name
        schema_version: schema version to use (default "v2")
        region: AWS region for the bucket (default "us-west-2")
        anonymous: If True, use unsigned requests for public bucket
            access. Set to False to use standard AWS credential resolution
            (environment variables, ~/.aws/credentials, IAM role, etc.).
        root_prefix: Top-level folder in the bucket under which all models
            are stored (default "routee-powertrain-model-library").
    """

    def __init__(
        self,
        bucket: str = DEFAULT_BUCKET,
        schema_version: str = SCHEMA_VERSION_STRING,
        region: str = DEFAULT_REGION,
        anonymous: bool = False,
        root_prefix: str = DEFAULT_ROOT_PREFIX,
    ) -> None:
        self.bucket = bucket
        self.schema_version = schema_version
        self.region = region
        self.anonymous = anonymous
        self.root_prefix = root_prefix.strip("/")
        self._client = None

    def _get_client(self):
        if self._client is None:
            kwargs = {"region_name": self.region}
            if self.anonymous:
                kwargs["config"] = Config(signature_version=UNSIGNED)
            self._client = boto3.client("s3", **kwargs)
        return self._client

    def _fetch_bytes(self, key: str) -> bytes:
        client = self._get_client()
        response = client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def _s3_prefix(self) -> str:
        """Return the full S3 key prefix for schema-versioned models."""
        if self.root_prefix:
            return f"{self.root_prefix}/{self.schema_version}"
        return self.schema_version

    def _list_metadata_keys(self) -> List[str]:
        """List all metadata.json keys under the schema prefix (for build_index)."""
        client = self._get_client()
        prefix = f"{self._s3_prefix()}/"
        keys: List[str] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(f"/{METADATA_FILENAME}"):
                    keys.append(key)
        return keys

    def _fetch_index(self) -> List[ModelInfo]:
        """Fetch and parse the index.json at the schema root.

        Raises:
            IndexMissingError: if the index is missing or unreadable. Callers
                should surface this with guidance to run ``build_index``.
        """
        key = f"{self._s3_prefix()}/{INDEX_FILENAME}"
        try:
            data = self._fetch_bytes(key)
        except Exception as exc:
            raise IndexMissingError(
                f"Could not read '{key}' from s3://{self.bucket}. "
                "The S3 registry requires an index.json at the schema root — "
                "run routee.powertrain.registry.s3.build_index to generate one."
            ) from exc
        try:
            index_dict = json.loads(data)
            return [ModelInfo.model_validate(m) for m in index_dict.get("models", [])]
        except Exception as exc:
            raise IndexMissingError(f"Could not parse index at '{key}': {exc}") from exc

    def list_models(
        self,
        version_strategy: VersionStrategy = "latest",
    ) -> List[ModelId]:
        ids = [m.model_id for m in self._fetch_index()]
        if version_strategy == "latest":
            return latest_model_ids(ids)
        return ids

    def query(
        self,
        make: Optional[str] = None,
        model: Optional[str] = None,
        year: Optional[int] = None,
        config_slug: Optional[str] = None,
        feature_names: Optional[Sequence[str]] = None,
        powertrain_type: Optional[str] = None,
        fuel_type: Optional[str] = None,
        drivetrain: Optional[str] = None,
        engine: Optional[str] = None,
        trim: Optional[str] = None,
        version: Optional[int] = None,
        version_strategy: VersionStrategy = "latest",
        custom_filters: Optional[Sequence[Callable[[ModelInfo], bool]]] = None,
        fuzzy: bool = True,
        fuzzy_threshold: int = 80,
    ) -> List[ModelInfo]:
        return filter_models(
            self._fetch_index(),
            make=make,
            model=model,
            year=year,
            config_slug=config_slug,
            feature_names=feature_names,
            powertrain_type=powertrain_type,
            fuel_type=fuel_type,
            drivetrain=drivetrain,
            engine=engine,
            trim=trim,
            version=version,
            version_strategy=version_strategy,
            custom_filters=custom_filters,
            fuzzy=fuzzy,
            fuzzy_threshold=fuzzy_threshold,
        )

    def load(self, model_id: Union[str, ModelId]) -> Model:
        model_id = _resolve_model_id(model_id)
        dir_key = f"{self._s3_prefix()}/{model_id.to_path()}"
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

    def get_metadata(self, model_id: Union[str, ModelId]) -> dict:
        model_id = _resolve_model_id(model_id)
        dir_key = f"{self._s3_prefix()}/{model_id.to_path()}"
        meta_key = f"{dir_key}/{METADATA_FILENAME}"
        data = self._fetch_bytes(meta_key)
        return json.loads(data)


def build_index(
    bucket: str = DEFAULT_BUCKET,
    schema_version: str = SCHEMA_VERSION_STRING,
    region: str = DEFAULT_REGION,
    root_prefix: str = DEFAULT_ROOT_PREFIX,
    dry_run: bool = False,
) -> dict:
    """
    Scan the S3 bucket for all models and build a ModelInfo index.
    """
    registry = S3Registry(
        bucket=bucket,
        schema_version=schema_version,
        region=region,
        root_prefix=root_prefix,
    )
    client = registry._get_client()

    log = logging.getLogger(__name__)
    log.info("Scanning s3://%s/%s for models...", bucket, registry._s3_prefix())

    models = []
    for key in registry._list_metadata_keys():
        try:
            model_id = _parse_model_id_from_key(key, schema_version, root_prefix)
            data = registry._fetch_bytes(key)
            metadata_dict = json.loads(data)
            # path is the directory prefix (key without /metadata.json)
            dir_key = key[: -len(f"/{METADATA_FILENAME}")]
            info = _model_info_from_metadata(metadata_dict, model_id, dir_key)
            models.append(info.model_dump(mode="json"))
        except Exception:
            continue

    index = {
        "schema_version": schema_version,
        "models": models,
    }

    index_key = f"{registry._s3_prefix()}/{INDEX_FILENAME}"
    if dry_run:
        log.info(
            "[dry-run] Would write index with %d models to s3://%s/%s",
            len(models),
            bucket,
            index_key,
        )
    else:
        log.info(
            "Writing index with %d models to s3://%s/%s", len(models), bucket, index_key
        )
        client.put_object(
            Bucket=bucket,
            Key=index_key,
            Body=json.dumps(index, indent=2),
            ContentType="application/json",
        )

    return index
