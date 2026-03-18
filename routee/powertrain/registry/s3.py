from __future__ import annotations

import json
import logging
import re
from typing import List, Optional, Union

from routee.powertrain.core.metadata import SCHEMA_VERSION_STRING
from routee.powertrain.core.model import Model
from routee.powertrain.core.year import parse_year, year_contains
from routee.powertrain.io.archive import (
    _model_from_metadata_and_bytes,
    METADATA_FILENAME,
)
from routee.powertrain.registry.filtering import _matches, filter_models
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
        [<root_prefix>/]<schema_version>/<make>/<model>/<year>/<variant>/<feature_set_id>/v<N>/metadata.json
    """
    if root_prefix:
        full_prefix = root_prefix + "/" + schema_version + "/"
    else:
        full_prefix = schema_version + "/"
    if not key.startswith(full_prefix):
        raise ValueError(f"Key {key} does not start with {full_prefix}")

    rel = key[len(full_prefix) :]
    parts = rel.split("/")
    # parts: [make, model, year, variant, feature_set_id, vN, metadata.json]
    if len(parts) != 7 or parts[-1] != METADATA_FILENAME:
        raise ValueError(
            f"Unexpected S3 key structure: {key}. "
            f"Expected <schema>/<make>/<model>/<year>/<variant>/<feature_set_id>/v<N>/{METADATA_FILENAME}"
        )

    make, model, year_str, variant, feature_set_id, version_dir, _ = parts
    match = VERSION_RE.match(version_dir)
    if not match:
        raise ValueError(f"Version directory '{version_dir}' does not match v<N>")

    return ModelId(
        make=make,
        model=model,
        year=parse_year(year_str),
        variant=variant,
        feature_set_id=feature_set_id,
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
        feature_names=feature_names,
        target_names=target_names,
        powertrain_type=config["powertrain_type"],
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

        s3://<bucket>/<root_prefix>/<schema_version>/<make>/<model>/<year>/<variant>/<feature_set_id>/v<N>/
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

    def _list_children(self, prefix: str) -> List[str]:
        """List immediate child directory names under an S3 prefix.

        Uses ``Delimiter='/'`` so S3 returns ``CommonPrefixes`` without
        downloading any objects.  Handles pagination.

        Args:
            prefix: S3 key prefix ending with ``/``

        Returns:
            List of child directory names (without trailing ``/``)
        """
        client = self._get_client()
        children: List[str] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self.bucket, Prefix=prefix, Delimiter="/"
        ):
            for cp in page.get("CommonPrefixes", []):
                # cp["Prefix"] looks like "<prefix><child>/"
                child = cp["Prefix"][len(prefix) :].rstrip("/")
                if child:
                    children.append(child)
        return children

    def _narrow_prefixes(
        self,
        prefixes: List[str],
        query_value: Optional[str],
        fuzzy: bool,
        threshold: int,
        is_year: bool = False,
        year_query: Optional[int] = None,
    ) -> List[str]:
        """Expand prefixes by one hierarchy level, optionally filtering.

        For each prefix, lists child directories via S3 and applies
        fuzzy or exact matching against ``query_value`` to narrow the
        result set.  For the year level, set ``is_year=True`` and pass
        the numeric year as ``year_query`` to use ``year_contains``.

        Args:
            prefixes: S3 key prefixes to expand
            query_value: string filter (make, model, etc.) or None
            fuzzy: whether to use fuzzy string matching
            threshold: fuzzy match threshold (0–100)
            is_year: if True, filter using ``year_contains`` instead
            year_query: numeric year used when ``is_year`` is True

        Returns:
            List of narrowed S3 prefixes (one level deeper)
        """
        next_prefixes: List[str] = []
        for prefix in prefixes:
            children = self._list_children(prefix)
            for child in children:
                child_prefix = f"{prefix}{child}/"
                if is_year and year_query is not None:
                    try:
                        child_year = parse_year(child)
                    except (ValueError, TypeError):
                        continue
                    if year_contains(child_year, year_query):
                        next_prefixes.append(child_prefix)
                elif query_value is not None:
                    if _matches(query_value, child.lower(), fuzzy, threshold):
                        next_prefixes.append(child_prefix)
                else:
                    next_prefixes.append(child_prefix)
        return next_prefixes

    def _list_metadata_keys(self) -> List[str]:
        """List all metadata.json keys under the schema prefix using pagination."""
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

    def _fetch_index(self) -> Optional[List[ModelInfo]]:
        """Fetch the index.json from the bucket if it exists."""
        key = f"{self._s3_prefix()}/{INDEX_FILENAME}"
        try:
            data = self._fetch_bytes(key)
            index_dict = json.loads(data)
            return [ModelInfo.from_dict(m) for m in index_dict.get("models", [])]
        except Exception:
            return None

    def _scan_models(self) -> List[ModelInfo]:
        """Scan the bucket for all models and return their metadata."""
        index = self._fetch_index()
        if index is not None:
            return index

        results: List[ModelInfo] = []
        for key in self._list_metadata_keys():
            try:
                model_id = _parse_model_id_from_key(
                    key, self.schema_version, self.root_prefix
                )
                data = self._fetch_bytes(key)
                metadata_dict = json.loads(data)
                # path is the directory prefix (key without /metadata.json)
                dir_key = key[: -len(f"/{METADATA_FILENAME}")]
                info = _model_info_from_metadata(metadata_dict, model_id, dir_key)
                results.append(info)
            except Exception:
                continue
        return results

    def list_models(self) -> List[ModelId]:
        index = self._fetch_index()
        if index is not None:
            return [m.model_id for m in index]

        results: List[ModelId] = []
        for key in self._list_metadata_keys():
            try:
                model_id = _parse_model_id_from_key(
                    key, self.schema_version, self.root_prefix
                )
                results.append(model_id)
            except Exception:
                continue
        return results

    def query(
        self,
        make: Optional[str] = None,
        model: Optional[str] = None,
        year: Optional[int] = None,
        variant: Optional[str] = None,
        feature_set_id: Optional[str] = None,
        fuzzy: bool = True,
        fuzzy_threshold: int = 80,
    ) -> List[ModelInfo]:
        index = self._fetch_index()
        if index is not None:
            return filter_models(
                index,
                make=make,
                model=model,
                year=year,
                variant=variant,
                feature_set_id=feature_set_id,
                fuzzy=fuzzy,
                fuzzy_threshold=fuzzy_threshold,
            )

        has_filters = any(
            v is not None for v in (make, model, year, variant, feature_set_id)
        )
        if not has_filters:
            return self._scan_models()

        # Walk the S3 hierarchy level-by-level, narrowing at each step.
        # Levels: make / model / year / variant / feature_set_id / version
        prefixes = [f"{self._s3_prefix()}/"]

        # Level 1: make
        prefixes = self._narrow_prefixes(prefixes, make, fuzzy, fuzzy_threshold)
        # Level 2: model
        prefixes = self._narrow_prefixes(prefixes, model, fuzzy, fuzzy_threshold)
        # Level 3: year
        prefixes = self._narrow_prefixes(
            prefixes,
            None,
            fuzzy,
            fuzzy_threshold,
            is_year=(year is not None),
            year_query=year,
        )
        # Level 4: variant
        prefixes = self._narrow_prefixes(prefixes, variant, fuzzy, fuzzy_threshold)
        # Level 5: feature_set_id
        prefixes = self._narrow_prefixes(
            prefixes, feature_set_id, fuzzy, fuzzy_threshold
        )
        # Level 6: version (expand all)
        prefixes = self._narrow_prefixes(prefixes, None, fuzzy, fuzzy_threshold)

        # Fetch metadata for narrowed results
        results: List[ModelInfo] = []
        for prefix in prefixes:
            meta_key = f"{prefix}{METADATA_FILENAME}"
            try:
                model_id = _parse_model_id_from_key(
                    meta_key, self.schema_version, self.root_prefix
                )
                data = self._fetch_bytes(meta_key)
                metadata_dict = json.loads(data)
                dir_key = prefix.rstrip("/")
                info = _model_info_from_metadata(metadata_dict, model_id, dir_key)
                results.append(info)
            except Exception:
                continue
        return results

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

    # We use _list_metadata_keys directly to avoid using an old index if it exists
    models = []
    for key in registry._list_metadata_keys():
        try:
            model_id = _parse_model_id_from_key(key, schema_version, root_prefix)
            data = registry._fetch_bytes(key)
            metadata_dict = json.loads(data)
            # path is the directory prefix (key without /metadata.json)
            dir_key = key[: -len(f"/{METADATA_FILENAME}")]
            info = _model_info_from_metadata(metadata_dict, model_id, dir_key)
            models.append(info.to_dict())
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
