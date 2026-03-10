from __future__ import annotations

import os
from pathlib import Path

from routee.powertrain.registry.registry import ModelRegistry

DEFAULT_BUCKET = "routee-powertrain-models"
DEFAULT_SCHEMA_VERSION = "v2"
DEFAULT_REGION = "us-west-2"


def get_default_registry() -> ModelRegistry:
    """
    Build the default model registry from environment variables.

    Environment variables:
        ROUTEE_REGISTRY_BACKEND: "s3" (default) or "local"
        ROUTEE_S3_BUCKET: S3 bucket name (default: routee-powertrain-models)
        ROUTEE_S3_REGION: AWS region (default: us-west-2)
        ROUTEE_SCHEMA_VERSION: schema version (default: v2)
        ROUTEE_CACHE_DIR: local cache directory (default: ~/.routee/cache/)
        ROUTEE_LOCAL_REGISTRY_ROOT: root directory for local registry backend

    Returns: a ModelRegistry (typically CachedRegistry wrapping S3Registry)
    """
    backend = os.environ.get("ROUTEE_REGISTRY_BACKEND", "s3")
    schema_version = os.environ.get("ROUTEE_SCHEMA_VERSION", DEFAULT_SCHEMA_VERSION)
    cache_dir_str = os.environ.get("ROUTEE_CACHE_DIR")
    cache_dir = Path(cache_dir_str) if cache_dir_str else None

    inner: ModelRegistry
    if backend == "s3":
        from routee.powertrain.registry.s3 import S3Registry

        bucket = os.environ.get("ROUTEE_S3_BUCKET", DEFAULT_BUCKET)
        region = os.environ.get("ROUTEE_S3_REGION", DEFAULT_REGION)
        inner = S3Registry(
            bucket=bucket,
            schema_version=schema_version,
            region=region,
        )
    elif backend == "local":
        from routee.powertrain.registry.local import LocalRegistry

        root = os.environ.get("ROUTEE_LOCAL_REGISTRY_ROOT")
        if root is None:
            raise ValueError(
                "ROUTEE_LOCAL_REGISTRY_ROOT must be set when using the 'local' backend"
            )
        inner = LocalRegistry(root=root, schema_version=schema_version)
    else:
        raise ValueError(f"Unknown registry backend: '{backend}'. Use 's3' or 'local'.")

    from routee.powertrain.registry.cache import CachedRegistry

    return CachedRegistry(inner=inner, cache_dir=cache_dir)
