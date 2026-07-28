from __future__ import annotations

import os
from pathlib import Path

from routee.powertrain.core.metadata import SCHEMA_VERSION_STRING
from routee.powertrain.registry.registry import ModelRegistry

DEFAULT_BUCKET = "routeecore-bucket"
DEFAULT_REGION = "us-west-2"
DEFAULT_ROOT_PREFIX = "routee-powertrain-model-library"

DEFAULT_HF_REPO_ID = "nreinicke/routee-powertrain-model-library"
DEFAULT_HF_REPO_TYPE = "model"


def _bundled_registry_root() -> Path:
    """Return the path to the bundled local model registry shipped with the package."""
    from routee.powertrain.resources.bundled_registry import bundled_registry_root

    return bundled_registry_root()


def get_default_registry() -> ModelRegistry:
    """
    Build the default model registry from environment variables.

    The default backend is ``"hf"``, which fetches models from a public
    HuggingFace Hub repository. Set ``ROUTEE_REGISTRY_BACKEND=s3`` to read the
    S3 bucket instead (this needs the optional ``routee.powertrain[s3]``
    install), or ``ROUTEE_REGISTRY_BACKEND=local`` to use a local filesystem
    registry (e.g. for CI or offline use). When the local backend is selected,
    the bundled registry shipped with the package is used unless
    ``ROUTEE_LOCAL_REGISTRY_ROOT`` is set.

    Environment variables:
        ROUTEE_REGISTRY_BACKEND: "hf" (default), "s3", or "local"
        ROUTEE_SCHEMA_VERSION: schema version (default: v2)
        ROUTEE_HF_REPO_ID: HuggingFace repo holding the model library
            (default: NatLabRockies/routee-powertrain-model-library)
        ROUTEE_HF_REPO_TYPE: "model" (default) or "dataset"
        ROUTEE_HF_REVISION: branch, tag, or commit sha to read
            (default: the repo's default branch). Pin a commit sha to freeze
            the whole library to an exact state.
        ROUTEE_HF_TOKEN: Hub access token; unset reads anonymously, which is
            all a public repo needs
        ROUTEE_S3_BUCKET: S3 bucket name (default: routeecore-bucket)
        ROUTEE_S3_REGION: AWS region (default: us-west-2)
        ROUTEE_S3_ROOT_PREFIX: Top-level folder in the S3 bucket
            (default: routee-powertrain-model-library)
        ROUTEE_LOCAL_REGISTRY_ROOT: root directory for local registry backend
            (default: bundled registry shipped with the package)

    Returns: a ModelRegistry (HFRegistry, S3Registry, or LocalRegistry)
    """
    backend = os.environ.get("ROUTEE_REGISTRY_BACKEND", "hf")
    schema_version = os.environ.get("ROUTEE_SCHEMA_VERSION", SCHEMA_VERSION_STRING)

    inner: ModelRegistry
    if backend == "hf":
        from routee.powertrain.registry.hf import HFRegistry

        inner = HFRegistry(
            repo_id=os.environ.get("ROUTEE_HF_REPO_ID", DEFAULT_HF_REPO_ID),
            schema_version=schema_version,
            repo_type=os.environ.get("ROUTEE_HF_REPO_TYPE", DEFAULT_HF_REPO_TYPE),
            revision=os.environ.get("ROUTEE_HF_REVISION"),
            token=os.environ.get("ROUTEE_HF_TOKEN"),
        )
    elif backend == "s3":
        from routee.powertrain.registry.s3 import S3Registry

        bucket = os.environ.get("ROUTEE_S3_BUCKET", DEFAULT_BUCKET)
        region = os.environ.get("ROUTEE_S3_REGION", DEFAULT_REGION)
        root_prefix = os.environ.get("ROUTEE_S3_ROOT_PREFIX", DEFAULT_ROOT_PREFIX)
        inner = S3Registry(
            bucket=bucket,
            schema_version=schema_version,
            region=region,
            root_prefix=root_prefix,
        )
    elif backend == "local":
        from routee.powertrain.registry.local import LocalRegistry

        root_str = os.environ.get("ROUTEE_LOCAL_REGISTRY_ROOT")
        root = Path(root_str) if root_str else _bundled_registry_root()
        inner = LocalRegistry(root=root, schema_version=schema_version)
    else:
        raise ValueError(
            f"Unknown registry backend: '{backend}'. Use 'hf', 's3', or 'local'."
        )

    return inner
