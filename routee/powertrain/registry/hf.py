from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Union

from routee.powertrain.core.metadata import SCHEMA_VERSION_STRING
from routee.powertrain.core.model import Model
from routee.powertrain.io.archive import (
    _model_filename,
    _model_from_metadata_and_bytes,
    METADATA_FILENAME,
)
from routee.powertrain.registry.default import (
    DEFAULT_HF_REPO_ID,
    DEFAULT_HF_REPO_TYPE,
)
from routee.powertrain.registry.entry import (
    model_info_from_metadata,
    parse_model_id_from_metadata_key,
)
from routee.powertrain.registry.filtering import (
    VersionStrategy,
    filter_models,
    latest_model_ids,
)
from routee.powertrain.registry.model_id import ModelId, ModelInfo
from routee.powertrain.registry.registry import (
    INDEX_FILENAME,
    IndexMissingError,
    ModelRegistry,
    _resolve_model_id,
)
from routee.powertrain.registry.slug import assert_metadata_matches_id

log = logging.getLogger(__name__)

_MISSING_HF_HUB = (
    "The HuggingFace registry backend requires huggingface_hub, which is not "
    "installed. Install it with: pip install huggingface_hub"
)


def _import_hf_api() -> Any:
    """Import ``HfApi`` lazily so an import error is reported in context."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(_MISSING_HF_HUB) from exc
    return HfApi


class HFRegistry(ModelRegistry):
    """
    A model registry backed by a HuggingFace Hub repository.

    The repository holds the same directory tree as the S3 backend, so a model
    is addressable by the very same ``ModelId`` path. Discovery requires an
    ``index.json`` at the schema root; build it with :func:`build_index`.

    Repository layout::

        <repo_id>/<root_prefix>/<schema_version>/
            index.json
            <make>/<vehicle_slug>/<year>/<config_slug>/v<N>/
                metadata.json
                model.onnx

    Two things this backend gets that the S3 one cannot express:

    * **Caching.** Downloads land in the shared HuggingFace cache
      (``~/.cache/huggingface`` unless ``HF_HOME`` says otherwise), so loading
      the same model twice hits the network once.
    * **Reproducible pins.** ``revision`` accepts a branch, tag, or commit sha.
      Pinning a sha freezes the entire library — every model and the index —
      to an exact state.

    Args:
        repo_id: the Hub repository holding the model library
            (e.g. ``"NatLabRockies/routee-powertrain-model-library"``)
        schema_version: schema version to use (default "v2")
        repo_type: ``"model"`` (default) or ``"dataset"``
        revision: branch, tag, or commit sha to read. ``None`` reads the
            repository's default branch.
        token: Hub access token. ``None`` (default) reads anonymously, which
            is all a public repository needs.
        root_prefix: folder in the repository under which the schema-versioned
            tree lives. Empty (default) puts ``<schema_version>/`` at the
            repository root.
    """

    def __init__(
        self,
        repo_id: str = DEFAULT_HF_REPO_ID,
        schema_version: str = SCHEMA_VERSION_STRING,
        repo_type: str = DEFAULT_HF_REPO_TYPE,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        root_prefix: str = "",
    ) -> None:
        self.repo_id = repo_id
        self.schema_version = schema_version
        self.repo_type = repo_type
        self.revision = revision
        self.token = token
        self.root_prefix = root_prefix.strip("/")
        self._client = None

    def _get_client(self):
        if self._client is None:
            hf_api_cls = _import_hf_api()
            self._client = hf_api_cls(token=self.token)
        return self._client

    def _fetch_bytes(self, path: str) -> bytes:
        """Download one file from the repository and return its bytes.

        The file is fetched into the local HuggingFace cache, so repeat calls
        for the same revision do not re-download.
        """
        client = self._get_client()
        local_path = client.hf_hub_download(
            repo_id=self.repo_id,
            filename=path,
            repo_type=self.repo_type,
            revision=self.revision,
        )
        return Path(local_path).read_bytes()

    def _repo_prefix(self) -> str:
        """Return the repo-relative prefix for schema-versioned models."""
        if self.root_prefix:
            return f"{self.root_prefix}/{self.schema_version}"
        return self.schema_version

    def _list_metadata_paths(self) -> List[str]:
        """List every metadata.json path under the schema prefix (for build_index)."""
        client = self._get_client()
        prefix = f"{self._repo_prefix()}/"
        files = client.list_repo_files(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            revision=self.revision,
        )
        return sorted(
            f
            for f in files
            if f.startswith(prefix) and f.endswith(f"/{METADATA_FILENAME}")
        )

    def _fetch_index(self) -> List[ModelInfo]:
        """Fetch and parse the index.json at the schema root.

        Raises:
            IndexMissingError: if the index is missing or unreadable. Callers
                should surface this with guidance to run ``build_index``.
        """
        path = f"{self._repo_prefix()}/{INDEX_FILENAME}"
        try:
            data = self._fetch_bytes(path)
        except Exception as exc:
            raise IndexMissingError(
                f"Could not read '{path}' from the HuggingFace repo "
                f"'{self.repo_id}'. The HuggingFace registry requires an "
                "index.json at the schema root — run "
                "routee.powertrain.registry.hf.build_index to generate one."
            ) from exc
        try:
            index_dict = json.loads(data)
            return [ModelInfo.model_validate(m) for m in index_dict.get("models", [])]
        except Exception as exc:
            raise IndexMissingError(
                f"Could not parse index at '{path}': {exc}"
            ) from exc

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
        model_digest: Optional[str] = None,
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
            model_digest=model_digest,
            version_strategy=version_strategy,
            custom_filters=custom_filters,
            fuzzy=fuzzy,
            fuzzy_threshold=fuzzy_threshold,
        )

    def load(self, model_id: Union[str, ModelId]) -> Model:
        model_id = _resolve_model_id(model_id)
        dir_path = f"{self._repo_prefix()}/{model_id.to_path()}"
        # Fetch metadata to learn the model filename
        meta_path = f"{dir_path}/{METADATA_FILENAME}"
        meta_bytes = self._fetch_bytes(meta_path)
        metadata_dict = json.loads(meta_bytes)

        model_filename = _model_filename(metadata_dict)

        model_bytes = self._fetch_bytes(f"{dir_path}/{model_filename}")
        model = _model_from_metadata_and_bytes(metadata_dict, model_bytes)
        assert_metadata_matches_id(model.metadata, model_id)
        return model

    def get_metadata(self, model_id: Union[str, ModelId]) -> dict:
        model_id = _resolve_model_id(model_id)
        dir_path = f"{self._repo_prefix()}/{model_id.to_path()}"
        data = self._fetch_bytes(f"{dir_path}/{METADATA_FILENAME}")
        return json.loads(data)


def build_index(
    repo_id: str = DEFAULT_HF_REPO_ID,
    schema_version: str = SCHEMA_VERSION_STRING,
    repo_type: str = DEFAULT_HF_REPO_TYPE,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    root_prefix: str = "",
    dry_run: bool = False,
) -> dict:
    """
    Scan the HuggingFace repo for all models and build a ModelInfo index.

    Writing the index requires a token with write access to ``repo_id``.
    """
    registry = HFRegistry(
        repo_id=repo_id,
        schema_version=schema_version,
        repo_type=repo_type,
        revision=revision,
        token=token,
        root_prefix=root_prefix,
    )
    client = registry._get_client()

    log.info("Scanning %s/%s for models...", repo_id, registry._repo_prefix())

    models = []
    for path in registry._list_metadata_paths():
        try:
            model_id = parse_model_id_from_metadata_key(path, registry._repo_prefix())
            data = registry._fetch_bytes(path)
            metadata_dict = json.loads(data)
            # path is the directory (the metadata path without /metadata.json)
            dir_path = path[: -len(f"/{METADATA_FILENAME}")]
            info = model_info_from_metadata(metadata_dict, model_id, dir_path)
            models.append(info.model_dump(mode="json"))
        except Exception:
            log.warning("Skipping unreadable registry entry: %s", path, exc_info=True)
            continue

    index = {
        "schema_version": schema_version,
        "models": models,
    }

    index_path = f"{registry._repo_prefix()}/{INDEX_FILENAME}"
    if dry_run:
        log.info(
            "[dry-run] Would write index with %d models to %s/%s",
            len(models),
            repo_id,
            index_path,
        )
    else:
        log.info(
            "Writing index with %d models to %s/%s", len(models), repo_id, index_path
        )
        client.upload_file(
            path_or_fileobj=json.dumps(index, indent=2).encode(),
            path_in_repo=index_path,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            commit_message=f"Rebuild {schema_version} model index ({len(models)} models)",
        )

    return index
