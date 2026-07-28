from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

from routee.powertrain.core.model import Model
from routee.powertrain.io.archive import (
    load_model_directory,
    read_directory_metadata,
    METADATA_FILENAME,
)
from routee.powertrain.core.metadata import SCHEMA_VERSION_STRING
from routee.powertrain.registry.entry import (
    model_info_from_metadata,
    parse_model_id_from_segments,
    VERSION_RE,
)
from routee.powertrain.registry.filtering import (
    VersionStrategy,
    filter_models,
    latest_model_ids,
)
from routee.powertrain.registry.model_id import ModelId, ModelInfo
from routee.powertrain.registry.registry import ModelRegistry, _resolve_model_id
from routee.powertrain.registry.slug import assert_metadata_matches_id

__all__ = ["LocalRegistry", "VERSION_RE"]


def _parse_model_id_from_path(model_dir: Path, schema_root: Path) -> ModelId:
    """
    Derive a ModelId from the directory path relative to the schema root.

    Expected structure: <make>/<vehicle_slug>/<year>/<config_slug>/v<N>/
    """
    rel = model_dir.relative_to(schema_root)
    return parse_model_id_from_segments(rel.parts, str(model_dir))


class LocalRegistry(ModelRegistry):
    """
    A model registry backed by a local filesystem directory.

    Models are stored as flat directories containing ``metadata.json``
    and a binary model file.  Discovery is done by scanning for
    ``metadata.json`` files rather than reading a catalog index.

    Directory layout::

        <root>/<schema_version>/<make>/<vehicle_slug>/<year>/<config_slug>/v<N>/
            metadata.json
            model.onnx  (or other binary)

    Args:
        root: path to the top-level directory
        schema_version: schema version to use (default "v2")
    """

    def __init__(
        self,
        root: Union[str, Path],
        schema_version: str = SCHEMA_VERSION_STRING,
    ) -> None:
        self.root = Path(root)
        self.schema_version = schema_version

    @property
    def _schema_root(self) -> Path:
        return self.root / self.schema_version

    def _scan_models(self) -> List[ModelInfo]:
        """Walk the directory tree and build ModelInfo from every metadata.json."""
        schema_root = self._schema_root
        if not schema_root.exists():
            return []

        results: List[ModelInfo] = []
        for meta_path in sorted(schema_root.glob(f"**/{METADATA_FILENAME}")):
            model_dir = meta_path.parent
            try:
                model_id = _parse_model_id_from_path(model_dir, schema_root)
                metadata_dict = read_directory_metadata(model_dir)
                rel_path = str(model_dir.relative_to(self.root))
                info = model_info_from_metadata(metadata_dict, model_id, rel_path)
                results.append(info)
            except Exception:
                continue  # skip malformed entries
        return results

    def list_models(
        self,
        version_strategy: VersionStrategy = "latest",
    ) -> List[ModelId]:
        schema_root = self._schema_root
        if not schema_root.exists():
            return []

        results: List[ModelId] = []
        for meta_path in sorted(schema_root.glob(f"**/{METADATA_FILENAME}")):
            model_dir = meta_path.parent
            try:
                model_id = _parse_model_id_from_path(model_dir, schema_root)
                results.append(model_id)
            except Exception:
                continue
        if version_strategy == "latest":
            return latest_model_ids(results)
        return results

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
            self._scan_models(),
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
            model_digest=model_digest,
            version=version,
            version_strategy=version_strategy,
            custom_filters=custom_filters,
            fuzzy=fuzzy,
            fuzzy_threshold=fuzzy_threshold,
        )

    def load(self, model_id: Union[str, ModelId]) -> Model:
        model_id = _resolve_model_id(model_id)
        full_path = self._schema_root / model_id.to_path()
        if not full_path.exists():
            raise FileNotFoundError(f"Model directory not found: {full_path}")
        model = load_model_directory(full_path)
        assert_metadata_matches_id(model.metadata, model_id)
        return model

    def get_metadata(self, model_id: Union[str, ModelId]) -> dict:
        model_id = _resolve_model_id(model_id)
        full_path = self._schema_root / model_id.to_path()
        if not full_path.exists():
            raise FileNotFoundError(f"Model directory not found: {full_path}")
        return read_directory_metadata(full_path)
