from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Union

from routee.powertrain.core.model import Model
from routee.powertrain.core.year import parse_year
from routee.powertrain.io.archive import (
    load_model_directory,
    read_directory_metadata,
    METADATA_FILENAME,
)
from routee.powertrain.core.metadata import SCHEMA_VERSION_STRING
from routee.powertrain.registry.filtering import filter_models
from routee.powertrain.registry.model_id import ModelId, ModelInfo
from routee.powertrain.registry.registry import ModelRegistry, _resolve_model_id

# Pattern to extract version number from directory name like v1, v2
VERSION_RE = re.compile(r"^v(\d+)$")


def _parse_model_id_from_path(model_dir: Path, schema_root: Path) -> ModelId:
    """
    Derive a ModelId from the directory path relative to the schema root.

    Expected structure: <make>/<model>/<year>/<variant>/<feature_set_id>/v<N>/
    """
    rel = model_dir.relative_to(schema_root)
    parts = list(rel.parts)

    if len(parts) != 6:
        raise ValueError(
            f"Unexpected path depth for {model_dir}. "
            f"Expected <make>/<model>/<year>/<variant>/<feature_set_id>/v<N>, "
            f"got {'/'.join(parts)}"
        )

    make, model_name, year_str, variant, feature_set_id, version_dir = parts

    match = VERSION_RE.match(version_dir)
    if not match:
        raise ValueError(
            f"Directory name '{version_dir}' does not match expected pattern v<N>"
        )

    return ModelId(
        make=make,
        model_name=model_name,
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


class LocalRegistry(ModelRegistry):
    """
    A model registry backed by a local filesystem directory.

    Models are stored as flat directories containing ``metadata.json``
    and a binary model file.  Discovery is done by scanning for
    ``metadata.json`` files rather than reading a catalog index.

    Directory layout::

        <root>/<schema_version>/<make>/<model>/<year>/<variant>/<feature_set_id>/v<N>/
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
                info = _model_info_from_metadata(metadata_dict, model_id, rel_path)
                results.append(info)
            except Exception:
                continue  # skip malformed entries
        return results

    def list_models(self) -> List[ModelId]:
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
        return results

    def query(
        self,
        make: Optional[str] = None,
        model_name: Optional[str] = None,
        year: Optional[int] = None,
        variant: Optional[str] = None,
        feature_set_id: Optional[str] = None,
        fuzzy: bool = True,
        fuzzy_threshold: int = 80,
    ) -> List[ModelInfo]:
        results = self._scan_models()
        return filter_models(
            results,
            make=make,
            model_name=model_name,
            year=year,
            variant=variant,
            feature_set_id=feature_set_id,
            fuzzy=fuzzy,
            fuzzy_threshold=fuzzy_threshold,
        )

    def load(self, model_id: Union[str, ModelId]) -> Model:
        model_id = _resolve_model_id(model_id)
        full_path = self._schema_root / model_id.to_path()
        if not full_path.exists():
            raise FileNotFoundError(f"Model directory not found: {full_path}")
        return load_model_directory(full_path)

    def get_metadata(self, model_id: Union[str, ModelId]) -> dict:
        model_id = _resolve_model_id(model_id)
        full_path = self._schema_root / model_id.to_path()
        if not full_path.exists():
            raise FileNotFoundError(f"Model directory not found: {full_path}")
        return read_directory_metadata(full_path)
