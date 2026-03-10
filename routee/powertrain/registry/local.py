from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from routee.powertrain.core.model import Model
from routee.powertrain.io.archive import load_archive, read_archive_metadata
from routee.powertrain.registry.catalog import Catalog
from routee.powertrain.registry.model_id import ModelId, ModelInfo
from routee.powertrain.registry.registry import ModelRegistry

CATALOG_FILENAME = "catalog.json"


class LocalRegistry(ModelRegistry):
    """
    A model registry backed by a local filesystem directory.

    The directory follows the same path convention as S3:
        <root>/<schema_version>/catalog.json
        <root>/<schema_version>/<make>/<model>/<year>/<trim>/<variant>/v<N>.zip

    Args:
        root: path to the top-level directory
        schema_version: schema version to use (default "v2")
    """

    def __init__(
        self,
        root: Union[str, Path],
        schema_version: str = "v2",
    ) -> None:
        self.root = Path(root)
        self.schema_version = schema_version

    @property
    def _schema_root(self) -> Path:
        return self.root / self.schema_version

    def _catalog_path(self) -> Path:
        return self._schema_root / CATALOG_FILENAME

    def _load_catalog(self) -> Catalog:
        catalog_path = self._catalog_path()
        if not catalog_path.exists():
            raise FileNotFoundError(
                f"Catalog not found at {catalog_path}. "
                "Run the catalog generation script first."
            )
        return Catalog.from_json(catalog_path)

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
        full_path = self.root / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Model archive not found: {full_path}")
        return load_archive(full_path)

    def get_metadata(self, model_id: ModelId) -> dict:
        rel_path = model_id.to_path(self.schema_version)
        full_path = self.root / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Model archive not found: {full_path}")
        return read_archive_metadata(full_path)
