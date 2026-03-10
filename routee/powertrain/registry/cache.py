from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from routee.powertrain.core.model import Model
from routee.powertrain.io.archive import load_archive
from routee.powertrain.registry.catalog import Catalog
from routee.powertrain.registry.model_id import ModelId, ModelInfo
from routee.powertrain.registry.registry import ModelRegistry

DEFAULT_CACHE_DIR = Path.home() / ".routee" / "cache"
DEFAULT_CATALOG_TTL_SECONDS = 3600  # 1 hour


class CachedRegistry(ModelRegistry):
    """
    A caching wrapper around any ModelRegistry implementation.

    Caches model zip files and the catalog index locally to avoid
    re-downloading on repeated queries/loads.

    Args:
        inner: the registry to wrap
        cache_dir: local directory for cached files (default: ~/.routee/cache/)
        catalog_ttl: time-to-live in seconds for the cached catalog (default: 1 hour)
    """

    def __init__(
        self,
        inner: ModelRegistry,
        cache_dir: Optional[Path] = None,
        catalog_ttl: int = DEFAULT_CATALOG_TTL_SECONDS,
    ) -> None:
        self.inner = inner
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.catalog_ttl = catalog_ttl

    def _catalog_cache_path(self) -> Path:
        return self.cache_dir / "catalog.json"

    def _catalog_timestamp_path(self) -> Path:
        return self.cache_dir / "catalog.timestamp"

    def _model_cache_path(self, model_id: ModelId, schema_version: str = "v2") -> Path:
        rel_path = model_id.to_path(schema_version)
        return self.cache_dir / rel_path

    def _is_catalog_fresh(self) -> bool:
        ts_path = self._catalog_timestamp_path()
        if not ts_path.exists():
            return False
        try:
            cached_time = float(ts_path.read_text())
            return (time.time() - cached_time) < self.catalog_ttl
        except (ValueError, OSError):
            return False

    def _load_cached_catalog(self) -> Optional[Catalog]:
        cache_path = self._catalog_cache_path()
        if cache_path.exists() and self._is_catalog_fresh():
            return Catalog.from_json(cache_path)
        return None

    def _save_catalog_to_cache(self, catalog: Catalog) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._catalog_cache_path()
        catalog.to_json(cache_path)
        self._catalog_timestamp_path().write_text(str(time.time()))

    def query(
        self,
        make: Optional[str] = None,
        model_name: Optional[str] = None,
        year: Optional[int] = None,
        trim: Optional[str] = None,
        variant: Optional[str] = None,
    ) -> List[ModelInfo]:
        catalog = self._load_cached_catalog()
        if catalog is not None:
            return catalog.query(
                make=make,
                model_name=model_name,
                year=year,
                trim=trim,
                variant=variant,
            )

        # Cache miss — fetch from inner registry (which fetches the full catalog)
        results = self.inner.query(
            make=make,
            model_name=model_name,
            year=year,
            trim=trim,
            variant=variant,
        )

        # Also cache the full catalog for next time
        try:
            all_models = self.inner.query()
            full_catalog = Catalog(schema_version="v2", models=all_models)
            self._save_catalog_to_cache(full_catalog)
        except Exception:
            pass  # Don't fail the query if caching doesn't work

        return results

    def load(self, model_id: ModelId) -> Model:
        cache_path = self._model_cache_path(model_id)
        if cache_path.exists():
            return load_archive(cache_path)

        model = self.inner.load(model_id)

        # Cache the model for next time by saving the archive
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            from routee.powertrain.io.archive import save_archive

            save_archive(model, cache_path)
        except Exception:
            pass  # Don't fail the load if caching doesn't work

        return model

    def get_metadata(self, model_id: ModelId) -> dict:
        return self.inner.get_metadata(model_id)

    def clear_cache(self) -> None:
        """Remove all cached files."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
