from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from routee.powertrain.core.model import Model
from routee.powertrain.core.year import year_contains
from routee.powertrain.io.archive import load_archive, save_archive
from routee.powertrain.registry.model_id import ModelId, ModelInfo
from routee.powertrain.registry.registry import ModelRegistry

DEFAULT_CACHE_DIR = Path.home() / ".routee" / "cache"
DEFAULT_CATALOG_TTL_SECONDS = 3600  # 1 hour


class CachedRegistry(ModelRegistry):
    """
    A caching wrapper around any ModelRegistry implementation.

    Caches model zip files and a query-results index locally to avoid
    re-downloading on repeated queries/loads.

    Args:
        inner: the registry to wrap
        cache_dir: local directory for cached files (default: ~/.routee/cache/)
        catalog_ttl: time-to-live in seconds for the cached query index (default: 1 hour)
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

    def _query_cache_path(self) -> Path:
        return self.cache_dir / "query_cache.json"

    def _query_timestamp_path(self) -> Path:
        return self.cache_dir / "query_cache.timestamp"

    def _model_cache_path(self, model_id: ModelId, schema_version: str = "v2") -> Path:
        rel_path = model_id.to_path(schema_version)
        return self.cache_dir / (rel_path + ".zip")

    def _is_query_cache_fresh(self) -> bool:
        ts_path = self._query_timestamp_path()
        if not ts_path.exists():
            return False
        try:
            cached_time = float(ts_path.read_text())
            return (time.time() - cached_time) < self.catalog_ttl
        except (ValueError, OSError):
            return False

    def _load_cached_query(self) -> Optional[List[ModelInfo]]:
        cache_path = self._query_cache_path()
        if cache_path.exists() and self._is_query_cache_fresh():
            with cache_path.open("r") as f:
                data = json.load(f)
            return [ModelInfo.from_dict(d) for d in data]
        return None

    def _save_query_cache(self, models: List[ModelInfo]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._query_cache_path()
        with cache_path.open("w") as f:
            json.dump([m.to_dict() for m in models], f)
        self._query_timestamp_path().write_text(str(time.time()))

    def _filter(
        self,
        models: List[ModelInfo],
        make: Optional[str] = None,
        model_name: Optional[str] = None,
        year: Optional[int] = None,
        variant: Optional[str] = None,
        feature_set_id: Optional[str] = None,
    ) -> List[ModelInfo]:
        results = models
        if make is not None:
            make_lower = make.lower()
            results = [m for m in results if m.model_id.make == make_lower]
        if model_name is not None:
            model_name_lower = model_name.lower()
            results = [m for m in results if m.model_id.model_name == model_name_lower]
        if year is not None:
            results = [m for m in results if year_contains(m.model_id.year, year)]
        if variant is not None:
            variant_lower = variant.lower()
            results = [m for m in results if m.model_id.variant == variant_lower]
        if feature_set_id is not None:
            fs_lower = feature_set_id.lower()
            results = [m for m in results if m.model_id.feature_set_id == fs_lower]
        return results

    def query(
        self,
        make: Optional[str] = None,
        model_name: Optional[str] = None,
        year: Optional[int] = None,
        variant: Optional[str] = None,
        feature_set_id: Optional[str] = None,
    ) -> List[ModelInfo]:
        cached = self._load_cached_query()
        if cached is not None:
            return self._filter(
                cached,
                make=make,
                model_name=model_name,
                year=year,
                variant=variant,
                feature_set_id=feature_set_id,
            )

        # Cache miss — fetch all models from inner registry
        all_models = self.inner.query()

        try:
            self._save_query_cache(all_models)
        except Exception:
            pass  # Don't fail the query if caching doesn't work

        return self._filter(
            all_models,
            make=make,
            model_name=model_name,
            year=year,
            variant=variant,
            feature_set_id=feature_set_id,
        )

    def load(self, model_id: ModelId) -> Model:
        cache_path = self._model_cache_path(model_id)
        if cache_path.exists():
            return load_archive(cache_path)

        model = self.inner.load(model_id)

        # Cache the model as a .zip for compactness
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
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
