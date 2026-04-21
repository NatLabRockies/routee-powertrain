from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

from rapidfuzz import fuzz

from routee.powertrain.core.year import serialize_year, year_contains
from routee.powertrain.registry.model_id import ModelId, ModelInfo

VersionStrategy = Literal["latest", "all"]


def _matches(query: str, candidate: str, fuzzy: bool, threshold: int) -> bool:
    """Check whether a query string matches a candidate string."""
    query_lower = query.lower()
    if not fuzzy:
        return candidate == query_lower
    return fuzz.WRatio(query_lower, candidate) >= threshold


def _group_key(model_id: ModelId) -> Tuple[str, str, object, str]:
    """Build a hashable grouping key that collapses versions of the same model."""
    return (
        model_id.make,
        model_id.model,
        serialize_year(model_id.year),
        model_id.config_slug,
    )


def filter_models(
    models: List[ModelInfo],
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
    """
    Filter a list of ModelInfo by the given criteria.
    Support for fuzzy string matching with configurable threshold.

    ``feature_names`` filters to models whose ``feature_names`` contains every
    listed name (subset match, exact column name).

    ``version`` pins results to an exact version. When set, ``version_strategy``
    is ignored. ``version_strategy`` collapses multiple versions of the same
    ``(make, model, year, config_slug)`` group to the highest version when set
    to ``"latest"`` (default), or returns all versions when set to ``"all"``.
    """
    results = models
    if make is not None:
        results = [
            m
            for m in results
            if _matches(make, m.model_id.make, fuzzy, fuzzy_threshold)
        ]
    if model is not None:
        results = [
            m
            for m in results
            if _matches(model, m.model_id.model, fuzzy, fuzzy_threshold)
        ]
    if year is not None:
        results = [m for m in results if year_contains(m.model_id.year, year)]
    if config_slug is not None:
        results = [
            m
            for m in results
            if _matches(config_slug, m.model_id.config_slug, fuzzy, fuzzy_threshold)
        ]
    if feature_names:
        required = {n.lower() for n in feature_names}
        results = [
            m
            for m in results
            if required.issubset({fn.lower() for fn in m.feature_names})
        ]
    if powertrain_type is not None:
        results = [
            m
            for m in results
            if _matches(
                powertrain_type, m.powertrain_type.lower(), fuzzy, fuzzy_threshold
            )
        ]
    if fuel_type is not None:
        results = [
            m
            for m in results
            if m.fuel_type is not None
            and _matches(fuel_type, m.fuel_type.lower(), fuzzy, fuzzy_threshold)
        ]
    if drivetrain is not None:
        results = [
            m
            for m in results
            if m.drivetrain is not None
            and _matches(drivetrain, m.drivetrain.lower(), fuzzy, fuzzy_threshold)
        ]
    if engine is not None:
        results = [
            m
            for m in results
            if m.engine is not None
            and _matches(engine, m.engine.lower(), fuzzy, fuzzy_threshold)
        ]
    if trim is not None:
        results = [
            m
            for m in results
            if m.trim is not None
            and _matches(trim, m.trim.lower(), fuzzy, fuzzy_threshold)
        ]
    if version is not None:
        results = [m for m in results if m.model_id.version == version]
    if custom_filters is not None:
        for fn in custom_filters:
            results = [m for m in results if fn(m)]

    # An explicit version filter overrides the strategy — the caller is
    # asking for exactly that version.
    effective_strategy: VersionStrategy = (
        "all" if version is not None else version_strategy
    )
    if effective_strategy == "all":
        return results
    if effective_strategy == "latest":
        best: Dict[Tuple[str, str, object, str], ModelInfo] = {}
        for m in results:
            key = _group_key(m.model_id)
            current = best.get(key)
            if current is None or m.model_id.version > current.model_id.version:
                best[key] = m
        return list(best.values())
    raise ValueError(
        f"Unknown version_strategy '{version_strategy}'. "
        "Expected one of: 'latest', 'all'."
    )


def latest_model_ids(ids: List[ModelId]) -> List[ModelId]:
    """Reduce a list of ModelIds to the highest version per model group."""
    best: Dict[Tuple[str, str, object, str], ModelId] = {}
    for mid in ids:
        key = _group_key(mid)
        current = best.get(key)
        if current is None or mid.version > current.version:
            best[key] = mid
    return list(best.values())
