from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from rapidfuzz import fuzz

from routee.powertrain.core.year import year_contains
from routee.powertrain.registry.model_id import ModelInfo


def _matches(query: str, candidate: str, fuzzy: bool, threshold: int) -> bool:
    """Check whether a query string matches a candidate string."""
    query_lower = query.lower()
    if not fuzzy:
        return candidate == query_lower
    return fuzz.WRatio(query_lower, candidate) >= threshold


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
    custom_filters: Optional[Sequence[Callable[[ModelInfo], bool]]] = None,
    fuzzy: bool = True,
    fuzzy_threshold: int = 80,
) -> List[ModelInfo]:
    """
    Filter a list of ModelInfo by the given criteria.
    Support for fuzzy string matching with configurable threshold.

    ``feature_names`` filters to models whose ``feature_names`` contains every
    listed name (subset match, exact column name).
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
    if custom_filters is not None:
        for fn in custom_filters:
            results = [m for m in results if fn(m)]
    return results
