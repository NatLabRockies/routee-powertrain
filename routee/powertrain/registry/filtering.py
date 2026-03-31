from __future__ import annotations

from typing import List, Optional

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
    variant: Optional[str] = None,
    feature_set_id: Optional[str] = None,
    powertrain_type: Optional[str] = None,
    fuzzy: bool = True,
    fuzzy_threshold: int = 80,
) -> List[ModelInfo]:
    """
    Filter a list of ModelInfo by the given criteria.
    Support for fuzzy string matching with configurable threshold.
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
    if variant is not None:
        results = [
            m
            for m in results
            if _matches(variant, m.model_id.variant, fuzzy, fuzzy_threshold)
        ]
    if feature_set_id is not None:
        results = [
            m
            for m in results
            if _matches(
                feature_set_id, m.model_id.feature_set_id, fuzzy, fuzzy_threshold
            )
        ]
    if powertrain_type is not None:
        results = [
            m
            for m in results
            if _matches(
                powertrain_type, m.powertrain_type.lower(), fuzzy, fuzzy_threshold
            )
        ]
    return results
