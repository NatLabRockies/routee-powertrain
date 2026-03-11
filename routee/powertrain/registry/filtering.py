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
    model_name: Optional[str] = None,
    year: Optional[int] = None,
    variant: Optional[str] = None,
    feature_set_id: Optional[str] = None,
    fuzzy: bool = True,
    fuzzy_threshold: int = 80,
) -> List[ModelInfo]:
    """
    Filter a list of ModelInfo by the given criteria.

    When *fuzzy* is True, string fields (make, model_name, variant,
    feature_set_id) are matched using ``rapidfuzz.fuzz.partial_ratio``
    with the given *fuzzy_threshold* (0–100).  When False, exact
    equality (after lowercasing) is required.

    The fuzzy scorer is ``rapidfuzz.fuzz.WRatio`` which combines
    multiple heuristics (ratio, partial ratio, token sort/set) and
    picks the best one, providing robust matching for abbreviations,
    substrings, and near-misspellings.

    Year filtering always uses ``year_contains`` regardless of the
    fuzzy flag.
    """
    results = models
    if make is not None:
        results = [
            m
            for m in results
            if _matches(make, m.model_id.make, fuzzy, fuzzy_threshold)
        ]
    if model_name is not None:
        results = [
            m
            for m in results
            if _matches(model_name, m.model_id.model_name, fuzzy, fuzzy_threshold)
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
    return results
