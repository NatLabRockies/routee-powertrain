from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from routee.powertrain.core.year import format_year

if TYPE_CHECKING:
    from routee.powertrain.core.metadata import Metadata
    from routee.powertrain.registry.model_id import ModelId

#: Maps the coarse ``architecture_tag`` (set by each ``Trainer``) to the short
#: prefix used in a ``config_slug``. Unknown tags fall back to the tag itself.
ARCHITECTURE_SHORT_CODES = {
    "random_forest": "rf",
    "ngboost": "ngb",
    "cnn": "cnn",
}

#: Number of hex characters of the feature-set hash included in the slug.
_FEATURE_HASH_LEN = 8


def architecture_short_code(architecture_tag: str) -> str:
    """Return the short slug prefix for a coarse architecture tag."""
    return ARCHITECTURE_SHORT_CODES.get(architecture_tag, architecture_tag)


def feature_set_hash(features_id: str) -> str:
    """Return a short, stable hash of a feature-set id (``&``-joined names)."""
    digest = hashlib.sha1(features_id.encode("utf-8")).hexdigest()
    return digest[:_FEATURE_HASH_LEN]


def derive_config_slug(metadata: Metadata) -> str:
    """
    Derive the canonical ``config_slug`` from a model's metadata.

    The slug is a pure function of metadata — it is never stored redundantly.
    It composes three parts, joined with ``_``:

    - the architecture short code (e.g. ``rf``, ``ngb``, ``cnn``)
    - the optional ``config.variant`` label, when set (e.g. ``steady``)
    - a short hash of the feature set

    e.g. ``rf_steady_a1b2c3d4`` or, with no variant, ``ngb_96224f1f``.

    Args:
        metadata: the model metadata to derive a slug from

    Returns: the derived config slug
    """
    parts = [architecture_short_code(metadata.architecture_tag)]

    variant = metadata.config.variant
    if variant:
        parts.append(variant)

    parts.append(feature_set_hash(metadata.config.feature_set.features_id))

    return "_".join(parts)


def assert_metadata_matches_id(metadata: Metadata, model_id: ModelId) -> None:
    """
    Raise if a path-derived ``ModelId`` is inconsistent with the metadata.

    Since the path is a frozen cache of an identity that metadata is the source
    of truth for, a mismatch means the directory was moved, hand-edited, or the
    slug-derivation algorithm changed. Surfacing it loudly beats a silent lie.

    Args:
        metadata: the loaded model metadata (source of truth)
        model_id: the identity parsed from the registry path

    Raises:
        ValueError: if any identity field disagrees with the metadata
    """
    config = metadata.config
    derived_slug = derive_config_slug(metadata)

    mismatches = []
    if model_id.make != config.make:
        mismatches.append(f"make: path='{model_id.make}' metadata='{config.make}'")
    if model_id.model != config.model:
        mismatches.append(f"model: path='{model_id.model}' metadata='{config.model}'")
    if format_year(model_id.year) != format_year(config.year):
        mismatches.append(
            f"year: path='{format_year(model_id.year)}' "
            f"metadata='{format_year(config.year)}'"
        )
    if model_id.config_slug != derived_slug:
        mismatches.append(
            f"config_slug: path='{model_id.config_slug}' derived='{derived_slug}'"
        )

    if mismatches:
        raise ValueError(
            f"Model at '{model_id.to_path()}' has metadata inconsistent with its "
            "registry path: " + "; ".join(mismatches)
        )
