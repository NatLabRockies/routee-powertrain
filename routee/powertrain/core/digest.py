from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import pandas as pd

    from routee.powertrain.core.metadata import Metadata

#: Version of the digest payload layout. The spec-1 payload built by
#: ``digest_payload`` is frozen forever — any change to which fields feed the
#: digest, or how they are encoded, must introduce a spec-2 builder instead of
#: editing this one, so that digests recorded under spec 1 remain verifiable.
DIGEST_SPEC = 1

DIGEST_PREFIX = "sha256:"

#: Length of the truncated digest used for display (``short_digest``).
SHORT_DIGEST_LEN = 12


def estimator_sha256(data: bytes) -> str:
    """Return the bare lowercase-hex sha256 of serialized estimator bytes.

    This is a pure content address of the binary artifact (the exact bytes
    written to e.g. ``model.onnx``). It is verified on load against the raw
    file bytes as read — never against a re-serialization.

    Args:
        data: the serialized estimator bytes

    Returns: the 64-character lowercase hex digest
    """
    return hashlib.sha256(data).hexdigest()


def digest_payload(metadata: Metadata) -> dict:
    """Build the canonical spec-1 identity payload for a model digest.

    The payload is constructed field-by-field (never from a full pydantic
    ``model_dump``) so its byte layout stays under our control as the schema
    evolves. Inclusion rule: *what the model is + what produced it*.
    Deliberately excluded: ``errors`` (derived metrics), ``routee_version``
    (environment), descriptive vehicle attributes (legitimately editable),
    ``input_spec`` (already baked into the estimator bytes), ``model_file``
    and ``schema_version`` (storage details), and ``trip_column`` (only feeds
    the excluded errors).

    Args:
        metadata: the model metadata to derive the payload from. Its
            ``estimator.estimator_sha256`` should already be populated so the
            payload transitively pins the binary artifact.

    Returns: a JSON-serializable dict
    """
    from routee.powertrain.core.year import format_year

    vehicle = metadata.vehicle
    contract = metadata.contract
    estimator = metadata.estimator
    training = metadata.training

    return {
        "digest_spec": DIGEST_SPEC,
        "vehicle": {
            "make": vehicle.make,
            "model": vehicle.model,
            "year": format_year(vehicle.year),
            "variant": vehicle.variant,
        },
        "contract": {
            "features": [
                {"name": f.name, "units": f.units}
                for f in contract.feature_set.features
            ],
            "distance": {
                "name": contract.distance.name,
                "units": contract.distance.units,
            },
            "targets": [
                {"name": t.name, "units": t.units} for t in contract.target.targets
            ],
            "predict_method": contract.predict_method.value,
            "real_world_adjustment_factor": contract.real_world_adjustment_factor,
        },
        "estimator": {
            "estimator_type": estimator.estimator_type,
            "architecture_tag": estimator.architecture_tag,
            "estimator_sha256": estimator.estimator_sha256,
        },
        "training": {
            "trained_date": training.trained_date,
            "random_seed": training.random_seed,
            "test_size": training.test_size,
            "validation_size": training.validation_size,
            "dataset_name": training.dataset_name,
            "dataset_hash": training.dataset_hash,
        },
    }


def compute_model_digest(metadata: Metadata) -> str:
    """Compute the composed model digest from metadata alone.

    Because the payload embeds ``estimator.estimator_sha256``, this digest is
    recomputable from a ``metadata.json`` in hand — no binary needed — while
    still transitively pinning the artifact bytes.

    Args:
        metadata: the model metadata (with ``estimator_sha256`` populated)

    Returns: the digest in ``sha256:<64 hex>`` form
    """
    canonical = json.dumps(
        digest_payload(metadata),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return DIGEST_PREFIX + hashlib.sha256(canonical).hexdigest()


def stamp_digest(metadata: Metadata, estimator_bytes: bytes) -> None:
    """Mint and stamp both identity fields onto metadata, in place.

    Sets ``metadata.estimator.estimator_sha256`` from the given bytes, then
    ``metadata.model_digest`` over the spec-1 payload. Registry-independent
    and offline — called at train time and re-checked at save time.

    Args:
        metadata: the metadata to stamp
        estimator_bytes: the exact serialized estimator bytes being persisted
    """
    metadata.estimator.estimator_sha256 = estimator_sha256(estimator_bytes)
    metadata.model_digest = compute_model_digest(metadata)


def normalize_digest(value: str) -> str:
    """Normalize a digest string to canonical ``sha256:<hex>`` form.

    Accepts input with or without the ``sha256:`` prefix and in any case.

    Args:
        value: a digest string

    Returns: the canonical lowercase prefixed form
    """
    value = value.strip().lower()
    if value.startswith(DIGEST_PREFIX):
        value = value[len(DIGEST_PREFIX) :]
    return DIGEST_PREFIX + value


def short_digest(digest: Optional[str]) -> Optional[str]:
    """Return a truncated display form of a digest (``sha256:<12 hex>``)."""
    if digest is None:
        return None
    normalized = normalize_digest(digest)
    return normalized[: len(DIGEST_PREFIX) + SHORT_DIGEST_LEN]


def hash_dataframe(df: pd.DataFrame) -> str:
    """Fingerprint a training DataFrame for dataset provenance.

    Convenience for populating ``ModelConfig.dataset_hash``. The hash covers
    the row values (not the index) via ``pandas.util.hash_pandas_object``, so
    it is sensitive to row order, column order, and dtypes.

    Args:
        df: the training data to fingerprint

    Returns: a bare lowercase-hex sha256 string
    """
    import pandas as pd

    row_hashes = pd.util.hash_pandas_object(df, index=False)
    return hashlib.sha256(row_hashes.to_numpy().tobytes()).hexdigest()
