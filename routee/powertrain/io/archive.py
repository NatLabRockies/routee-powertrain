from __future__ import annotations

import json
import logging
import re
import tarfile
import warnings
import zipfile
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from routee.powertrain.__about__ import MIGRATION_GUIDE_URL
from routee.powertrain.core.metadata import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_STRING,
    Metadata,
)

if TYPE_CHECKING:
    from routee.powertrain.core.model import Model
    from routee.powertrain.registry.model_id import ModelId

METADATA_FILENAME = "metadata.json"

_VERSION_DIR_RE = re.compile(r"^v(\d+)$")

log = logging.getLogger(__name__)


def _get_estimator_registry():
    from routee.powertrain.core.model import REGISTERED_ESTIMATORS

    return REGISTERED_ESTIMATORS


def _estimator_section(metadata_dict: dict) -> dict:
    """Return the ``estimator`` section of a serialized metadata dict.

    The estimator descriptor (``estimator_type``, ``model_file``, …) is grouped
    under an ``estimator`` key in the schema-v2 layout.
    """
    section = metadata_dict.get("estimator")
    if not isinstance(section, dict):
        raise ValueError("Archive metadata must contain an 'estimator' section")
    return section


def _model_filename(metadata_dict: dict) -> str:
    """Return the estimator binary filename from a serialized metadata dict."""
    filename = _estimator_section(metadata_dict).get("model_file")
    if filename is None:
        raise ValueError(
            "Archive metadata 'estimator' section must contain 'model_file'"
        )
    return filename


#: Contract fields every persisted estimator must carry (required on save).
_REQUIRED_CONTRACT_FIELDS = (
    "input_columns",
    "output_columns",
    "predict_method",
    "distance_column",
)


def _require_input_contract(model: Model) -> None:
    """Raise unless the estimator carries a complete input/output contract.

    The contract is *required on persist*: any model written to disk (directory,
    zip, tar, or registry) must be self-describing about its positional input and
    output order. Trained models get this automatically — ``Trainer.train`` calls
    ``estimator.bind_io_contract(config)`` — so this only fires for an estimator
    built by hand and never bound.
    """
    spec = model.estimator.input_spec
    missing = [f for f in _REQUIRED_CONTRACT_FIELDS if getattr(spec, f) is None]
    if missing:
        raise ValueError(
            "Cannot save a model whose estimator has an incomplete input/output "
            f"contract (missing: {missing}). Trained models bind this "
            "automatically; if you built the estimator directly, call "
            "estimator.bind_io_contract(config) before saving."
        )


def _build_metadata_dict(model: Model) -> dict:
    """Build the metadata dictionary for serialization.

    Requires a complete input/output contract (see ``_require_input_contract``).
    Only the estimator *mechanics* — ``lookback`` / ``grouping_column`` /
    ``pad_strategy`` — are persisted into ``estimator.input_spec``. The ordered
    input/output columns are deliberately **not** duplicated here: ``contract``
    is the single ordered source of truth in ``metadata.json`` (and the resolved
    positional contract still travels embedded in the estimator binary for
    consumers that only have the binary). ``input_spec`` is excluded from the
    digest payload, so this never affects ``model_digest``.

    The version-less ``model_key`` is stamped here too, so every artifact this
    package writes carries it. Neither it nor ``input_spec`` is part of the
    digest payload.
    """
    _require_input_contract(model)
    model.metadata.estimator.input_spec = model.estimator.input_spec.model_dump(
        mode="json", include={"lookback", "grouping_column", "pad_strategy"}
    )
    # Re-derive rather than carry a loaded value forward, so an edit to the
    # identity fields reaches what gets written.
    model.metadata.model_key = model.metadata.derived_model_key
    return model.metadata.model_dump(mode="json")


def _ensure_digest(model: Model, model_bytes: bytes) -> None:
    """Guarantee the digest fields are consistent with the bytes being written.

    Every save path funnels through this so that any artifact on disk carries
    an ``estimator_sha256`` matching its own binary and a ``model_digest``
    matching its own metadata. Absent fields are stamped (legacy or
    hand-constructed models); a previously-set value that no longer matches —
    re-serialization drift on a load→save round trip, or an identity-relevant
    metadata edit — is re-stamped with a warning, since it means the model's
    instance identity has changed.
    """
    from routee.powertrain.core.digest import compute_model_digest, estimator_sha256

    metadata = model.metadata
    current_sha = estimator_sha256(model_bytes)
    prior_sha = metadata.estimator.estimator_sha256
    if prior_sha is not None and prior_sha != current_sha:
        warnings.warn(
            "estimator bytes differ from the recorded estimator_sha256 "
            "(re-serialization drift or a replaced estimator); re-stamping the "
            "model's digest — its instance identity has changed."
        )
    metadata.estimator.estimator_sha256 = current_sha

    current_digest = compute_model_digest(metadata)
    prior_digest = metadata.model_digest
    if (
        prior_digest is not None
        and prior_digest != current_digest
        and prior_sha == current_sha
    ):
        warnings.warn(
            "identity-relevant metadata changed since model_digest was minted; "
            "re-stamping — the model's instance identity has changed."
        )
    metadata.model_digest = current_digest


def _verify_digest(metadata: Metadata, model_bytes: bytes) -> None:
    """Verify a loaded artifact against its recorded digests, when present.

    A binary that does not hash to its recorded ``estimator_sha256`` is corrupt
    or tampered with — raise. A ``model_digest`` that no longer matches a
    recomputation from the (validated) metadata means identity-relevant fields
    were edited after minting — warn. Models saved before digests existed have
    neither field and skip both checks.
    """
    from routee.powertrain.core.digest import compute_model_digest, estimator_sha256

    expected_sha = metadata.estimator.estimator_sha256
    if expected_sha is not None:
        actual_sha = estimator_sha256(model_bytes)
        if actual_sha != expected_sha:
            raise ValueError(
                "Estimator binary does not match its recorded estimator_sha256 "
                f"(expected {expected_sha}, got {actual_sha}). "
                "The artifact is corrupt or was modified after saving."
            )

    if metadata.model_digest is not None:
        recomputed = compute_model_digest(metadata)
        if recomputed != metadata.model_digest:
            warnings.warn(
                f"model_digest mismatch: stored {metadata.model_digest}, "
                f"recomputed {recomputed}. Identity-relevant metadata was "
                "edited after the digest was minted."
            )


def _verify_model_key(metadata: Metadata) -> None:
    """Check a loaded artifact's ``model_key`` against a fresh derivation.

    The stored value caches an identity the metadata fields own, so a
    disagreement means those fields were edited after the artifact was written,
    or the slug derivation changed. Warn and let the derived value stand, which
    matches how a ``model_digest`` mismatch is handled. Artifacts saved without
    the field skip the check.
    """
    if metadata.model_key is None:
        return
    derived = metadata.derived_model_key
    if metadata.model_key != derived:
        warnings.warn(
            f"model_key mismatch: stored '{metadata.model_key}', derived "
            f"'{derived}'. Identity-relevant metadata was edited after the "
            "artifact was written, or the slug derivation changed. The derived "
            "value is authoritative."
        )


def _verify_input_contract(metadata: Metadata, estimator) -> None:
    """Cross-check the estimator binary's embedded input order against metadata.

    When the binary is self-describing (its ``input_spec`` carries an ordered
    ``input_columns`` list — the ONNX case since the contract was introduced),
    that order must match the one implied by the metadata contract
    (``feature_set`` order, plus the distance column for RAW). A disagreement
    means the binary and ``metadata.json`` were minted from, or edited to,
    different feature orders — predictions would be silently transposed — so we
    **raise**. Binaries with no embedded contract (legacy artifacts, or backends
    that don't embed one) skip the check.
    """
    spec = estimator.input_spec
    if spec.input_columns is None:
        return
    embedded = [c.name for c in spec.input_columns]
    expected = metadata.config.all_feature_names
    if embedded != expected:
        raise ValueError(
            "Estimator input contract does not match metadata. The binary's "
            f"embedded input order is {embedded} but metadata.json implies "
            f"{expected}. The artifact's estimator and metadata disagree on "
            "feature order; predictions would be silently wrong."
        )


def _model_from_metadata_and_bytes(metadata_dict: dict, model_bytes: bytes) -> Model:
    """Reconstruct a Model from a parsed metadata dict and raw model bytes."""
    from routee.powertrain.core.model import Model

    schema_version = metadata_dict.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported archive schema version: {schema_version}. "
            f"Expected: {SCHEMA_VERSION}"
        )

    estimator_section = _estimator_section(metadata_dict)
    estimator_type_str = estimator_section.get("estimator_type")
    if estimator_type_str is None:
        raise ValueError(
            "Archive metadata 'estimator' section must contain 'estimator_type'"
        )

    registry = _get_estimator_registry()
    estimator_cls = registry.get(estimator_type_str)
    if estimator_cls is None:
        raise ValueError(
            f"Estimator type '{estimator_type_str}' is not registered. "
            f"Available types: {list(registry.keys())}"
        )

    metadata = Metadata.model_validate(metadata_dict)
    # Verify against the raw bytes as read, before deserializing the binary.
    _verify_digest(metadata, model_bytes)
    _verify_model_key(metadata)

    estimator = estimator_cls.from_bytes(model_bytes)
    _verify_input_contract(metadata, estimator)

    # Ensure the loaded estimator carries the full contract in memory even when
    # its binary format doesn't embed one (e.g. NGBoost's joblib blob). We
    # rebuild it from the metadata ``contract`` — the single ordered source —
    # rather than a duplicated copy, so a load → save round trip stays
    # contract-complete. Windowed estimators embed their own contract, so they
    # never reach this branch (their binary already carries ``input_columns``).
    if estimator.input_spec.input_columns is None:
        estimator.bind_io_contract(metadata.config)

    return Model(estimator=estimator, metadata=metadata)


# ---------------------------------------------------------------------------
# Directory format (default)
# ---------------------------------------------------------------------------


def save_model_directory(model: Model, path: Union[str, Path]) -> None:
    """
    Save a model as a flat directory containing metadata.json and a binary model file.

    Args:
        model: the model to save
        path: the directory path to write to
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    model_bytes = model.estimator.to_bytes()
    _ensure_digest(model, model_bytes)

    metadata_dict = _build_metadata_dict(model)
    model_filename = _model_filename(metadata_dict)

    (path / METADATA_FILENAME).write_text(json.dumps(metadata_dict, indent=2))
    (path / model_filename).write_bytes(model_bytes)


def _next_version(config_dir: Path) -> int:
    """Return the next unused version under a config_slug directory (max+1)."""
    if not config_dir.exists():
        return 1
    versions = [
        int(m.group(1))
        for p in config_dir.iterdir()
        if p.is_dir() and (m := _VERSION_DIR_RE.match(p.name))
    ]
    return max(versions, default=0) + 1


def _find_existing_version_by_digest(config_dir: Path, digest: str) -> Optional[int]:
    """Return the version under a config_slug directory whose stored
    ``model_digest`` matches, or ``None``. Makes registry publish idempotent:
    re-publishing an identical model resolves to its existing version instead
    of minting a duplicate.
    """
    if not config_dir.exists():
        return None
    for p in sorted(config_dir.iterdir()):
        m = _VERSION_DIR_RE.match(p.name)
        if m is None or not p.is_dir():
            continue
        meta_path = p / METADATA_FILENAME
        if not meta_path.exists():
            continue
        try:
            existing = json.loads(meta_path.read_text()).get("model_digest")
        except (OSError, json.JSONDecodeError):
            continue
        if existing is not None and existing == digest:
            return int(m.group(1))
    return None


def save_to_registry(
    model: Model,
    registry_root: Union[str, Path],
    config_slug: Optional[str] = None,
    version: Optional[int] = None,
    schema_version: str = SCHEMA_VERSION_STRING,
    overwrite: bool = False,
) -> ModelId:
    """
    Save a model into a local registry directory tree.

    Builds the canonical path
    ``<registry_root>/<schema_version>/<make>/<vehicle_slug>/<year>/<config_slug>/v<N>/``
    from ``model.metadata``. Both slugs are *derived* from metadata: the
    ``vehicle_slug`` from the model name plus optional engine/drivetrain/trim,
    and the ``config_slug`` from architecture + optional ``config.variant`` +
    feature-set hash (unless an explicit override is passed). The resulting directory is directly loadable by ``LocalRegistry``
    — or by ``pt.load_model(...)`` when ``ROUTEE_REGISTRY_BACKEND=local`` and
    ``ROUTEE_LOCAL_REGISTRY_ROOT`` points at ``registry_root``.

    Args:
        model: the trained model to save
        registry_root: filesystem root of the local registry (the directory
            that contains the ``<schema_version>/`` subtree)
        config_slug: optional override for the derived slug. Leave ``None`` to
            use the canonical derived value (recommended); a mismatching
            override emits a warning. Use ``config.variant`` to distinguish
            configs that share an architecture and feature set.
        version: positive integer version. Bump when retraining the same
            ``config_slug``. Defaults to the next unused version (max+1) under
            the slug directory when ``None`` — unless an existing version
            already holds an identical model (same ``model_digest``), in which
            case that version's ``ModelId`` is returned without writing
            (idempotent publish).
        schema_version: registry schema directory name (default ``"v2"``)
        overwrite: if False (default), raise ``FileExistsError`` when the
            target directory already contains a saved model. If True, the
            existing files are replaced.

    Returns:
        The ``ModelId`` that was written. Its ``to_path()`` matches what
        ``LocalRegistry.load()`` expects.

    Raises:
        FileExistsError: if ``overwrite`` is False and the target directory
            already contains a saved model.
        ValueError: if ``version`` is not a positive integer.
    """
    # Local import to avoid a circular import at package load time:
    # routee.powertrain.registry imports Model from core.model, and core.model
    # imports from this module.
    from routee.powertrain.core.year import format_year
    from routee.powertrain.registry.model_id import ModelId
    from routee.powertrain.registry.slug import (
        derive_config_slug,
        derive_vehicle_slug,
    )

    if version is not None and version < 1:
        raise ValueError(f"version must be a positive integer, got {version}")

    vehicle_slug = derive_vehicle_slug(model.metadata)
    derived_slug = derive_config_slug(model.metadata)
    if config_slug is not None and config_slug.lower() != derived_slug:
        warnings.warn(
            f"config_slug override '{config_slug}' does not match the slug "
            f"derived from metadata ('{derived_slug}'). The registry validates "
            "against the derived slug on load, so this override may fail to load."
        )
    effective_slug = config_slug if config_slug is not None else derived_slug

    config = model.metadata.config
    config_dir = (
        Path(registry_root)
        / schema_version
        / config.make.lower()
        / vehicle_slug
        / format_year(config.year)
        / effective_slug.lower()
    )
    if version is None:
        # Idempotent publish: an identical model (same instance digest) already
        # under this key resolves to its existing version instead of a new one.
        _ensure_digest(model, model.estimator.to_bytes())
        assert model.metadata.model_digest is not None
        existing_version = _find_existing_version_by_digest(
            config_dir, model.metadata.model_digest
        )
        if existing_version is not None:
            model_id = ModelId(
                make=config.make,
                vehicle_slug=vehicle_slug,
                year=config.year,
                config_slug=effective_slug,
                version=existing_version,
            )
            log.info(
                "model with digest %s already published at %s; returning the "
                "existing version instead of writing a new one",
                model.metadata.short_digest,
                model_id.to_path(),
            )
            return model_id
        version = _next_version(config_dir)

    model_id = ModelId(
        make=config.make,
        vehicle_slug=vehicle_slug,
        year=config.year,
        config_slug=effective_slug,
        version=version,
    )

    target_dir = Path(registry_root) / schema_version / model_id.to_path()
    if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
        existing = sorted(p.name for p in target_dir.parent.iterdir() if p.is_dir())
        raise FileExistsError(
            f"Registry slot already exists: {target_dir}. "
            f"Existing versions for this config_slug: {existing}. "
            "Pass overwrite=True to replace, or bump version."
        )

    save_model_directory(model, target_dir)
    return model_id


def load_model_directory(path: Union[str, Path]) -> Model:
    """
    Load a model from a directory containing metadata.json and a binary model file.

    Args:
        path: the directory path to read from

    Returns: a Model instance
    """
    path = Path(path)
    metadata_path = path / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {path}")

    metadata_dict = json.loads(metadata_path.read_text())
    model_filename = _model_filename(metadata_dict)

    model_bytes = (path / model_filename).read_bytes()
    return _model_from_metadata_and_bytes(metadata_dict, model_bytes)


def read_directory_metadata(path: Union[str, Path]) -> dict:
    """
    Read only the metadata.json from a model directory.

    Args:
        path: the directory path

    Returns: the parsed metadata dictionary
    """
    path = Path(path)
    metadata_path = path / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {path}")
    return json.loads(metadata_path.read_text())


# ---------------------------------------------------------------------------
# ZIP format
# ---------------------------------------------------------------------------


def save_archive(model: Model, path: Union[str, Path]) -> None:
    """
    Save a model as a ZIP archive containing metadata.json and a binary model file.

    Args:
        model: the model to save
        path: the path to write the .zip file to
    """
    path = Path(path)
    if path.suffix != ".zip":
        raise ValueError("Model archive must be a .zip file")

    model_bytes = model.estimator.to_bytes()
    _ensure_digest(model, model_bytes)

    metadata_dict = _build_metadata_dict(model)
    model_filename = _model_filename(metadata_dict)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(METADATA_FILENAME, json.dumps(metadata_dict))
        zf.writestr(model_filename, model_bytes)


def load_archive(path: Union[str, Path]) -> Model:
    """
    Load a model from a ZIP archive.

    Args:
        path: the path to the .zip file

    Returns: a Model instance
    """
    path = Path(path)
    if path.suffix != ".zip":
        raise ValueError("Model archive must be a .zip file")

    with zipfile.ZipFile(path, "r") as zf:
        metadata_dict = json.loads(zf.read(METADATA_FILENAME))
        model_filename = _model_filename(metadata_dict)
        model_bytes = zf.read(model_filename)
        return _model_from_metadata_and_bytes(metadata_dict, model_bytes)


def load_archive_bytes(data: bytes) -> Model:
    """
    Load a model from in-memory ZIP archive bytes.

    Args:
        data: the raw bytes of the zip archive

    Returns: a Model instance
    """
    with zipfile.ZipFile(BytesIO(data), "r") as zf:
        metadata_dict = json.loads(zf.read(METADATA_FILENAME))
        model_filename = _model_filename(metadata_dict)
        model_bytes = zf.read(model_filename)
        return _model_from_metadata_and_bytes(metadata_dict, model_bytes)


def read_archive_metadata(path: Union[str, Path]) -> dict:
    """
    Read only the metadata.json from a ZIP archive without loading the model binary.

    Args:
        path: the path to the .zip file

    Returns: the parsed metadata dictionary
    """
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        return json.loads(zf.read(METADATA_FILENAME))


# ---------------------------------------------------------------------------
# Tar format (.tar.gz)
# ---------------------------------------------------------------------------


def save_tar_archive(model: Model, path: Union[str, Path]) -> None:
    """
    Save a model as a .tar.gz archive containing metadata.json and a binary model file.

    Args:
        model: the model to save
        path: the path to write the .tar.gz file to
    """
    path = Path(path)

    model_bytes = model.estimator.to_bytes()
    _ensure_digest(model, model_bytes)

    metadata_dict = _build_metadata_dict(model)
    model_filename = _model_filename(metadata_dict)
    metadata_bytes = json.dumps(metadata_dict).encode()

    with tarfile.open(path, "w:gz") as tf:
        meta_info = tarfile.TarInfo(name=METADATA_FILENAME)
        meta_info.size = len(metadata_bytes)
        tf.addfile(meta_info, BytesIO(metadata_bytes))

        model_info = tarfile.TarInfo(name=model_filename)
        model_info.size = len(model_bytes)
        tf.addfile(model_info, BytesIO(model_bytes))


def load_tar_archive(path: Union[str, Path]) -> Model:
    """
    Load a model from a .tar.gz archive.

    Args:
        path: the path to the .tar.gz file

    Returns: a Model instance
    """
    path = Path(path)
    with tarfile.open(path, "r:gz") as tf:
        meta_member = tf.getmember(METADATA_FILENAME)
        meta_file = tf.extractfile(meta_member)
        if meta_file is None:
            raise ValueError(f"Could not extract {METADATA_FILENAME} from {path}")
        metadata_dict = json.loads(meta_file.read())

        model_filename = _model_filename(metadata_dict)
        model_member = tf.getmember(model_filename)
        model_file = tf.extractfile(model_member)
        if model_file is None:
            raise ValueError(f"Could not extract {model_filename} from {path}")
        model_bytes = model_file.read()

        return _model_from_metadata_and_bytes(metadata_dict, model_bytes)


# ---------------------------------------------------------------------------
# Polymorphic helpers
# ---------------------------------------------------------------------------


def _unsupported_format_message(path: Path) -> str:
    """Build the error text for a path that isn't a v2 model archive.

    A ``.json`` path is almost always a v1 single-file model, so point at the
    converter rather than restating the list of accepted suffixes.
    """
    base = (
        f"Unsupported model format: {path}. "
        "Expected a directory, .zip file, or .tar.gz file."
    )
    if path.suffix != ".json":
        return base
    return (
        f"{path} looks like a routee-powertrain v1 model file. v1 stored a whole "
        "model as a single .json; v2 models are directories, .zip, or .tar.gz "
        "archives, and v1 files can no longer be loaded directly.\n"
        "\n"
        "Convert it first (one v1 file becomes one v2 model per feature set):\n"
        f"    routee-powertrain convert-v1 {path} out/ "
        "--make toyota --model camry --year 2016\n"
        "\n"
        "or from Python:\n"
        "    pt.convert_legacy_model(\n"
        f'        "{path}", "out/", make="toyota", model="camry", year=2016\n'
        "    )\n"
        "\n"
        f"See {MIGRATION_GUIDE_URL}"
    )


def load_model_from_path(path: Union[str, Path]) -> Model:
    """
    Auto-detect format (directory, .zip, or .tar.gz) and load a model.

    Args:
        path: path to a model directory, .zip file, or .tar.gz file

    Returns: a Model instance
    """
    path = Path(path)
    if path.is_dir():
        return load_model_directory(path)
    elif path.suffix == ".zip":
        return load_archive(path)
    elif path.name.endswith(".tar.gz") or path.suffix == ".tar":
        return load_tar_archive(path)
    else:
        raise ValueError(_unsupported_format_message(path))


def read_metadata(path: Union[str, Path]) -> dict:
    """
    Read metadata.json from any supported format (directory, .zip, or .tar.gz).

    Args:
        path: path to a model directory, .zip file, or .tar.gz file

    Returns: the parsed metadata dictionary
    """
    path = Path(path)
    if path.is_dir():
        return read_directory_metadata(path)
    elif path.suffix == ".zip":
        return read_archive_metadata(path)
    elif path.name.endswith(".tar.gz") or path.suffix == ".tar":
        with tarfile.open(path, "r:gz") as tf:
            meta_member = tf.getmember(METADATA_FILENAME)
            meta_file = tf.extractfile(meta_member)
            if meta_file is None:
                raise ValueError(f"Could not extract {METADATA_FILENAME} from {path}")
            return json.loads(meta_file.read())
    else:
        raise ValueError(_unsupported_format_message(path))
