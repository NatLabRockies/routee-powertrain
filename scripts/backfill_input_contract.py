"""Backfill the embedded input/output contract onto existing registry entries.

Walks a local registry tree (``<root>/<schema_version>/.../v<N>/``), and for
every model whose estimator binary predates the self-describing input/output
contract, re-derives the contract from the model's own metadata, stamps it onto
the estimator (:meth:`Estimator.bind_io_contract`), and rewrites both the binary
(with the contract embedded in its ONNX ``metadata_props``) and ``metadata.json``
in place. Re-embedding changes the binary bytes, so ``estimator_sha256`` and
``model_digest`` are re-stamped as part of the save.

Entries whose binary already carries an ``input_columns`` contract are left
untouched (pass ``--force`` to re-stamp them anyway).

Usage:
    python scripts/backfill_input_contract.py <registry_root> [--schema-version v2] [--force]

To backfill the bundled registry:
    python scripts/backfill_input_contract.py routee/powertrain/resources/bundled_registry
"""

from __future__ import annotations

import argparse
from pathlib import Path

from routee.powertrain.core.metadata import SCHEMA_VERSION_STRING
from routee.powertrain.io.archive import (
    METADATA_FILENAME,
    load_model_directory,
    save_model_directory,
)


def backfill_registry(
    registry_root: Path,
    schema_version: str = SCHEMA_VERSION_STRING,
    force: bool = False,
) -> int:
    """Embed the input/output contract into every entry under a registry root.

    Returns the number of models rewritten.
    """
    schema_root = registry_root / schema_version
    if not schema_root.exists():
        raise FileNotFoundError(f"No such registry schema root: {schema_root}")

    updated = 0
    for meta_path in sorted(schema_root.glob(f"**/{METADATA_FILENAME}")):
        model_dir = meta_path.parent
        model = load_model_directory(model_dir)

        if model.estimator.input_spec.input_columns is not None and not force:
            print(f"skip (already has contract): {meta_path}")
            continue

        model.estimator.bind_io_contract(model.metadata.config)
        save_model_directory(model, model_dir)
        print(f"stamped contract {model.metadata.short_digest}: {meta_path}")
        updated += 1

    return updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry_root", type=Path)
    parser.add_argument("--schema-version", default=SCHEMA_VERSION_STRING)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-stamp entries whose binary already carries a contract",
    )
    args = parser.parse_args()

    count = backfill_registry(args.registry_root, args.schema_version, args.force)
    print(f"updated {count} model(s)")
