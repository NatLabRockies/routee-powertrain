"""Backfill digest fields onto existing registry entries.

Walks a local registry tree (``<root>/<schema_version>/.../v<N>/``), and for
every model whose ``metadata.json`` lacks a ``model_digest``, hashes the
existing estimator binary, stamps ``estimator.estimator_sha256`` and
``model_digest``, and rewrites the metadata file in place. Entries that already
carry a digest are left untouched (pass ``--force`` to re-stamp them).

Usage:
    python scripts/backfill_digests.py <registry_root> [--schema-version v2] [--force]

To backfill the bundled registry:
    python scripts/backfill_digests.py routee/powertrain/resources/bundled_registry
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from routee.powertrain.core.digest import compute_model_digest, estimator_sha256
from routee.powertrain.core.metadata import SCHEMA_VERSION_STRING, Metadata
from routee.powertrain.io.archive import METADATA_FILENAME


def backfill_registry(
    registry_root: Path,
    schema_version: str = SCHEMA_VERSION_STRING,
    force: bool = False,
) -> int:
    """Stamp digest fields onto every entry under a registry root.

    Returns the number of metadata files rewritten.
    """
    schema_root = registry_root / schema_version
    if not schema_root.exists():
        raise FileNotFoundError(f"No such registry schema root: {schema_root}")

    updated = 0
    for meta_path in sorted(schema_root.glob(f"**/{METADATA_FILENAME}")):
        metadata_dict = json.loads(meta_path.read_text())
        if metadata_dict.get("model_digest") is not None and not force:
            print(f"skip (already stamped): {meta_path}")
            continue

        metadata = Metadata.model_validate(metadata_dict)
        model_file = meta_path.parent / metadata.estimator.model_file
        if not model_file.exists():
            print(f"skip (missing binary {metadata.estimator.model_file}): {meta_path}")
            continue

        metadata.estimator.estimator_sha256 = estimator_sha256(model_file.read_bytes())
        metadata.model_digest = compute_model_digest(metadata)

        meta_path.write_text(
            json.dumps(metadata.model_dump(mode="json"), indent=2) + "\n"
        )
        print(f"stamped {metadata.short_digest}: {meta_path}")
        updated += 1

    return updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry_root", type=Path)
    parser.add_argument("--schema-version", default=SCHEMA_VERSION_STRING)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-stamp entries that already carry a digest",
    )
    args = parser.parse_args()

    count = backfill_registry(args.registry_root, args.schema_version, args.force)
    print(f"updated {count} metadata file(s)")
