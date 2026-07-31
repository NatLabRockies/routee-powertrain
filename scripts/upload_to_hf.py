#!/usr/bin/env python3
"""
Upload a local model database to a HuggingFace Hub model registry repo.

Walks a local model directory tree (matching the registry directory layout)
and uploads each model's files (metadata.json + binary) to the Hub. Unlike the
S3 uploader, which PUTs one object at a time, this pushes the whole tree in a
single commit so the repo never sits in a half-published state.

Usage
-----
::

    # Upload all models from a local registry directory:
    python scripts/upload_to_hf.py /path/to/local/model-library

    # Specify a different repo:
    python scripts/upload_to_hf.py /path/to/local/model-library \\
        --repo-id my-org/my-model-library

    # Dry-run to see what would be uploaded:
    python scripts/upload_to_hf.py /path/to/local/model-library --dry-run

    # Upload only models matching a specific make:
    python scripts/upload_to_hf.py /path/to/local/model-library --prefix v2/toyota

Authentication uses the standard huggingface_hub token resolution: the
``HF_TOKEN`` environment variable, or a token stored by ``huggingface-cli login``.
Writing requires a token with write access to the target repo.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from routee.powertrain.core.metadata import SCHEMA_VERSION_STRING
from routee.powertrain.io.archive import METADATA_FILENAME
from routee.powertrain.registry.default import (
    DEFAULT_HF_REPO_ID,
    DEFAULT_HF_REPO_TYPE,
)
from routee.powertrain.registry.hf import build_index

log = logging.getLogger(__name__)


def discover_model_dirs(root: Path, schema_version: str) -> list[Path]:
    """Find all model directories (those containing metadata.json) under the schema root."""
    schema_root = root / schema_version
    if not schema_root.exists():
        log.error("Schema directory not found: %s", schema_root)
        return []
    return sorted(d.parent for d in schema_root.glob(f"**/{METADATA_FILENAME}"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Upload a local model database to a HuggingFace registry repo.",
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Path to the local model database root directory.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_HF_REPO_ID,
        help=f"HuggingFace repo id (default: {DEFAULT_HF_REPO_ID}).",
    )
    parser.add_argument(
        "--repo-type",
        default=DEFAULT_HF_REPO_TYPE,
        choices=["model", "dataset"],
        help=f"HuggingFace repo type (default: {DEFAULT_HF_REPO_TYPE}).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Branch to commit to (default: the repo's default branch).",
    )
    parser.add_argument(
        "--schema-version",
        default=SCHEMA_VERSION_STRING,
        help=f"Schema version subdirectory (default: {SCHEMA_VERSION_STRING}).",
    )
    parser.add_argument(
        "--root-prefix",
        default="",
        help="Folder in the repo to place the schema-versioned tree under "
        "(default: the repo root).",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Only upload models whose relative path starts with this prefix "
        "(e.g. 'v2/toyota' to upload only Toyota models).",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create the repo (public) if it does not already exist.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the uploaded tree instead of merging into it: files in "
        "the repo that are not present locally are deleted in the same commit. "
        "Use when republishing a regenerated library, so retired models and "
        "stale versions don't linger.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading.",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Do not rebuild the index after upload.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    root = args.root.resolve()
    if not root.is_dir():
        log.error("Root path is not a directory: %s", root)
        return 1

    model_dirs = discover_model_dirs(root, args.schema_version)
    if not model_dirs:
        log.error("No models found under %s/%s", root, args.schema_version)
        return 1

    # Apply prefix filter if specified. allow_patterns is matched against
    # repo-relative paths, which equal the root-relative paths on disk.
    if args.prefix:
        prefix = args.prefix.strip("/")
        model_dirs = [
            d for d in model_dirs if str(d.relative_to(root)).startswith(prefix)
        ]
        if not model_dirs:
            log.error("No models matched prefix '%s'", args.prefix)
            return 1

    allow_patterns = [f"{d.relative_to(root)}/*" for d in model_dirs]

    # ``delete_patterns`` is evaluated against the uploaded folder's scope, so
    # every repo file under it that isn't in this upload is removed as part of
    # the same commit — the repo never sits in a half-replaced state. Scope the
    # deletion to --prefix when one is given, so a partial upload can't wipe the
    # models it never looked at.
    delete_patterns = None
    if args.replace:
        delete_patterns = (
            f"{args.prefix.strip('/')}/*" if args.prefix else f"{args.schema_version}/*"
        )

    log.info(
        "Found %d model(s) to upload to %s (%s)",
        len(model_dirs),
        args.repo_id,
        args.repo_type,
    )

    if args.dry_run:
        log.info("--- DRY RUN (nothing will be uploaded) ---")
        if args.replace:
            log.info(
                "[dry-run] --replace: repo files under '%s' not in this upload "
                "would be DELETED",
                delete_patterns,
            )
        for d in model_dirs:
            log.info("[dry-run] Would upload %s -> %s", d, d.relative_to(root))
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))

    if args.create:
        log.info("Ensuring repo %s exists...", args.repo_id)
        api.create_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            exist_ok=True,
            private=False,
        )

    if args.replace:
        log.warning(
            "--replace: repo files under '%s' not in this upload will be deleted",
            delete_patterns,
        )

    commit_message = (
        f"Replace library with {len(model_dirs)} model(s)"
        if args.replace
        else f"Publish {len(model_dirs)} model(s)"
    )

    try:
        api.upload_folder(
            folder_path=str(root),
            path_in_repo=args.root_prefix.strip("/"),
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            allow_patterns=allow_patterns,
            delete_patterns=delete_patterns,
            commit_message=commit_message,
        )
    except Exception:
        log.exception("Failed to upload models to %s", args.repo_id)
        return 1

    log.info("Uploaded %d model(s) to %s", len(model_dirs), args.repo_id)

    if not args.no_index:
        log.info("Updating the registry index...")
        try:
            build_index(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                token=os.environ.get("HF_TOKEN"),
                schema_version=args.schema_version,
                root_prefix=args.root_prefix,
            )
        except Exception:
            log.exception("Failed to rebuild the registry index")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
