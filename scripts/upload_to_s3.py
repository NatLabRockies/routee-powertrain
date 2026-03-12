#!/usr/bin/env python3
"""
Upload a local model database to the S3 model registry bucket.

Walks a local model directory tree (matching the registry directory layout)
and uploads each model's files (metadata.json + binary) to S3.

Usage
-----
::

    # Upload all models from a local registry directory:
    python scripts/upload_to_s3.py /path/to/local/model-library

    # Specify a custom bucket or region:
    python scripts/upload_to_s3.py /path/to/local/model-library \\
        --bucket my-bucket --region us-east-1

    # Dry-run to see what would be uploaded:
    python scripts/upload_to_s3.py /path/to/local/model-library --dry-run

    # Upload only models matching a specific make:
    python scripts/upload_to_s3.py /path/to/local/model-library --prefix v2/toyota

AWS credentials are resolved via the standard boto3 credential chain
(environment variables, ~/.aws/credentials, IAM role, etc.).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import boto3

from routee.powertrain.core.metadata import SCHEMA_VERSION_STRING
from routee.powertrain.io.archive import METADATA_FILENAME
from routee.powertrain.registry.default import (
    DEFAULT_BUCKET,
    DEFAULT_REGION,
    DEFAULT_ROOT_PREFIX,
)

log = logging.getLogger(__name__)


def discover_model_dirs(root: Path, schema_version: str) -> list[Path]:
    """Find all model directories (those containing metadata.json) under the schema root."""
    schema_root = root / schema_version
    if not schema_root.exists():
        log.error("Schema directory not found: %s", schema_root)
        return []
    return sorted(d.parent for d in schema_root.glob(f"**/{METADATA_FILENAME}"))


def upload_model_dir(
    client,
    model_dir: Path,
    root: Path,
    bucket: str,
    dry_run: bool = False,
    root_prefix: str = DEFAULT_ROOT_PREFIX,
) -> int:
    """
    Upload all files in a model directory to S3.

    Returns the number of files uploaded.
    """
    # Build the S3 key prefix from the relative path (e.g. v2/toyota/camry/2016/default/speed_grade/v1)
    rel = model_dir.relative_to(root)
    uploaded = 0

    for file_path in sorted(model_dir.iterdir()):
        if not file_path.is_file():
            continue
        rel_key = str(rel / file_path.name)
        key = f"{root_prefix}/{rel_key}" if root_prefix else rel_key
        if dry_run:
            log.info("[dry-run] Would upload %s -> s3://%s/%s", file_path, bucket, key)
        else:
            log.info("Uploading %s -> s3://%s/%s", file_path, bucket, key)
            client.upload_file(str(file_path), bucket, key)
        uploaded += 1

    return uploaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Upload a local model database to the S3 registry bucket.",
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Path to the local model database root directory.",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"S3 bucket name (default: {DEFAULT_BUCKET}).",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION}).",
    )
    parser.add_argument(
        "--schema-version",
        default=SCHEMA_VERSION_STRING,
        help=f"Schema version subdirectory (default: {SCHEMA_VERSION_STRING}).",
    )
    parser.add_argument(
        "--root-prefix",
        default=DEFAULT_ROOT_PREFIX,
        help=f"Top-level folder in the bucket under which all models are stored "
        f"(default: {DEFAULT_ROOT_PREFIX}).",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Only upload models whose relative path starts with this prefix "
        "(e.g. 'v2/toyota' to upload only Toyota models).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading.",
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
        log.error(
            "No models found under %s/%s",
            root,
            args.schema_version,
        )
        return 1

    # Apply prefix filter if specified
    if args.prefix:
        prefix = args.prefix.strip("/")
        model_dirs = [
            d for d in model_dirs if str(d.relative_to(root)).startswith(prefix)
        ]
        if not model_dirs:
            log.error("No models matched prefix '%s'", args.prefix)
            return 1

    log.info(
        "Found %d model(s) to upload to s3://%s",
        len(model_dirs),
        args.bucket,
    )

    if args.dry_run:
        log.info("--- DRY RUN (no files will be uploaded) ---")

    client = None if args.dry_run else boto3.client("s3", region_name=args.region)

    total_files = 0
    failed = 0
    for model_dir in model_dirs:
        rel = model_dir.relative_to(root)
        try:
            count = upload_model_dir(
                client,
                model_dir,
                root,
                args.bucket,
                args.dry_run,
                root_prefix=args.root_prefix.strip("/") if args.root_prefix else "",
            )
            total_files += count
            log.debug("  %s: %d file(s)", rel, count)
        except Exception:
            log.exception("Failed to upload %s", rel)
            failed += 1

    action = "Would upload" if args.dry_run else "Uploaded"
    log.info(
        "%s %d file(s) from %d model(s) to s3://%s",
        action,
        total_files,
        len(model_dirs) - failed,
        args.bucket,
    )
    if failed:
        log.warning("%d model(s) failed to upload", failed)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
