#!/usr/bin/env python3
"""
Scan an S3 model registry bucket and build a root-level index.json file.
"""

from __future__ import annotations

import argparse
import logging
import sys

from routee.powertrain.core.metadata import SCHEMA_VERSION_STRING
from routee.powertrain.registry.default import (
    DEFAULT_BUCKET,
    DEFAULT_REGION,
    DEFAULT_ROOT_PREFIX,
)
from routee.powertrain.registry.s3 import build_index

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a root-level index.json for an S3 model registry.",
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
        help=f"Schema version (default: {SCHEMA_VERSION_STRING}).",
    )
    parser.add_argument(
        "--root-prefix",
        default=DEFAULT_ROOT_PREFIX,
        help=f"Root prefix in the bucket (default: {DEFAULT_ROOT_PREFIX}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the index but don't upload it to S3.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        build_index(
            bucket=args.bucket,
            region=args.region,
            schema_version=args.schema_version,
            root_prefix=args.root_prefix,
            dry_run=args.dry_run,
        )
    except Exception:
        log.exception("Failed to build index")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
