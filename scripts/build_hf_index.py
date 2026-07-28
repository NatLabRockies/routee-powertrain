#!/usr/bin/env python3
"""
Scan a HuggingFace Hub model registry repo and build a root-level index.json file.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from routee.powertrain.core.metadata import SCHEMA_VERSION_STRING
from routee.powertrain.registry.default import (
    DEFAULT_HF_REPO_ID,
    DEFAULT_HF_REPO_TYPE,
)
from routee.powertrain.registry.hf import build_index

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a root-level index.json for a HuggingFace model registry.",
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
        help="Branch or tag to write to (default: the repo's default branch).",
    )
    parser.add_argument(
        "--schema-version",
        default=SCHEMA_VERSION_STRING,
        help=f"Schema version (default: {SCHEMA_VERSION_STRING}).",
    )
    parser.add_argument(
        "--root-prefix",
        default="",
        help="Folder in the repo holding the schema-versioned tree "
        "(default: the repo root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the index but don't upload it to the Hub.",
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
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            token=os.environ.get("HF_TOKEN"),
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
