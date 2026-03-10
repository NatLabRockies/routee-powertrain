#!/usr/bin/env python
"""
Migrate legacy .json model files to the new .zip archive format.

Usage:
    python scripts/migrate_to_zip.py model.json                  # single file
    python scripts/migrate_to_zip.py models_dir/                 # all .json in directory
    python scripts/migrate_to_zip.py model.json -o output_dir/   # custom output directory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ensure the package is importable when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from routee.powertrain.core.model import Model
from routee.powertrain.io.archive import save_archive


def migrate_file(src: Path, dest_dir: Path) -> Path:
    """
    Convert a single legacy .json model file to a .zip archive.

    Args:
        src: path to the .json file
        dest_dir: directory to write the .zip into

    Returns: path to the created .zip file
    """
    with src.open("r") as f:
        input_dict = json.load(f)

    model = Model.from_dict(input_dict)

    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / (src.stem + ".zip")
    save_archive(model, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Migrate legacy .json model files to .zip archives"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="A .json model file or a directory containing .json model files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: same directory as input)",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if input_path.is_file():
        sources = [input_path]
    elif input_path.is_dir():
        sources = sorted(input_path.glob("*.json"))
        if not sources:
            print(f"No .json files found in {input_path}")
            return
    else:
        print(f"Input path does not exist: {input_path}")
        sys.exit(1)

    for src in sources:
        dest_dir = args.output if args.output else src.parent
        try:
            out = migrate_file(src, dest_dir)
            print(f"  {src.name} -> {out}")
        except Exception as e:
            print(f"  FAILED {src.name}: {e}")


if __name__ == "__main__":
    main()
