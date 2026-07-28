#!/usr/bin/env python3
"""
Compatibility shim for the v1 -> v2 model converter.

The converter now ships inside the package at ``routee.powertrain.io.legacy`` and
is exposed as ``pt.convert_legacy_model()`` and the ``routee-powertrain
convert-v1`` console script, so users who ``pip install routee.powertrain`` can
reach it without cloning this repo.

This file remains so ``scripts/convert_nlr_library.py`` (which imports
``convert_legacy_json`` / ``VehicleIdentity`` by module name) keeps working.

Prefer::

    routee-powertrain convert-v1 path/to/model.json output_dir/ \\
        --make toyota --model camry --year 2016 --trim 4cyl_2wd
"""

from __future__ import annotations

import sys

from routee.powertrain.cli import main as _cli_main
from routee.powertrain.io.legacy import (
    ARCHITECTURE_TAG_MAP,
    ESTIMATOR_FILE_MAP,
    VehicleIdentity,
    convert_legacy_json,
    convert_legacy_model,
)

__all__ = [
    "ARCHITECTURE_TAG_MAP",
    "ESTIMATOR_FILE_MAP",
    "VehicleIdentity",
    "convert_legacy_json",
    "convert_legacy_model",
]


if __name__ == "__main__":
    # Forward to the packaged CLI, injecting the subcommand so the old
    # invocation form keeps working.
    raise SystemExit(_cli_main(["convert-v1", *sys.argv[1:]]))
