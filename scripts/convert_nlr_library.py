#!/usr/bin/env python3
"""
Batch-convert the NREL legacy model library to v2 format.

This script applies NREL-specific naming conventions to resolve
make / model / year / trim / variant from the legacy JSON filenames,
then delegates to ``convert_legacy_models.convert_legacy_json`` for the
actual conversion.

Usage
-----
::

    python scripts/convert_nrel_library.py old-json-models.ignore/ output_dir/
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from convert_legacy_models import VehicleIdentity, convert_legacy_json

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vehicle attribute extraction
# ---------------------------------------------------------------------------

# Maps (make, base_model) → drivetrain for models that use "2wd" in their name.
# 2WD is ambiguous (could be FWD or RWD), so we resolve per vehicle.
_2WD_DRIVETRAIN_MAP: Dict[tuple, str] = {
    ("audi", "a3"): "FWD",
    ("bmw", "328d"): "RWD",
    ("chevrolet", "colorado"): "RWD",
    ("chevrolet", "malibu"): "FWD",
    ("ford", "escape"): "FWD",
    ("ford", "explorer"): "RWD",
    ("hyundai", "elantra"): "FWD",
    ("maruti", "swift"): "FWD",
    ("tesla", "model_s60"): "RWD",
    ("toyota", "camry"): "FWD",
    ("toyota", "corolla"): "FWD",
}

# Engine spec patterns to search for in model names (order matters: longest first)
_ENGINE_PATTERNS = [
    re.compile(r"(\d+\.\d+_dci)"),  # 1.5_dci
    re.compile(r"(\d+\.\d+_mpi)"),  # 1.0_mpi
    re.compile(r"(\d+\.\d+tsi)"),  # 1.5tsi
    re.compile(r"(\d+\.\d+tdi)"),  # 2.0tdi
    re.compile(r"(\d+\.\d+_l)"),  # 3.5_l
    # Note: kwh values (24_kwh, 30_kwh) are treated as trim, not engine
    re.compile(r"(\d+kw)\b"),  # 300kw, 400kw
    re.compile(r"(\d+cyl)"),  # 4cyl
]

# Known trim values that appear as suffixes in model names
_KNOWN_TRIMS = {
    "active",
    "sport",
    "le",
    "mid",
    "g",
    "vdi",
    "i-stop",
    "authentique",
    "e_j_mt",
    "hardtop_2_door",
    "double_cab",
    "daycab",
    "sleeper",
    "24_kwh",
    "30_kwh",
}


def _extract_drivetrain(make: str, model: str) -> Optional[str]:
    """Extract drivetrain from the model name.

    Returns a Drivetrain enum name string or None.
    """
    name = model.lower()

    # Explicit drivetrain patterns
    if "_rwd" in name or name.endswith("_rwd"):
        return "RWD"
    if "_fwd" in name or name.endswith("_fwd"):
        return "FWD"
    if "_4wd" in name or name.endswith("_4wd"):
        return "FOURWD"
    if "xdrive" in name:
        return "AWD"
    if "dual_motor" in name:
        return "AWD"
    if name.endswith("_twin"):
        return "AWD"

    # 2WD: resolve per vehicle using lookup table
    if "_2wd" in name or name.endswith("_2wd"):
        # Find the base model by stripping everything after common suffixes
        for (m, base), dt in _2WD_DRIVETRAIN_MAP.items():
            if make.lower() == m and base in name:
                return dt
        log.warning(
            "Could not resolve 2wd drivetrain for %s/%s; leaving as None",
            make,
            model,
        )
        return None

    return None


def _extract_engine(model: str) -> Optional[str]:
    """Extract engine/motor specification from the model name."""
    name = model.lower()
    for pattern in _ENGINE_PATTERNS:
        m = pattern.search(name)
        if m:
            return m.group(1)

    # Special case: vios_1.5_g → engine is "1.5", trim is "g"
    if "vios" in name:
        m = re.search(r"(\d+\.\d+)", name)
        if m:
            return m.group(1)

    return None


def _extract_trim(make: str, model: str) -> Optional[str]:
    """Extract trim level from the model name using known trim values."""
    name = model.lower()

    # Heavy duty: cab type is trim
    if make.lower() == "generic_heavy_duty":
        if "daycab" in name:
            return "daycab"
        if "sleeper" in name:
            return "sleeper"

    # Check for known trim values anywhere in the model name as a
    # delimited segment (not just at the end, since drivetrain suffixes
    # like _4wd may follow the trim, e.g. "hilux_double_cab_4wd").
    # Try multi-word trims first (longest match).
    for trim in sorted(_KNOWN_TRIMS, key=len, reverse=True):
        # Match as _trim_ in middle, or _trim at end
        needle = f"_{trim}"
        pos = name.find(needle)
        if pos != -1:
            after = pos + len(needle)
            # Ensure it's a full segment (followed by _ or end of string)
            if after == len(name) or name[after] == "_":
                return trim

    return None


# ---------------------------------------------------------------------------
# Filename → VehicleIdentity parsing
# ---------------------------------------------------------------------------


def _parse_library_filename(filename: str) -> VehicleIdentity:
    """
    Parse an NREL library filename into VehicleIdentity.

    Handles:
    - Year-prefixed: ``2016_TOYOTA_Camry_4cyl_2WD.json``
    - Variants: ``*_Charge_Depleting.json``, ``*_Charge_Sustaining.json``,
      ``*_Stochastic.json``, ``*_steady.json``, ``*_transient.json``
    - Heavy-duty: ``Sleeper_*.json``, ``Daycab_*.json``
    - Transit: ``Transit_Bus_*.json``
    - No-year models: ``BYD_ATTO_3.json``, ``Maruti_Swift_4cyl_2WD.json``, ...
    """
    stem = filename.removesuffix(".json")

    # --- Detect and strip variant suffixes ---
    variant = "default"
    variant_suffixes = {
        "_Charge_Depleting": "charge_depleting",
        "_Charge_Sustaining": "charge_sustaining",
        "_Stochastic": "stochastic",
    }
    for suffix, var_name in variant_suffixes.items():
        if stem.endswith(suffix):
            variant = var_name
            stem = stem[: -len(suffix)]
            break

    # Steady/transient are detected by checking for the pattern before temp range
    # e.g. "2016_Nissan_Leaf_30_kWh_0F_110F_steady"
    if stem.endswith("_steady"):
        variant = "steady"
        stem = stem.removesuffix("_steady")
    elif stem.endswith("_transient"):
        variant = "transient"
        stem = stem.removesuffix("_transient")

    # --- Heavy-duty: Sleeper / Daycab ---
    if stem.startswith("Sleeper_") or stem.startswith("Daycab_"):
        parts = stem.split("_")  # e.g. ["Sleeper", "new", "300kW"]
        cab_type = parts[0].lower()  # "sleeper" or "daycab"
        age_class = parts[1].lower() if len(parts) > 1 else ""
        power_rating = "_".join(parts[2:]).lower() if len(parts) > 2 else ""
        year: int | str
        if age_class == "old":
            year = "2000-2010"
        elif age_class == "new":
            year = "2010-2020"
        else:
            year = 0
        model = f"class_8_{cab_type}"
        if power_rating:
            model = f"{model}_{power_rating}"
        return VehicleIdentity(
            make="generic_heavy_duty",
            model=model,
            year=year,
            variant=variant,
        )

    # --- Transit Bus ---
    if stem.startswith("Transit_Bus_"):
        rest = stem.removeprefix("Transit_Bus_")
        bus_type = rest.lower().replace(" ", "_")
        model = f"40_foot_{bus_type}" if bus_type != "default" else "40_foot"
        return VehicleIdentity(
            make="generic_transit",
            model=model,
            year="2020-2025",
            variant=variant,
        )

    # --- Standard vehicles: try to extract year from prefix ---
    parts = stem.split("_")
    year = 0
    start_idx = 0

    # Check if first token is a 4-digit year
    if parts[0].isdigit() and len(parts[0]) == 4:
        year = int(parts[0])
        start_idx = 1

    remaining = "_".join(parts[start_idx:])

    # Parse make/model/trim from the remaining string
    make, model, trim = _parse_vehicle_name(remaining)

    # Fold trim into model when it's not default
    if trim != "default":
        model = f"{model}_{trim}"

    # Strip temperature range patterns (e.g. "_0f_110f") from model names
    # that appear in steady/transient variants
    model = re.sub(r"_\d+f_\d+f$", "", model)

    return VehicleIdentity(
        make=make,
        model=model,
        year=year,
        variant=variant,
    )


# ---------------------------------------------------------------------------
# Vehicle name → make / model / trim
# ---------------------------------------------------------------------------

# Known make prefixes (order matters: longest match first)
_KNOWN_MAKES = [
    "CHEVROLET",
    "Chevrolet",
    "TOYOTA",
    "Toyota",
    "FORD",
    "Ford",
    "HYUNDAI",
    "Hyundai",
    "MITSUBISHI",
    "Mitsubishi",
    "BMW",
    "AUDI",
    "Audi",
    "TESLA",
    "Tesla",
    "KIA",
    "Kia",
    "Nissan",
    "NISSAN",
    "VW",
    "Volkswagen",
    "Volvo",
    "Renault",
    "Honda",
    "Maruti",
    "Mazda",
    "BYD",
    "MINI",
    "Peugot",
    "Peugeot",
    "Fiat",
    "Cupra",
    "Polestar",
    "VinFast",
]

_MAKE_ALIASES = {
    "vw": "volkswagen",
    "peugot": "peugeot",
}

# Normalize model names that refer to the same vehicle
_MODEL_ALIASES = {
    ("chevrolet", "bolt"): "bolt_ev",
}

# Known trim suffixes (checked in order)
_TRIM_PATTERNS = [
    "4cyl_2WD",
    "4cyl_2wd",
    "2WD",
    "2wd",
    "4WD",
    "4wd",
    "FWD",
    "fwd",
    "RWD",
    "rwd",
    "VDI",
    "Double_Cab_4WD",
    "Hardtop_2_door",
    "i-Stop",
]


def _parse_vehicle_name(name: str) -> tuple:
    """
    Parse a vehicle name string into (make, model, trim).

    Examples::

        'TOYOTA_Camry_4cyl_2WD'  → ('toyota', 'camry', '4cyl_2wd')
        'CHEVROLET_Spark_EV'     → ('chevrolet', 'spark_ev', 'default')
        'BMW_i3_REx_PHEV'        → ('bmw', 'i3_rex_phev', 'default')
    """
    make = "unknown"
    rest = name

    for m in _KNOWN_MAKES:
        prefix = m + "_"
        if name.startswith(prefix):
            make = m.lower()
            rest = name[len(prefix) :]
            break

    # Special case: "Leaf" without prefix → Nissan
    if make == "unknown" and rest.startswith("Leaf"):
        make = "nissan"

    # Special case: "Prius" without explicit make prefix → Toyota
    if make == "unknown" and rest.startswith("Prius"):
        make = "toyota"

    # Normalize some makes
    make = _MAKE_ALIASES.get(make, make)

    # Normalize model names
    model_str_lower = rest.lower().strip("_")
    for tp in _TRIM_PATTERNS:
        suffix = "_" + tp
        if rest.endswith(suffix):
            model_str_lower = rest[: -len(suffix)].lower().strip("_")
            break
    alias_key = (make, model_str_lower)
    if alias_key in _MODEL_ALIASES:
        # Replace the model portion so the trim parsing below still works
        new_model = _MODEL_ALIASES[alias_key]
        # Reconstruct rest with the alias
        rest = (
            new_model + rest[len(model_str_lower) :]
            if rest.lower().startswith(model_str_lower)
            else rest
        )

    # Split rest into model and trim
    trim = "default"
    model_str = rest

    for tp in _TRIM_PATTERNS:
        suffix = "_" + tp
        if rest.endswith(suffix):
            trim = tp.lower()
            model_str = rest[: -len(suffix)]
            break

    model = model_str.lower().strip("_")

    return make, model, trim


# ---------------------------------------------------------------------------
# Best-guess years for models without a year in the filename
# ---------------------------------------------------------------------------

_YEARLESS_MODEL_YEARS: Dict[str, int | str] = {
    "BYD_ATTO_3": 2022,
    "Maruti_Swift_4cyl_2WD": 2018,
    "Nissan_Navara": 2020,
    "Renault_Clio_IV_diesel": 2016,
    "Renault_Megane_1.5_dCi_Authentique": 2016,
    "Toyota_Corolla_Cross_Hybrid": 2022,
    "Toyota_Etios_Liva_diesel": 2015,
    "Toyota_Hilux_Double_Cab_4WD": 2020,
    "Toyota_Mirai": 2021,
}


def _resolve_year(identity: VehicleIdentity, original_stem: str) -> VehicleIdentity:
    """Fill in year=0 from the best-guess table when the filename has no year."""
    if identity.year != 0 and identity.year != "0":
        return identity

    # Strip variant suffixes for lookup
    lookup = original_stem.removesuffix(".json")
    for suffix in [
        "_Charge_Depleting",
        "_Charge_Sustaining",
        "_Stochastic",
        "_steady",
        "_transient",
    ]:
        lookup = lookup.removesuffix(suffix)

    guessed = _YEARLESS_MODEL_YEARS.get(lookup, 0)
    if guessed:
        log.info("Guessed year %d for %s", guessed, original_stem)
        return VehicleIdentity(
            make=identity.make,
            model=identity.model,
            year=guessed,
            variant=identity.variant,
        )

    return identity


# ---------------------------------------------------------------------------
# Enrich identity with vehicle attributes
# ---------------------------------------------------------------------------


def _enrich_vehicle_attributes(identity: VehicleIdentity) -> VehicleIdentity:
    """Populate fuel_type, drivetrain, engine, and trim on the identity.

    fuel_type is left as None here — it is auto-inferred inside
    ``convert_legacy_json`` from the powertrain_type and target metrics
    in the actual model JSON (which has more information than the filename).
    """
    return VehicleIdentity(
        make=identity.make,
        model=identity.model,
        year=identity.year,
        variant=identity.variant,
        # fuel_type is inferred later in convert_legacy_json
        fuel_type=None,
        drivetrain=_extract_drivetrain(identity.make, identity.model),
        engine=_extract_engine(identity.model),
        trim=_extract_trim(identity.make, identity.model),
    )


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------


def convert_library(
    input_dir: Path,
    output_dir: Path,
) -> List[Path]:
    """
    Convert all legacy JSON models in *input_dir* using NREL library naming.

    Returns list of all created model directories.
    """
    all_created: List[Path] = []
    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        log.warning("No .json files found in %s", input_dir)
        return all_created

    for json_path in json_files:
        log.info("Converting %s ...", json_path.name)
        try:
            identity = _parse_library_filename(json_path.name)
            identity = _resolve_year(identity, json_path.name)
            identity = _enrich_vehicle_attributes(identity)

            created = convert_legacy_json(
                json_path=json_path,
                output_dir=output_dir,
                identity=identity,
            )
            all_created.extend(created)
        except Exception:
            log.exception("Failed to convert %s", json_path.name)

    return all_created


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Batch-convert the NREL legacy model library to v2 format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_dir", type=Path, help="Directory containing legacy .json files."
    )
    parser.add_argument("output_dir", type=Path, help="Root output directory.")

    args = parser.parse_args()

    created = convert_library(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    print(f"\nDone. Created {len(created)} model(s) total.")


if __name__ == "__main__":
    main()
