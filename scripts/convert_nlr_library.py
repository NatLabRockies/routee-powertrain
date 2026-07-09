#!/usr/bin/env python3
"""
Batch-convert the NREL legacy model library to v2 format.

This script applies NREL-specific naming conventions to parse
make / model / year / trim / variant / drivetrain / engine from the legacy
JSON filenames, then delegates to ``convert_legacy_models.convert_legacy_json``
for the actual conversion.

The v2 on-disk layout is
``v2/<make>/<vehicle_slug>/<year>/<config_slug>/v<N>/``. Both slugs are
derived from metadata by the package (``ModelId.from_metadata``): the
``vehicle_slug`` is the model name plus the coarse powertrain family (e.g.
``camry_ice``, ``leaf_24_kwh_bev``, ``volt_phev``), and ``variant`` is folded
into the derived ``config_slug`` (e.g. ``rf_charge_depleting_c3326385``)
rather than being its own path segment. Legacy filenames lump spec tokens
into the vehicle name; this script keeps the commercial designation in the
model name, strips bare spec tokens (``4cyl``, drivetrain markers, redundant
trailing powertrain flavors), and records engine/drivetrain/trim as
descriptive, non-identity metadata. See ``convert_legacy_models.py``.

Usage
-----
::

    python scripts/convert_nlr_library.py old-json-models.ignore/ output_dir/
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import replace
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


def _extract_drivetrain(make: str, model: str) -> tuple[Optional[str], Optional[str]]:
    """Extract drivetrain from the model name.

    Returns ``(drivetrain, strip_token)``: the Drivetrain enum name string (or
    None) plus the literal name token to strip from the model name (or None
    when the marker is part of the model designation, e.g. ``xdrive40``).
    """
    name = model.lower()

    # Explicit drivetrain patterns
    if "_rwd" in name or name.endswith("_rwd"):
        return "RWD", "rwd"
    if "_fwd" in name or name.endswith("_fwd"):
        return "FWD", "fwd"
    if "_4wd" in name or name.endswith("_4wd"):
        return "FOURWD", "4wd"
    if "xdrive" in name:
        return "AWD", None  # xdrive<NN> is part of the model designation
    if "dual_motor" in name:
        return "AWD", "dual_motor"
    if name.endswith("_twin"):
        return "AWD", "twin"

    # 2WD: resolve per vehicle using lookup table
    if "_2wd" in name or name.endswith("_2wd"):
        # Find the base model by stripping everything after common suffixes
        for (m, base), dt in _2WD_DRIVETRAIN_MAP.items():
            if make.lower() == m and base in name:
                return dt, "2wd"
        log.warning(
            "Could not resolve 2wd drivetrain for %s/%s; leaving as None",
            make,
            model,
        )
        return None, "2wd"

    return None, None


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


# Models whose name contains a _KNOWN_TRIMS token that is actually part of the
# model designation, not a trim (e.g. the Pajero Sport is a distinct vehicle,
# not a Pajero with a "sport" trim).
_MODELS_WITHOUT_TRIM_EXTRACTION = {
    ("mitsubishi", "pajero_sport"),
}


def _extract_trim(make: str, model: str) -> Optional[str]:
    """Extract trim level from the model name using known trim values."""
    name = model.lower()

    if (make.lower(), name) in _MODELS_WITHOUT_TRIM_EXTRACTION:
        return None

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


# Powertrain-flavor tokens that are redundant with the vehicle_slug's family
# suffix — stripped from the END of a model name only (e.g. spark_ev → spark,
# which derives to spark_bev; optima_hybrid → optima → optima_hev).
_FLAVOR_TOKENS = {"ev", "phev", "hev", "bev", "hybrid"}

# Names whose trailing flavor token is part of the designation itself and must
# not be stripped (e.g. "Panda Mild Hybrid" is the commercial name of an
# ICE-typed mild hybrid — stripping would leave "panda_mild").
_FLAVOR_KEEP = {
    ("fiat", "panda_mild_hybrid"),
}


def _strip_token(name: str, token: str) -> str:
    """Remove the first occurrence of *token* from an ``_``-separated name,
    matching whole segment runs only (so stripping ``le`` can't damage
    ``lightning``). Returns the name unchanged if the token isn't present as a
    segment run or if stripping would leave the name empty.
    """
    segments = name.split("_")
    token_segments = token.split("_")
    for i in range(len(segments) - len(token_segments) + 1):
        if segments[i : i + len(token_segments)] == token_segments:
            remaining = segments[:i] + segments[i + len(token_segments) :]
            stripped = "_".join(remaining).strip("_")
            return stripped if stripped else name
    return name


def _extract_structured_fields(
    make: str, name: str
) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Split a legacy lumped vehicle name into a model name + descriptive fields.

    Extracts engine / drivetrain / trim into their structured metadata fields
    (filterable, correctable — they do NOT feed identity). The model name keeps
    the vehicle's commercial designation — engine designations like ``1.5tsi``
    and trims like ``24_kwh`` stay in it, since ``derive_vehicle_slug`` is only
    ``<model>_<powertrain_family>`` and the model name must distinguish
    same-year stablemates (``golf_1.5tsi`` vs ``golf_2.0tdi``). Stripped from
    the name: bare spec tokens that aren't part of any designation (``4cyl``,
    drivetrain markers like ``2wd``/``dual_motor``), and a trailing powertrain
    flavor token that the family suffix makes redundant (``spark_ev`` →
    ``spark`` → slug ``spark_bev``).

    Returns ``(model, engine, drivetrain, trim)``.
    """
    engine = _extract_engine(name)
    drivetrain, drivetrain_token = _extract_drivetrain(make, name)
    trim = _extract_trim(make, name)

    model = name
    if engine == "4cyl":
        # A cylinder count is a spec, not part of the commercial designation;
        # displacement/motor designations (1.5tsi, 3.5_l, 300kw) stay in the name.
        model = _strip_token(model, engine)
    if engine is None and "diesel" in model.split("_"):
        # Names like colorado_2wd_diesel carry the powerplant designation —
        # record it as the engine (kept in the name).
        engine = "diesel"
    if drivetrain_token is not None:
        model = _strip_token(model, drivetrain_token)

    segments = model.split("_")
    if (
        len(segments) > 1
        and segments[-1] in _FLAVOR_TOKENS
        and (make, model) not in _FLAVOR_KEEP
    ):
        model = "_".join(segments[:-1])

    return model, engine, drivetrain, trim


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
        variant = "steady_thermal"
        stem = stem.removesuffix("_steady")
    elif stem.endswith("_transient"):
        variant = "transient_thermal"
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
        # The cab type and power rating ARE the designation of these generic
        # truck classes, so they stay in the model name (the slug appends only
        # the powertrain family: class_8_daycab_300kw_heavy_duty). They are
        # also recorded as descriptive engine/trim metadata for filtering.
        model = f"class_8_{cab_type}"
        if power_rating:
            model = f"{model}_{power_rating}"
        return VehicleIdentity(
            make="generic_heavy_duty",
            model=model,
            year=year,
            variant=variant,
            engine=power_rating or None,
            trim=cab_type,
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
    # Parentheses in legacy names (e.g. "C-MAX_(PHEV)") are dropped — they'd
    # otherwise land in the derived vehicle_slug path segment.
    stem = stem.replace("(", "").replace(")", "")
    parts = stem.split("_")
    year = 0
    start_idx = 0

    # Check if first token is a 4-digit year
    if parts[0].isdigit() and len(parts[0]) == 4:
        year = int(parts[0])
        start_idx = 1

    remaining = "_".join(parts[start_idx:])

    # Parse make and the (still-lumped) vehicle name, then split the lump into
    # bare model + structured engine/drivetrain/trim fields.
    make, base_model, suffix = _parse_vehicle_name(remaining)
    lumped = base_model if suffix == "default" else f"{base_model}_{suffix}"

    # Strip temperature range patterns (e.g. "_0f_110f") from model names
    # that appear in steady/transient variants
    lumped = re.sub(r"_\d+f_\d+f$", "", lumped)

    model, engine, drivetrain, trim = _extract_structured_fields(make, lumped)

    return VehicleIdentity(
        make=make,
        model=model,
        year=year,
        variant=variant,
        # fuel_type is inferred later in convert_legacy_json from the
        # powertrain type and target metrics in the model JSON itself.
        engine=engine,
        drivetrain=drivetrain,
        trim=trim,
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
    Parse a vehicle name string into (make, base_model, suffix).

    The suffix is the trailing engine/drivetrain token run from
    ``_TRIM_PATTERNS`` (historically called "trim"); the caller re-joins it
    with the base model and hands the lump to ``_extract_structured_fields``,
    which does the real engine/drivetrain/trim split.

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
        log.info("Guessed year %s for %s", guessed, original_stem)
        return replace(identity, year=guessed)

    return identity


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
