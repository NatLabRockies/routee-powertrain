from __future__ import annotations

from typing import Union

Year = Union[int, tuple]


def parse_year(value) -> Year:
    """
    Parse a year value from various formats.

    Accepts:
        - int: single year (e.g. 2020)
        - tuple/list of two ints: year range (e.g. (2020, 2026))
        - str: single year "2020" or range "2020-2026"

    Returns: int for a single year, tuple[int, int] for a range.
    """
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(
                f"Year range tuple must have exactly 2 elements, got {len(value)}"
            )
        start, end = int(value[0]), int(value[1])
        if start > end:
            raise ValueError(f"Year range start ({start}) must be <= end ({end})")
        return (start, end)
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError(
                f"Year range list must have exactly 2 elements, got {len(value)}"
            )
        start, end = int(value[0]), int(value[1])
        if start > end:
            raise ValueError(f"Year range start ({start}) must be <= end ({end})")
        return (start, end)
    if isinstance(value, str):
        if "-" in value:
            parts = value.split("-")
            if len(parts) != 2:
                raise ValueError(
                    f"Year range string must be 'YYYY-YYYY', got '{value}'"
                )
            start, end = int(parts[0]), int(parts[1])
            if start > end:
                raise ValueError(f"Year range start ({start}) must be <= end ({end})")
            return (start, end)
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    raise ValueError(f"Cannot parse year from {type(value).__name__}: {value}")


def format_year(year: Year) -> str:
    """Format a year for display and filesystem path usage."""
    if isinstance(year, tuple):
        return f"{year[0]}-{year[1]}"
    return str(year)


def serialize_year(year: Year):
    """
    Serialize year for JSON/dict storage.

    Returns int for a single year, "YYYY-YYYY" string for a range.
    """
    if isinstance(year, tuple):
        return f"{year[0]}-{year[1]}"
    return year


def year_contains(year: Year, query_year: int) -> bool:
    """Check if a year value contains/matches a specific query year."""
    if isinstance(year, tuple):
        return year[0] <= query_year <= year[1]
    return year == query_year
