from pathlib import Path


def bundled_registry_root() -> Path:
    """Return the path to the bundled local model registry."""
    return Path(__file__).parent
