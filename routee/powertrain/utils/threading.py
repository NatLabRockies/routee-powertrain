from __future__ import annotations

import os


def get_restricted_threads() -> int | None:
    """
    Return the number of restricted threads if in a restricted environment.

    In resource-constrained environments, os.sched_getaffinity may return fewer threads
    than os.cpu_count. If we detect this restriction, return the restricted count so
    ONNX respects the environment limit. Otherwise return None to let ONNX use its
    default behavior.
    """
    try:
        # os.sched_getaffinity is Linux-only; guarded at runtime for other
        # platforms (e.g. macOS) where the attribute is absent.
        restricted = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        return None

    total = os.cpu_count() or 1
    return restricted if restricted < total else None
