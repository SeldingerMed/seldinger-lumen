"""Shared constructor validation for navigation environments."""

from __future__ import annotations

import numpy as np


def validate_boolean(value, name: str) -> bool:
    """Return a navigation flag without accepting truthy non-booleans."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def validate_action_scale(value, name: str) -> float:
    """Return a finite positive navigation action scale."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        scale = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite positive number") from None
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return scale
