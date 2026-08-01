"""Shared navigation action parsing helpers."""

from __future__ import annotations

import numpy as np


def parse_nav_action(action) -> tuple[float, float]:
    """Return clipped ``(insertion, twist)`` commands for navigation envs."""
    act = np.asarray(action, dtype=float).reshape(-1)
    if len(act) < 1:
        raise ValueError("action must contain at least an insertion command")
    if len(act) > 2:
        raise ValueError("action must contain insertion and optional twist only")
    if not np.isfinite(act).all():
        raise ValueError("action values must be finite")
    insertion = float(np.clip(act[0], -1.0, 1.0))
    # Backward compatibility: scalar actions from old policies mean no commanded twist.
    twist = float(np.clip(act[1] if len(act) > 1 else 0.0, -1.0, 1.0))
    return insertion, twist
