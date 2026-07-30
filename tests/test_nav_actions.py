"""Unit tests for shared navigation action contracts."""

import numpy as np
import pytest

from lumen.envs._validation import validate_action_scale


@pytest.mark.parametrize("value", [0, -1, True, np.bool_(False), float("nan"), float("inf"), "invalid"])
def test_validate_action_scale_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="finite positive number"):
        validate_action_scale(value, "max_insertion")


@pytest.mark.parametrize("value", [0.25, 2, "1.5"])
def test_validate_action_scale_accepts_positive_finite_numbers(value):
    assert validate_action_scale(value, "max_twist") == pytest.approx(float(value))
