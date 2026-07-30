"""Unit tests for shared navigation action contracts."""

import numpy as np
import pytest

from lumen.envs._validation import validate_action_scale, validate_boolean
from lumen.envs.nav_gym import NavEnv
from lumen.envs.tree_nav import TreeNavEnv


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, np.bool_(False)])
def test_validate_boolean_rejects_truthy_and_boolean_like_values(value):
    with pytest.raises(ValueError, match="terminate_on_unsafe must be a boolean"):
        validate_boolean(value, "terminate_on_unsafe")


@pytest.mark.parametrize("value", [False, True])
def test_validate_boolean_accepts_booleans(value):
    assert validate_boolean(value, "terminate_on_unsafe") is value


def test_navigation_envs_reject_non_boolean_termination_before_setup():
    with pytest.raises(ValueError, match="terminate_on_unsafe must be a boolean"):
        NavEnv(terminate_on_unsafe="false")
    with pytest.raises(ValueError, match="terminate_on_unsafe must be a boolean"):
        TreeNavEnv(None, terminate_on_unsafe="false")


@pytest.mark.parametrize("value", [0, -1, True, np.bool_(False), float("nan"), float("inf"), "invalid"])
def test_validate_action_scale_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="finite positive number"):
        validate_action_scale(value, "max_insertion")


@pytest.mark.parametrize("value", [0.25, 2, "1.5"])
def test_validate_action_scale_accepts_positive_finite_numbers(value):
    assert validate_action_scale(value, "max_twist") == pytest.approx(float(value))
