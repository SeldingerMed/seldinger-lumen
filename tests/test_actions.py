"""Tests for shared navigation action parsing."""

import numpy as np
import pytest

from lumen.envs._actions import parse_nav_action


@pytest.mark.parametrize("action", [[np.nan], [np.inf], [0.0, -np.inf]])
def test_parse_nav_action_rejects_non_finite_values(action):
    with pytest.raises(ValueError, match="action values must be finite"):
        parse_nav_action(action)
