"""The simulator's rest shape option keeps its older positional arguments' meaning."""

import inspect

import numpy as np
import pytest

pytest.importorskip("newton")

from lumen.newton.sim import NewtonGuidewireSim  # noqa: E402


def test_rest_points_is_keyword_only():
    p = inspect.signature(NewtonGuidewireSim).parameters
    assert p["rest_points"].kind is inspect.Parameter.KEYWORD_ONLY
    names = list(p)
    assert names.index("bend_damping") == names.index("bend_stiffness") + 1


def test_rest_points_must_match_the_wire():
    line = np.c_[np.zeros(30), np.zeros(30), np.linspace(0, 60, 30)]
    wire = np.c_[np.zeros(5), np.zeros(5), np.arange(5) * 2.0]
    with pytest.raises(ValueError, match="as many points"):
        NewtonGuidewireSim(line, 2.0, wire, device="cpu", rest_points=wire[:4])
