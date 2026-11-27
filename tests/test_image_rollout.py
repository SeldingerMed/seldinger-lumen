import base64

import numpy as np
import pytest

from lumen.experiments import image_rollout as run


def test_pixels_decide_between_two_actions_or_no_exposure():
    image = np.zeros((64, 64), dtype=np.uint8)
    image[47:55, 9:17] = 95
    assert run.choose(image) == "left"
    assert run.choose(np.fliplr(image)) == "right"
    assert run.choose(image + np.fliplr(image)) == "stop"


# case-b is the fast insertion (6 mm a push, open-loop). It crossed the wall when the wall
# let the wire through; with the wire held in the lumen it enters the branch.
@pytest.mark.parametrize("cid,expected", [("case-a", "reach"), ("case-b", "reach"), ("case-c", "abstain")])
def test_actual_cpu_rollouts(cid, expected):
    pytest.importorskip("newton")
    result = run.run_case(cid)
    s = result["scores"]
    assert result["hidden"]["outcome"] == expected
    frame = result["frames"][0]
    pixels = np.frombuffer(base64.b64decode(frame["data"]), np.uint8).reshape(frame["shape"])
    assert run.choose(pixels) == result["action"]
    assert s["target_reached"] == (expected == "reach")
    assert (s["max_penetration_mm"] > 0.3) == (expected == "contact")
    assert s["abstained"] == (expected == "abstain")
    assert len(result["hidden"]["step_evidence"]) == len(result["hidden"]["tip_xyz_mm"]) - 1


def test_crossing_the_wall_stop_ends_the_insertion(monkeypatch):
    pytest.importorskip("newton")
    monkeypatch.setattr(run, "CONTACT_LIMIT_MM", -1.0)   # any step is over the stop
    result = run.run_case("case-b")
    assert result["hidden"]["outcome"] == "contact"
    assert len(result["hidden"]["step_evidence"]) == 1
    assert not result["scores"]["target_reached"]
