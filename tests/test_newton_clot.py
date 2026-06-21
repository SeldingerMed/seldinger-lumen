"""Task #6: INSIST/Luraghi clot constitutive + two-way flow (doc §3.4.4, §3.4.3).

Parameters grounded in Luraghi et al. (Interface Focus 2020): Ogden clot bulk
(μ≈0.5 kPa, α≈0.3), clot–device friction ≈0.1, fragmentation during retrieval.
Pure numpy (no newton needed).
"""

import numpy as np

from lumen.newton.clot import ClotParams, ClotModel, ogden_stress, downstream_flow


def test_ogden_constitutive():
    p = ClotParams()
    assert abs(ogden_stress(1.0, p)) < 1e-9            # zero at rest
    assert ogden_stress(1.3, p) > 0                    # tension positive
    assert ogden_stress(0.8, p) < 0                    # compression negative
    # monotone increasing
    s = [ogden_stress(l, p) for l in (0.8, 1.0, 1.2, 1.4)]
    assert all(a < b for a, b in zip(s, s[1:]))


def _retrieve(v_retract, aspiration=0.0):
    clot = ClotModel(s_clot=0.040, params=ClotParams(), engage_radius=3e-3)
    s_tip, dt = 0.020, 1e-3
    for _ in range(40):                                 # advance & engage
        s_tip = min(s_tip + 0.0008, 0.041)
        clot.step(s_tip, dt)
    r = None
    for _ in range(60):                                 # retract
        s_tip -= v_retract * dt
        r = clot.step(s_tip, dt, aspiration=aspiration)
    return r


def test_slow_retraction_retrieves_clot():
    r = _retrieve(v_retract=0.1)
    assert not r["fragmented"]
    assert r["retrieved"] > 1e-3                        # clot pulled along with the device


def test_clot_does_not_teleport_to_tip():
    # device pushes 5mm PAST the clot, then retracts: the clot must follow the
    # device velocity, never snapping forward (retrieved must stay >= 0).
    clot = ClotModel(s_clot=0.040, params=ClotParams(), engage_radius=3e-3)
    s_tip, dt = 0.030, 1e-3
    for _ in range(40):                                 # advance well past the clot
        s_tip = min(s_tip + 0.0008, 0.045)
        clot.step(s_tip, dt)
    worst = 0.0
    for _ in range(40):                                 # retract
        s_tip -= 0.1 * dt
        r = clot.step(s_tip, dt)
        worst = min(worst, r["retrieved"])
    assert worst >= -1e-9                               # never teleported forward


def test_fast_yank_fragments_clot():
    r = _retrieve(v_retract=0.8)
    assert r["fragmented"]                              # retrieval load exceeds failure


def test_aspiration_rescues_retrieval():
    # the same fast pull that fragments without aspiration succeeds with it
    assert _retrieve(v_retract=0.8)["fragmented"]
    r = _retrieve(v_retract=0.8, aspiration=0.03)
    assert not r["fragmented"] and r["retrieved"] > 1e-3


def test_two_way_flow_occlusion_and_aspiration():
    assert downstream_flow(4.0, clot_present=False) == 4.0
    assert downstream_flow(4.0, clot_present=True) < 0.5            # clot occludes
    partial = downstream_flow(4.0, clot_present=True, aspiration_fraction=0.5)
    assert downstream_flow(4.0, True) < partial < 4.0              # aspiration recovers flow
    # #17 — aspiration_fraction is bounded to [0,1]: cannot exceed base flow
    assert downstream_flow(4.0, True, aspiration_fraction=2.0) <= 4.0 + 1e-9
