"""Task: stent-retriever — engagement and clot retrieval/slip/fragmentation (§3.4.1)."""

import numpy as np
import pytest

from lumen.newton.clot import ClotField, ClotParams
from lumen.newton.devices import Stentriever


def _clot(grip=0.15):
    return ClotField(80.0, 40, 8, 2.0, 35, 45, 1.6, ClotParams(grip_coeff=grip))


def test_engagement_scales_with_overlap_and_force():
    c = _clot()
    on = Stentriever(deployed_center=40, radial_force=0.2, n_struts=6)   # over the clot
    off = Stentriever(deployed_center=10, radial_force=0.2, n_struts=6)  # away from clot
    weak = Stentriever(deployed_center=40, radial_force=0.05, n_struts=6)
    assert on.engagement_strength(c) > 0
    assert off.engagement_strength(c) == 0.0                 # no overlap -> no grip
    assert on.engagement_strength(c) > weak.engagement_strength(c)


def test_strong_engagement_retrieves_clot():
    c = _clot()
    st = Stentriever(deployed_center=40, radial_force=0.2, n_struts=6)
    r = c.retrieve(2.0, st.engagement_strength(c), aspiration=0.0)
    assert r["status"] == "retrieve" and r["retrieved"] == 2.0


def test_weak_engagement_slips():
    c = _clot()
    st = Stentriever(deployed_center=40, radial_force=0.003, n_struts=2)
    r = c.retrieve(2.0, st.engagement_strength(c), aspiration=0.0)
    assert r["status"] == "slip" and r["retrieved"] == 0.0


def test_high_grip_fragments_and_aspiration_rescues():
    st = Stentriever(deployed_center=40, radial_force=0.2, n_struts=6)
    c = _clot(grip=0.4)
    assert c.retrieve(2.0, st.engagement_strength(c), aspiration=0.0)["status"] == "fragment"
    c2 = _clot(grip=0.4)
    r = c2.retrieve(2.0, st.engagement_strength(c2), aspiration=0.06)   # aspiration lowers the hold
    assert r["status"] == "retrieve" and r["retrieved"] == 2.0


def test_batched_retrieve_keeps_fragmentation_independent():
    c = ClotField(80.0, 40, 8, 2.0, 35, 45, 1.6,
                  ClotParams(grip_coeff=0.4), n_envs=2, device="cpu")
    st = Stentriever(deployed_center=40, radial_force=0.2, n_struts=6)
    engagement = [st.engagement_strength_for_mask(c.s_grid, c.mask_env[e]) for e in range(2)]
    r = c.retrieve_batched(2.0, engagement, aspiration=np.array([0.06, 0.0]))
    assert [x["status"] for x in r] == ["retrieve", "fragment"]
    assert c.retrieved_env.tolist() == [2.0, 0.0]
    assert c.D_env[0].max() == 0.0
    assert c.D_env[1].max() > 0.0


def test_stentriever_retrieves_clot_in_full_sim():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from lumen.newton.sim import NewtonGuidewireSim
    from lumen.newton.flow import NewtonFlow
    M, L, R, n = 60, 120.0, 2.0, 11
    vessel = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    dev = np.stack([np.zeros(n), np.zeros(n), np.linspace(40, 40 + 2 * (n - 1), n)], axis=1)
    flow = NewtonFlow()
    st = Stentriever(deployed_center=62, radial_force=0.2, n_struts=6)
    sim = NewtonGuidewireSim(vessel, R, dev, radius=0.2, kappa=3e3, d_hat=0.3,
                             flow=flow, clot_segment=(55, 70), clot_height=1.6,
                             stentriever=st, device="cpu")
    sim.step(dt=2.5e-2, substeps=2)                          # settle; clot establishes occlusion
    assert flow.occlusion > 0.5                              # clot occludes downstream flow
    centre0 = sim.clot.s_grid[sim.clot.mask].mean()
    for _ in range(5):
        sim.step(dt=2.5e-2, substeps=2, insertion=-2.0, aspiration=0.02)   # retract w/ aspiration
    assert sim.last_retrieval["status"] == "retrieve"
    assert sim.clot.retrieved > 0.0                          # clot dragged proximally
    assert sim.clot.s_grid[sim.clot.mask].mean() < centre0  # the occlusion moved toward the catheter
