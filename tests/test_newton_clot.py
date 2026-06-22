"""Task: real clot field — Ogden constitutive, finite extent, R-collapse,
progressive damage, flow occlusion (doc §3.4.4)."""

import numpy as np
import pytest

from lumen.newton.clot import ClotField, ClotParams, ogden_stress


def _clot():
    return ClotField(s_max=80.0, n_s=40, n_th=8, R_base=2.0, s0=35, s1=45,
                     height=1.6, params=ClotParams())


def _load_grid(c, total_per_cell):
    wl = np.zeros(c.n_s * c.n_th)
    W = wl.reshape(c.n_s, c.n_th)
    W[c.mask, :] = total_per_cell / c.n_th               # spread over θ so each s sums to total
    return wl


def test_ogden_constitutive():
    p = ClotParams()
    assert abs(ogden_stress(1.0, p)) < 1e-9
    assert ogden_stress(1.3, p) > 0 and ogden_stress(0.8, p) < 0
    s = [ogden_stress(l, p) for l in (0.8, 1.0, 1.2, 1.4)]
    assert all(a < b for a, b in zip(s, s[1:]))          # monotone


def test_clot_has_finite_extent():
    c = _clot()
    assert c.o.max() == 1.6 and c.o[0] == 0.0            # occluded only in [s0,s1]
    grid = c.occlusion_grid()
    assert grid.shape == (40 * 8,) and grid.max() == 1.6


def test_compression_follows_ogden_curve_not_constant():
    os = []
    for L in (0.0, 5e-3, 2e-2, 5e-2):
        c = _clot()
        c.update(_load_grid(c, L), dt=1e-3)
        os.append(c.o.max())
    assert all(a >= b for a, b in zip(os, os[1:]))       # more load -> more compression
    assert os[0] > os[1] > os[2]                         # genuinely varies (not a constant)


def test_progressive_damage_then_fragmentation():
    c = _clot()
    over = 2.0 * c.p.failure_stress * c.p.area           # sustained over-failure contact load
    wl = _load_grid(c, over)
    Ds = []
    for _ in range(30):
        c.update(wl, dt=1e-2)
        Ds.append(c.max_damage())
    assert 0.0 < Ds[2] < Ds[10] < 1.0                    # PROGRESSIVE (not a boolean threshold)
    assert Ds[-1] == 1.0 and c.o.max() == 0.0           # fully damaged -> occlusion cleared


def test_friction_uses_contact_normal_force_not_bulk_stress():
    # #review — friction = μ · (actual contact normal force), not μ · bulk Ogden stress
    c = _clot()
    Fn = 0.05
    assert abs(c.friction_resistance(_load_grid(c, Fn)) - c.p.friction_mu * (Fn * c.mask.sum())) < 1e-9


def test_retrieve_fragmentation_scales_with_dt():
    # M2: tearing damage now integrates with dt via the damage law (was a fixed 0.3
    # jump that ignored the timestep). failure_stress chosen so wall-grip hold > R_coh.
    p = ClotParams(failure_stress=3.0e3)
    def frag_D(dt):
        c = ClotField(80.0, 40, 8, 2.0, 35, 45, 1.6, p)
        r = c.retrieve(delta_s=1.0, engagement=0.0, aspiration=0.0, dt=dt)
        assert r["status"] == "fragment"
        return c.max_damage()
    d1, d2 = frag_D(1e-2), frag_D(2e-2)
    assert d2 > d1 and abs(d2 - 2.0 * d1) < 0.2 * d2     # ~linear in dt, not a fixed jump


def test_clot_occludes_lumen_and_drives_flow_in_sim():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from lumen.newton.sim import NewtonGuidewireSim
    from lumen.newton.flow import NewtonFlow
    M, L, R, n = 60, 120.0, 2.0, 11
    vessel = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    dev = np.stack([np.zeros(n), np.zeros(n), np.linspace(4, 4 + 2 * (n - 1), n)], axis=1)
    flow = NewtonFlow()
    sim = NewtonGuidewireSim(vessel, R, dev, radius=0.2, kappa=3e3, d_hat=0.3,
                             flow=flow, clot_segment=(55, 70), clot_height=1.6,
                             device="cpu")
    sim.step(dt=2.5e-2, substeps=2)
    r0 = sim.solver._wall.r0_field.numpy().reshape(sim.solver._wall.n_s, sim.solver._wall.n_th)
    s_grid = np.linspace(0, sim.solver._wall.s_max, sim.solver._wall.n_s)
    in_clot = (s_grid >= 55) & (s_grid <= 70)
    assert r0[in_clot].max() < R - 1.0                  # R-collapse reaches the contact kernel
    assert r0[~in_clot].min() > R - 0.3                 # lumen open away from the clot
    assert flow.occlusion > 0.5                          # clot occludes -> downstream flow drops
    assert flow.downstream_Q() < flow.Q()
