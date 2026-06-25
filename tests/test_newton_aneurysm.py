"""L0d.3 — flow diverter + saccular aneurysm (two-way flow diversion, §3.4.3).

The compartment math is pure numpy (no Newton); the sim integration needs the
1-D FlowField, so it pytest.importorskips warp/newton (matches the flow tests)."""

import math

import numpy as np
import pytest

from lumen.newton.aneurysm import Aneurysm, AneurysmSac
from lumen.newton.devices import FlowDiverter


# ---- the device overlap (pure) -----------------------------------------------
def test_flow_diverter_diversion_tracks_neck_overlap():
    an = Aneurysm(s_neck=50.0, neck_width=4.0)
    full = FlowDiverter(deployed_center=50.0, span=20.0, metal_coverage=0.4)
    assert full.diversion(an) == pytest.approx(0.4)          # span covers the whole neck
    half = FlowDiverter(deployed_center=50.0 + 1.0, span=2.0, metal_coverage=0.4)
    assert half.diversion(an) == pytest.approx(0.2)          # span covers half the neck
    missed = FlowDiverter(deployed_center=10.0, span=8.0, metal_coverage=0.4)
    assert missed.diversion(an) == 0.0                       # placed away from the neck


# ---- the sac compartment (pure numpy, sinusoidal drive) ----------------------
def _drive(sac, diversion, cycles=8, hr=1.5, P0=100.0, dP=40.0, steps=400):
    """Run the sac under a sinusoidal lumen pressure; return (peak inflow, turnover)."""
    dt = cycles / hr / steps
    for k in range(steps):
        P = P0 + dP * math.sin(2 * math.pi * hr * k * dt)
        sac.update(P, dt, diversion=diversion)
    return sac.inflow_peak(), sac.turnover_time()


def test_diverter_cuts_inflow_and_lengthens_turnover():
    an = Aneurysm(s_neck=50.0, neck_width=4.0, sac_volume=100.0)
    i_open, t_open = _drive(AneurysmSac(an), diversion=0.0)
    i_div, t_div = _drive(AneurysmSac(an), diversion=0.35)
    i_dense, t_dense = _drive(AneurysmSac(an), diversion=0.6)
    # a flow diverter throttles the neck: less inflow jet, longer washout (stasis)
    assert i_div < 0.85 * i_open                             # meaningful inflow reduction
    assert t_div > 1.15 * t_open                             # turnover lengthens
    # denser coverage diverts more (monotone) — the dose-response a clinician picks on
    assert i_dense < i_div and t_dense > t_div


def test_sac_inflow_is_pulse_driven_not_a_charging_transient():
    # lazy-init seats P_sac at the first lumen pressure, so the metric is the cyclic
    # exchange, not the one-off charge-up — a steady (no-pulse) lumen drives ~no inflow.
    an = Aneurysm(s_neck=50.0)
    steady, _ = _drive(AneurysmSac(an), diversion=0.0, dP=0.0)
    pulsed, _ = _drive(AneurysmSac(an), diversion=0.0, dP=40.0)
    assert steady < 1e-6 < pulsed


# ---- sim integration (needs the 1-D FlowField) -------------------------------
def _flow_sim(flow_diverter, hr=1.5):
    from lumen.newton.flow import FlowField, FlowFieldParams
    from lumen.newton.sim import NewtonGuidewireSim
    M, L, R, n = 50, 100.0, 2.0, 11
    vessel = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    dev = np.stack([np.zeros(n), np.zeros(n), np.linspace(4, 4 + 2 * (n - 1), n)], axis=1)
    return NewtonGuidewireSim(vessel, R, dev, radius=0.2, kappa=3e3, d_hat=0.3,
                              flow=FlowField(FlowFieldParams(heart_rate=hr)),
                              aneurysm=Aneurysm(s_neck=50.0), flow_diverter=flow_diverter,
                              device="cpu")


def test_flow_diverter_reduces_sac_inflow_in_sim():
    pytest.importorskip("warp")
    pytest.importorskip("newton")

    def run(fd):
        sim = _flow_sim(fd)
        for _ in range(120):
            sim.step(dt=2.5e-2, substeps=2)
        return sim.sac_inflow_peak(), sim.sac_turnover_time(), sim.sac_diversion()

    i0, t0, d0 = run(None)
    i1, t1, d1 = run(FlowDiverter(deployed_center=50.0, span=20.0, metal_coverage=0.6))
    im, tm, dm = run(FlowDiverter(deployed_center=10.0, span=8.0, metal_coverage=0.6))  # misses
    assert d0 == 0.0 and d1 == pytest.approx(0.6) and dm == 0.0
    assert i1 < 0.7 * i0 and t1 > 1.3 * t0                   # the diverter works through the live P(s)
    assert im == pytest.approx(i0) and tm == pytest.approx(t0)   # a missed diverter does nothing


def test_aneurysm_sac_resets_with_the_sim():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    sim = _flow_sim(FlowDiverter(deployed_center=50.0, span=20.0, metal_coverage=0.5))
    for _ in range(40):
        sim.step(dt=2.5e-2, substeps=2)
    assert sim.sac_inflow_peak() > 0.0
    sim.reset()
    assert sim.sac_inflow_peak() == 0.0 and sim.aneurysm_sac.P_sac is None


def test_aneurysm_requires_flow_field_and_single_env():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from lumen.newton.flow import FlowField
    from lumen.newton.sim import NewtonGuidewireSim
    vessel = np.stack([np.zeros(40), np.zeros(40), np.linspace(0, 80, 40)], axis=1)
    dev = np.stack([np.zeros(11), np.zeros(11), np.linspace(4, 24, 11)], axis=1)
    with pytest.raises(NotImplementedError, match="1-D FlowField"):     # no flow at all
        NewtonGuidewireSim(vessel, 2.0, dev, aneurysm=Aneurysm(s_neck=40.0), device="cpu")
    with pytest.raises(ValueError, match="without an aneurysm"):        # diverter, no sac
        NewtonGuidewireSim(vessel, 2.0, dev, flow=FlowField(),
                           flow_diverter=FlowDiverter(40.0), device="cpu")
    with pytest.raises(NotImplementedError, match="single-env"):        # batched
        NewtonGuidewireSim(vessel, 2.0, dev, flow=FlowField(), n_envs=2,
                           aneurysm=Aneurysm(s_neck=40.0), device="cpu")
