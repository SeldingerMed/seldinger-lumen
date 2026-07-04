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


def _run(sac, diversion, k0, n, dt, hr=1.5, P0=100.0, dP=40.0):
    """Drive an existing sac for n steps continuing the phase clock from step k0."""
    for k in range(k0, k0 + n):
        sac.update(P0 + dP * math.sin(2 * math.pi * hr * k * dt), dt, diversion=diversion)
    return k0 + n


def test_measurement_window_isolates_post_deployment():
    # the realistic workflow: observe the open aneurysm, DEPLOY the diverter, then
    # measure the post-deployment stasis. mark_window() must isolate the second phase
    # (M1/M2) — without it the running peak/turnover would stay blended with phase 1.
    an = Aneurysm(s_neck=50.0)
    sac = AneurysmSac(an)
    dt = 8 / 1.5 / 400
    k = _run(sac, 0.0, 0, 200, dt)                  # phase 1: open neck
    pre_peak, pre_turn = sac.inflow_peak(), sac.turnover_time()
    sac.mark_window()                               # "deploy" the diverter here
    P_kept = sac.P_sac                              # equilibrium preserved (not reset)
    _run(sac, 0.6, k, 200, dt)                      # phase 2: throttled neck
    assert sac.P_sac is not None and P_kept is not None
    assert sac.inflow_peak() < 0.7 * pre_peak       # window shows the POST-deploy peak
    assert sac.turnover_time() > 1.3 * pre_turn     # ...and the post-deploy stasis


def test_rc_integration_substeps_a_stiff_sac_that_would_blow_up():
    # L1: a STIFF sac (small C_sac) shrinks the stability limit 2τ=2·R·C; with a coarse
    # caller dt ≫ 2τ, plain forward Euler would diverge — the internal sub-stepping must
    # engage (n_sub>1) and keep it finite.
    an = Aneurysm(s_neck=50.0, sac_volume=10.0, wall_stiffness=1.0e5)   # C_sac=1e-4 -> τ=1e-4
    sac = AneurysmSac(an)
    tau = sac.R_neck_base * sac.C_sac
    dt = 0.05
    assert dt > 2.0 * tau                              # plain Euler (n_sub=1) would blow up here
    assert int(dt / (0.5 * tau)) + 1 > 1               # so the sub-stepping path is taken
    for k in range(200):
        sac.update(100.0 + 40.0 * math.sin(2 * math.pi * 1.5 * k * dt), dt)
    assert math.isfinite(sac.P_sac) and math.isfinite(sac.inflow_peak())


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


def test_batched_aneurysm_flow_diverters_keep_independent_sac_state():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from lumen.newton.flow import FlowField, FlowFieldParams
    from lumen.newton.sim import NewtonGuidewireSim

    M, L, R, n = 50, 100.0, 2.0, 11
    vessel = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    dev = np.stack([np.zeros(n), np.zeros(n), np.linspace(4, 4 + 2 * (n - 1), n)], axis=1)
    aneurysms = [
        Aneurysm(s_neck=45.0, neck_width=4.0, sac_volume=90.0, wall_stiffness=1800.0),
        Aneurysm(s_neck=55.0, neck_width=6.0, sac_volume=160.0, wall_stiffness=2600.0),
    ]
    diverters = [
        FlowDiverter(deployed_center=45.0, span=20.0, metal_coverage=0.6),
        FlowDiverter(deployed_center=55.0, span=3.0, metal_coverage=0.4),
    ]
    sim = NewtonGuidewireSim(
        vessel, R, dev, radius=0.2, kappa=3e3, d_hat=0.3,
        flow=FlowField(FlowFieldParams(heart_rate=1.5)),
        aneurysm=aneurysms, flow_diverter=diverters, n_envs=2, device="cpu",
    )

    for _ in range(120):
        sim.step(dt=2.5e-2, substeps=2)

    inflow = np.asarray(sim.sac_inflow_peak())
    turnover = np.asarray(sim.sac_turnover_time())
    diversion = np.asarray(sim.sac_diversion())
    sac_pressure = np.array([sac.P_sac for sac in sim.aneurysm_sacs])
    assert inflow.shape == turnover.shape == diversion.shape == (2,)
    assert diversion == pytest.approx([0.6, 0.2])
    assert np.all(inflow > 0.0) and np.all(np.isfinite(turnover))
    assert sac_pressure[0] != pytest.approx(sac_pressure[1])

    sim.reset()
    assert sim.sac_inflow_peak() == pytest.approx([0.0, 0.0])
    assert [sac.P_sac for sac in sim.aneurysm_sacs] == [None, None]


def test_aneurysm_sac_resets_with_the_sim():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    sim = _flow_sim(FlowDiverter(deployed_center=50.0, span=20.0, metal_coverage=0.5))
    for _ in range(40):
        sim.step(dt=2.5e-2, substeps=2)
    assert sim.sac_inflow_peak() > 0.0
    sim.reset()
    assert sim.sac_inflow_peak() == 0.0 and sim.aneurysm_sac.P_sac is None


def test_aneurysm_requires_flow_field_and_valid_per_env_config():
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
    with pytest.raises(ValueError, match="outside the vessel"):         # s_neck past the end (L2)
        NewtonGuidewireSim(vessel, 2.0, dev, flow=FlowField(),
                           aneurysm=Aneurysm(s_neck=200.0), device="cpu")
    with pytest.raises(ValueError, match="aneurysm length"):            # per-env config must align
        NewtonGuidewireSim(vessel, 2.0, dev, flow=FlowField(), n_envs=2,
                           aneurysm=[Aneurysm(s_neck=40.0)], device="cpu")
